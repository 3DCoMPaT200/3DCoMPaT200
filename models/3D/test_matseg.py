"""
Train a part segmentation model.
"""

import argparse
import datetime
import importlib
import json
import logging
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import provider
import torch
from compat_loader import CompatLoader3D as CompatSeg
from compat_utils import compute_overall_iou, inplace_relu, to_categorical
from tqdm import tqdm
from timm.scheduler import CosineLRScheduler

import clip
import torch.nn.functional as F
import pudb
import torch
import torch.nn as nn

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, "models"))

seg_classes = {}



def parse_args():
    """
    Parse input arguments.
    """
    parser = argparse.ArgumentParser("Model")
    parser.add_argument(
        "--model", type=str, default="pointnet2_part_seg_ssg", help="model name"
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="batch Size during training"
    )
    parser.add_argument("--gpu", type=str, default="0,1,2", help="specify GPU devices")
    parser.add_argument("--log_dir", type=str, default=None, help="log path")
    parser.add_argument("--npoint", type=int, default=2048, help="point Number")
    parser.add_argument(
        "--normal", action="store_true", default=False, help="use normals"
    )
    parser.add_argument(
        "--shape_prior", action="store_true", default=False, help="use shape prior"
    )
    parser.add_argument(
        "--verbose", action="store_true", default=False, help="print accuracies"
    )
    parser.add_argument("--data_name", type=str, default="fine", help="data_name")
    return parser.parse_args()


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    batch_size = args.batch_size

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Create output directory
    exp_dir = Path("./log/")
    exp_dir.mkdir(exist_ok=True)
    path = (
        "mat_seg"
        + "/"
        + args.model
        + "_"
        + args.data_name
    )
    exp_dir = exp_dir.joinpath(path)
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath("checkpoints/")
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath("logs/")
    log_dir.mkdir(exist_ok=True)

    # Logging
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler = logging.FileHandler("%s/%s.txt" % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string("PARAMETER ...")
    log_string(args)

    root = os.path.join(os.getcwd(), "data/" + args.data_name + "_grained/")

    if args.data_name == "coarse":
        metadata = json.load(open("metadata/coarse_seg_classes.json"))
        meterial_names = json.load(open("metadata/materials.json"))
        parts_names = json.load(open("metadata/parts_coarse.json"))
    else:
        metadata = json.load(open("metadata/fine_seg_classes.json"))
        meterial_names = json.load(open("metadata/materials.json"))
        parts_names = json.load(open("metadata/parts_fine.json"))

    num_classes = metadata["num_classes"]
    num_part = metadata["num_part"]
    seg_classes = metadata["seg_classes"]
    print("obj_classes", num_classes)
    print("seg_classes", num_part)

    parts_ls = []
    for cat in seg_classes.keys():
        parts_ls = parts_ls + seg_classes[cat]
    print("part_classes", min(parts_ls), max(parts_ls))

    seg_label_to_cat = {}
    for cat in seg_classes.keys():
        for label in seg_classes[cat]:
            seg_label_to_cat[label] = cat

    # Model loading
            
    MODEL = importlib.import_module(args.model)
    shutil.copy("models/%s.py" % args.model, str(exp_dir))
    shutil.copy("models/%s.py" % args.model, str(exp_dir))

    shape_prior = args.shape_prior
    log_string("Shape Prior is:")
    log_string(shape_prior)
    if args.lang:
        num_feats = 512
    else:
        num_feats = 13
    classifier = MODEL.get_model(
        num_feats,
        shape_prior=shape_prior,
        num_shapes=200,
        normal_channel=True,
    ).cuda()

    # use DataParallel
    classifier = torch.nn.DataParallel(classifier)

    classifier.apply(inplace_relu)

    print(checkpoints_dir.joinpath("best_model.pth"))
    checkpoint = torch.load(checkpoints_dir.joinpath("best_model.pth"))

    start_epoch = checkpoint["epoch"]
    print("loading model")
    classifier.load_state_dict(checkpoint["model_state_dict"])

    log_string("Use pretrain model")

    filter_classes = None
    TEST_DATASET = CompatSeg(
        data_root=root,
        num_points=args.npoint,
        split="test",
        transform=None,
        semantic_level=args.data_name,
        filter_json=filter_classes,
        material=True,
    )

    testDataLoader = torch.utils.data.DataLoader(
        TEST_DATASET, batch_size=batch_size, shuffle=False, num_workers=3
    )

    all_predictions = []

    with torch.no_grad():
        test_metrics = {}
        total_correct = 0
        total_seen = 0
        total_seen_class = [0 for i in range(num_part)]
        total_correct_class = [0 for i in range(num_part)]
        shape_ious = {str(cat): [] for cat in seg_classes.keys()}
        seg_label_to_cat = {}
        general_miou = []

        classifier = classifier.eval()

        for _, (points, label, _, target, _,_) in tqdm(
            enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9
        ):
            cur_batch_size, NUM_POINT, _ = points.size()
            points, label, target = (
                points.float().cuda(),
                label.long().cuda(),
                target.long().cuda(),
            )
            points = points.transpose(2, 1)
            if shape_prior:
                seg_pred, _, seg_feat = classifier(
                    points, to_categorical(label, num_classes)
                )
            else:
                seg_pred, _, seg_feat = classifier(points)

            

            cur_pred_val = seg_pred.cpu().data.numpy()
            cur_pred_val = np.argmax(cur_pred_val, -1)
            
            cur_pred_val = cur_pred_val.reshape(-1, NUM_POINT)
            # Store predictions
            all_predictions.append(cur_pred_val)
            target = target.cpu().data.numpy()

            correct = np.sum(cur_pred_val == target)
            total_correct += correct
            total_seen += cur_batch_size * NUM_POINT
            for partk_k in range(num_part):
                total_seen_class[partk_k] += np.sum(target == partk_k)
                total_correct_class[partk_k] += np.sum(
                    (cur_pred_val == partk_k) & (target == partk_k)
                )
            # calculate the mIoU given shape prior knowledge and without it
            miou = compute_overall_iou(cur_pred_val, target, num_part)
            general_miou = general_miou + miou
            for i in range(cur_batch_size):
                segp = cur_pred_val[i, :]
                segl = target[i, :]
                shape = str(label[i].item())
                part_ious = {}
                for shape_k in seg_classes[shape]:
                    if (np.sum(segl == shape_k) == 0) and (
                        np.sum(segp == shape_k) == 0
                    ):  # part is not present, no prediction as well
                        part_ious[shape_k] = 1.0
                    else:
                        part_ious[shape_k] = np.sum(
                            (segl == shape_k) & (segp == shape_k)
                        ) / float(np.sum((segl == shape_k) | (segp == shape_k)))
                # Convert the dictionary to a list
                part_ious = list(part_ious.values())
                shape_ious[shape].append(np.mean(part_ious))

        all_shape_ious = []
        for cat in shape_ious.keys():
            for iou in shape_ious[cat]:
                all_shape_ious.append(iou)
            shape_ious[cat] = np.mean(shape_ious[cat])
        mean_shape_ious = np.mean(list(shape_ious.values()))
        test_metrics["accuracy"] = total_correct / float(total_seen)

        # Create a mask to filter out positions where both arrays have '0' entries
        total_correct_class = np.array(total_correct_class)
        total_seen_class = np.array(total_seen_class, dtype=float)+1e-6
        mask = (total_correct_class != 0) | (total_seen_class != 0)

        test_metrics["class_avg_accuracy"] = np.mean(
            total_correct_class / total_seen_class
        )
        test_metrics["class_avg_iou"] = mean_shape_ious
        test_metrics["inctance_avg_iou"] = np.mean(all_shape_ious)
        test_metrics["avg_iou_wihtout_shape"] = np.nanmean(general_miou)

        log_string("Test accuracy: %f " % (test_metrics["accuracy"] * 100))
        log_string(
            "Test class avg mIoU: %f " % (test_metrics["class_avg_iou"] * 100)
        )
        log_string(
            "Test instance avg mIoU: %f " % (test_metrics["inctance_avg_iou"] * 100)
        )
        log_string(
            "Test general mIoU: %f " % (test_metrics["avg_iou_wihtout_shape"] * 100)
        )


if __name__ == "__main__":
    args = parse_args()
    main(args)
