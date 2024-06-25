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


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.weight = (
            weight  # Class weights, if required for addressing class imbalance
        )
        self.gamma = gamma  # Focusing parameter to adjust the rate at which easy examples are down-weighted
        self.reduction = reduction  # Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'

    def forward(self, inputs, targets):
        """
        Apply the Focal Loss
        :param inputs: Logits from the model (before applying softmax).
        :param targets: True class labels.
        :return: Computed focal loss.
        """
        # Apply log-softmax to convert logits to log-probabilities
        # log_probs = F.log_softmax(inputs, dim=-1)

        # Compute the negative log likelihood loss without reduction
        # Note: nll_loss expects log-probabilities
        NLL_loss = F.nll_loss(inputs, targets, reduction="none", weight=self.weight)

        # Convert NLL loss to probabilities (for focal modulation)
        pt = torch.exp(
            -NLL_loss
        )  # pt is the probability of being classified to the true class

        # Compute the focal loss
        F_loss = ((1 - pt) ** self.gamma) * NLL_loss

        if self.reduction == "mean":
            return torch.mean(F_loss)
        elif self.reduction == "sum":
            return torch.sum(F_loss)
        else:  # 'none'
            return F_loss


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
    parser.add_argument("--epoch", default=251, type=int, help="epoch to run")
    parser.add_argument(
        "--learning_rate", default=0.005, type=float, help="initial learning rate"
    )
    parser.add_argument("--gpu", type=str, default="0,1,2", help="specify GPU devices")
    parser.add_argument("--optimizer", type=str, default="Adam", help="Adam or SGD")
    parser.add_argument("--log_dir", type=str, default=None, help="log path")
    parser.add_argument("--decay_rate", type=float, default=1e-4, help="weight decay")
    parser.add_argument("--npoint", type=int, default=2048, help="point Number")
    parser.add_argument(
        "--normal", action="store_true", default=False, help="use normals"
    )
    parser.add_argument(
        "--shape_prior", action="store_true", default=False, help="use shape prior"
    )
    parser.add_argument(
        "--lang", action="store_true", default=False, help="use language encoder"
    )
    parser.add_argument(
        "--verbose", action="store_true", default=False, help="print accuracies"
    )
    parser.add_argument(
        "--focal", action="store_true", default=False, help="print accuracies"
    )
    parser.add_argument(
        "--step_size", type=int, default=20, help="decay step for lr decay"
    )
    parser.add_argument(
        "--lr_decay", type=float, default=0.5, help="decay rate for lr decay"
    )
    parser.add_argument("--data_name", type=str, default="fine", help="data_name")
    parser.add_argument("--loss", type=str, default="ce", help="data_name")
    parser.add_argument(
        "--temperature", type=int, default=10, help="decay step for lr decay"
    )
    return parser.parse_args()


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    batch_size = args.batch_size
    # if args.model == "pct_seg":
    #     print("changin batch size to 64 for pct_seg ")
    #     batch_size = 64
    # elif args.model == "curvenet_seg":
    #     print("changin batch size to 32 for curvenet_seg ")
    #     batch_size = 32

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Create output directory
    exp_dir = Path("./log2/")
    exp_dir.mkdir(exist_ok=True)
    shape_str = "_shape_prior" if args.shape_prior else "_no_shape"
    path = (
        "mat_seg2"
        + "_"
        + args.model
        + "_"
        + args.data_name
        + "_"
        + args.loss
        + shape_str
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

    root = "/ibex/project/c2106/Mahmoud/datasets/3D/hdf5/"
    # root = os.path.join(os.getcwd(), "data/" + args.data_name + "_grained/")

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

    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = args.step_size
    for subset in [None]:
        if subset is not None:
            filter_classes = json.load(
                open("metadata/{}_classes.json".format(subset))
            )
        else:
            filter_classes = None
        VAL_DATASET = CompatSeg(
            data_root=root,
            num_points=args.npoint,
            split="test",
            transform=None,
            semantic_level=args.data_name,
            filter_json=filter_classes,
            material=True,
        )

        valDataLoader = torch.utils.data.DataLoader(
            VAL_DATASET, batch_size=batch_size, shuffle=False, num_workers=3
        )
        print("*" * 40)
        print("Processing subset: {}".format(subset))
        best_acc = 0
        global_epoch = 0
        best_class_avg_iou = 0
        best_inctance_avg_iou = 0
        best_avg_iou_wihtout_shape = 0
        best_avg_iou_wihtout_shape = 0

        device = "cuda" if torch.cuda.is_available() else "cpu"
        clip_model, preprocess = clip.load("ViT-B/32", device=device)
        all_predictions = []

        with torch.no_grad():
            test_metrics = {}
            total_correct = 0
            total_seen = 0
            total_seen_class = [0 for i in range(num_part)]
            total_correct_class = [0 for i in range(num_part)]
            # print (filter_classes)
            if filter_classes is not None:
                shape_ious = {
                    str(cat): []
                    for cat in seg_classes.keys()
                    if int(cat) in filter_classes
                }
            else:
                shape_ious = {str(cat): [] for cat in seg_classes.keys()}
            # print(shape_ious)
            seg_label_to_cat = {}
            general_miou = []

            classifier = classifier.eval()

            for _, (points, label, _, target, _,_) in tqdm(
                enumerate(valDataLoader), total=len(valDataLoader), smoothing=0.9
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

                if args.lang:
                    with torch.no_grad():
                        text_inputs = clip.tokenize(parts_names).to(seg_pred.device)
                        text_emb = clip_model.encode_text(text_inputs).float()
                    # text_emb = classifier.module.lang(text_emb)
                    seg_pred = seg_feat.reshape(-1, seg_feat.shape[-1])
                    seg_pred = seg_pred / seg_pred.norm(dim=-1, keepdim=True)
                    text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
                    similarity = args.temperature * seg_pred @ text_emb.T
                    # cur_pred_val = similarity.max(1)[1]
                    # log_string(str(cur_pred_val.shape))
                    seg_pred = F.log_softmax(similarity, dim=-1)
                    # log_string(str(seg_pred.shape))

                cur_pred_val = seg_pred.cpu().data.numpy()
                cur_pred_val = np.argmax(cur_pred_val, -1)
                
                cur_pred_val = cur_pred_val.reshape(-1, NUM_POINT)
                # Store predictions
                all_predictions.append(cur_pred_val)
                target = target.cpu().data.numpy()

                correct = np.sum(cur_pred_val == target)
                total_correct += correct
                total_seen += cur_batch_size * NUM_POINT
                # print(target.shape, cur_pred_val.shape)
                for partk_k in range(num_part):
                    total_seen_class[partk_k] += np.sum(target == partk_k)
                    total_correct_class[partk_k] += np.sum(
                        (cur_pred_val == partk_k) & (target == partk_k)
                    )
                # import pdb; pdb.set_trace()
                # calculate the mIoU given shape prior knowledge and without it
                miou = compute_overall_iou(cur_pred_val, target, num_part)
                general_miou = general_miou + miou
                for i in range(cur_batch_size):
                    segp = cur_pred_val[i, :]
                    segl = target[i, :]
                    shape = str(label[i].item())
                    # print(shape, seg_classes)
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
            # import pdb; pdb.set_trace()
            test_metrics["accuracy"] = total_correct / float(total_seen)
            # import pdb; pdb.set_trace()
            # Create a mask to filter out positions where both arrays have '0' entries
            total_correct_class = np.array(total_correct_class)
            total_seen_class = np.array(total_seen_class, dtype=float)+1e-6
            mask = (total_correct_class != 0) | (total_seen_class != 0)
            # Filter arrays using the mask
            filtered_correct_class = total_correct_class[mask]

            filtered_seen_class = total_seen_class[mask].astype(float)
            test_metrics["class_avg_accuracy"] = np.mean(
                total_correct_class / total_seen_class
            )
            # if args.verbose:
            #     for cat in sorted(shape_ious.keys()):
            #         log_string(
            #             "eval mIoU of %s %f"
            #             % (cat + " " * (14 - len(cat)), shape_ious[cat])
            #         )
            test_metrics["class_avg_iou"] = mean_shape_ious
            test_metrics["inctance_avg_iou"] = np.mean(all_shape_ious)
            test_metrics["avg_iou_wihtout_shape"] = np.nanmean(general_miou)
            all_predictions = np.concatenate(all_predictions, axis=0)
            np.save('./predictions/mat/'+args.data_name+'_'+args.model+'_mat_pred.npy', all_predictions)  # Save as NumPy array
            print('saved results in:', './predictions/mat/'+args.data_name+'_'+args.model+'_mat_pred.npy')

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
