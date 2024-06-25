"""
Evaluate 3D shape classification.
"""

import argparse
import importlib
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
from compat_loader import CompatLoader3DCls as Compat
from tqdm import tqdm
import json
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import clip
import torch.nn.functional as F
import pudb
import torch
import torch.nn as nn

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, "models"))
# import clip
import torch.nn.functional as F
import pandas as pd

def plot_confusion_matrix(
    conf_matrix,
    classes,
    normalize=False,
    title="Confusion Matrix",
    cmap=plt.cm.Blues,
    save_path="confusion_matrix.png",
):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        conf_matrix = (
            conf_matrix.astype("float") / conf_matrix.sum(axis=1)[:, np.newaxis]
        )
    df_conf_matrix = pd.DataFrame(conf_matrix, index=classes, columns=classes)
    df_conf_matrix.to_csv('/home/ahmems0a/repos/3DCoMPaT200/models/3D/log2/cls/pointnet2_cls_msg_ce3_coarse_ce/confusion_matrix.csv')
    plt.figure(figsize=(50, 50))
    sns.heatmap(
        conf_matrix,
        annot=False,
        fmt=".2f" if normalize else "d",
        cmap=cmap,
        xticklabels=classes,
        yticklabels=classes,
    )
    plt.title(title)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()

    # Save the figure
    plt.savefig(os.path.join('/home/ahmems0a/repos/3DCoMPaT200/models/3D/log2/cls/pointnet2_cls_msg_ce3_coarse_ce/', save_path))
    # plt.show()


def parse_args():
    """
    Parse input arguments.
    """
    parser = argparse.ArgumentParser("Testing")
    parser.add_argument(
        "--use_cpu", action="store_true", default=False, help="use cpu mode"
    )
    parser.add_argument("--gpu", type=str, default="0", help="specify gpu device")
    parser.add_argument(
        "--batch_size", type=int, default=24, help="batch size in training"
    )
    parser.add_argument(
        "--num_category",
        default=42,
        type=int,
        choices=[10, 40],
        help="training on ModelNet10/40",
    )
    parser.add_argument(
        "--model",
        default="pointnet2_cls_msg",
        help="model name [default: pointnet_cls]",
    )
    parser.add_argument(
        "--lang", action="store_true", default=False, help="use language encoder"
    )
    parser.add_argument("--num_point", type=int, default=2048, help="Point Number")
    # parser.add_argument("--log_dir", type=str, required=True, help="Experiment root")
    parser.add_argument(
        "--use_normals", action="store_true", default=False, help="use normals"
    )
    parser.add_argument(
        "--use_uniform_sample",
        action="store_true",
        default=False,
        help="use uniform sampiling",
    )
    parser.add_argument(
        "--num_votes",
        type=int,
        default=1,
        help="Aggregate classification scores with voting",
    )
    parser.add_argument("--data_name", type=str, default="coarse", help="data_name")

    return parser.parse_args()


def test(args, model, loader, clip_model, classes, num_class=40, vote_num=1, subset=None):
    mean_correct = []
    classifier = model.eval()
    class_acc = np.zeros((num_class, 3))
    correct_preds_per_class = {}
    total_instances_per_class = {}
    all_preds = []
    all_targets = []
    class_counts = json.load(open("metadata/class_counts.json"))
    all_predictions = []

    for points, target in tqdm(loader, total=len(loader)):
        if not args.use_cpu:
            points, target = points.cuda(), target.cuda()

        points = points[:,:,:3]
        points = points.transpose(2, 1)
        vote_pool = torch.zeros(target.size()[0], num_class).cuda()

        for _ in range(vote_num):
            pred_logits, _, feat = classifier(points)
            if args.lang:
                with torch.no_grad():
                    text_inputs = clip.tokenize(classes).to(pred_logits.device)
                    text_emb = clip_model.encode_text(text_inputs).float()
                text_emb = classifier.module.lang(text_emb)
                pred_feat = feat.reshape(-1, feat.shape[-1])
                pred_feat = pred_feat / pred_feat.norm(dim=-1, keepdim=True)
                text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
                similarity = 10 * pred_feat @ text_emb.T  # .softmax(dim=-1)
                pred = F.log_softmax(similarity, dim=-1)
                # pass
            else:
                pred = F.log_softmax(pred_logits, -1)
            # print(pred.shape, vote_pool.shape)
            vote_pool += pred
        pred = vote_pool / vote_num
        pred_choice = pred.data.max(1)[1]

        all_preds.extend(pred_choice.view(-1).cpu().numpy())

        all_targets.extend(target.cpu().numpy())

        for cat in np.unique(target.cpu()):
            is_cat = target == cat
            if cat not in correct_preds_per_class:
                correct_preds_per_class[cat] = 0
                total_instances_per_class[cat] = 0
            correct_preds_per_class[cat] += (
                pred_choice[is_cat].eq(target[is_cat].long().data).cpu().sum().item()
            )
            total_instances_per_class[cat] += is_cat.sum().item()
        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item() / float(points.size()[0]))
    # print("Total instances per class: ", total_instances_per_class)
    class_acc_dict = {}
    class_acc = 0
    for key in correct_preds_per_class.keys():
        class_acc_dict[int(key)] = (correct_preds_per_class[key] / total_instances_per_class[key]) * 100
        class_acc += (correct_preds_per_class[key] / total_instances_per_class[key])
    print("Accuracy per class: ", class_acc_dict)
    # for i, acc in enumerate(class_acc):
    #     class_counts[str(i)] = acc * 100.0
    file_name = f"/{subset}_class_acc.json" if subset is not None else "/class_acc.json"
    # with open(log_dir+ file_name, "w") as f:
    #     json.dump(class_acc_dict, f)
    overall_class_acc = class_acc / len(correct_preds_per_class) # Mean accuracy over all classes
    instance_acc = np.mean(mean_correct)
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    np.save(args.model+'_cls_pred.npy', all_preds)  # Save as NumPy array
    # conf_matrix = confusion_matrix(
    #     all_targets, all_preds, labels=np.arange(num_class)
    # )

    # You can now return or print the confusion matrix
    # Define your class names or IDs (update according to your dataset)
    # class_names = [
    #     classes[i] for i in range(200)
    # ]  # Example for numeric class IDs

    # Now plot and save the confusion matrix
    # plot_confusion_matrix(
    #     conf_matrix,
    #     class_names,
    #     normalize=True,
    #     title="Normalized Confusion Matrix",
    #     save_path="confusion_matrix.png",
    # )
    return instance_acc, overall_class_acc


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    exp_dir = Path("./log2/")
    exp_dir.mkdir(exist_ok=True)
    lang_str = "_lang" if args.lang else ""
    shape_str = "_rgb" if args.use_normals else ""
    path = (
        "cls"
        + "/"
        + args.model
        + "_ce3"
        + "_"
        + args.data_name
        + "_"
        + "ce"
        + shape_str
        + lang_str
    )
    exp_dir = exp_dir.joinpath(path)
    log_dir = exp_dir.joinpath("logs/")
    checkpoints_dir = exp_dir.joinpath("checkpoints/")

    # Logging
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler = logging.FileHandler("%s/eval.txt" % log_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string("PARAMETER ...")
    log_string(args)

    # Dataloader
    log_string("Load dataset ...")
    root = "/ibex/project/c2106/Mahmoud/datasets/3D/hdf5"
    
    # Loading the models
    if args.lang:
        num_class = 512
    else:
        num_class = 200
    classes = json.load(open("classes.json"))
    # model_name = os.listdir(exp_dir + "/logs")[1].split(".")[0]
    model = importlib.import_module(args.model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # clip_model = None
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    if args.model == "curvenet_cls":
        classifier = model.get_model(num_class, npoints=args.num_point)
    else:
        classifier = model.get_model(num_class, False)
    if not args.use_cpu:
        classifier = classifier.cuda()
    classifier = torch.nn.DataParallel(classifier)
    checkpoint = torch.load(checkpoints_dir.joinpath("best_model.pth"))
    classifier.load_state_dict(checkpoint["model_state_dict"])
    for subset in [None]:
        if subset is not None:
            filter_classes = json.load(
                open("metadata/{}_classes.json".format(subset))
            )
        else:
            filter_classes = None
        # print(filter_classes)
        # VAL_DATASET = CompatSeg(
        #     data_root=root,
        #     num_points=args.npoint,
        #     split="test",
        #     transform=None,
        #     semantic_level=args.data_name,
        #     filter_json=filter_classes,
        # )

        # valDataLoader = torch.utils.data.DataLoader(
        #     VAL_DATASET, batch_size=batch_size, shuffle=False, num_workers=3
        # )
        print("*" * 40)
        print("Processing subset: {}".format(subset))
        TEST_DATASET = Compat(
            data_root=root,
            num_points=args.num_point,
            split="test",
            transform=None,
            semantic_level=args.data_name,
            filter_json=filter_classes,
        )
        testDataLoader = torch.utils.data.DataLoader(
            TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=3
        )
        # all_predictions = []

        with torch.no_grad():
            instance_acc_avg, class_acc_avg = 0.0, 0.0
            for _ in range(1):
                instance_acc, class_acc = test(
                    args,
                    classifier.eval(),
                    testDataLoader,
                    clip_model,
                    classes,
                    vote_num=args.num_votes,
                    num_class=200,
                    subset= subset
                )
                log_string(
                    "Test Instance Accuracy: %f, Class Accuracy: %f"
                    % (instance_acc * 100, class_acc * 100)
                )
                instance_acc_avg += instance_acc
                class_acc_avg += class_acc
            print("Running 5 times average: ", instance_acc_avg/1.0*100 , class_acc_avg/1.0*100 )


if __name__ == "__main__":
    args = parse_args()
    main(args)
