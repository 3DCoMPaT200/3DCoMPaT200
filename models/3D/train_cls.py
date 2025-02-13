"""
Training script for 3D classification.
"""

import argparse
import datetime
import importlib
import logging
import os
import shutil
import sys
from pathlib import Path
import json
import numpy as np
import provider
import torch
from compat_loader import CompatLoader3DCls as Compat
from tqdm import tqdm
import clip
import torch.nn.functional as F
import pudb
from timm.scheduler import CosineLRScheduler
from torch.utils.data import WeightedRandomSampler
import torch
import torch.nn as nn
import random
from torch.utils.data import Dataset

# import focal_loss
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from MixedPrioritizedSampler import MixedPrioritizedSampler
from sampler_utils import *





BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, "models"))



def parse_args():
    """
    Parse input arguments.
    """
    parser = argparse.ArgumentParser("training")
    parser.add_argument(
        "--use_cpu", action="store_true", default=False, help="use cpu mode"
    )
    parser.add_argument("--gpu", type=str, default="0", help="specify gpu device")
    parser.add_argument(
        "--batch_size", type=int, default=64, help="batch size in training"
    )
    parser.add_argument(
        "--model",
        default="pointnet2_cls_msg",
        help="model name [default: pointnet_cls]",
    )
    parser.add_argument("--dataset", default="3DCompat", help="3DCompat or ModelNet40")
    parser.add_argument(
        "--num_category",
        default=200,
        type=int,
        choices=[10, 40],
        help="training on ModelNet10/40",
    )
    parser.add_argument(
        "--epoch", default=200, type=int, help="number of epoch in training"
    )
    parser.add_argument(
        "--learning_rate", default=0.0005, type=float, help="learning rate in training"
    )
    parser.add_argument("--num_point", type=int, default=2048, help="Point Number")
    parser.add_argument(
        "--optimizer", type=str, default="Adam", help="optimizer for training"
    )
    parser.add_argument("--log_dir", type=str, default=None, help="experiment root")
    parser.add_argument("--decay_rate", type=float, default=1e-4, help="decay rate")
    parser.add_argument(
        "--use_normals", action="store_true", default=False, help="use normals"
    )
    parser.add_argument(
        "--process_data", action="store_true", default=False, help="save data offline"
    )
    parser.add_argument("--data_name", type=str, default="coarse", help="data_name")
    parser.add_argument("--loss", type=str, default="ce", help="loss")

    return parser.parse_args()


def inplace_relu(m):
    """
    Set all ReLU layers in a model to be inplace.
    """
    classname = m.__class__.__name__
    if classname.find("ReLU") != -1:
        m.inplace = True


import pandas as pd


def test(
    epoch,
    model,
    loader,
    classes,
    clip_model,
    num_class=200,
    plot=False,
    class_accuracy_df=None,
):
    # create model and move it to GPU with id rank
    rank = dist.get_rank()
    device_id = rank % torch.cuda.device_count()
    mean_correct = []
    class_acc = np.zeros((num_class, 3))
    classifier = model.eval()
    all_preds = []
    all_targets = []
    for _, (points, target) in tqdm(enumerate(loader), total=len(loader)):
        if not args.use_cpu:
            points, target = points.to(
                "cuda:" + str(device_id)
            ).contiguous(), target.to("cuda:" + str(device_id))
            if not args.use_normals:
                points = points[:, :, :3]
        pred_logits, _, _ = classifier(points, noaug=True)
        pred = F.log_softmax(pred_logits, -1)
        pred_choice = pred.data.max(1)[1]
        all_preds.extend(pred_choice.view(-1).cpu().numpy())
        all_targets.extend(target.cpu().numpy())
        for cat in np.unique(target.cpu()):
            classacc = (
                pred_choice[target == cat]
                .eq(target[target == cat].long().data)
                .cpu()
                .sum()
            )
            class_acc[cat, 0] += classacc.item() / float(
                points[target == cat].size()[0]
            )
            class_acc[cat, 1] += 1

        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item() / float(points.size()[0]))
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
    class_accuracy_series = pd.Series(
        class_acc[:, 2], index=[f"class_{i}" for i in range(num_class)]
    )
    class_accuracy_series["epoch"] = epoch

    print("saving df")
    class_accuracy_df = pd.concat(
        [class_accuracy_df, class_accuracy_series.to_frame().T], ignore_index=True
    )
    print("saved df")
    class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
    class_acc = np.mean(class_acc[:, 2])
    instance_acc = np.mean(mean_correct)

    return instance_acc, class_acc, class_accuracy_df


LOG_DIR = ""


def main(args):
    def log_string(str):
        logger.info(str)
        if dist.get_rank() == 0:
            print(str)

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    if dist.get_rank() == 0:
        print(f"Start running basic DDP example on rank {rank}.")

    # create model and move it to GPU with id rank
    rank = dist.get_rank()
    device_id = rank % torch.cuda.device_count()

    """HYPER PARAMETER"""
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Create output directory
    exp_dir = Path("./log/")
    exp_dir.mkdir(exist_ok=True)
    path = (
        "cls"
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
    if dist.get_rank() == 0:
        log_string("PARAMETER ...")
        log_string(args)

    # Dataloader
    log_string("Load dataset ...")
    root = "./data/"
    TRAIN_DATASET = Compat(
        data_root=root,
        num_points=args.num_point,
        split="train",
        transform=None,
        semantic_level=args.data_name,
    )

    VAL_DATASET = Compat(
        data_root=root,
        num_points=args.num_point,
        split="valid",
        transform=None,
        semantic_level=args.data_name,
    )



    trainDataLoader = torch.utils.data.DataLoader(
        TRAIN_DATASET,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=3,
        drop_last=True,
    )

    valDataLoader = torch.utils.data.DataLoader(
        VAL_DATASET,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=3,
    )
    log_string("The number of training data is: %d" % len(TRAIN_DATASET))
    log_string("The number of validation data is: %d" % len(VAL_DATASET))

    if args.data_name == "coarse":
        metadata = json.load(open("metadata/coarse_seg_classes.json"))
        classes = json.load(open("classes.json"))
        meterials = json.load(open("metadata/materials.json"))
        parts_ls = json.load(open("metadata/parts_coarse.json"))
    else:
        metadata = json.load(open("metadata/fine_seg_classes.json"))
        classes = json.load(open("classes.json"))
        meterials = json.load(open("metadata/materials.json"))
        parts_ls = json.load(open("metadata/parts_fine.json"))

    num_class = metadata["num_classes"]
    model = importlib.import_module(args.model)
    shutil.copy("./models/%s.py" % args.model, str(exp_dir))
    shutil.copy("models/pointnet2_utils.py", str(exp_dir))
    shutil.copy("./train_cls.py", str(exp_dir))

    output_dim = 200
    if args.model == "curvenet_cls":
        classifier = model.get_model(output_dim, npoints=args.num_point)
    else:
        classifier = model.get_model(output_dim, normal_channel=args.use_normals)
    criterion = model.get_loss()

    classifier.apply(inplace_relu)
    if not args.use_cpu:
        classifier = classifier.to("cuda:" + str(device_id))
        criterion = criterion.to("cuda:" + str(device_id))

    classifier = classifier.to(device_id)
    classifier = DDP(classifier, device_ids=[device_id])

    start_epoch = 0
    if args.optimizer == "Adam":
        optimizer = torch.optim.AdamW(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate,
        )
    else:
        optimizer = torch.optim.SGD(
            classifier.parameters(),
            lr=args.learning_rate * 100,
            momentum=0.9,
            weight_decay=args.decay_rate,
        )

    if args.optimizer == "Adam":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)

    elif args.optimizer == "SGD":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, args.epoch, eta_min=args.learning_rate
        )

    global_epoch = 0
    global_step = 0
    best_instance_acc = 0.0
    best_class_acc = 0.0
    if dist.get_rank() == 0:
        class_accuracy_df = pd.DataFrame()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Training
    if dist.get_rank() == 0:
        logger.info("Start training...")
    for epoch in range(start_epoch, args.epoch):
        if dist.get_rank() == 0:
            log_string("Epoch %d (%d/%s):" % (global_epoch + 1, epoch + 1, args.epoch))
        mean_correct = []
        mean_loss = []
        classifier = classifier.train()

        for _, (points, target) in tqdm(
            enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9
        ):
            optimizer.zero_grad()
            points = points.data.numpy()
            points = provider.random_point_dropout(points)
            points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
            points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
            points = torch.Tensor(points)
            if not args.use_normals:
                points = points[:, :, :3]
            points = points.transpose(2, 1)

            if not args.use_cpu:
                points, target = points.to(
                    "cuda:" + str(device_id)
                ).contiguous(), target.to("cuda:" + str(device_id))
            pred_logits, trans_feat, feat = classifier(points)
            pred = F.log_softmax(pred_logits, -1)
            loss = criterion(pred, target.long())
            mean_loss.append(loss.item())

            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item() / float(points.size()[0]))
            mean_loss.append(loss.item())
            loss.backward()
            optimizer.step()
            global_step += 1
            
        try:
            scheduler.step()
        except:
            scheduler.step(epoch)

        train_instance_acc = np.mean(mean_correct)
        train_instance_loss = np.mean(mean_loss)
        if not args.pretrain:
            if dist.get_rank() == 0:
                log_string("Train Instance Accuracy: %f" % train_instance_acc)
        if dist.get_rank() == 0:
            log_string("Train loss: %f" % train_instance_loss)
        if dist.get_rank() == 0:
            with torch.no_grad():
                if epoch % 10 == 0:
                    plot = True
                else:
                    plot = False
                instance_acc, class_acc, class_accuracy_df = test(
                    epoch,
                    classifier.eval(),
                    valDataLoader,
                    num_class=num_class,
                    classes=classes,
                    plot=plot,
                    class_accuracy_df=class_accuracy_df,
                )

                if class_acc >= best_class_acc:
                    best_instance_acc = instance_acc
                    best_epoch = epoch + 1
                    best_class_acc = class_acc

                # if class_acc >= best_class_acc:
                log_string(
                    "Validation Instance Accuracy: %f, Class Accuracy: %f"
                    % (instance_acc, class_acc)
                )
                log_string(
                    "Best Instance Accuracy: %f, Class Accuracy: %f"
                    % (best_instance_acc, best_class_acc)
                )

                if class_acc >= best_class_acc:
                    logger.info("Save model... %f" % (class_acc))
                    savepath = str(checkpoints_dir) + "/best_model.pth"
                    log_string("Saving at %s" % savepath)
                    state = {
                        "epoch": best_epoch,
                        "instance_acc": instance_acc,
                        "class_acc": class_acc,
                        "model_state_dict": classifier.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                    }
                    torch.save(state, savepath)
            class_accuracy_df.to_csv(exp_dir.joinpath("class_accuracy.csv"))
        global_epoch += 1

    logger.info("End of training...")
    if dist.get_rank() == 0:
        log_string("Best Instance Accuracy: %f" % best_instance_acc)
        log_string("Best Class Accuracy: %f" % best_class_acc)
        class_accuracy_df.to_csv(exp_dir.joinpath("class_accuracy.csv"))
        logger.info("Save class accuracy...")


if __name__ == "__main__":
    args = parse_args()
    main(args)
