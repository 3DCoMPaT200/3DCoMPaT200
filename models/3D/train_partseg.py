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
import pandas as pd
import random

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, "models"))

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DistributedSampler

from MixedPrioritizedSampler import MixedPrioritizedSampler


seg_classes = {}


class Mix3DDataset(Dataset):
    def __init__(self, original_dataset):
        self.original_dataset = original_dataset

    def mix3d(self, pc1, pc2):
        mixed_pc = np.concatenate([pc1, pc2], axis=0)
        return mixed_pc

    def __getitem__(self, index):
        pc1, label1, seg1, _ = self.original_dataset[index]
        random_index = random.randint(0, len(self.original_dataset) - 1)
        pc2, label2, seg2, _ = self.original_dataset[random_index]
        mixed_pc = self.mix3d(pc1, pc2)
        seg = np.concatenate([seg1, seg2], axis=0)
        return mixed_pc, label1, seg, label2

    def __len__(self):
        return len(self.original_dataset)


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
        "--batch_size", type=int, default=128, help="batch Size during training"
    )
    parser.add_argument("--epoch", default=51, type=int, help="epoch to run")
    parser.add_argument(
        "--lr", default=0.001, type=float, help="initial learning rate"
    )
    parser.add_argument(
        "--gpu", type=str, default="0", help="specify GPU devices"
    )
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
        "--mix", action="store_true", default=False, help="Use Mix3D Technique"
    )
    parser.add_argument(
        "--mps",
        action="store_true",
        default=False,
        help="Use Mixed Priority Sampling Technique",
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


def hubless_loss(cls_scores, num_classes):
    # xp_yall_prob   (batch_size, num_classes)
    # xp_yall_prob.T (num_classes, batch_size
    # xp_yall_prob.expand(0, 1, -1, 1)
    # xp_yall_probT_average_reshape = xp_yall_probT_reshaped.mean(axis=2)
    # hubness_dist = xp_yall_probT_average_reshape - hubness_blob
    # hubness_dist_sqr = hubness_dist.pow(2)
    # hubness_dist_sqr_scaled = hubness_dist_sqr * cfg.TRAIN.HUBNESS_SCALE
    cls_scores = F.softmax(cls_scores, dim=1)
    # print(torch.max(cls_scores), torch.min(cls_scores))
    hubness_blob = 1.0 / num_classes
    cls_scores_T = cls_scores.transpose(0, 1)
    # cls_scores_T = cls_scores_T.unsqueeze(1).unsqueeze(3).expand(-1, 1, -1, 1)
    cls_scores_T = cls_scores_T.mean(dim=1, keepdim=True)
    hubness_dist = cls_scores_T - hubness_blob
    hubness_dist = hubness_dist.pow(2) * 1000
    hubless_loss = hubness_dist.mean()
    return hubless_loss


def vl_contrastive_loss(image_feat, text_feat, temperature=1):
    image_feat = F.normalize(image_feat, dim=-1)
    text_feat = F.normalize(text_feat, dim=-1)

    logits = torch.matmul(image_feat, text_feat.t())
    logit_scale = temperature.exp().clamp(max=100)

    gt = torch.arange(logits.shape[0], device=logits.device)
    loss1 = F.cross_entropy(logit_scale * logits, gt)
    loss2 = F.cross_entropy(logit_scale * logits.t(), gt)
    return (loss1 + loss2) / 2  # scale it up by the number of GPUs


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    dist.init_process_group("nccl")
    rank = dist.get_rank()

    # create model and move it to GPU with id rank
    rank = dist.get_rank()
    device_id = rank % torch.cuda.device_count()

    batch_size = args.batch_size
    if args.model == "pct_seg":
        batch_size = 64
    elif args.model == "curvenet_seg":
        batch_size = 32

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Create output directory
    exp_dir = Path("./log2/")
    exp_dir.mkdir(exist_ok=True)
    shape_str = "_shape_prior" if args.shape_prior else "_no_shape"
    lang_str = "_lang" if args.lang else ""
    focal_str = "_focal" if args.focal else ""
    mps_str = "_mps" if args.mps else ""
    path = (
        "part_seg"
        + "/"
        + args.model
        + "_ce_test"
        + "_"
        + args.data_name
        + "_"
        + args.loss
        + shape_str
        + lang_str
        + focal_str
        + mps_str
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
        log_string(args)

    root = "/ibex/project/c2106/Mahmoud/datasets/3D/hdf5/"
    # root = os.path.join(os.getcwd(), "data/" + args.data_name + "_grained/")
    TRAIN_DATASET = CompatSeg(
        data_root=root,
        num_points=args.npoint,
        split="train",
        transform=None,
        semantic_level=args.data_name,
    )
    VAL_DATASET = CompatSeg(
        data_root=root,
        num_points=args.npoint,
        split="valid",
        transform=None,
        semantic_level=args.data_name,
    )
    if args.data_name == "fine":
        part_weights = json.load(
            open("/home/ahmems0a/repos/3DCoMPaT200/metadata/part_fine_weight.json")
        )
    else:
        part_weights = json.load(
            open(
                "/home/ahmems0a/repos/3DCoMPaT200/metadata/part_coarse_weight.json"
            )
        )
    part_weights = torch.Tensor(np.array(list(part_weights.values())))
    part_weights = part_weights.to(device_id)
    print(part_weights.shape)
    # train_sampler = DistributedSampler(
    #     TRAIN_DATASET, num_replicas=torch.cuda.device_count(), rank=rank
    # )
    train_sampler = MixedPrioritizedSampler(
        TRAIN_DATASET,
        epochs=args.epoch // 4,
        lam=1.0,
    )
    if args.mix:
        TRAIN_DATASET = Mix3DDataset(TRAIN_DATASET)
    if args.mps:
        trainDataLoader = torch.utils.data.DataLoader(
            TRAIN_DATASET,
            batch_size=batch_size,
            shuffle=False,
            num_workers=3,
            drop_last=True,
            sampler=train_sampler,
        )
    else:
        trainDataLoader = torch.utils.data.DataLoader(
            TRAIN_DATASET,
            batch_size=batch_size,
            shuffle=True,
            num_workers=3,
            drop_last=True,
        )
    valDataLoader = torch.utils.data.DataLoader(
        VAL_DATASET, batch_size=batch_size, shuffle=False, num_workers=3
    )
    log_string("The number of training data is: %d" % len(TRAIN_DATASET))
    log_string("The number of validation data is: %d" % len(VAL_DATASET))

    if args.data_name == "coarse":
        metadata = json.load(open("metadata/coarse_seg_classes.json"))
        meterial_names = json.load(open("metadata/materials.json"))
        parts_names = json.load(open("metadata/parts_coarse.json"))
    else:
        metadata = json.load(open("metadata/fine_seg_classes.json"))
        meterial_names = json.load(open("metadata/materials.json"))
        parts_names = json.load(open("metadata/parts_fine.json"))

    parts_names = [name.replace("_", " ") for name in parts_names]

    num_classes = metadata["num_classes"]
    num_part = metadata["num_part"]
    seg_classes = metadata["seg_classes"]
    if dist.get_rank() == 0:
        print("obj_classes", num_classes)
        print("seg_classes", num_part)

    parts_ls = []
    for cat in seg_classes.keys():
        parts_ls = parts_ls + seg_classes[cat]
    if dist.get_rank() == 0:
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
    if dist.get_rank() == 0:

        log_string("Shape Prior is:")
        log_string(shape_prior)
    if args.lang:
        num_feats = 512
    else:
        num_feats = num_part
    classifier = MODEL.get_model(
        num_feats,
        shape_prior=shape_prior,
        num_shapes=metadata["num_classes"],
        normal_channel=args.normal,
    ).to(device_id)

    # use DataParallel
    classifier = classifier.to(device_id)
    classifier = DDP(classifier, device_ids=[device_id])
    if args.loss == "ce":
        criterion = MODEL.get_loss().to(device_id)
    elif args.loss == "focal":
        # print("using focal loss")
        criterion = FocalLoss(gamma=4).to(device_id)
    elif args.loss == "hubless":
        # criterion = torch.hub.load(
        #     "adeelh/pytorch-multi-class-focal-loss",
        #     model="FocalLoss",
        #     gamma=2,
        #     reduction="mean",
        #     force_reload=False,
        # )
        if args.focal:
            criterion = FocalLoss(gamma=4).to(device_id)
        else:
            criterion = MODEL.get_loss().to(device_id)

    classifier.apply(inplace_relu)
    # log_string(classifier)

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find("Conv2d") != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find("Linear") != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.0)

    # try:
    # if args.data_name == 'fine':
    # checkpoint = torch.load(
    #     "/ibex/project/c2106/part_seg/pointnet2_coarse_N2048_tem10_noprior/checkpoints/best_model.pth"
    # )
    # # else:
    # #     checkpoint = torch.load("/ibex/project/c2106/part_seg/pointnet2_coarse_N2048_tem10/checkpoints/best_model.pth")
    # # start_epoch = checkpoint["epoch"]
    # # print('loading model')
    # # classifier.load_state_dict(checkpoint["model_state_dict"])
    # model_state_dict = classifier.state_dict()
    # filtered_state_dict = {
    #     k: v
    #     for k, v in checkpoint["model_state_dict"].items()
    #     if k in model_state_dict and model_state_dict[k].size() == v.size()
    # }

    # # # # Load the filtered state dict
    # classifier.load_state_dict(filtered_state_dict, strict=False)
    # log_string("Use pretrain model")
    # except:
    #     log_string("No existing model, starting training from scratch...")
    start_epoch = 0
    classifier = classifier.apply(weights_init)
    if args.lang:
        print("Using Language Encoder")
        classifier.module.lang.reset_parameters()
        torch.nn.init.eye_(classifier.module.lang.weight)
        # Set biases to zero
        if classifier.module.lang.bias is not None:
            torch.nn.init.zeros_(classifier.module.lang.bias)

    if args.optimizer == "Adam":
        optimizer = torch.optim.AdamW(
            classifier.parameters(),
            lr=args.lr,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate,
        )
    else:
        optimizer = torch.optim.SGD(
            classifier.parameters(), lr=args.lr * 100, momentum=0.9
        )

    def bn_momentum_adjust(m, momentum):
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum

    if args.optimizer == "Adam":
        # scheduler = scheduler = CosineLRScheduler(
        #     optimizer,
        #     t_initial=args.epoch,
        #     lr_min=1e-6,
        #     warmup_lr_init=1e-6,
        #     warmup_t=10,
        #     cycle_limit=1,
        #     t_in_epochs=True,
        # )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    elif args.optimizer == "SGD":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, args.epoch, eta_min=args.lr
        )

    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = args.step_size

    best_acc = 0
    global_epoch = 0
    best_class_avg_iou = 0
    best_inctance_avg_iou = 0
    best_avg_iou_wihtout_shape = 0
    best_avg_iou_wihtout_shape = 0

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, preprocess = clip.load("ViT-B/32", device=device_id)
    if dist.get_rank() == 0:
        class_accuracy_df = pd.DataFrame()
    for epoch in range(start_epoch, args.epoch):
        mean_correct = []
        if dist.get_rank() == 0:
            log_string("Epoch %d (%d/%s):" % (global_epoch + 1, epoch + 1, args.epoch))
            log_string("Learning rate:%f" % scheduler.get_last_lr()[0])
        momentum = MOMENTUM_ORIGINAL * (
            MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP)
        )
        if momentum < 0.01:
            momentum = 0.01
        if dist.get_rank() == 0:
            print("BN momentum updated to: %f" % momentum)
        classifier = classifier.apply(lambda x: bn_momentum_adjust(x, momentum))
        classifier = classifier.train()

        # '''learning one epoch'''
        for _, (points, label, target, _) in tqdm(
            enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9
        ):
            optimizer.zero_grad()
            cur_batch_size, NUM_POINT, _ = points.size()
            unique_classes, counts = label.unique(return_counts=True)

            # Convert to a Python dict for easy viewing/usage
            frequency_dict = {
                int(class_label): int(count)
                for class_label, count in zip(unique_classes, counts)
            }
            # print(frequency_dict)
            # print(points.shape)
            points = points.data.numpy()
            points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
            points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
            points = torch.Tensor(points)
            points, label, target = (
                points.float().to(device_id),
                label.long().to(device_id),
                target.long().to(device_id),
            )
            points = points.transpose(2, 1)
            # print(points.shape)
            if shape_prior:
                seg_pred, trans_feat, seg_feat = classifier(
                    points, to_categorical(label, num_classes)
                )
            else:
                seg_pred, trans_feat, seg_feat = classifier(points)
            # print(seg_pred.shape, seg_feat.shape)
            # import pdb; pdb.set_trace()
            if args.lang:
                # print(seg_pred.shape, seg_feat.shape)
                with torch.no_grad():
                    text_inputs = clip.tokenize(parts_names).to(seg_pred.device)
                    text_emb = clip_model.encode_text(text_inputs).float()
                text_emb = classifier.module.lang(text_emb)
                seg_pred = seg_feat.reshape(-1, seg_feat.shape[-1])
                # loss = vl_contrastive_loss(seg_pred, text_emb, temperature=torch.full((num_part,), args.temperature).to(device_id))
                seg_pred = seg_pred / seg_pred.norm(dim=-1, keepdim=True)
                text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
                similarity = args.temperature * seg_pred @ text_emb.T
                # cur_pred_val = similarity.max(1)[1]
                # log_string(str(cur_pred_val.shape))
                seg_pred = F.log_softmax(similarity, dim=-1)
                # log_string(str(seg_pred.shape))
            
            target = target.view(-1, 1)[:, 0]
            # print(seg_pred.shape, target.shape)
            # import pdb; pdb.set_trace()
            seg_pred = seg_pred.contiguous().view(-1, num_part)

            pred_choice = seg_pred.data.max(1)[1]
            
            # print(torch.unique(pred_choice))
            # print(torch.unique(target))
            # print(num_part)
            # part_weights = None
            if args.loss == "hubless":
                loss = criterion(seg_pred, target, part_weights)
                loss += hubless_loss(similarity, num_part)
            else:

                loss = criterion(seg_pred, target, part_weights)
            # calculate acc
            correct = pred_choice.eq(target.data).cpu().sum()
            mean_correct.append(correct.item() / (batch_size * args.npoint))

            loss.backward()
            optimizer.step()

        train_instance_acc = np.mean(mean_correct)
        if dist.get_rank() == 0:
            log_string("Train accuracy is: %.5f" % train_instance_acc)
            log_string("Train loss is: %.5f" % loss.item())
        try:
            scheduler.step()
        except:
            scheduler.step(epoch)
        if dist.get_rank() == 0:
            with torch.no_grad():
                test_metrics = {}
                total_correct = 0
                total_seen = 0
                total_seen_class = [0 for _ in range(num_part)]
                total_correct_class = [0 for _ in range(num_part)]
                shape_ious = {cat: [] for cat in seg_classes.keys()}
                seg_label_to_cat = {}
                general_miou = []

                classifier = classifier.eval()

                for _, (points, label, target, _) in tqdm(
                    enumerate(valDataLoader), total=len(valDataLoader), smoothing=0.9
                ):
                    cur_batch_size, NUM_POINT, _ = points.size()
                    points, label, target = (
                        points.float().to(device_id),
                        label.long().to(device_id),
                        target.long().to(device_id),
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
                        text_emb = classifier.module.lang(text_emb)
                        seg_pred = seg_feat.reshape(-1, seg_feat.shape[-1])
                        seg_pred = seg_pred / seg_pred.norm(dim=-1, keepdim=True)
                        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
                        similarity = args.temperature * seg_pred @ text_emb.T
                        # cur_pred_val = similarity.max(1)[1]
                        # log_string(str(cur_pred_val.shape))
                        seg_pred = F.log_softmax(similarity, dim=-1)
                        log_string(str(seg_pred.shape))

                    cur_pred_val = seg_pred.cpu().data.numpy()
                    cur_pred_val = np.argmax(cur_pred_val, -1)
                    cur_pred_val = cur_pred_val.reshape(-1, NUM_POINT)
                    target = target.cpu().data.numpy()
                    # print(target, cur_pred_val)
                    # import pdb; pdb.set_trace()
                    correct = np.sum(cur_pred_val == target)
                    total_correct += correct
                    total_seen += cur_batch_size * NUM_POINT
                    # print(target.shape, cur_pred_val.shape)
                    for partk_k in range(num_part):
                        total_seen_class[partk_k] += np.sum(target == partk_k) + 1e-9
                        total_correct_class[partk_k] += np.sum(
                            (cur_pred_val == partk_k) & (target == partk_k)
                        )
                    # pdb.set_trace()
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
                # print(shape_ious.keys())
                class_accuracy_series = pd.Series(shape_ious)
                class_accuracy_series["epoch"] = epoch
                # class_accuracy_df = class_accuracy_df.append(
                #     class_accuracy_series, ignore_index=True
                # )
                # print("saving df")
                class_accuracy_df = pd.concat(
                    [class_accuracy_df, class_accuracy_series.to_frame().T],
                    ignore_index=True,
                )
                mean_shape_ious = np.mean(list(shape_ious.values()))
                test_metrics["accuracy"] = total_correct / float(total_seen)
                test_metrics["class_avg_accuracy"] = np.mean(
                    np.array(total_correct_class)
                    / np.array(total_seen_class, dtype=float)
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

            log_string(
                "Epoch %d validation class avg mIoU: %f "
                % (epoch + 1, test_metrics["class_avg_iou"])
            )
            if test_metrics["class_avg_iou"] >= best_class_avg_iou:
                logger.info("Save model...")
                savepath = str(checkpoints_dir) + "/best_model.pth"
                log_string("Saving at %s" % savepath)
                state = {
                    "epoch": epoch,
                    "train_acc": train_instance_acc,
                    "val_acc": test_metrics["accuracy"],
                    "class_avg_iou": test_metrics["class_avg_iou"],
                    "inctance_avg_iou": test_metrics["inctance_avg_iou"],
                    "avg_iou_wihtout_shape": test_metrics["avg_iou_wihtout_shape"],
                    "model_state_dict": classifier.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                }
                torch.save(state, savepath)
                log_string("Saving model....")

            if test_metrics["class_avg_iou"] > best_class_avg_iou:
                best_inctance_avg_iou = test_metrics["inctance_avg_iou"]
                best_class_avg_iou = test_metrics["class_avg_iou"]
                best_avg_iou_wihtout_shape = test_metrics["avg_iou_wihtout_shape"]
                best_acc = test_metrics["accuracy"]

            log_string("Best accuracy is: %.5f" % best_acc)
            log_string("Best class avg mIOU is: %.5f" % best_class_avg_iou)
            log_string("Best inctance avg mIOU is: %.5f" % best_inctance_avg_iou)
            log_string("Best general avg mIOU is: %.5f" % best_avg_iou_wihtout_shape)
            class_accuracy_df.to_csv(exp_dir.joinpath("class_mIoU.csv"))
            global_epoch += 1


if __name__ == "__main__":
    args = parse_args()
    main(args)
