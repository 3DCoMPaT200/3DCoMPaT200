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
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import clip
import torch.nn.functional as F

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
        "--batch_size", type=int, default=128, help="batch Size during training"
    )
    parser.add_argument("--epoch", default=251, type=int, help="epoch to run")
    parser.add_argument(
        "--learning_rate", default=0.001, type=float, help="initial learning rate"
    )
    parser.add_argument("--gpu", type=str, default="0", help="specify GPU devices")
    parser.add_argument("--optimizer", type=str, default="Adam", help="Adam or SGD")
    parser.add_argument("--log_dir", type=str, default=None, help="log path")
    parser.add_argument("--decay_rate", type=float, default=1e-4, help="weight decay")
    parser.add_argument("--npoint", type=int, default=2048, help="point Number")
    parser.add_argument(
        "--normal", action="store_true", default=True, help="use normals"
    )
    parser.add_argument(
        "--shape_prior", action="store_true", default=False, help="use shape prior"
    )
    parser.add_argument(
        "--step_size", type=int, default=20, help="decay step for lr decay"
    )
    parser.add_argument(
        "--lr_decay", type=float, default=0.5, help="decay rate for lr decay"
    )
    parser.add_argument("--data_name", type=str, default="fine", help="data_name")

    return parser.parse_args()





def main(args):
    def log_string(str):
        logger.info(str)
        print(str)
    dist.init_process_group("nccl")
    rank = dist.get_rank()

    # create model and move it to GPU with id rank
    rank = dist.get_rank()
    device_id = rank % torch.cuda.device_count()
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
    TRAIN_DATASET = CompatSeg(
        data_root=root,
        num_points=args.npoint,
        split="train",
        transform=None,
        semantic_level=args.data_name,
        material=True,
    )
    VAL_DATASET = CompatSeg(
        data_root=root,
        num_points=args.npoint,
        split="valid",
        transform=None,
        semantic_level=args.data_name,
        material=True,
    )

    trainDataLoader = torch.utils.data.DataLoader(
        TRAIN_DATASET,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=3,
        drop_last=True,
    )
    valDataLoader = torch.utils.data.DataLoader(
        VAL_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=10
    )
    log_string("The number of training data is: %d" % len(TRAIN_DATASET))
    log_string("The number of validation data is: %d" % len(VAL_DATASET))

    if args.data_name == "coarse":
        metadata = json.load(open("metadata/coarse_mat_seg_classes.json"))
        meterials = json.load(open("metadata/materials.json"))
        parts_ls = json.load(open("metadata/parts_coarse.json"))
    else:
        metadata = json.load(open("metadata/fine_mat_seg_classes.json"))
        meterials = json.load(open("metadata/materials.json"))
        parts_ls = json.load(open("metadata/parts_fine.json"))

    num_classes = metadata["num_classes"]
    num_part = metadata["num_mat"]
    seg_classes = metadata["seg_mat_classes"]
    if dist.get_rank() == 0:
        print("obj_classes", num_classes)
        print("seg_classes", num_part)
    parts_ls = []
    for i in seg_classes.keys():
        parts_ls = parts_ls + seg_classes[i]
    if dist.get_rank() == 0:
        print("seg_classes", min(parts_ls), max(parts_ls))
    seg_label_to_cat = {}
    for cat in seg_classes.keys():
        for label in seg_classes[cat]:
            seg_label_to_cat[label] = cat

    # Model loading
    MODEL = importlib.import_module(args.model)
    shutil.copy("models/%s.py" % args.model, str(exp_dir))
    shutil.copy("models/pointnet2_utils.py", str(exp_dir))

    shape_prior = args.shape_prior
    if dist.get_rank() == 0:
        log_string("Shape Prior is:")
        log_string(shape_prior)

    classifier = MODEL.get_model(
        num_part,
        shape_prior=shape_prior,
        num_shapes=metadata["num_classes"],
        normal_channel=args.normal,
    ).to(device_id)

    # use DataParallel
    classifier = classifier.to(device_id)
    classifier = DDP(classifier, device_ids=[device_id])

    criterion = MODEL.get_loss().to(device_id)


    classifier.apply(inplace_relu)

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

    try:
        checkpoint = torch.load(str(exp_dir) + "/checkpoints/best_model.pth")
        start_epoch = checkpoint["epoch"]
        classifier.load_state_dict(checkpoint["model_state_dict"])
        log_string("Use pretrain model")
    except:
        log_string("No existing model, starting training from scratch...")
    start_epoch = 0
    classifier = classifier.apply(weights_init)

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
            classifier.parameters(), lr=args.learning_rate * 100, momentum=0.9
        )

    def bn_momentum_adjust(m, momentum):
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum

    if args.optimizer == "Adam":
        scheduler = scheduler = CosineLRScheduler(
            optimizer,
            t_initial=args.epoch,
            lr_min=1e-6,
            warmup_lr_init=1e-6,
            warmup_t=10,
            cycle_limit=1,
            t_in_epochs=True,
        )
    elif args.optimizer == "SGD":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, args.epoch, eta_min=args.learning_rate
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
    clip_model, preprocess = clip.load("ViT-B/32", device=device)

    for epoch in range(start_epoch, args.epoch):
        mean_correct = []

        log_string("Epoch %d (%d/%s):" % (global_epoch + 1, epoch + 1, args.epoch))
        momentum = MOMENTUM_ORIGINAL * (
            MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP)
        )
        if momentum < 0.01:
            momentum = 0.01
        print("BN momentum updated to: %f" % momentum)
        classifier = classifier.apply(lambda x: bn_momentum_adjust(x, momentum))
        classifier = classifier.train()

        # '''learning one epoch'''
        for _, (points, label, _, target, _) in tqdm(
            enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9
        ):
            optimizer.zero_grad()

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
            if shape_prior:
                seg_pred, trans_feat, _ = classifier(
                    points, to_categorical(label, num_classes)
                )
            else:
                seg_pred, trans_feat, _ = classifier(points)
            seg_pred = seg_pred.contiguous().view(-1, num_part)
            target = target.view(-1, 1)[:, 0]
            pred_choice = seg_pred.data.max(1)[1]
            correct = pred_choice.eq(target.data).cpu().sum()
            mean_correct.append(correct.item() / (args.batch_size * args.npoint))

            loss = criterion(seg_pred, target)
            
            loss.backward()
            optimizer.step()

        train_instance_acc = np.mean(mean_correct)
        log_string("Train accuracy is: %.5f" % train_instance_acc)
        try:
            scheduler.step()
        except:
            scheduler.step(epoch)
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

            for _, (points, label, _, target, _) in tqdm(
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
                    seg_pred, _, _ = classifier(
                        points, to_categorical(label, num_classes)
                    )
                else:
                    seg_pred, _, _ = classifier(points)
                cur_pred_val = seg_pred.cpu().data.numpy()
                cur_pred_val = np.argmax(cur_pred_val, -1)
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
            test_metrics["class_avg_accuracy"] = np.mean(
                np.array(total_correct_class) / np.array(total_seen_class, dtype=float)
            )

            test_metrics["class_avg_iou"] = mean_shape_ious
            test_metrics["inctance_avg_iou"] = np.mean(all_shape_ious)
            test_metrics["avg_iou_wihtout_shape"] = np.nanmean(general_miou)

        log_string(
            "Epoch %d validation mIoU: %f "
            % (epoch + 1, test_metrics["avg_iou_wihtout_shape"])
        )
        if test_metrics["avg_iou_wihtout_shape"] >= best_avg_iou_wihtout_shape:
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

        if test_metrics["avg_iou_wihtout_shape"] > best_avg_iou_wihtout_shape:
            best_class_avg_iou = test_metrics["class_avg_iou"]
            best_inctance_avg_iou = test_metrics["inctance_avg_iou"]
            best_avg_iou_wihtout_shape = test_metrics["avg_iou_wihtout_shape"]
            best_acc = test_metrics["accuracy"]

        log_string("Best accuracy is: %.5f" % best_acc)
        log_string("Best class avg mIOU is: %.5f" % best_class_avg_iou)
        log_string("Best inctance avg mIOU is: %.5f" % best_inctance_avg_iou)
        log_string("Best general avg mIOU is: %.5f" % best_avg_iou_wihtout_shape)
        global_epoch += 1


if __name__ == "__main__":
    args = parse_args()
    main(args)
