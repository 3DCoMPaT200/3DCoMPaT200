import torch.nn as nn
import torch
import torch.nn.functional as F
from models.pointnet2_utils import PointNetSetAbstraction, PointNetFeaturePropagation
import logging

logger = logging.getLogger("Model")


class get_model(nn.Module):
    def __init__(
        self,
        num_classes,
        num_materials=0,
        num_shapes=200,
        shape_prior=False,
        normal_channel=False,
        pretrain=False,
    ):
        super(get_model, self).__init__()
        if normal_channel:
            additional_channel = 3
        else:
            additional_channel = 0
        self.num_classes = num_classes
        self.normal_channel = normal_channel
        self.num_shapes = num_shapes
        self.shape_prior = shape_prior
        self.num_materials = num_materials
        self.sa1 = PointNetSetAbstraction(
            npoint=512,
            radius=0.2,
            nsample=32,
            in_channel=6 + additional_channel,
            mlp=[64, 64, 128],
            group_all=False,
        )
        self.sa2 = PointNetSetAbstraction(
            npoint=128,
            radius=0.4,
            nsample=64,
            in_channel=128 + 3,
            mlp=[128, 128, 256],
            group_all=False,
        )
        self.sa3 = PointNetSetAbstraction(
            npoint=None,
            radius=None,
            nsample=None,
            in_channel=256 + 3,
            mlp=[256, 512, 1024],
            group_all=True,
        )
        self.fp3 = PointNetFeaturePropagation(in_channel=1280, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=384, mlp=[256, 128])
        self.fp1 = PointNetFeaturePropagation(
            in_channel=128
            + 6
            + additional_channel
            + (num_shapes if shape_prior else 0),
            mlp=[128, 128, 128],
        )  # +16

        self.conv1 = nn.Conv1d(128, 512, 1)
        self.bn1 = nn.BatchNorm1d(512)
        if num_classes != 512:
            self.drop1 = nn.Dropout(0.5)
            self.conv2 = nn.Conv1d(512, num_classes, 1)
        if num_classes == 512:
            self.lang = nn.Linear(512, 512)
            self.lang.reset_parameters()
            torch.nn.init.eye_(self.lang.weight)
            # Set biases to zero
            if self.lang.bias is not None:
                torch.nn.init.zeros_(self.lang.bias)

    def forward(self, xyz, cls_label=None):
        # Set Abstraction layers
        B, C, N = xyz.shape
        if self.normal_channel:
            l0_points = xyz
            l0_xyz = xyz[:, :3, :]
        else:
            l0_points = xyz[:, :3, :]
            l0_xyz = xyz[:, :3, :]
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        # print(l3_points.shape)
        # Feature Propagation layers
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        if self.shape_prior:
            # print(cls_label.shape)
            cls_label_one_hot = cls_label.unsqueeze(2).repeat(1, 1, N)
            # print(l0_xyz.shape,l0_points.shape, cls_label_one_hot.shape)
            l0_points = self.fp1(
                l0_xyz,
                l1_xyz,
                torch.cat([l0_xyz, l0_points, cls_label_one_hot], 1),
                l1_points,
            )
        else:
            l0_points = self.fp1(
                l0_xyz, l1_xyz, torch.cat([l0_xyz, l0_points], 1), l1_points
            )
        # FC layers
        feat = F.relu(self.bn1(self.conv1(l0_points)))
        if self.num_classes == 512:
            return feat, l3_points, feat.transpose(1, 2)
        else:
            x = self.drop1(feat)
            x = self.conv2(x)
            x = F.log_softmax(x, dim=1)
            x = x.permute(0, 2, 1)
            return x, l3_points, feat.transpose(1, 2)
        # if self.num_materials == 0:
        # mat_feat = F.relu(self.mat_bn1(self.mat_conv1(l0_points)))
        # mat_x = self.mat_drop1(mat_feat)
        # mat_x = self.mat_conv2(mat_x)
        # if self.num_classes != 512 :
        #     mat_x = F.log_softmax(mat_x, dim=1)
        # mat_x = mat_x.permute(0, 2, 1)
        # return x, mat_x, l0_points


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, weights=None):
        if weights is None:
            total_loss = F.nll_loss(pred, target)
        else:
            total_loss = F.nll_loss(pred, target, weight=weights)

        return total_loss
