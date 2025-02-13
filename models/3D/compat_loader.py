""""
Dataloaders for the preprocessed point clouds from 3DCoMPaT dataset.
"""

import os

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import Sampler
from collections import defaultdict


def pc_normalize(pc):
    """
    Center and scale the point cloud.
    """
    xyz = pc[:, :3]
    other_feature = pc[:, 3:] 
    centroid = np.mean(xyz, axis=0)
    xyz = xyz - centroid
    m = np.max(np.sqrt(np.sum(xyz ** 2, axis=1)))
    xyz = xyz / m

    pc = np.concatenate((xyz, other_feature), axis=1)
    return pc


def load_data(data_dir, partition, semantic_level, material=False):
    """
    Pre-load and process the pointcloud data into memory.
    """
    h5_name = os.path.join(data_dir, "{}_{}.hdf5".format(partition, semantic_level))
    # h5_name = data_dir
    with h5py.File(h5_name, "r") as f:
        points = np.array(f["points"][:]).astype("float32")
        points_labels = np.array(f["points_part_labels"][:]).astype("uint16")
        shape_ids = f["shape_id"][:].astype("str")
        style_ids = f["style_id"][:].astype("str")
        shape_labels = np.array(f["shape_label"][:]).astype("uint8")
        if material:
            point_materials = np.array(f["points_mat_labels"][:]).astype("uint16")
        normalized_points = np.zeros(points.shape)
        for i in range(points.shape[0]):
            normalized_points[i] = pc_normalize(points[i])
    data = {}
    if material:
        data["normalized_points"] = normalized_points
        data["points_labels"] = points_labels
        data["point_materials"] = point_materials
        data["shape_ids"] = shape_ids
        data["style_ids"] = style_ids
        data["shape_labels"] = shape_labels
        return data
    data["normalized_points"] = normalized_points
    data["points_labels"] = points_labels
    data["shape_ids"] = shape_ids
    data["shape_labels"] = shape_labels
    return data


class CompatLoader3D(Dataset):
    """
    Base class for loading preprocessed 3D point clouds.

    Args:
    ----
        data_root:   Base dataset URL containing data split shards
        split:       One of {train, valid}.
        num_points:  Number of sampled points
        transform:   data transformations
    """

    def __init__(
        self,
        data_root,
        split="train",
        num_points=4096,
        semantic_level="fine",
        transform=None,
        material=False,
        filter_json=None,
    ):
        # train, test, valid
        self.material = material
        self.partition = split.lower()
        print('filtered classes', filter_json)
        if self.material:
            print('Using materials')
            samples = load_data(data_root, self.partition, semantic_level, True)
            if filter_json is None:
                self.data, self.seg, self.mat, self.shape_ids, self.label, self.style_ids = (
                    np.array(samples["normalized_points"]),
                    np.array(samples["points_labels"]),
                    np.array(samples["point_materials"]),
                    samples["shape_ids"],
                    np.array(samples["shape_labels"]),
                    samples["style_ids"],
                )
            else:
                filtered_data = {
                    "normalized_points": [],
                    "points_labels": [],
                    "point_materials": [],
                    "shape_ids": [],
                    "style_ids": [],
                    "shape_labels": [],
                }

                for i, shape_label in enumerate(samples["shape_labels"]):
                    if shape_label in filter_json:
                        filtered_data["normalized_points"].append(
                            samples["normalized_points"][i]
                        )
                        filtered_data["points_labels"].append(
                            samples["points_labels"][i]
                        )
                        filtered_data["point_materials"].append(
                            samples["point_materials"][i]
                        )
                        filtered_data["shape_ids"].append(samples["shape_ids"][i])
                        filtered_data["shape_labels"].append(shape_label)
                        filtered_data["style_ids"].append(samples["style_ids"][i])
                self.data, self.seg, self.mat, self.shape_ids, self.label, self.style_ids = (
                    np.array(filtered_data["normalized_points"]),
                    np.array(filtered_data["points_labels"]),
                    np.array(samples["point_materials"]),
                    filtered_data["shape_ids"],
                    np.array(filtered_data["shape_labels"]),
                    filtered_data["style_ids"],
                )

        else:
            samples = load_data(data_root, self.partition, semantic_level)
            if filter_json is None:
                self.data, self.seg, self.shape_ids, self.label = (
                    np.array(samples["normalized_points"]),
                    np.array(samples["points_labels"]),
                    samples["shape_ids"],
                    np.array(samples["shape_labels"]),
                )
            else:
                filtered_data = {
                    "normalized_points": [],
                    "points_labels": [],
                    "shape_ids": [],
                    "shape_labels": [],
                }

                for i, shape_label in enumerate(samples["shape_labels"]):
                    if shape_label in filter_json:
                        filtered_data["normalized_points"].append(
                            samples["normalized_points"][i]
                        )
                        filtered_data["points_labels"].append(
                            samples["points_labels"][i]
                        )
                        filtered_data["shape_ids"].append(samples["shape_ids"][i])
                        filtered_data["shape_labels"].append(shape_label)
                self.data, self.seg, self.shape_ids, self.label = (
                    np.array(filtered_data["normalized_points"]),
                    np.array(filtered_data["points_labels"]),
                    filtered_data["shape_ids"],
                    np.array(filtered_data["shape_labels"]),
                )


        self.num_points = num_points
        self.transform = transform

    def __getitem__(self, item):
        pointcloud = self.data[item]
        label = self.label[item]
        seg = self.seg[item].astype(np.int32)
        if self.material:
            mat = self.mat[item].astype(np.int32)
        shape_id = self.shape_ids[item]
        pointcloud = torch.from_numpy(pointcloud)
        seg = torch.from_numpy(seg)

        if self.material:
            style_id = self.style_ids[item]
            return pointcloud, label, seg, mat, shape_id, style_id

        return pointcloud, label, seg, shape_id

    def __len__(self):
        return self.data.shape[0]

    def num_classes(self):
        return np.max(self.label) + 1

    def num_segments(self):
        return np.max(self.seg) + 1

    def get_shape_label(self):
        return self.label


class CompatLoader3DCls(CompatLoader3D):
    """
    Classification data loader using preprocessed 3D point clouds.

    Args:
    ----
        data_root:   Base dataset URL containing data split shards
        split:       One of {train, valid}.
        num_points:  Number of sampled points
        transform:   data transformations
    """

    def __init__(
        self,
        data_root="data/compat",
        split="train",
        num_points=4096,
        transform=None,
        semantic_level="fine",
        filter_json=None,
    ):
        super().__init__(
            data_root, split, num_points, semantic_level, transform, False,  filter_json
        )

    def __getitem__(self, item):
        pointcloud = self.data[item].astype(np.float32)
        label = self.label[item]
        seg = self.seg[item].astype(np.int32)

        pointcloud = torch.from_numpy(pointcloud)
        seg = torch.from_numpy(seg)
        return pointcloud, label


