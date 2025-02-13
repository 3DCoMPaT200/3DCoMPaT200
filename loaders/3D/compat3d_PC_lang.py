import os
import json
import torch
from compat3D_PC import StylizedShapeLoader_PC
import numpy as np
class CaptioningDataloader(StylizedShapeLoader_PC):
    """
    Dataloader for the captioning challenge that loads both point cloud data and prompts.
    
    Args:
    ----
        root_dir:          Root directory containing HDF5 files
        prompts_path:      Path to the JSON file containing prompts
        split:             One of {train, valid, test}
        semantic_level:    Semantic level to use for segmentations. One of {fine, medium, coarse}
        num_points:        Number of points to sample
        transform:         Data transformations
        half_precision:    Use half precision floats
        normalize_points:  Normalize point clouds
        is_rgb:           The HDF5 to load has RGB features
    """
    def __init__(
        self,
        root_dir,
        prompts_path,
        split,
        semantic_level,
        metadata_path,
        num_points=4096,
        transform=None,
        half_precision=False,
        normalize_points=False,
        is_rgb=True,
    ):
        super().__init__(
            root_dir=root_dir,
            split=split,
            semantic_level=semantic_level,
            num_points=num_points,
            transform=transform,
            half_precision=half_precision,
            normalize_points=normalize_points,
            is_rgb=is_rgb,
        )
        self.metadata_path = metadata_path
        parts_path = os.path.join(metadata_path, 'parts_fine.json')
        self.parts = json.load(open(parts_path))
        materials_path = os.path.join(metadata_path, 'mat_categories.json')
        self.materials = list(json.load(open(materials_path)).keys())
        # Load prompts from JSON file
        with open(prompts_path, 'r') as f:
            all_prompts = json.load(f)
        
        # Get prompts for current split
        self.prompts = all_prompts
        self.part2label = {part: i for i, part in enumerate(self.parts)}
        self.mat2label = {mat: i for i, mat in enumerate(self.materials)}
    def __getitem__(self, item):
        # Get point cloud data from parent class
        shape_id, style_id, shape_label, points, points_part_labels, points_mat_labels = super().__getitem__(item)
        
        # Get corresponding prompt
        model_style_id = f"{shape_id}__{style_id}"
        prompt = self.prompts[model_style_id]
        if prompt['task'] == 'part':
            masks_idx = [self.part2label[mask] for mask in prompt['masks']]
            masks_idx = np.array(masks_idx)
            masks = np.array([points_part_labels == idx for idx in masks_idx]) # [num_masks, num_points]
            
        elif prompt['task'] == 'material':
            masks_idx = [self.mat2label[mask] for mask in prompt['masks']]
            masks_idx = np.array(masks_idx)
            masks = np.array([points_mat_labels == idx for idx in masks_idx]) # [num_masks, num_points]
        
        return (
            shape_id,
            style_id, 
            shape_label,
            points,
            masks,
            prompt
        )
    

if __name__ == '__main__':
    dataloader = CaptioningDataloader(
        root_dir="data/hdf5",
        prompts_path='challenge_extraction/train_prompts.json',
        split='train',
        semantic_level='fine',
        metadata_path='metadata',
        num_points=2048,
        normalize_points=True,
    )
    for i in range(10):
        data = dataloader[i]
        print(data)
