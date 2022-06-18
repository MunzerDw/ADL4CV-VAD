from pathlib import Path
import json

import numpy as np
import torch


class ShapeNet(torch.utils.data.Dataset):
    num_classes = 8
    #dataset_sdf_path = Path("exercise_3/data/shapenet_dim32_sdf")  # path to voxel data
    dataset_df_path = Path("data/shapenet_dim32_df")  # path to voxel data
    class_name_mapping = json.loads(Path("data/shape_info.json").read_text())  # mapping for ShapeNet ids -> names
    classes = sorted(class_name_mapping.keys())

    def __init__(self, split,  filter_class=None):
        super().__init__()
        assert split in ['train', 'val', 'test', 'overfit']
        self.truncation_distance = 3

        if filter_class:
            self.items = Path(f"data/splits/shapenet/{filter_class}_{split}.txt").read_text().splitlines()
        else:
            self.items = Path(f"data/splits/shapenet/{split}.txt").read_text().splitlines()  # keep track of shapes based on split

    def __getitem__(self, index):
        df_id = self.items[index]
        #input_sdf = ShapeNet.get_shape_sdf(sdf_id)
        target_df = ShapeNet.get_shape_df(df_id)

        #input_sdf = np.clip(input_sdf, a_min=-3, a_max=3)
        #input_sdf = np.stack([np.abs(input_sdf),np.sign(input_sdf)])
        target_df = np.log(np.clip(target_df, a_min=-3, a_max=3)+1)
        return {
            'name': f'{df_id}',
            #'input_sdf': input_sdf,
            'target_df': target_df,
            'index' : index
        }

    def __len__(self):
        return len(self.items)

    @staticmethod
    def move_batch_to_device(batch, device):
        #batch["input_sdf"] = batch['input_sdf'].to(device, torch.float32)
        batch['target_df'] = batch['target_df'].to(device, torch.float32)


    @staticmethod
    def get_shape_sdf(shapenet_id):
        # TODO implement sdf data loading
        prepath_points = str(Path(ShapeNet.dataset_sdf_path)) + "/" + str(shapenet_id) +  ".sdf"
        shape = np.fromfile(file=prepath_points, count= 3, dtype=np.uint64)
        sdf = np.fromfile(file=prepath_points, dtype=np.float32, offset=24)
        sdf = np.reshape(sdf, (shape[0],shape[1], shape[2])).astype(np.float32)
        return sdf

    @staticmethod
    def get_shape_df(shapenet_id):
        df = None
        # TODO implement df data loading
        prepath_points = str(Path(ShapeNet.dataset_df_path)) + "/" + str(shapenet_id)
        shape = np.fromfile(file=prepath_points, count=3, dtype=np.uint64)
        df = np.fromfile(file=prepath_points, dtype=np.float32, offset=24)
        df = np.reshape(df, (shape[0], shape[1], shape[2])).astype(np.float32)

        return df
