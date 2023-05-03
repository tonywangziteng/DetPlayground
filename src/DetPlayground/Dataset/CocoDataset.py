import os
from typing import Any, Dict, Optional, Callable, List
import logging

from pycocotools.coco import COCO

from PIL import Image

import numpy as np
import torch
from torchvision import transforms
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader

from DetPlayground.Dataset.Transforms import YoloXCocoDataTransform
from DetPlayground.Dataset.Transforms import YoloXCocoTargetTransform


class CocoDataset(Dataset):
    def __init__(
        self, 
        args: Dict[str, Any], 
        data_transform: Optional[Callable] = None, 
        target_transform: Optional[Callable] = None, 
    ) -> None:
        """Coco structure dataset

        Args:
            args (Dict[str: Any]): model arguments
            transform (Optional[Callable], optional):A function/transform that  takes in an PIL image
                and returns a transformed version. E.g, ``transforms.ToTensor``. Defaults to None.
            target_transform (Optional[Callable], optional): A function/transform that takes in the
                target and transforms it. Defaults to None.
        """
        self._root_path: str = args["rootPath"]
        # image folder path
        self._img_path: str = os.path.join(self._root_path, args["imgPath"])
        # json annotation file path
        self._anno_path: str = os.path.join(self._root_path, args["annoPath"])
        self._coco_dataset = COCO(self._anno_path)
        self._ids: List[str] = list(self._coco_dataset.imgs.keys())
        
        if data_transform is not None:
            self._data_transform = data_transform
        else:
            self._data_transform = YoloXCocoDataTransform(args["imgSize"])
                        
        if target_transform is not None:
            self._target_transform = target_transform
        else:
            self._target_transform = YoloXCocoTargetTransform(args["imgSize"])
        
    def __len__(self) -> int:
        return len(self._ids)
        
    def __getitem__(self, index):
        img_id = self._ids[index]
        ann_ids = self._coco_dataset.getAnnIds(imgIds=img_id)
        target = self._coco_dataset.loadAnns(ann_ids)

        img_info: Dict = self._coco_dataset.loadImgs(img_id)[0]
        path: str = img_info['file_name']

        img_origin = Image.open(os.path.join(self._img_path, path)).convert('RGB')
        original_img_size = img_origin.size
        if self._data_transform is not None:
            img = self._data_transform(img_origin)

        if self._target_transform is not None:
            target = self._target_transform(target, original_img_size)

        return img, target
    

def get_coco_dataloader(args: Dict) -> DataLoader:
    dataset = CocoDataset(args=args["data"])
    dataloader = DataLoader(
        dataset=dataset, 
        batch_size=args["batchSize"], 
    )
    return dataloader
    
    
# def coco_collate_fn(batch):
#     images, annotations = zip(*batch)
#     max_ann_len = max(len(ann) for ann in annotations)

#     padded_annotations = []
#     for ann in annotations:
#         pad_len = max_ann_len - len(ann)
#         padded_ann = ann + [None] * pad_len
#         padded_annotations.append(padded_ann)

#     images = torch.stack(images, 0)
#     return images, padded_annotations

if __name__ == "__main__":
    import yaml
    with open("../Config/YoloxL.yaml") as f:
        args = yaml.load(f, Loader=yaml.SafeLoader)
        
    transform = transforms.Compose([
        transforms.Resize([300, 300]), 
        transforms.ToTensor()
    ])
    coco_dataset = CocoDataset(
        args=args["training"]["data"], 
        data_transform=transform
    )
    
    coco_dataloader = DataLoader(
        coco_dataset, 8, shuffle=True, 
    )
    
    for data, target in coco_dataloader:
        print(data)
        