from typing import List, Optional, Tuple

import numpy as np
import torch
from torchvision import transforms

from DetPlayground.Utils.Bboxes import xyxy2cxcywh


class YoloXCocoDataTransform:
    def __init__(
        self, 
        img_size
    ) -> None:
        self._img_size = img_size
        self._base_transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Resize(self._img_size, antialias=True)
        ])
        
    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        return self._base_transform(data)


class YoloXCocoTargetTransform:
    def __init__(
        self, 
        img_size
    ) -> None:
        self._img_size = img_size
        self._width: int = self._img_size[0]
        self._height: int = self._img_size[1]
    
    def __call__(
        self, 
        target: List, 
        original_img_size: Tuple[int, int], 
        format: str = "cxcywh"    # "xyxy"
    ) -> torch.Tensor:
        """
        Params:
            target: List of targets. bbox in [left_top_x, left_top_y, w, h]
        return:
            bboxes: torch.Tensor
        """
        # category(1) + bbox(4)
        bboxes = torch.zeros([120, 5])
        valid_target_idx: int = 0
        for instance in target:
            bbox: List[float] = instance.get("bbox")
            bbox = self._rectify_bbox(bbox, original_img_size[0], original_img_size[1])
            if bbox is not None and instance.get("area") > 0:
                category_id: int = instance.get("category_id")
                bboxes[valid_target_idx, 4] = category_id
                bboxes[valid_target_idx, :4] = torch.tensor(bbox)
                valid_target_idx += 1
                
        # rescale the annotation
        # TODO[Ziteng]: rescale with padding
        original_width, original_height = original_img_size
        x_scale: float = 1.0 * self._width / original_width
        y_scale: float = 1.0 * self._height / original_height
        scale: torch.Tensor = torch.tensor(
            [x_scale, y_scale, x_scale, y_scale]
        ).unsqueeze(0)
        
        bboxes[:, :4] *= scale
        
        if format == "cxcywh":
            bboxes = xyxy2cxcywh(bboxes)
        
        return bboxes
        

    def _rectify_bbox(
        self, bbox: List[float], 
        original_width: int, 
        original_height: int
    ) -> Optional[List[float]]:
        """
        clip the bbox into the image and change it to x1y1x2y2 format
        """
        x1 = max(0, bbox[0])
        y1 = max(0, bbox[1])
        x2 = min(original_width, x1 + max((0, bbox[2])))
        y2 = min(original_height, y1 + max((0, bbox[3])))
        if x2 >= x1 and y2 >= y1:
            return [x1, y1, x2, y2]
        else:
            return None
