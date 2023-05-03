from typing import Tuple
import os
import logging

import torch

def correct_output_and_get_grid(
    output: torch.Tensor, 
    stride: int, 
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    add offset to output, exponetial bbox width and height, and apply stride
    """
    batch_size, channels, height, width = output.shape

    yv, xv = torch.meshgrid([torch.arange(height), torch.arange(width)])
    xv = xv.to(device=output.device, dtype=output.dtype)
    yv = yv.to(device=output.device, dtype=output.dtype)
    grid = torch.stack([xv, yv], 2).view(1, height * width, 2)

    output = output.permute(0, 2, 3, 1)
    output = output.reshape(batch_size, height * width, channels)

    output[..., :2] = (output[..., :2] + grid) * stride
    output[..., 2:4] = torch.exp(output[..., 2:4]) * stride

    return output, grid


def get_geometry_constraint(
    gt_bboxes_per_img: torch.Tensor,    # [n_gt, 4]
    expanded_strides: torch.Tensor, # [1, n_anchors_all]
    x_shifts: torch.Tensor, # [1, n_anchors_all]
    y_shifts: torch.Tensor, # [1, n_anchors_all]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate whether the center of an object is located in a fixed range of
    an anchor. This is used to avert inappropriate matching. It can also reduce
    the number of candidate anchors so that the GPU memory is saved.
    """
    x_centers_per_img = ((x_shifts + 0.5) * expanded_strides)
    y_centers_per_img = ((y_shifts + 0.5) * expanded_strides)

    # in fixed center (with inflation)
    center_radius = 1.5
    center_dist = expanded_strides * center_radius  
    # gt in format cxcywhm, so this is the center ball
    gt_bboxes_per_img_l = (gt_bboxes_per_img[:, 0:1]) - center_dist
    gt_bboxes_per_img_r = (gt_bboxes_per_img[:, 0:1]) + center_dist
    gt_bboxes_per_img_t = (gt_bboxes_per_img[:, 1:2]) - center_dist
    gt_bboxes_per_img_b = (gt_bboxes_per_img[:, 1:2]) + center_dist

    c_l = x_centers_per_img - gt_bboxes_per_img_l   # anchor center to gt center left
    c_r = gt_bboxes_per_img_r - x_centers_per_img   # anchor center to gt center right
    c_t = y_centers_per_img - gt_bboxes_per_img_t   # anchor center to gt center top
    c_b = gt_bboxes_per_img_b - y_centers_per_img   # anchor center to gt center bottom
    center_deltas = torch.stack([c_l, c_t, c_r, c_b], 2)    # [n_gt, n_anchors_all, 4]
    is_in_centers = center_deltas.min(dim=-1).values > 0.0  # all four number > 0
    anchor_filter = is_in_centers.sum(dim=0) > 0    # filter out anchors with gt
    geometry_relation = is_in_centers[:, anchor_filter]

    return anchor_filter, geometry_relation
