from typing import Tuple, List
import logging

import torch
import torch.nn as nn

from Loss.LossUtils import correct_output_and_get_grid
from Loss.IOULoss import IOUloss
from Loss.LossUtils import get_geometry_constraint


class LossCalculator:
    def __init__(
        self, 
        num_classes: int, 
        strides: List[int]
    ) -> None:
        self.l1_loss = nn.L1Loss(reduction="none")
        self.bcewithlog_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.iou_loss = IOUloss(reduction="none")
        self.strides = strides

        self._logger = logging.getLogger("LossCalculator")
        print("Calculator logger level: ", self._logger.level)

    def calculate_losses(
        self, 
        outputs: List[torch.Tensor], 
        targets: torch.Tensor,  # [bs, 120, 5]
    ):
        x_shifts, y_shifts, expanded_strides = [], [], []
        processed_outputs: List[torch.Tensor] = []
        for i, (output, stride) in enumerate(zip(outputs, self.strides)):
            # output: [bs, n_anchors, 85]
            output, grid = correct_output_and_get_grid(output, stride, output.device)
            x_shifts.append(grid[:, :, 0])
            y_shifts.append(grid[:, :, 1])
            expanded_strides.append(
                torch.full([1, grid.shape[1]], stride).type_as(output)
            )

            # TODO[Ziteng]: store original preds for l1 loss 

            processed_outputs.append(output)
        # [bs, n_anchors_all, 85]
        processed_outputs: torch.Tensor = torch.cat(processed_outputs, dim=1)

        bbox_preds = processed_outputs[:, :, :4]  # [batch, n_anchors_all, 4]
        obj_preds = processed_outputs[:, :, 4:5]  # [batch, n_anchors_all, 1]
        cls_preds = processed_outputs[:, :, 5:]  # [batch, n_anchors_all, n_cls]

        # target: [bs, 120, 5], zero padding, nlabel: [bs]
        nlabel = (targets.sum(dim=2) > 0).sum(dim=1)  # number of objects

        total_num_anchors = processed_outputs.shape[1]
        x_shifts = torch.cat(x_shifts, 1)  # [1, n_anchors_all]
        y_shifts = torch.cat(y_shifts, 1)  # [1, n_anchors_all]
        expanded_strides = torch.cat(expanded_strides, 1) # [1, n_anchors_all]

        cls_targets = []
        reg_targets = []
        l1_targets = []
        obj_targets = []
        fg_masks = []

        num_fore_ground: int = 0
        num_groud_truth: int = 0

        for batch_idx in range(processed_outputs.shape[0]):
            num_gt_per_img = int(nlabel[batch_idx])
            num_groud_truth += num_gt_per_img
            if num_gt_per_img == 0:
                cls_target = processed_outputs.new_zeros((0, self.num_classes))
                reg_target = processed_outputs.new_zeros((0, 4))
                l1_target = processed_outputs.new_zeros((0, 4))
                obj_target = processed_outputs.new_zeros((total_num_anchors, 1))
                fg_mask = processed_outputs.new_zeros(total_num_anchors).bool()
            else:
                gt_bboxes_per_img = targets[batch_idx, :num_gt_per_img, 1:5]
                gt_classes_per_img = targets[batch_idx, :num_gt_per_img, 0]
                bboxes_preds_per_img = bbox_preds[batch_idx]  # bbox_preds: [bs, n_anchors_all, 4]

                # assign ground truth to preds
                try:
                    (
                        gt_matched_classes,
                        fg_mask,
                        pred_ious_this_matching,
                        matched_gt_inds,
                        num_fg_img,
                    ) = self.get_assignments(  # noqa
                        batch_idx,
                        num_gt_per_img,
                        gt_bboxes_per_img,
                        gt_classes_per_img,
                        bboxes_preds_per_img,
                        expanded_strides,
                        x_shifts,
                        y_shifts,
                        cls_preds,
                        obj_preds,
                    )
                except RuntimeError as e:
                    # TODO: the string might change, consider a better way
                    if "CUDA out of memory. " not in str(e):
                        raise  # RuntimeError might not caused by CUDA OOM

                    self._logger.error(
                        "OOM RuntimeError is raised due to the huge memory cost during label assignment. \
                           CPU mode is applied in this batch. If you want to avoid this issue, \
                           try to reduce the batch size or image size."
                    )
                    torch.cuda.empty_cache()
                    # (
                    #     gt_matched_classes,
                    #     fg_mask,
                    #     pred_ious_this_matching,
                    #     matched_gt_inds,
                    #     num_fg_img,
                    # ) = self.get_assignments(  # noqa
                    #     batch_idx,
                    #     num_gt,
                    #     gt_bboxes_per_image,
                    #     gt_classes_per_img,
                    #     bboxes_preds_per_img,
                    #     expanded_strides,
                    #     x_shifts,
                    #     y_shifts,
                    #     cls_preds,
                    #     obj_preds,
                    #     "cpu",
                    # )


        cls_targets = torch.cat(cls_targets, 0)
        reg_targets = torch.cat(reg_targets, 0)
        obj_targets = torch.cat(obj_targets, 0)
        fg_masks = torch.cat(fg_masks, 0)

    @torch.no_grad()
    def get_assignments(
        self,
        batch_idx: int,
        num_gt_per_img: int,
        gt_bboxes_per_img: torch.Tensor,    # [n_gt, 4]
        gt_classes_per_img: torch.Tensor,   # [n_gt]
        bboxes_preds_per_img: torch.Tensor,   # [n_anchors_all, 4]
        expanded_strides: torch.Tensor, # [1, n_anchors_all]
        x_shifts: torch.Tensor, # [1, n_anchors_all]
        y_shifts: torch.Tensor, # [1, n_anchors_all]
        cls_preds: torch.Tensor,    # [n_anchors_all, n_cls]
        obj_preds: torch.Tensor,    # [n_anchors_all, 1]
        mode="gpu",
    ): 
        if mode == "cpu":
            self._logger.warning("-----------Using CPU for the Current Batch-------------")
            gt_bboxes_per_img = gt_bboxes_per_img.cpu().float()
            bboxes_preds_per_img = bboxes_preds_per_img.cpu().float()
            gt_classes = gt_classes.cpu().float()
            expanded_strides = expanded_strides.cpu().float()
            x_shifts = x_shifts.cpu()
            y_shifts = y_shifts.cpu()
            
        fg_mask, geometry_relation = get_geometry_constraint(
            gt_bboxes_per_img,
            expanded_strides,
            x_shifts,
            y_shifts,
        )
        
    