from typing import Tuple, List
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from Loss.LossUtils import correct_output_and_get_grid
from Loss.IOULoss import IOUloss
from Loss.LossUtils import get_geometry_constraint

from Utils.Bboxes import bboxes_iou


class LossCalculator:
    def __init__(
        self, 
        num_classes: int, 
        strides: List[int]
    ) -> None:
        self.l1_loss = nn.L1Loss(reduction="none")
        self.bcewithlog_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.iou_loss = IOUloss(reduction="none")
        self.num_classes = num_classes
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
                gt_bboxes_per_img = targets[batch_idx, :num_gt_per_img, :4]
                gt_classes_per_img = targets[batch_idx, :num_gt_per_img, 4]
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
                        num_gt_per_img,
                        gt_bboxes_per_img,
                        gt_classes_per_img,
                        bboxes_preds_per_img,
                        expanded_strides,
                        x_shifts,
                        y_shifts,
                        cls_preds[batch_idx],
                        obj_preds[batch_idx]
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
                    (
                        gt_matched_classes,
                        fg_mask,
                        pred_ious_this_matching,
                        matched_gt_inds,
                        num_fg_img,
                    ) = self.get_assignments(  # noqa
                        num_gt_per_img,
                        gt_bboxes_per_img,
                        gt_classes_per_img,
                        bboxes_preds_per_img,
                        expanded_strides,
                        x_shifts,
                        y_shifts,
                        cls_preds[batch_idx],
                        obj_preds[batch_idx],
                        "cpu",
                    )
                torch.cuda.empty_cache()
                num_fore_ground += num_fg_img

                cls_target = F.one_hot(
                    gt_matched_classes.to(torch.int64), self.num_classes
                ) * pred_ious_this_matching.unsqueeze(-1)   # class prediction actually predicts the IoU
                obj_target = fg_mask.unsqueeze(-1)
                reg_target = gt_bboxes_per_img[matched_gt_inds]
                
            cls_targets.append(cls_target)
            reg_targets.append(reg_target)
            obj_targets.append(obj_target.type_as(cls_target))
            fg_masks.append(fg_mask)

        cls_targets = torch.cat(cls_targets, 0)
        reg_targets = torch.cat(reg_targets, 0)
        obj_targets = torch.cat(obj_targets, 0)
        fg_masks = torch.cat(fg_masks, 0)
        
        num_fore_ground = max(num_fore_ground, 1)
        loss_iou = (
            self.iou_loss(bbox_preds.view(-1, 4)[fg_masks], reg_targets)
        ).sum() / num_fore_ground
        loss_obj = (
            self.bcewithlog_loss(obj_preds.view(-1, 1), obj_targets)
        ).sum() / num_fore_ground
        loss_cls = (
            self.bcewithlog_loss(
                cls_preds.view(-1, self.num_classes)[fg_masks], cls_targets
            )
        ).sum() / num_fore_ground
        
        # TODO[Ziteng]: loss weights magic number
        reg_weight = 5.0
        loss = reg_weight * loss_iou + loss_obj + loss_cls

        return (
            loss,
            reg_weight * loss_iou,
            loss_obj,
            loss_cls,
            num_fore_ground / max(num_groud_truth, 1),
        )

    @torch.no_grad()
    def get_assignments(
        self,
        num_gt_per_img: int,
        gt_bboxes_per_img: torch.Tensor,    # [n_gt, 4]
        gt_classes_per_img: torch.Tensor,   # [n_gt]
        bboxes_preds_per_img: torch.Tensor,   # [n_anchors_all, 4]
        expanded_strides: torch.Tensor, # [1, n_anchors_all]
        x_shifts: torch.Tensor, # [1, n_anchors_all]
        y_shifts: torch.Tensor, # [1, n_anchors_all]
        cls_preds_per_img: torch.Tensor,    # [n_anchors_all, n_cls]
        obj_preds_per_img: torch.Tensor,    # [n_anchors_all, 1]
        mode="gpu",
    ): 
        if mode == "cpu":
            self._logger.warning("-----------Using CPU for the Current Batch-------------")
            gt_bboxes_per_img = gt_bboxes_per_img.cpu().float()
            bboxes_preds_per_img = bboxes_preds_per_img.cpu().float()
            gt_classes_per_img = gt_classes_per_img.cpu().float()
            expanded_strides = expanded_strides.cpu().float()
            x_shifts = x_shifts.cpu()
            y_shifts = y_shifts.cpu()
            
        # forground_mask (from gt)
        fg_mask, geometry_relation = get_geometry_constraint(
            gt_bboxes_per_img,
            expanded_strides,
            x_shifts,
            y_shifts,
        )
        
        # select foreground predictions
        bboxes_preds_per_img = bboxes_preds_per_img[fg_mask]
        cls_preds_per_img_ = cls_preds_per_img[fg_mask]
        obj_preds_per_img_ = obj_preds_per_img[fg_mask]
        num_in_boxes_anchor = bboxes_preds_per_img.shape[0]
        
        if mode == "cpu":
            gt_bboxes_per_img = gt_bboxes_per_img.cpu()
            bboxes_preds_per_img = bboxes_preds_per_img.cpu()
        
        # IOU Loss
        pair_wise_ious = bboxes_iou(
            gt_bboxes_per_img, 
            bboxes_preds_per_img, 
            xyxy = False
        )
        pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)
        
        # Classisfication loss (binary cross entropy) 
        gt_classes_per_img_one_hot = F.one_hot(
            gt_classes_per_img.to(torch.int64), self.num_classes
        ).float()
        
        if mode == "cpu":
            cls_preds_per_img_ = cls_preds_per_img_.cpu()
            obj_preds_per_img_ = obj_preds_per_img_.cpu()
            
        with torch.cuda.amp.autocast(enabled=False):
            cls_preds_per_img_ = (
                cls_preds_per_img_.float().sigmoid_() * obj_preds_per_img_.float().sigmoid_()
            ).sqrt()
            pair_wise_cls_loss = F.binary_cross_entropy(
                cls_preds_per_img_.unsqueeze(0).repeat(num_gt_per_img, 1, 1),
                gt_classes_per_img_one_hot.unsqueeze(1).repeat(1, num_in_boxes_anchor, 1),
                reduction="none"
            ).sum(-1)
        del cls_preds_per_img_
        
        # [n_gt, n_matched_anchor]
        cost = (
            pair_wise_cls_loss
            + 3.0 * pair_wise_ious_loss
            + float(1e6) * (~geometry_relation)
        )
        (
            num_fg,
            gt_matched_classes,
            pred_ious_this_matching,
            matched_gt_inds,
        ) = self.simota_matching(
            cost, 
            pair_wise_ious, 
            gt_classes_per_img, 
            num_gt_per_img, 
            fg_mask
        )
        
        if mode != "cpu":
            gt_matched_classes = gt_matched_classes.cuda()
            fg_mask = fg_mask.cuda()
            pred_ious_this_matching = pred_ious_this_matching.cuda()
            matched_gt_inds = matched_gt_inds.cuda()
            
        return (
            gt_matched_classes,
            fg_mask,
            pred_ious_this_matching,
            matched_gt_inds,
            num_fg,
        )
        
    def simota_matching(
        self, 
        cost: torch.Tensor, # [n_gt, n_matched_anchor]
        pair_wise_ious: torch.Tensor,   # [n_gt, n_matched_anchor]
        gt_classes: torch.Tensor,   # [n_gt]
        num_gt: int,  
        fg_mask: torch.Tensor   # [n_anchors_all]
    ) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
        matching_matrix = torch.zeros_like(cost, dtype=torch.uint8)

        n_candidate_k: int = min(10, pair_wise_ious.size(1))
        topk_ious, _ = torch.topk(pair_wise_ious, n_candidate_k, dim=1)
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1) # [n_gt]
        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(
                cost[gt_idx], k=dynamic_ks[gt_idx], largest=False
            )
            matching_matrix[gt_idx][pos_idx] = 1

        del topk_ious, dynamic_ks, pos_idx

        anchor_matching_gt = matching_matrix.sum(0)
        # deal with the case that one anchor matches multiple ground-truths
        if anchor_matching_gt.max() > 1:
            multiple_match_mask = anchor_matching_gt > 1
            _, cost_argmin = torch.min(cost[:, multiple_match_mask], dim=0)
            matching_matrix[:, multiple_match_mask] *= 0
            matching_matrix[cost_argmin, multiple_match_mask] = 1
        fg_mask_inboxes = anchor_matching_gt > 0
        num_fg = fg_mask_inboxes.sum().item()

        fg_mask[torch.where(fg_mask)] = fg_mask_inboxes

        matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
        gt_matched_classes = gt_classes[matched_gt_inds]

        pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(0)[
            fg_mask_inboxes
        ]
        return num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds