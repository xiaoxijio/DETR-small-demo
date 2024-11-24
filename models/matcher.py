# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        """

        :param cost_class: 分类错误的相对权重
        :param cost_bbox: 边界框坐标L1误差的相对权重
        :param cost_giou: 边界框的giou损失的相对权重
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """
        目标匹配 根据 分类结果 和 框位置 的相似度来匹配预测框和真实框
        :param outputs: This is a dict that contains at least these entries:
                    "pred_logits": 预测分类结果  [batch_size, num_queries, num_classes]
                    "pred_boxes": 预测框坐标的  [batch_size, num_queries, 4]
        :param targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:\
                    "labels": 目标类别标签 [num_target_boxes] num_target_boxes  图中真实目标数量
                    "boxes": 目标框坐标 [num_target_boxes, 4]
        :return: A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i 预测的索引
                - index_j 真实的索引
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes) 不会超过 100个
        """

        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes 将每个目标的类别标签和框坐标拼接在一起，形成整体的标签和框
        tgt_ids = torch.cat([v["labels"] for v in targets])  # 标签里的类别
        tgt_bbox = torch.cat([v["boxes"] for v in targets])  # 标签里的锚框

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        cost_class = -out_prob[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)  # 计算预测框和真实框之间的 L1距离（曼哈顿距离）

        # Compute the giou cost betwen boxes  计算框的重叠程度 GIOU
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        # Final cost matrix 分类损失、L1损失、GIoU损失按权重合并
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]  # 每个图片中的目标框的数量
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]  # 匈牙利算法
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_matcher(args):
    return HungarianMatcher(cost_class=args.set_cost_class, cost_bbox=args.set_cost_bbox, cost_giou=args.set_cost_giou)
