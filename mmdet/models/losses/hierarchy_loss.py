# Copyright (c) OpenMMLab. All rights reserved.
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES


def _load_class_hierarchy(hierarchy_file):

    assert os.path.exists(hierarchy_file), "File {} dose not exist.".format(hierarchy_file)
    
    print('Loading', hierarchy_file)
    hierarchy_data = json.load(open(hierarchy_file, 'r'))
    is_childs = torch.Tensor(hierarchy_data['is_childs']).float()  # (C + 1) x (C + 1), the last row / column is the background
    is_parents = torch.Tensor(hierarchy_data['is_parents']).float()  # C x C
    
    assert (is_childs[:-1, :-1] * is_parents).sum() == 0 and (is_childs[-1, :].sum() + is_childs[:, -1].sum()) == 0
    hierarchy_weight = (1 - is_childs, is_parents)
    return hierarchy_weight


@LOSSES.register_module()
class HierarchyLoss(nn.Module):
    def __init__(self,
                 hierarchy_file="./label_spaces/hierarchy.json",
                 loss_weight=1.0):
        super(HierarchyLoss, self).__init__()
        self.hierarchy_file = hierarchy_file
        self.loss_weight = loss_weight
        self.hierarchy_weight, self.is_parents = _load_class_hierarchy(hierarchy_file)

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):

        assert reduction_override in (None, 'none'), "reduction should be none"

        if pred.numel() == 0:
            return pred.new_zeros([1])[0]  # This is more robust than .sum() * 0.

        B = pred.shape[0]
        C = pred.shape[1] - 1

        target_refined = pred.new_zeros(B, C + 1)
        target_refined[range(len(target)), target] = 1  # B x (C + 1)
        
        # set all parents to positive
        is_parents = self.is_parents.to(target_refined.device).detach()  # C x C
        target_refined[:, :C] = torch.mm(target_refined[:, :C], is_parents) + target_refined[:, :C]  # B x C

        cls_loss = F.binary_cross_entropy_with_logits(pred, target_refined, reduction='none')  # B x (C + 1)

        # ignore all childern
        hierarchy_w = self.hierarchy_weight.to(target_refined.device).detach()  # (C + 1) x (C + 1)
        hierarchy_w = hierarchy_w[target]  # B x (C + 1)

        cls_loss = torch.sum(cls_loss * hierarchy_w) / B
        
        return self.loss_weight * cls_loss