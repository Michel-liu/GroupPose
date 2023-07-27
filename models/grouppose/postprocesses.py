import torch
from torch import nn
from torchvision.ops.boxes import nms


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    def __init__(self, num_select=100, num_body_points=17) -> None:
        super().__init__()
        self.num_select = num_select
        self.num_body_points = num_body_points

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        num_select = self.num_select
        out_logits, out_keypoints= outputs['pred_logits'], outputs['pred_keypoints']
        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), num_select, dim=1)
        scores = topk_values

        # keypoints
        topk_keypoints = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        keypoints = torch.gather(out_keypoints, 1, topk_keypoints.unsqueeze(-1).repeat(1, 1, self.num_body_points*3))
        Z_pred = keypoints[:, :, :self.num_body_points*2]  # bs, nq, 34
        V_pred = keypoints[:, :, self.num_body_points*2:]  # bq, nq, 17
        img_h, img_w = target_sizes.unbind(1)
        Z_pred = Z_pred * torch.stack([img_w, img_h], dim=1).repeat(1, self.num_body_points)[:, None, :]
        keypoints_res = torch.zeros_like(keypoints)
        keypoints_res[..., 0::3] = Z_pred[..., 0::2]
        keypoints_res[..., 1::3] = Z_pred[..., 1::2]
        keypoints_res[..., 2::3] = V_pred[..., 0::1]

        results = [{'scores': s, 'labels': l, 'keypoints': k} for s, l, k in zip(scores, labels, keypoints_res)]
        return results
