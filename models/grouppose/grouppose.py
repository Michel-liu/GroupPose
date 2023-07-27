# ------------------------------------------------------------------------
# Modified from Conditional DETR model and criterion classes.
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------




import copy

import math
import torch
import torch.nn.functional as F
from torch import nn

from util.keypoint_ops import keypoint_xyzxyz_to_xyxyzz
from util.misc import (NestedTensor, nested_tensor_from_tensor_list, inverse_sigmoid)

from .backbones import build_backbone
from .matcher import build_matcher
from .transformer import build_transformer
from .utils import MLP
from .postprocesses import PostProcess
from .criterion import SetCriterion

from ..registry import MODULE_BUILD_FUNCS

class GroupPose(nn.Module):
    """ This is the Cross-Attention Detector module that performs object detection """
    def __init__(self, backbone, transformer, num_classes, num_queries, 
                    aux_loss=False,
                    num_feature_levels=1,
                    nheads=8,
                    two_stage_type='no',
                    dec_pred_class_embed_share=False,
                    dec_pred_pose_embed_share=False,
                    two_stage_class_embed_share=True,
                    two_stage_bbox_embed_share=True,
                    cls_no_bias = False,
                    num_body_points = 17
                    ):
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim = transformer.d_model
        self.num_feature_levels = num_feature_levels
        self.nheads = nheads
        self.num_body_points = num_body_points      

        # prepare input projection layers
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.num_channels)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            assert two_stage_type == 'no', "two_stage_type should be no if num_feature_levels=1 !!!"
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[-1], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])

        self.backbone = backbone
        self.aux_loss = aux_loss

        # prepare class
        _class_embed = nn.Linear(hidden_dim, num_classes, bias=(not cls_no_bias))
        if not cls_no_bias:
            prior_prob = 0.01
            bias_value = -math.log((1 - prior_prob) / prior_prob)
            _class_embed.bias.data = torch.ones(self.num_classes) * bias_value

        _point_embed = MLP(hidden_dim, hidden_dim, 2, 3)
        nn.init.constant_(_point_embed.layers[-1].weight.data, 0)
        nn.init.constant_(_point_embed.layers[-1].bias.data, 0)
        
        if dec_pred_class_embed_share:
            class_embed_layerlist = [_class_embed for i in range(transformer.num_decoder_layers)]
        else:
            class_embed_layerlist = [copy.deepcopy(_class_embed) for i in range(transformer.num_decoder_layers)]
        if dec_pred_pose_embed_share:
            pose_embed_layerlist = [_point_embed for i in range(transformer.num_decoder_layers)]
        else:
            pose_embed_layerlist = [copy.deepcopy(_point_embed) for i in range(transformer.num_decoder_layers)]

        self.class_embed = nn.ModuleList(class_embed_layerlist)
        self.pose_embed = nn.ModuleList(pose_embed_layerlist)
        self.transformer.decoder.pose_embed = self.pose_embed
        self.transformer.decoder.class_embed = self.class_embed
        self.transformer.decoder.num_body_points = num_body_points

        # two stage
        _keypoint_embed = MLP(hidden_dim, 2*hidden_dim, 2*num_body_points, 4)
        nn.init.constant_(_keypoint_embed.layers[-1].weight.data, 0)
        nn.init.constant_(_keypoint_embed.layers[-1].bias.data, 0)
        
        if two_stage_bbox_embed_share:
            self.transformer.enc_pose_embed = _keypoint_embed
        else:
            self.transformer.enc_pose_embed = copy.deepcopy(_keypoint_embed)

        if two_stage_class_embed_share:
            self.transformer.enc_out_class_embed = _class_embed
        else:
            self.transformer.enc_out_class_embed = copy.deepcopy(_class_embed)

        self._reset_parameters()

    def _reset_parameters(self):
        # init input_proj
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

    def forward(self, samples: NestedTensor):
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, poss = self.backbone(samples)

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                poss.append(pos_l)


        hs_pose, refpoint_pose, mix_refpoint, mix_embedding = self.transformer(srcs, masks, poss)
        
        outputs_class=[]
        outputs_keypoints_list = []
        
        for dec_lid, (hs_pose_i, refpoint_pose_i, layer_pose_embed, layer_cls_embed) in enumerate(zip(hs_pose, refpoint_pose, self.pose_embed, self.class_embed)):
            # pose
            bs, nq, np = refpoint_pose_i.shape
            refpoint_pose_i = refpoint_pose_i.reshape(bs, nq, np // 2, 2)
            delta_pose_unsig = layer_pose_embed(hs_pose_i[:, :, 1:])
            layer_outputs_pose_unsig = inverse_sigmoid(refpoint_pose_i[:, :, 1:]) + delta_pose_unsig
            vis_flag = torch.ones_like(layer_outputs_pose_unsig[..., -1:], device=layer_outputs_pose_unsig.device)
            layer_outputs_pose_unsig = torch.cat([layer_outputs_pose_unsig, vis_flag], dim=-1).flatten(-2)
            layer_outputs_pose_unsig = layer_outputs_pose_unsig.sigmoid()
            outputs_keypoints_list.append(keypoint_xyzxyz_to_xyxyzz(layer_outputs_pose_unsig))
            
            # cls
            layer_cls = layer_cls_embed(hs_pose_i[:, :, 0])
            outputs_class.append(layer_cls)

        out = {'pred_logits': outputs_class[-1], 'pred_keypoints': outputs_keypoints_list[-1]}

        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_keypoints_list)

        # for encoder output
        if mix_refpoint is not None and mix_embedding is not None:
            # prepare intermediate outputs
            interm_class = self.transformer.enc_out_class_embed(mix_embedding)
            out['interm_outputs'] = {'pred_logits': interm_class, 'pred_keypoints': mix_refpoint}
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_keypoints):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_keypoints': c}
                for a, c in zip(outputs_class[:-1], outputs_keypoints[:-1])]



@MODULE_BUILD_FUNCS.registe_with_name(module_name='grouppose')
def build_grouppose(args):

    num_classes = args.num_classes
    device = torch.device(args.device)

    backbone = build_backbone(args)

    transformer = build_transformer(args)

    model = GroupPose(
        backbone,
        transformer,
        aux_loss=args.aux_loss,
        num_classes=num_classes,
        nheads=args.nheads,
        num_queries=args.num_queries,
        num_feature_levels=args.num_feature_levels,
        dec_pred_class_embed_share=args.dec_pred_class_embed_share,
        dec_pred_pose_embed_share=args.dec_pred_pose_embed_share,
        two_stage_type=args.two_stage_type,
        two_stage_bbox_embed_share=args.two_stage_bbox_embed_share,
        two_stage_class_embed_share=args.two_stage_class_embed_share,
        cls_no_bias=args.cls_no_bias,
        num_body_points=args.num_body_points
    )

    matcher = build_matcher(args)

    # prepare weight dict
    weight_dict = {
        'loss_ce': args.cls_loss_coef, 
        "loss_keypoints":args.keypoints_loss_coef,
        "loss_oks":args.oks_loss_coef
    }
    clean_weight_dict = copy.deepcopy(weight_dict)

    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            for k, v in clean_weight_dict.items():
                aux_weight_dict.update({k + f'_{i}': v})
        weight_dict.update(aux_weight_dict)

    if args.two_stage_type != 'no':
        interm_weight_dict = {}
        no_interm_loss = args.no_interm_loss
        _coeff_weight_dict = {
            'loss_ce': 1.0 if not no_interm_loss else 0.0,
            'loss_keypoints': 1.0 if not no_interm_loss else 0.0,
            'loss_oks': 1.0 if not no_interm_loss else 0.0,
        }
        interm_weight_dict.update({k + f'_interm': v * args.interm_loss_coef * _coeff_weight_dict[k] for k, v in clean_weight_dict.items()})
        weight_dict.update(interm_weight_dict)
    
    losses = ['labels', "keypoints", "matching"]

    criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict, focal_alpha=args.focal_alpha, losses=losses, num_body_points=args.num_body_points)
    criterion.to(device)
    postprocessors = {
        'keypoints': PostProcess(num_select=args.num_select, num_body_points=args.num_body_points)
    }

    return model, criterion, postprocessors