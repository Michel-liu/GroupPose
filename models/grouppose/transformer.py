# ------------------------------------------------------------------------
# Modified from Conditional DETR Transformer class.
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

import copy
from typing import Optional
from util.misc import inverse_sigmoid
import torch
from torch import nn, Tensor
from .transformer_deformable import DeformableTransformerEncoderLayer, DeformableTransformerDecoderLayer

from .utils import gen_encoder_output_proposals, MLP, gen_sineembed_for_position
from .ops.modules.ms_deform_attn import MSDeformAttn


class Transformer(nn.Module):

    def __init__(self, d_model=256, 
                 nhead=8, 
                 num_queries=300, 
                 num_encoder_layers=6,
                 num_decoder_layers=6, 
                 dim_feedforward=2048, 
                 dropout=0.0,
                 activation="relu", 
                 normalize_before=False,
                 return_intermediate_dec=False, 
                 num_feature_levels=1,
                 enc_n_points=4,
                 dec_n_points=4,
                 learnable_tgt_init=False,
                 two_stage_type='no',
                 num_body_points=17
                 ):
        super().__init__()
        self.num_feature_levels = num_feature_levels
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.num_queries = num_queries

        encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward, dropout, activation, num_feature_levels, nhead, enc_n_points)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm, 
                                          d_model=d_model, 
                                          num_queries=num_queries)           
        decoder_layer = DeformableTransformerDecoderLayer(d_model, dim_feedforward, dropout, activation, num_feature_levels, nhead, dec_n_points)
        
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers,
                                        return_intermediate=return_intermediate_dec,
                                        d_model=d_model,
                                        num_body_points=num_body_points)
        self.d_model = d_model
        self.nhead = nhead
        self.dec_layers = num_decoder_layers
        self.num_queries = num_queries
        if num_feature_levels > 1:
            if self.num_encoder_layers > 0:
                self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))
            else:
                self.level_embed = None
        
        # shared prior between instances
        self.num_body_points = num_body_points
        self.keypoint_embedding = nn.Embedding(num_body_points, self.d_model)
        self.instance_embedding = nn.Embedding(1, self.d_model)
        
        self.learnable_tgt_init = learnable_tgt_init
        if learnable_tgt_init:
            self.register_buffer("tgt_embed", torch.zeros(self.num_queries, d_model))
        else:
            self.tgt_embed = None
            
        # for two stage
        self.two_stage_type = two_stage_type
        if two_stage_type in ['standard']:
            # anchor selection at the output of encoder
            self.enc_output = nn.Linear(d_model, d_model)
            self.enc_output_norm = nn.LayerNorm(d_model)
        self.enc_out_class_embed = None
        self.enc_pose_embed = None

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()

        if self.num_feature_levels > 1 and self.level_embed is not None:
            nn.init.normal_(self.level_embed)

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio
    
    def forward(self, srcs, masks, pos_embeds):
        """
        Input:
            - srcs: List of multi features [bs, ci, hi, wi]
            - masks: List of multi masks [bs, hi, wi]
            - refpoint_embed: [bs, num_dn, 4]. None in infer
            - pos_embeds: List of multi pos embeds [bs, ci, hi, wi]
            - tgt: [bs, num_dn, d_model]. None in infer
            
        """
        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)

            src = src.flatten(2).transpose(1, 2)                # bs, hw, c
            mask = mask.flatten(1)                              # bs, hw
            pos_embed = pos_embed.flatten(2).transpose(1, 2)    # bs, hw, c
            if self.num_feature_levels > 1 and self.level_embed is not None:
                lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            else:
                lvl_pos_embed = pos_embed
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)    # bs, \sum{hxw}, c 
        mask_flatten = torch.cat(mask_flatten, 1)   # bs, \sum{hxw}
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1) # bs, \sum{hxw}, c 
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        memory = self.encoder(
                src_flatten, 
                pos=lvl_pos_embed_flatten, 
                level_start_index=level_start_index, 
                spatial_shapes=spatial_shapes,
                valid_ratios=valid_ratios,
                key_padding_mask=mask_flatten)

        if self.two_stage_type in ['standard']:
            output_memory, output_proposals = gen_encoder_output_proposals(memory, mask_flatten, spatial_shapes)
            output_memory = self.enc_output_norm(self.enc_output(output_memory))
            # top-k select index
            topk = self.num_queries
            enc_outputs_class_unselected = self.enc_out_class_embed(output_memory)
            topk_proposals = torch.topk(enc_outputs_class_unselected.max(-1)[0], topk, dim=1)[1]
            # calculate K, e.g., 17 for COCO, points for keypoint
            bs, nq = output_memory.shape[:2]
            delta_unsig_keypoint = self.enc_pose_embed(output_memory).reshape(bs, nq, -1, 2)
            enc_outputs_pose_coord_unselected = (delta_unsig_keypoint + output_proposals[..., :2].unsqueeze(-2)).sigmoid()
            enc_outputs_center_coord_unselected = torch.mean(enc_outputs_pose_coord_unselected, dim=2, keepdim=True)
            enc_outputs_pose_coord_unselected = torch.cat([enc_outputs_center_coord_unselected, enc_outputs_pose_coord_unselected], dim=2).flatten(-2)
            # gather pose
            enc_outputs_pose_coord_sigmoid = torch.gather(enc_outputs_pose_coord_unselected, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, enc_outputs_pose_coord_unselected.shape[-1]))
            refpoint_pose_sigmoid = enc_outputs_pose_coord_sigmoid.detach()
            # gather tgt
            tgt_undetach = torch.gather(output_memory, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, self.d_model))
            # combine pose embedding
            if self.learnable_tgt_init:
                tgt = self.tgt_embed.expand_as(tgt_undetach).unsqueeze(-2)
            else:
                tgt = tgt_undetach.detach().unsqueeze(-2)
            # query construction
            tgt_pose = self.keypoint_embedding.weight[None, None].repeat(1, topk, 1, 1).expand(bs, -1, -1, -1) + tgt
            tgt_global = self.instance_embedding.weight[None, None].repeat(1, topk, 1, 1).expand(bs, -1, -1, -1)
            tgt_pose = torch.cat([tgt_global, tgt_pose], dim=2)

        hs_pose, refpoint_pose = self.decoder(
                tgt=tgt_pose,
                memory=memory.transpose(0, 1), 
                memory_key_padding_mask=mask_flatten, 
                refpoints_sigmoid=refpoint_pose_sigmoid.transpose(0, 1), 
                level_start_index=level_start_index, 
                spatial_shapes=spatial_shapes,
                valid_ratios=valid_ratios)
        if self.two_stage_type == 'standard':
            mix_refpoint = enc_outputs_pose_coord_sigmoid
            mix_embedding = tgt_undetach
        else:
            mix_refpoint = None
            mix_embedding = None
        return hs_pose, refpoint_pose, mix_refpoint, mix_embedding


class TransformerEncoder(nn.Module):
    def __init__(self, 
        encoder_layer, num_layers, norm=None, d_model=256, 
        num_queries=300):
        super().__init__()
        if num_layers > 0:
            self.layers = _get_clones(encoder_layer, num_layers, layer_share=False)
        else:
            self.layers = []
            del encoder_layer
        self.num_queries = num_queries
        self.num_layers = num_layers
        self.norm = norm
        self.d_model = d_model

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):

            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, 
            src: Tensor, 
            pos: Tensor, 
            spatial_shapes: Tensor, 
            level_start_index: Tensor, 
            valid_ratios: Tensor, 
            key_padding_mask: Tensor):

        output = src
        # preparation and reshape
        if self.num_layers > 0:
            reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        # main process
        for layer_id, layer in enumerate(self.layers):
            output = layer(src=output, pos=pos, reference_points=reference_points, spatial_shapes=spatial_shapes, level_start_index=level_start_index, key_padding_mask=key_padding_mask)
        if self.norm is not None:
            output = self.norm(output)
        return output


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, 
                    return_intermediate=False, 
                    d_model=256,
                    num_body_points=17):   
        super().__init__()
        if num_layers > 0:
            self.layers = _get_clones(decoder_layer, num_layers, layer_share=False)
        else:
            self.layers = []
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_body_points = num_body_points
        self.return_intermediate = return_intermediate 
        self.class_embed = None
        self.pose_embed = None
        self.half_pose_ref_point_head = MLP(d_model, d_model, d_model, 2)

    def forward(self, tgt, memory,
                memory_key_padding_mask: Optional[Tensor] = None,
                refpoints_sigmoid: Optional[Tensor] = None,
                # for memory
                level_start_index: Optional[Tensor] = None,
                spatial_shapes: Optional[Tensor] = None,
                valid_ratios: Optional[Tensor] = None):
        
        output_pose = tgt.transpose(0, 1)
        refpoint_pose = refpoints_sigmoid
        intermediate_pose = []
        ref_pose_points = [refpoint_pose]
        for layer_id, layer in enumerate(self.layers):
            refpoint_pose_input = refpoint_pose[:, :, None] * torch.cat([valid_ratios] * (refpoint_pose.shape[-1] // 2), -1)[None, :]
            nq, bs, np = refpoint_pose.shape
            refpoint_pose_reshape = refpoint_pose_input[:, :, 0].reshape(nq, bs, np // 2, 2).reshape(nq * bs, np // 2, 2)
            pose_query_sine_embed = gen_sineembed_for_position(refpoint_pose_reshape).reshape(nq, bs, np // 2, self.d_model)
            pose_query_pos = self.half_pose_ref_point_head(pose_query_sine_embed)
            
            output_pose = layer(
                tgt_pose = output_pose,
                tgt_pose_query_pos = pose_query_pos[:, :, 1:],
                tgt_pose_reference_points = refpoint_pose_input,
                
                memory = memory,
                memory_key_padding_mask = memory_key_padding_mask,
                memory_level_start_index = level_start_index,
                memory_spatial_shapes = spatial_shapes,
            )
            intermediate_pose.append(output_pose)
            
            # iteration
            nq, bs, np = refpoint_pose.shape
            refpoint_pose = refpoint_pose.reshape(nq, bs, np // 2, 2)
            refpoint_pose_unsigmoid = inverse_sigmoid(refpoint_pose[:, :, 1:])
            delta_pose_unsigmoid = self.pose_embed[layer_id](output_pose[:, :, 1:])
            refpoint_pose_without_center = (refpoint_pose_unsigmoid + delta_pose_unsigmoid).sigmoid()
            # center of pose
            refpoint_center_pose = torch.mean(refpoint_pose_without_center, dim=2, keepdim=True)
            refpoint_pose = torch.cat([refpoint_center_pose, refpoint_pose_without_center], dim=2).flatten(-2)
            ref_pose_points.append(refpoint_pose)
            refpoint_pose = refpoint_pose.detach()

        return [
            [itm_out.transpose(0, 1) for itm_out in intermediate_pose],
            [itm_refpoint.transpose(0, 1) for itm_refpoint in ref_pose_points]
        ]


def _get_clones(module, N, layer_share=False):
    if layer_share:
        return nn.ModuleList([module for i in range(N)])
    else:
        return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        num_queries=args.num_queries,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=args.return_intermediate_dec,
        activation=args.transformer_activation,
        num_feature_levels=args.num_feature_levels,
        enc_n_points=args.enc_n_points,
        dec_n_points=args.dec_n_points,
        learnable_tgt_init=args.learnable_tgt_init,
        two_stage_type=args.two_stage_type,
        num_body_points=args.num_body_points)