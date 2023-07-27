# ------------------------------------------------------------------------
# Modified from Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

from typing import Optional
from torch import nn, Tensor
from .ops.modules import MSDeformAttn
from .utils import  _get_activation_fn


class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4,
                 ):
        super().__init__()
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation, d_model=d_ffn)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, key_padding_mask=None):
        src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index, key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src = self.forward_ffn(src)
        return src


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()
        # within-instance self-attention
        self.within_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.within_dropout = nn.Dropout(dropout)
        self.within_norm = nn.LayerNorm(d_model)
        # across-instance self-attention
        self.across_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.across_dropout = nn.Dropout(dropout)
        self.across_norm = nn.LayerNorm(d_model)
        # deformable cross-attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        # FFN
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation, d_model=d_ffn, batch_dim=1)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        if pos is not None:
            np = pos.shape[2]
            tensor[:, :, -np:] += pos
        return tensor

    def forward_FFN(self, tgt):
        tgt2 = self.linear2(self.dropout2(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm2(tgt)
        return tgt

    def forward(self, 
                # for tgt
                tgt_pose: Optional[Tensor],
                tgt_pose_query_pos: Optional[Tensor],
                tgt_pose_reference_points: Optional[Tensor],
                # for memory
                memory: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                memory_level_start_index: Optional[Tensor] = None,
                memory_spatial_shapes: Optional[Tensor] = None,
            ):
        # within-instance self-attention
        q = k = self.with_pos_embed(tgt_pose, tgt_pose_query_pos)
        tgt2 = self.within_attn(q.flatten(0, 1).transpose(0, 1), k.flatten(0, 1).transpose(0, 1), tgt_pose.flatten(0, 1).transpose(0, 1))[0].transpose(0, 1).reshape(q.shape)
        tgt_pose = tgt_pose + self.within_dropout(tgt2)
        tgt_pose = self.within_norm(tgt_pose)
        # across-instance self-attention
        q_pose = k_pose = tgt_pose
        tgt2_pose = self.across_attn(q_pose.flatten(1, 2), k_pose.flatten(1, 2), tgt_pose.flatten(1, 2))[0].reshape(q_pose.shape)
        tgt_pose = tgt_pose + self.across_dropout(tgt2_pose)
        tgt_pose = self.across_norm(tgt_pose)
        # deformable cross-attention
        nq, bs, np, d_model = tgt_pose.shape
        tgt2_pose = self.cross_attn(self.with_pos_embed(tgt_pose, tgt_pose_query_pos).transpose(0, 1).flatten(1, 2),
                                         tgt_pose_reference_points.transpose(0, 1),
                                         memory.transpose(0, 1), memory_spatial_shapes, 
                                         memory_level_start_index, memory_key_padding_mask).reshape(bs, nq, np, d_model).transpose(0, 1)
        tgt_pose = tgt_pose + self.dropout1(tgt2_pose)
        tgt_pose = self.norm1(tgt_pose)
        tgt_pose = self.forward_FFN(tgt_pose)
        return tgt_pose
