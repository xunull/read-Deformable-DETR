# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import copy
from typing import Optional, List
import math

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_

from util.misc import inverse_sigmoid
# 多尺度可变性Attention模块
from models.ops.modules import MSDeformAttn


class DeformableTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8,
                 num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 activation="relu", return_intermediate_dec=False,
                 num_feature_levels=4,
                 # 这两个4应该就是这个论文主要的Deformable的内容
                 dec_n_points=4,
                 enc_n_points=4,

                 two_stage=False, two_stage_num_proposals=300):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.two_stage = two_stage
        self.two_stage_num_proposals = two_stage_num_proposals

        encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, enc_n_points)
        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)

        decoder_layer = DeformableTransformerDecoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, dec_n_points)
        self.decoder = DeformableTransformerDecoder(decoder_layer, num_decoder_layers, return_intermediate_dec)

        # deformable detr 多的内容 论文中有提及
        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        # two_stage 使用的内容
        if two_stage:

            self.enc_output = nn.Linear(d_model, d_model)

            self.enc_output_norm = nn.LayerNorm(d_model)

            self.pos_trans = nn.Linear(d_model * 2, d_model * 2)

            self.pos_trans_norm = nn.LayerNorm(d_model * 2)
        else:
            self.reference_points = nn.Linear(d_model, 2)
        # 参数重制
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        if not self.two_stage:
            xavier_uniform_(self.reference_points.weight.data, gain=1.0)
            constant_(self.reference_points.bias.data, 0.)
        normal_(self.level_embed)

    def get_proposal_pos_embed(self, proposals):
        """
        获取proposal对应的位置嵌入 [bs,300,4]
        """

        num_pos_feats = 128

        temperature = 10000

        scale = 2 * math.pi
        # 0->128
        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=proposals.device)
        # 类似于Transformer的位置嵌入公式
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        # sigmoid然后缩放
        # N, L, 4
        proposals = proposals.sigmoid() * scale

        # N, L, 4, 128
        pos = proposals[:, :, :, None] / dim_t

        # N, L, 4, 64, 2 -> flatten(2) -> [bs,300,512]
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4).flatten(2)

        return pos

    def gen_encoder_output_proposals(self, memory, memory_padding_mask, spatial_shapes):
        """
        two stage的情况时会被调用
        """
        # bs, all hw, channel
        N_, S_, C_ = memory.shape
        base_scale = 4.0

        proposals = []
        # 定位在memory_padding_mask中的位置，memory_padding_mask这里是所有特征层的内容在一起了
        _cur = 0
        # 某个特征层的 高 宽
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            # 获取mask, memory_padding_mask is[bs,all hw] [bs,H_,W_,1]
            mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H_ * W_)].view(N_, H_, W_, 1)
            # 有效的高度
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            # 有效的宽度
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)
            # 生成网格点，这里没有0.5的做法 get_reference_points这个方法中是从0.5开始，这里是从0开始
            grid_y, grid_x = torch.meshgrid(torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
                                            torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device))
            # 生成网格二维点
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

            # scale [bs,1,1,2]
            scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N_, 1, 1, 2)
            # 获取相对值 [bs,h,w,2]
            grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale
            # [bs,h,w,2] 宽高的相对值，不同的特征层 wh不同,层级越高，wh相对也会更大
            wh = torch.ones_like(grid) * 0.05 * (2.0 ** lvl)
            # [bs,h,w,4] -> [bs,hw,4]
            proposal = torch.cat((grid, wh), -1).view(N_, -1, 4)

            proposals.append(proposal)

            _cur += (H_ * W_)
        # [bs,all hw,4]
        output_proposals = torch.cat(proposals, 1)
        # [bs,all hw,1] proposals 不需要太靠近边界
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
        # sigmoid的反函数
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        # 对于mask部分，填充inf
        output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float('inf'))
        # 靠近边界的部分，填充inf
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float('inf'))

        # 并不修改memory
        output_memory = memory
        # 对于memory的输出进行同样的填充
        output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
        # [bs,all hw,256]
        output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))

        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        # output_memory 是memory 经过填充inf，经过一层全连接后的结果
        # output_proposals 是制作的proposals，非法的位置填充了inf
        return output_memory, output_proposals

    def get_valid_ratio(self, mask):
        """
        图像的大小在一个batch内统一成了最大的高宽，但是具体的一个张图会占据其中左上的某个区域，其他的区域在Mask中是True
        这里就要求出真实的图像大小占据的比率
        mask [bs,h,w]
        有效高宽占总的高宽的比率
        """
        # 特征图的高 宽
        _, H, W = mask.shape

        # tensor的size就是bs的大小 [:,:,0] 就是取的第一列，那么就是高度的意思，~mask，有mask的位置是True，~mask取反
        # 有效的高度
        valid_H = torch.sum(~mask[:, :, 0], 1)
        # 有效的宽度
        valid_W = torch.sum(~mask[:, 0, :], 1)
        # 占总长度的比例
        valid_ratio_h = valid_H.float() / H
        # 占总宽度的比例
        valid_ratio_w = valid_W.float() / W

        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)

        return valid_ratio

    # 这里的内容比原始的DETR多了很多
    def forward(self, srcs, masks, pos_embeds, query_embed=None):
        """
        srcs backbone的特征
        masks seg使用的
        pos_embeds 位置编码
        query_embed decoder使用
        """
        # 有two_stage 或者不是two_stage但是要有query_embed
        assert self.two_stage or query_embed is not None

        # prepare input for encoder
        src_flatten = []

        mask_flatten = []

        lvl_pos_embed_flatten = []
        # 特征图的高宽
        spatial_shapes = []
        # src内的特征图是从大尺寸到小尺寸
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            # 特征图的高宽
            spatial_shape = (h, w)

            spatial_shapes.append(spatial_shape)
            # [bs,hw,256]
            src = src.flatten(2).transpose(1, 2)  # 这块维度的顺序与detr不同 detr 是 hw，bs，dim，这里是 bs,hw, dim
            # [bs,hw]
            mask = mask.flatten(1)
            # [bs,hw,256]
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            # level embed 会被加入到位置编码中,level是在论文中提到过的
            # 这个还多个level pos embed [bs,hw,256]
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)

            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)

        # [bs,all hw,256]
        # 所有特征层的拼在一起
        # 他们就是在维度1上长度不同，尺寸越大的特征层，1上的数量越多
        src_flatten = torch.cat(src_flatten, 1)

        # [bs,all hw]
        mask_flatten = torch.cat(mask_flatten, 1)

        # [bs, all hw,256]
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)

        # [特征层的数量，2] 存储的是高宽
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)

        # 各个src层 起始的位置, 第一个spatial_shapes.new_zeros((1,))是在起始位置填的0
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

        # 有效高宽占总的batch高宽的比率 [bs,4,2]
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        # src_flatten 类似detr中的src，mask_flatten应该是对应了src_key_padding_mask，lvl_pos_embed_flatten对应了pos
        # 其他的三个完全是这里多出来的参数
        # encoder
        # [bs,all hw,256]
        memory = self.encoder(src_flatten, spatial_shapes, level_start_index,
                              valid_ratios, lvl_pos_embed_flatten, mask_flatten)

        # prepare input for decoder
        bs, _, c = memory.shape

        if self.two_stage:
            # output_memory 是memory 经过填充inf，经过一层全连接后的结果 [bs,all hw,256]
            # output_proposals 是制作的proposals，非法的位置填充了inf [bs,all hw,4]
            output_memory, output_proposals = self.gen_encoder_output_proposals(memory, mask_flatten, spatial_shapes)

            # hack implementation for two-stage Deformable DETR
            # encoder的输出直接使用最后的分类头得到输出结果
            # 作为一个返回值
            # [bs,all hw,91]
            enc_outputs_class = self.decoder.class_embed[self.decoder.num_layers](output_memory)
            # encoder的输出直接使用最后的bbox头得到输出结果
            # 作为一个返回值
            # 原始代码这里decoder.bbox_embed 并未赋值 [2,all hw,4]
            enc_outputs_coord_unact = self.decoder.bbox_embed[self.decoder.num_layers](output_memory) + output_proposals

            topk = self.two_stage_num_proposals
            # 前300个proposals
            topk_proposals = torch.topk(enc_outputs_class[..., 0], topk, dim=1)[1]
            # 取出对应的300个proposals的预测值 [bs,300,4]
            topk_coords_unact = torch.gather(enc_outputs_coord_unact, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4))
            # 脱离gpu
            topk_coords_unact = topk_coords_unact.detach()
            # 相对值，都压制在0-1内
            reference_points = topk_coords_unact.sigmoid()

            init_reference_out = reference_points
            # [bs,300,512] 先获取pos_embed，然后经过网络，然后经过layer norm
            pos_trans_out = self.pos_trans_norm(self.pos_trans(self.get_proposal_pos_embed(topk_coords_unact)))
            # 分解出query_embed和tgt
            query_embed, tgt = torch.split(pos_trans_out, c, dim=2)
        else:
            # query_embed [300,512] -> [300,256] tgt [300,256]
            query_embed, tgt = torch.split(query_embed, c, dim=1)

            # 扩出第0维度，bs
            query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)
            # 扩充第0维度成为bs
            tgt = tgt.unsqueeze(0).expand(bs, -1, -1)

            # [bs,300,2]  reference_points为全连接层 512-2，生成出参考点位坐标
            # 这些坐标经过了sigmoid处理，在最后得到修正的点的坐标时，还会使用sigmoid的反函数
            reference_points = self.reference_points(query_embed).sigmoid()
            # 经过网络初始生成出的参考点坐标
            init_reference_out = reference_points

        # decoder
        # tgt 从query_embed分出来的 [bs,300,256]
        # reference_points  query_embed经过全连接网络生成的 [bs,300,2] 为初始的参考点坐标
        # memory encoder的输出 [bs, all hw,256]
        # spatial_shapes 各个特征层的高宽 [4,2]
        # level_start_index 各个特征层的起始下标
        # valid_ratios [bs,4,2]
        # query_embed [bs,300,256]
        # mask_flatten [bs,all hw]
        #
        # hs [6,bs,300,256]
        # inter_references [6,bs,300,2]
        hs, inter_references = self.decoder(tgt, reference_points, memory,
                                            spatial_shapes, level_start_index, valid_ratios,
                                            query_embed, mask_flatten)

        inter_references_out = inter_references

        # 后两个只有two_stage的时候有值
        if self.two_stage:
            # encoder的输出经过最后的分类头得到的输出
            # enc_outputs_class [bs,all hw,91]
            # encoder的输出经过最后的bbox头得到的输出
            # enc_outputs_coord_unact [bs,all hw,4]
            return hs, init_reference_out, inter_references_out, enc_outputs_class, enc_outputs_coord_unact

        return hs, init_reference_out, inter_references_out, None, None


class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8,
                 # Deformable的内容
                 n_points=4):
        super().__init__()

        # self attention
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn 的内容
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        # 一个全连接，一个激活，一个dropout，一个全连接
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        # add 和 dropout
        src = src + self.dropout3(src2)
        # norm
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        """
        src: [bs,all hw,256]
        pos: [bs,all hw,256]
        reference_points: [bs,all hw,4,2]
        spatial_shapes: [4,2] 4个特征层的高宽
        level_start_index: 各个特征层的起始index的下标 如: [    0,  8056, 10070, 10583]
        padding_mask: [bs,all hw]
        """
        # with_pos_embed 就是将src和pos相加, src是上一层encoder的输出
        # 1. self attention
        src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src,
                              spatial_shapes, level_start_index, padding_mask)
        # 2. add
        src = src + self.dropout1(src2)
        # 3. norm
        src = self.norm1(src)

        # 4. ffn + add & norm
        src = self.forward_ffn(src)

        return src


class DeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        # spatial_shapes [特征层数,2] valid_ratios [bs,特征层数,2]
        reference_points_list = []

        for lvl, (H_, W_) in enumerate(spatial_shapes):
            # 生成网格点,从0.5开始 到 减掉一个0.5
            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))

            # 坐标进行缩放 valid_ratios[:, None, lvl, 1] * H_是在H_基础上进一步缩减范围

            # reshape(-1) 拉平会变成一维的 shape=hw，[None]，会在最前面加上一个1维度 -> [1,hw] -> [2,hw]
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            # [2,hw]
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            # [2,hw,2]
            ref = torch.stack((ref_x, ref_y), -1)

            reference_points_list.append(ref)
        # 所有特征层的参考点拼在一起 [bs,all hw,2]
        reference_points = torch.cat(reference_points_list, 1)

        # reference_points[:,:,None] -> [2,all hw,1,2]
        # valid_ratios[:,None] -> [bs,1,特征层数量,2]
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        # [2,all hw,4,2]
        return reference_points

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None):
        """
        src: [bs,all hw,256] backbone的特征
        spatial_shapes: [特征层的数量,2] 各个特征层的高宽
        level_start_index: 各个层的 all hw中的起始坐标位置
        valid_ratios: [bs,4,2] todo 不知道什么用
        pos: [bs,all hw,256] 位置编码
        padding_mask: [bs, all hw]
        """
        output = src
        # 获取参考点
        # encoder的参考点是grid生成的，并且不会精炼，不会有迭代的更新
        # [bs,all hw,4,2]
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        for _, layer in enumerate(self.layers):
            # [bs,all hw,256]
            output = layer(output, pos, reference_points,
                           spatial_shapes, level_start_index, padding_mask)

        return output


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()
        # Deformable DETR实现的Attention
        # cross attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        # 标准的Attention
        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, tgt, query_pos, reference_points,
                src, src_spatial_shapes, level_start_index,
                src_padding_mask=None):
        """
        tgt 是上一层的输出 [bs,300,256]
        query_pos 就是外面的query_embed [bs,300,256]
        reference_points 各个image在300个query上每个特征层上的参考点的坐标 [bs,300,4,2]
        src是encoder的输出 [bs,all hw,256]
        src_spatial_shapes 各个特征层的高宽 [4,2]
        level_start_index 各个特征层的起始下标
        src_padding_mask mask
        """

        # self attention
        # tgt和query_pos相加, q k就是这两个构成了
        q = k = self.with_pos_embed(tgt, query_pos)
        # 1. self-attention, q k在上面创建了，value就还是tgt
        tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1))[0].transpose(0, 1)
        # 2. add
        tgt = tgt + self.dropout2(tgt2)
        # 3. norm
        tgt = self.norm2(tgt)

        # Deformable DETR实现的attention
        # 4. cross attention

        # reference_points 各个image在300个query上每个特征层上的目标点位 [bs,300,4,2]
        # src是encoder的输出 [bs,all hw,256]
        # src_spatial_shapes 各个特征层的高宽 [4,2]
        # level_start_index 各个特征层的起始下标
        # src_padding_mask mask
        tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos),
                               reference_points,
                               src,
                               src_spatial_shapes, level_start_index, src_padding_mask)

        # 5. add
        tgt = tgt + self.dropout1(tgt2)
        # 6. norm
        tgt = self.norm1(tgt)

        # 7. ffn add & norm
        tgt = self.forward_ffn(tgt)

        return tgt


class DeformableTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate

        # hack implementation for iterative bounding box refinement and two-stage Deformable DETR
        self.bbox_embed = None
        self.class_embed = None

    def forward(self, tgt, reference_points, src,
                src_spatial_shapes, src_level_start_index, src_valid_ratios,
                query_pos=None, src_padding_mask=None):
        """
        tgt 从query_embed分出来的 [bs,300,256]
        reference_points reference_point query_embed经过全连接网络生成的 [bs,300,2], 参考点的初始参考点
        src encoder的输出 [bs, all hw,256]
        src_spatial_shapes 各个特征层的高宽 [4,2]
        src_level_start_index 各个特征层的起始下标
        src_valid_ratios [bs,4,2] 各个图像真实的高宽占据Mask大小中的比率
        query_pos 就是外面的query_embed [bs,300,256]
        src_padding_mask [bs,all hw]
        """

        output = tgt

        intermediate = []

        intermediate_reference_points = []

        for lid, layer in enumerate(self.layers):

            if reference_points.shape[-1] == 4:
                # 不过现在代码中生成参考点的Linear输出就是维度就是2，并非设置，可能作者在实验中试验过=4的情况
                reference_points_input = reference_points[:, :, None] \
                                         * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:, None]
            else:
                # 参考点的坐标
                assert reference_points.shape[-1] == 2
                # [bs,300,2] -> [bs,300,1,2] [bs,4,2] -> [bs,1,4,2]
                # reference_points_input [bs,300,4,2]
                # 参考点坐标也要按比例的进行缩放
                reference_points_input = reference_points[:, :, None] * src_valid_ratios[:, None]

            # 如果没有使用bbox精炼，那么每次的reference_points_input 其实是相同的
            # 如果使用了bbox精炼，那么每次的reference_points_input 是不同的，会使用decoder的输出得到偏移量修正

            # output 是上一层的输出
            # query_pos 就是外面的query_embed
            # reference_points_input 各个image在300个query上每个特征层上的目标点位
            # src是encoder的输出
            # src_spatial_shapes 各个特征层的高宽
            # src_level_start_index 各个特征层的起始下标
            # src_padding_mask mask
            output = layer(output, query_pos, reference_points_input,
                           src, src_spatial_shapes, src_level_start_index,
                           src_padding_mask)

            # 如果使用了with_box_refine模式, 这个地方的bbox_embed 是非None的
            # hack implementation for iterative bounding box refinement
            if self.bbox_embed is not None:
                # output 是上一层decoder的输出，经过bbox预测网络 [bs,300,4]
                # 得到此次的bbox的偏移量修正
                tmp = self.bbox_embed[lid](output)

                if reference_points.shape[-1] == 4:
                    # inverse_sigmoid sigmoid的反函数
                    new_reference_points = tmp + inverse_sigmoid(reference_points)

                    new_reference_points = new_reference_points.sigmoid()
                else:
                    assert reference_points.shape[-1] == 2  # [bs,300,2]
                    # [bs,300,4]
                    new_reference_points = tmp
                    # 前两个是bbox的中心坐标，中心坐标用reference_points的内容+偏移量修正
                    new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
                    # [bs,300,4] 经过sigmoid 约束在0-1
                    new_reference_points = new_reference_points.sigmoid()
                # 替换了原来的reference_points
                reference_points = new_reference_points.detach()

            # 返回前几层，否则只返回最后一层
            if self.return_intermediate:
                intermediate.append(output)
                # 如果是精炼模式，reference_points在各个层之后是不同的
                # 如果不是精炼模型，reference_points在各个层之后还是同一份内容
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)

        return output, reference_points


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def build_deformable_transformer(args):
    return DeformableTransformer(
        d_model=args.hidden_dim,
        # 头的数量
        nhead=args.nheads,
        # encoder的层数
        num_encoder_layers=args.enc_layers,
        # decoder的层数
        num_decoder_layers=args.dec_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        # 激活函数
        activation="relu",
        # 返回decoder的中间层的输出
        return_intermediate_dec=True,
        num_feature_levels=args.num_feature_levels,
        dec_n_points=args.dec_n_points,
        enc_n_points=args.enc_n_points,
        # 双阶段方式
        two_stage=args.two_stage,
        two_stage_num_proposals=args.num_queries)
