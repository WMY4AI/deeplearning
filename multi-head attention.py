"""
Author:Mingyang Wu
Day:04.02.2024
Abstract:
Tips:
"""
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key, value, mask=None, dropout=None):
    # qkv shape(batch, n_heads, seq_len, dv)
    # dv 表示每个头部分的维度大小
    dv = query.size(-1)
    x = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(dv)
    # mask处理，将x对应到mask为0的位置，置为一个负的极大值
    if mask is not None:
        x = x.masked_fill(mask == 0, -1e9)
    attention_map = F.softmax(x, dim=-1)
    if dropout is not None:
        attention_map = dropout(attention_map)

    return torch.matmul(attention_map, value)


class multiheadattention(nn.Module):
    def __init__(self, hs, n_heads, dropout=None):
        super(multiheadattention, self).__init__()
        self.hs = hs
        self.n_heads = n_heads
        assert hs % n_heads == 0
        self.dv = hs // n_heads             # 正确计算了每个头的维度大小
        self.dropout = nn.Dropout(dropout) if dropout is not None else None
        self.linears = clones(nn.Linear(hs, hs), 4)

    def forward(self, query, key, value, mask=None):
        # qkv shape(batch, seqlen, hs)
        # mask shape(batch, seqlen, seqlen)
        if mask:
            mask = mask.unsequenze(1)       # 确保mask的维度和注意力分数的维度对齐
        batchsize = query.size(0)

        # qkv projection
        # l(x) (batch, seqlen, hs) view(batch, seqlen, n_heads, dv) transpose(batch, n_heads, seqlen, dv)
        query, key, value = [l(x).view(batchsize, -1, self.n_heads, self.dv).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key, value))]

        # scaled dot product
        # x(batch, n_heads, seqlen, hs)
        x = attention(query, key, value, mask, self.dropout)
        # (batch, seqlen, n_heads, hs) (batch, seqlen, hs)
        x = x.transpose(1, 2).contiguous().view(batchsize, -1, self.hs)

        # 最终合并后的结果还要经过一个linear层
        return self.linears[-1](x)






