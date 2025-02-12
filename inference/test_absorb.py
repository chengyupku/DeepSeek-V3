# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn.functional as F
import argparse
from functools import partial

import random
seed = 42
random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# import sys
# sys.path.append('/home/aiscuser/cy/tilelang/3rdparty/deepseek_v3/inference')

import math
from dataclasses import dataclass
from typing import Tuple, Optional, Literal

from torch import nn
from model import ModelArgs, world_size, ColumnParallelLinear, Linear, RMSNorm, RowParallelLinear, apply_rotary_emb, weight_dequant

attn_impl = "naive"

class MLA(nn.Module):
    """
    Multi-Headed Attention Layer (MLA).

    Attributes:
        dim (int): Dimensionality of the input features.
        n_heads (int): Number of attention heads.
        n_local_heads (int): Number of local attention heads for distributed systems.
        q_lora_rank (int): Rank for low-rank query projection.
        kv_lora_rank (int): Rank for low-rank key/value projection.
        qk_nope_head_dim (int): Dimensionality of non-positional query/key projections.
        qk_rope_head_dim (int): Dimensionality of rotary-positional query/key projections.
        qk_head_dim (int): Total dimensionality of query/key projections.
        v_head_dim (int): Dimensionality of value projections.
        softmax_scale (float): Scaling factor for softmax in attention computation.
    """
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.n_local_heads = args.n_heads // world_size
        self.q_lora_rank = args.q_lora_rank
        self.kv_lora_rank = args.kv_lora_rank
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.qk_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim
        self.v_head_dim = args.v_head_dim

        if self.q_lora_rank == 0:
            self.wq = ColumnParallelLinear(self.dim, self.n_heads * self.qk_head_dim)
        else:
            self.wq_a = Linear(self.dim, self.q_lora_rank)
            self.q_norm = RMSNorm(self.q_lora_rank)
            self.wq_b_nope = ColumnParallelLinear(self.q_lora_rank, self.n_heads * self.qk_nope_head_dim)
            self.wq_b_rope = ColumnParallelLinear(self.q_lora_rank, self.n_heads * self.qk_rope_head_dim)
            self.wq_b = ColumnParallelLinear(self.q_lora_rank, self.n_heads * self.qk_head_dim)
            wq_b_nope_weight_view = self.wq_b_nope.weight.view(self.n_heads, self.qk_nope_head_dim, self.q_lora_rank)
            wq_b_rope_weight_view = self.wq_b_rope.weight.view(self.n_heads, self.qk_rope_head_dim, self.q_lora_rank)
            self.wq_b.weight.data = torch.cat([wq_b_nope_weight_view, wq_b_rope_weight_view], dim=1).view(self.n_heads * self.qk_head_dim, self.q_lora_rank)
         
        self.wkv_a = Linear(self.dim, self.kv_lora_rank + self.qk_rope_head_dim)
        self.kv_norm = RMSNorm(self.kv_lora_rank)
        self.wkv_b = ColumnParallelLinear(self.kv_lora_rank, self.n_heads * (self.qk_nope_head_dim + self.v_head_dim))

        wkv_b_weight = self.wkv_b.weight.clone()
        wkv_b_weight = wkv_b_weight.view(self.n_heads, -1, self.kv_lora_rank)
        wkv_b_weight = wkv_b_weight[:, :self.qk_nope_head_dim, :].contiguous().view(self.n_heads, self.qk_nope_head_dim, self.kv_lora_rank)
        self.wqk_weight = torch.einsum("hdq,hdk->hqk", wq_b_nope_weight_view, wkv_b_weight).cuda().to(torch.float16)

        self.wo = RowParallelLinear(self.n_heads * self.v_head_dim, self.dim)
        self.softmax_scale = self.qk_head_dim ** -0.5
        if args.max_seq_len > args.original_seq_len:
            mscale = 0.1 * args.mscale * math.log(args.rope_factor) + 1.0
            self.softmax_scale = self.softmax_scale * mscale * mscale

        # if attn_impl == "naive":
        #     self.register_buffer("kv_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.kv_lora_rank), persistent=False)
        #     self.register_buffer("pe_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.qk_rope_head_dim), persistent=False)
        # else:
        #     pass
        self.register_buffer("kv_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.kv_lora_rank), persistent=False)
        self.register_buffer("pe_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.qk_rope_head_dim), persistent=False)
   
            

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor], attn_impl="naive") -> torch.Tensor:
        """
        Forward pass for the Multi-Headed Attention Layer (MLA).

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim).
            start_pos (int): Starting position in the sequence for caching.
            freqs_cis (torch.Tensor): Precomputed complex exponential values for rotary embeddings.
            mask (Optional[torch.Tensor]): Mask tensor to exclude certain positions from attention.

        Returns:
            torch.Tensor: Output tensor with the same shape as the input.
        """
        bsz, seqlen, _ = x.size()
        end_pos = start_pos + seqlen
        if self.q_lora_rank == 0:
            q = self.wq(x)
        else:
            if attn_impl == "naive":
                q = self.wq_b(self.q_norm(self.wq_a(x)))
                q = q.view(bsz, seqlen, self.n_local_heads, self.qk_head_dim)
                q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
            elif attn_impl == "absorb":
                compressed_q = self.q_norm(self.wq_a(x))
                q_pe = self.wq_b_rope(compressed_q).view(bsz, seqlen, self.n_local_heads, self.qk_rope_head_dim)
        q_pe = apply_rotary_emb(q_pe, freqs_cis)
        kv = self.wkv_a(x)
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis)
        if attn_impl == "naive":
            wkv_b = self.wkv_b.weight if self.wkv_b.scale is None else weight_dequant(self.wkv_b.weight, self.wkv_b.scale, block_size) 
            wkv_b = wkv_b.view(self.n_local_heads, -1, self.kv_lora_rank)
            q_nope = torch.einsum("bshd,hdc->bshc", q_nope, wkv_b[:, :self.qk_nope_head_dim])
            self.kv_cache[:bsz, start_pos:end_pos] = self.kv_norm(kv)
            self.pe_cache[:bsz, start_pos:end_pos] = k_pe.squeeze(2)
            scores = (torch.einsum("bshc,btc->bsht", q_nope, self.kv_cache[:bsz, :end_pos]) +
                      torch.einsum("bshr,btr->bsht", q_pe, self.pe_cache[:bsz, :end_pos])) * self.softmax_scale
        else:
            wkv_b = self.wkv_b.weight if self.wkv_b.scale is None else weight_dequant(self.wkv_b.weight, self.wkv_b.scale, block_size) 
            wkv_b = wkv_b.view(self.n_local_heads, -1, self.kv_lora_rank)
            q_nope = torch.einsum("bsq,hqk->bshk", compressed_q, self.wqk_weight)
            print("absorb q_nope:", q_nope)
            self.kv_cache[:bsz, start_pos:end_pos] = self.kv_norm(kv)
            self.pe_cache[:bsz, start_pos:end_pos] = k_pe.squeeze(2)
            scores = (torch.einsum("bshk,btk->bsht", q_nope, self.kv_cache[:bsz, :end_pos]) +
                      torch.einsum("bshr,btr->bsht", q_pe, self.pe_cache[:bsz, :end_pos])) * self.softmax_scale
        
        if mask is not None:
            scores += mask.unsqueeze(1)
        scores = scores.softmax(dim=-1, dtype=torch.float32).type_as(x)
        if attn_impl == "naive":
            x = torch.einsum("bsht,btc->bshc", scores, self.kv_cache[:bsz, :end_pos])
            x = torch.einsum("bshc,hdc->bshd", x, wkv_b[:, -self.v_head_dim:])
        else:
            x = torch.einsum("bsht,btc->bshc", scores, self.kv_cache[:bsz, :end_pos])
            x = torch.einsum("bshc,hdc->bshd", x, wkv_b[:, -self.v_head_dim:])
        x = self.wo(x.flatten(2))
        return x


def ref_program(x, freqs_cis, mla):
    return mla(x, 0, freqs_cis, None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=1, help='batch size')
    parser.add_argument('--seq_len', type=int, default=1024, help='sequence length')
    parser.add_argument('--dim', type=int, default=2048, help='dim')
    parser.add_argument('--is_causal', action='store_true', help='causal')
    args = parser.parse_args()

    batch_size, seq_len, dim = args.batch, args.seq_len, args.dim
    qk_rope_head_dim = 64

    args = ModelArgs(
        q_lora_rank=1536
    )
    
    # print(args)
    x = torch.randn(batch_size, seq_len, dim, dtype=torch.float16, device='cuda')
    freqs_cis = torch.randn(seq_len, qk_rope_head_dim // 2, dtype=torch.float16, device='cuda')

    mla = MLA(args).half().cuda()

    # print("x0:", x)
    # print("freqs_cis0:", freqs_cis)
    # print("x1:", x)
    # print("freqs_cis1:", freqs_cis)
    out_naive = mla(x, 0, freqs_cis, None, attn_impl="naive")
    out_absorb = mla(x, 0, freqs_cis, None, attn_impl="absorb")
    print("out_naive", out_naive)
    print("out_absorb", out_absorb)
    # torch.testing.assert_close(out_naive, out_absorb, rtol=1e-1, atol=1e-1)