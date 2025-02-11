from layers.NanoGPTBlock import Block as GPTDecoderBlock
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import math
from types import SimpleNamespace

import matplotlib.pyplot as plt

@torch.compile
def rotary_embedding(max_len, dim):
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
    position = torch.arange(0, max_len).unsqueeze(1)
    sinusoid_inp = torch.ger(position.squeeze(), inv_freq).float()
    return torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)

def apply_rotary_pos_emb(x, sincos, reverse=False):
    B, seq_len, dim = x.shape
    if reverse:
        sincos = sincos[0:seq_len].flip(0)
    else:
        sincos = sincos[0:seq_len]
    sin, cos = sincos[:, :dim//2].repeat_interleave(2, dim=-1), sincos[:, dim//2:].repeat_interleave(2, dim=-1)
    return (x * cos) + (torch.roll(x, shifts=1, dims=-1) * sin)

class TransformerEncoder(nn.Module):
    def __init__(self, emb_size=128, depth=2, heads=4, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.pre_norm = nn.LayerNorm(emb_size)
        self.final_norm = nn.LayerNorm(emb_size)
        for _ in range(depth):
            self.layers.append(GPTDecoderBlock(SimpleNamespace(n_embd=emb_size, bias=False, dropout=dropout, n_head=heads)))
            
    def forward(self, x, src_mask=None, output_attention=False):
        x = self.pre_norm(x)
        attention_maps = []
        for layer in self.layers:
            if output_attention:
                x, attention = layer(x, src_mask=src_mask, output_attention=output_attention)
                attention_maps.append(attention)
            else:
                x = layer(x, src_mask=src_mask)
        x = self.final_norm(x)
        if output_attention:
            return x, attention_maps
        return x

class ICPretrain(nn.Module):
    def __init__(self, configs):
        super().__init__()
        emb_init = 0.002
        
        self.lookback = configs.lookback
        self.future = configs.future
        self.d_model = configs.d_model
        self.tokenization_mode = configs.tokenization_mode    # 'compact', 'sequential'
        if self.tokenization_mode == 'compact':
            self.x_projection = nn.Linear(self.lookback, self.d_model//2)
            self.y_projection_reg = nn.Linear(self.future, self.d_model//2)
        elif self.tokenization_mode == 'sequential':
            self.x_projection = nn.Linear(self.lookback, self.d_model)
            self.y_projection_reg = nn.Linear(self.future, self.d_model)
        self.y_projection_cls = nn.Embedding(4096, self.future, padding_idx=0) # vocab size is not implemented: todo
        self.output_projection_reg = nn.Linear(self.d_model, self.future)
        self.output_projection_cls = nn.Linear(self.d_model, 4096)
        
        self.nan_embedding = nn.Parameter(emb_init*torch.randn(1, 1, self.lookback + self.future))
        self.ana_embedding = nn.Parameter(emb_init*torch.randn(1, 1, self.lookback + self.future))
        self.tgt_embedding = nn.Parameter(emb_init*torch.randn(1, 1, self.future))
        self.gtg_embedding = nn.Parameter(emb_init*torch.randn(1, 1, self.future))
        
        self.channel_embedding = nn.Embedding(configs.max_channel_vocab_size + 1, self.d_model, padding_idx=0)
        self.channel_embedding.weight.data = self.channel_embedding.weight.data * emb_init
        
        self.group_position_embedding = nn.Embedding(configs.max_position_vocab_size + 1, self.d_model, padding_idx=0)
        self.group_position_embedding.weight.data = self.group_position_embedding.weight.data * emb_init
        
        self.source_embedding = nn.Embedding(configs.max_source_vocab_size + 1, self.d_model, padding_idx=0)
        self.source_embedding.weight.data = self.source_embedding.weight.data * emb_init
        
        self.tag_embedding_projection = nn.Linear(configs.max_tag_vocab_size, self.d_model)
        
        self.task_embedding = nn.Embedding(32, self.d_model)
        self.task_embedding.weight.data = self.task_embedding.weight.data * emb_init
        
        self.enable_task_embedding = configs.enable_task_embedding
        
        #self.RoPE = nn.Parameter(rotary_embedding(configs.hard_token_limit, self.d_model), requires_grad=False)
        
        self.transformer = TransformerEncoder(self.d_model, configs.n_layers, configs.n_heads, configs.mlp_ratio, dropout=configs.dropout)
        self.process_output = False

        self.number_of_targets = getattr(configs, "number_of_targets", 0)

        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * configs.n_layers))

        torch.set_printoptions(
            edgeitems=10,
            linewidth=200
        )

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.002)
    
    def forward(self, 
                token_x_part, token_y_part, 
                channel_label, position_label, source_label, tag_multihot,
                y_true_shape, task_ids):
        # pad to normal tensors
        # token_x_part = torch.nested.to_padded_tensor(token_x_part, 0.0)
        # token_y_part = torch.nested.to_padded_tensor(token_y_part, 0.0)
        # channel_label = torch.nested.to_padded_tensor(channel_label, 0)
        # position_label = torch.nested.to_padded_tensor(position_label, 0)
        # source_label = torch.nested.to_padded_tensor(source_label, 0)
        # tag_multihot = torch.nested.to_padded_tensor(tag_multihot, 0)

        if self.tokenization_mode == 'sequential':
            token_y_part_snapshot = token_y_part.clone()

        with torch.no_grad():
            padding_mask_orig = torch.all(token_x_part == 0, dim=-1, keepdim=True).logical_not()   # B N 1
            padding_mask = padding_mask_orig.permute(0, 2, 1)                        # B 1 N
            padding_mask = padding_mask.unsqueeze(-1) & padding_mask.unsqueeze(-2)   # B 1 N N

            B, L, _ = token_x_part.shape
            tril_mask = torch.tril(torch.ones(L, L, dtype=torch.bool, device=token_x_part.device), diagonal=0).unsqueeze(0).unsqueeze(0)  # 1 1 N N
            position_label_unsqueeze = position_label.unsqueeze(-1)
            position_mask = position_label_unsqueeze.expand(B, L, L) != position_label_unsqueeze.transpose(1, 2).expand(B, L, L)
            final_mask = tril_mask & position_mask.unsqueeze(1)   # B 1 N N  
            
            if self.tokenization_mode == 'sequential':
                final_mask = final_mask.repeat_interleave(2, dim=-1).repeat_interleave(2, dim=-2)
                padding_mask = padding_mask.repeat_interleave(2, dim=-1).repeat_interleave(2, dim=-2)
                diag_mask = torch.eye(2*L, dtype=torch.bool, device=token_x_part.device).unsqueeze(0).unsqueeze(0)
                position_xy_mask_orig = position_label_unsqueeze.expand(B, L, L) == position_label_unsqueeze.transpose(1, 2).expand(B, L, L)

                position_xy_mask = torch.cat([position_xy_mask_orig.unsqueeze(-1), position_xy_mask_orig.logical_not().unsqueeze(-1)], dim=-1).view(B, L, 2*L)
                position_xy_mask = torch.cat([position_xy_mask.unsqueeze(-2), position_xy_mask.logical_not().unsqueeze(-2)], dim=2).view(B, 2*L, 2*L)

                position_xy_mask_samepos = position_xy_mask_orig.repeat_interleave(2, dim=-1).repeat_interleave(2, dim=-2)
                position_xy_mask = position_xy_mask & position_xy_mask_samepos
                
                position_xy_mask = position_xy_mask.unsqueeze(1)
                
                final_mask = final_mask | position_xy_mask
                final_mask = final_mask & padding_mask
                final_mask = final_mask | diag_mask

            else:
                diag_mask = torch.eye(L, dtype=torch.bool, device=token_x_part.device).unsqueeze(0).unsqueeze(0)
                final_mask = final_mask & padding_mask
                final_mask = final_mask | diag_mask

        # implement target embedding
        # assert L != 0, task_ids
        if task_ids[0] != 1:
            inf_mask = torch.isinf(token_y_part)#token_y_part == float('inf')
            if self.tokenization_mode == 'compact':
                tgt_embedding = self.tgt_embedding.expand_as(token_y_part)
                gtg_embedding = self.gtg_embedding.expand_as(token_y_part)
                token_y_part = token_y_part.masked_fill(inf_mask, 0) + tgt_embedding.masked_fill(~inf_mask, 0) + gtg_embedding.masked_fill(inf_mask, 0)
            elif self.tokenization_mode == 'sequential':
                token_y_part = torch.where(inf_mask, float('nan'), token_y_part)
        else:
            token_y_part = self.y_projection_cls(token_y_part)
            
        channel_embedding = self.channel_embedding(channel_label)                      # B N d
        group_position_embedding = self.group_position_embedding(position_label)       # B N d
        source_embedding = self.source_embedding(source_label)                         # B N d
        tag_embedding = 0 #self.tag_embedding_projection(tag_multihot)                 # B N d

        if self.tokenization_mode == 'compact':
            # construct input tokens
            tokens = torch.cat([token_x_part, token_y_part], dim=-1)        # B N lb+ft
    
            nan_mask = torch.isnan(tokens)
            nan_embedding = self.nan_embedding.expand_as(tokens).masked_fill(~nan_mask, 0)
            ana_embedding = self.ana_embedding.expand_as(tokens).masked_fill(nan_mask, 0)
    
            tokens = tokens.masked_fill(nan_mask, 0)
            tokens = tokens + nan_embedding + ana_embedding
            tokens = torch.cat([
                self.x_projection(tokens[:, :, 0:self.lookback]),
                self.y_projection_reg(tokens[:, :, -self.future:])
            ], dim=-1)                                                      # B N d

        elif self.tokenization_mode == 'sequential':
            nan_mask_x_part = torch.isnan(token_x_part)
            nan_embedding_x_part = self.nan_embedding[:, :, 0:self.lookback].expand_as(token_x_part).masked_fill(~nan_mask_x_part, 0)
            ana_embedding_x_part = self.ana_embedding[:, :, 0:self.lookback].expand_as(token_x_part).masked_fill(nan_mask_x_part, 0)
            nan_embedding_x_part = nan_embedding_x_part.masked_fill(~padding_mask_orig, 0)
            ana_embedding_x_part = ana_embedding_x_part.masked_fill(~padding_mask_orig, 0)
            token_x_part = token_x_part.masked_fill(nan_mask_x_part, 0) + nan_embedding_x_part + ana_embedding_x_part

            nan_mask_y_part = torch.isnan(token_y_part)
            nan_embedding_y_part = self.nan_embedding[:, :, -self.future:].expand_as(token_y_part).masked_fill(~nan_mask_y_part, 0)
            ana_embedding_y_part = self.ana_embedding[:, :, -self.future:].expand_as(token_y_part).masked_fill(nan_mask_y_part, 0)
            nan_embedding_y_part = nan_embedding_y_part.masked_fill(~padding_mask_orig, 0)
            ana_embedding_y_part = ana_embedding_y_part.masked_fill(~padding_mask_orig, 0)
            token_y_part = token_y_part.masked_fill(nan_mask_y_part, 0) + nan_embedding_y_part + ana_embedding_y_part
            
            token_x_part = self.x_projection(token_x_part)
            token_y_part = self.y_projection_reg(token_y_part)
            
            B, N, d = token_x_part.shape
            tokens = torch.cat([token_x_part.unsqueeze(2), token_y_part.unsqueeze(2)], dim=2).view(B, N * 2, d)  # B 2*N d

            # expand embeddings
            channel_embedding = channel_embedding.repeat_interleave(2, dim=1)
            group_position_embedding = group_position_embedding.repeat_interleave(2, dim=1)
            source_embedding = source_embedding.repeat_interleave(2, dim=1)
            tag_embedding = tag_embedding.repeat_interleave(2, dim=1) if tag_embedding else 0
            
        else:
            raise NotImplementedError

        if self.enable_task_embedding:
            task_id_label = task_ids.unsqueeze(1).expand(-1, tokens.shape[1])          # B -> B N
            task_embedding = self.task_embedding(task_id_label)
        else:
            task_embedding = 0
        
        tokens = tokens + channel_embedding + group_position_embedding + source_embedding + task_embedding + tag_embedding

        # rotary position embedding
        #tokens = apply_rotary_pos_emb(tokens, self.RoPE, reverse=True)

        # transformer blocks
        tokens = self.transformer(tokens, src_mask=final_mask) #

        if self.tokenization_mode == 'sequential':
            tokens = tokens[:, ::2, :]
            if task_ids[0] != 1:
                output = self.output_projection_reg(tokens)
                padding_mask = padding_mask_orig.expand_as(output) & nan_mask_y_part.logical_not()
                additional_loss = F.mse_loss(output[:, 1:, -self.number_of_targets:][padding_mask[:, 1:, -self.number_of_targets:]], 
                                             token_y_part_snapshot[:, 1:, -self.number_of_targets:][padding_mask[:, 1:, -self.number_of_targets:]])
            else:
                output = self.output_projection_cls(tokens)
                padding_mask = padding_mask_orig.expand_as(output)
                additional_loss = F.cross_entropy(output[padding_mask], token_y_part_snapshot[padding_mask])
            
        if self.process_output:
            # y_true_shape: B C L
            if task_ids[0] != 1:
                # prune the output
                n_max, ft_max = y_true_shape.max(dim=0)[0]
                n_max, ft_max = n_max.item(), ft_max.item()
                tokens = tokens[:, -n_max:]
                output = self.output_projection_reg(tokens) if self.tokenization_mode == 'compact' else output[:, -n_max:]
                pruned_inf_mask = inf_mask[:, -output.shape[1]:]
                output = output.masked_fill(~pruned_inf_mask, 0)
                output = output[:, -n_max:, 0:ft_max]
            else:
                # prune the output
                n_max = y_true_shape.max(dim=0)[0]
                n_max = n_max.item()
                tokens = tokens[:, -n_max:, :]
                output = self.output_projection_cls(tokens) if self.tokenization_mode == 'compact' else output[:, -n_max:]
        else:
            if task_ids[0] != 1:
                output = self.output_projection_reg(tokens) if self.tokenization_mode == 'compact' else output
                output = output.masked_fill(~inf_mask, 0)
            else: 
                output = self.output_projection_cls(tokens) if self.tokenization_mode == 'compact' else output

        if self.tokenization_mode == 'compact' or not self.training:
            return output
        elif self.tokenization_mode == 'sequential':
            return output, additional_loss