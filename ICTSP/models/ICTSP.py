import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_size, heads, dropout=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        self.dropout = nn.Dropout(dropout)

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, queries, mask=None):
        N = queries.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], queries.shape[1]
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = queries.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.dropout(self.keys(keys))
        queries = self.queries(queries)

        # Matrix multiplication of keys and queries for all heads
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))
        
        energy = energy / (self.embed_size ** (1 / 2))
        attention = F.softmax(energy, dim=-1)
        
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )

        out = self.fc_out(out)
        return out, attention

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadSelfAttention(embed_size, heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.heads = heads

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask=None, output_attention=False):
        x_norm = self.norm1(x)
        attention_out, attention = self.attention(x_norm, x_norm, x_norm, src_mask)
        attention_out = x + attention_out
        attention_norm = self.norm2(attention_out)
        forward = self.feed_forward(attention_norm)
        out = attention_out + forward
        if output_attention:
            return out, attention
        else:
            return out

class Tokenizer(nn.Module):
    def __init__(self, lookback=96, output=96, stride=None):
        super(Tokenizer, self).__init__()
        self.d = lookback + output   # length of chunks
        self.s = output if stride is None else stride   # stride
    
    def forward(self, tensor):
        # tensor: B C L
        # return: B C N d
        return tensor.flip(-1).unfold(dimension=2, size=self.d, step=self.s).flip(-1).flip(-2)
    
class TransformerEncoder(nn.Module):
    def __init__(self, emb_size=128, depth=2, heads=4, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.TransformerEncoderLayer(d_model=emb_size, nhead=heads, dim_feedforward=mlp_ratio*emb_size, batch_first=True, dropout=dropout, norm_first=False))
            #self.layers.append(TransformerBlock(emb_size, heads, dropout, mlp_ratio))
        
    def forward(self, x, mask=None, output_attention=False):
        attention_maps = []
        for layer in self.layers:
            #if output_attention:
            #    x, attention = layer(x, src_mask=mask, output_attention=output_attention)
            #    attention_maps.append(attention)
            #else:
            x = layer(x, src_mask=mask)
        if output_attention:
            return x, attention_maps
        return x
    
def normalize_vectors(x):
    norms = torch.norm(x, p=2, dim=-1, keepdim=True)
    normalized_x = x / norms
    return normalized_x
    
class TokenFusionLayerWithAdjustableRatio(nn.Module):
    def __init__(self, input_dim, fusion_ratio=0.25, init_reduce_rate=4, batch_size=1024):
        super(TokenFusionLayerWithAdjustableRatio, self).__init__()
        self.input_dim = input_dim
        self.fusion_ratio = fusion_ratio
        self.init_reduce_rate = init_reduce_rate
        self.batch_size = batch_size 
        self.input_q = nn.Linear(input_dim//2, 32)
        self.input_k = nn.Linear(input_dim//2, 32)
        self.silu = nn.ReLU()

    def forward(self, x, num_target_tokens, limit=512):
        target_tokens = x[:, -num_target_tokens:]
        other_tokens = x[:, :-num_target_tokens]

        attention_scores = torch.zeros((x.size(0), other_tokens.size(1)), device=x.device)

        for start_idx in range(0, other_tokens.size(1), self.batch_size):
            end_idx = min(start_idx + self.batch_size, other_tokens.size(1))
            batch_other_tokens = other_tokens[:, start_idx:end_idx]
            batch_other_tokens_expanded = normalize_vectors(self.input_q(batch_other_tokens[:, :, -batch_other_tokens.shape[-1]//2:])).unsqueeze(1)
            
            cosine_sims = F.cosine_similarity(batch_other_tokens_expanded, normalize_vectors(self.input_k(target_tokens[:, :, -batch_other_tokens.shape[-1]//2:])).unsqueeze(2), dim=-1)
            cosine_sims = self.silu(cosine_sims)

            attention_scores[:, start_idx:end_idx] = cosine_sims.mean(dim=1)

        sorted_scores, sorted_indices = torch.sort(attention_scores, descending=True, dim=1)
        
        
        num_keep = int(other_tokens.size(1) * self.fusion_ratio)
        top_indices = sorted_indices[:, :num_keep]
        top_tokens = other_tokens.gather(1, top_indices.unsqueeze(-1).expand(-1, -1, other_tokens.size(2)))
        
        top_scores = attention_scores.gather(1, top_indices)
        fused_tokens = [top_tokens * top_scores.unsqueeze(-1)]
            
        remaining_indices = sorted_indices[:, num_keep:]
        remaining = limit - num_keep
        fusion_step = self.init_reduce_rate

        while remaining > 0 and fusion_step <= remaining:
            num_to_fuse = max(int(remaining * self.fusion_ratio), fusion_step)
            num_groups = num_to_fuse // fusion_step

            group_tokens = torch.zeros((x.size(0), num_groups, x.size(2)), device=x.device)
            group_scores = torch.zeros((x.size(0), num_groups), device=x.device)
            
            for i in range(num_groups):
                start_idx = i * fusion_step
                end_idx = start_idx + fusion_step
                slice_tokens = other_tokens.gather(1, remaining_indices[:, start_idx:end_idx].unsqueeze(-1).expand(-1, -1, other_tokens.size(2)))
                slice_scores = sorted_scores.gather(1, remaining_indices[:, start_idx:end_idx])
                weights = F.softmax(slice_scores, dim=1)
                group_tokens[:, i, :] = torch.sum(slice_tokens * weights.unsqueeze(-1), dim=1)
                group_scores[:, i] = torch.sum(slice_scores * weights, dim=1)

            fused_tokens.append(group_tokens * group_scores.unsqueeze(-1))
            remaining_indices = remaining_indices[:, num_to_fuse:]
            remaining -= num_to_fuse
            fusion_step *= self.init_reduce_rate

        output_tokens = torch.cat(fused_tokens, dim=1)[:, 0:limit, :]
        output_tokens = torch.cat([torch.flip(output_tokens, [1]), target_tokens], dim=1)
        return output_tokens
    
class ICTSP(nn.Module):
    def __init__(self, lookback=96, output=96, depth=3, heads=8, mlp_ratio=4, d_model=0, emb_init=0.01, 
                 output_projection=False, external_stride=16, external_context=False, task_emb_dim=0, n_channels=0, 
                 channel_emb_dim=0, partial_mask=True, dropout=0.1, inter_series_latent_dim=16, number_of_targets=0, 
                 time_emb_dim=0, token_retriever_flag=True, linear_warmup_steps=5000, token_limit=1024, ICL_embedding=False, mask_length_for_comparison=0):
        super(ICTSP, self).__init__()
        self.lookback = lookback
        self.pred_len = output
        self.time_emb_dim = time_emb_dim
        
        self.lookback_pool = [lookback]
        self.future_pool = [output]
        self.external_stride = external_stride

        self.ICL_embedding = ICL_embedding
        if ICL_embedding:
            self.x_projection = nn.ModuleDict({str(lb):nn.Linear(lb, d_model//2) for lb in self.lookback_pool})
            self.y_projection = nn.ModuleDict({str(ft):nn.Linear(ft, d_model//2) for ft in self.future_pool})
        else:
            self.input_projection = nn.ModuleDict({str(lb)+'_'+str(ft):nn.Linear(lb+ft, d_model) for lb in self.lookback_pool for ft in self.future_pool})
        emb_size = d_model
        self.transformer_encoder = TransformerEncoder(emb_size, depth, heads, mlp_ratio, dropout=dropout)
        self.input_norm = nn.LayerNorm(emb_size)
        self.output_norm = nn.LayerNorm(emb_size)

        self.output_embedding = nn.Parameter(emb_init*torch.randn(1, 1, 1200))
        self.output_projection = nn.ModuleDict({str(ft):nn.Linear(emb_size, ft) for ft in self.future_pool})

        
        self.partial_mask = partial_mask

        self.n_channels = n_channels + time_emb_dim
        self.n_heads = heads

        self.channel_discerning_mask = nn.Parameter(emb_init*torch.randn(1, 1024, emb_size))
        
        self.number_of_targets = number_of_targets
        
        self.in_context_learning_type = 'concat' # ['concat', 'seperate']
        self.in_context_positional_embedding = nn.Parameter(emb_init*torch.randn(1, 8192, 1, emb_size))
        self.in_context_positional_embedding_after = nn.Parameter(emb_init*torch.randn(1, 8192, emb_size))
        
        self.external_stride = external_stride

        self.number_of_targets = number_of_targets 
        
        self.initialized = False
        
        self.token_retriever_flag = token_retriever_flag
        self.linear_warmup_steps = linear_warmup_steps
        self.token_merger = TokenFusionLayerWithAdjustableRatio(emb_size, 0.1, 8, batch_size=2048)
        
        self.token_limit = token_limit
        
        self.linear_warm_up_counter = 0

        self.mask_length_for_comparison = mask_length_for_comparison
        #self.linear_refill = nn.Parameter(torch.zeros(1, mask_length_for_comparison, 128)) # nn.Linear(512, mask_length_for_comparison)
        self.linear_refill = nn.Linear(512, mask_length_for_comparison)
        self.linear_refill_norm = nn.LayerNorm(mask_length_for_comparison)

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        # for pn, p in self.named_parameters():
        #     if pn.endswith('c_proj.weight'):
        #         torch.nn.init.normal_(p, mean=0.0, std=1e-5/np.sqrt(2 * depth)) # 0.02/math.sqrt(2 * config.n_layer)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, x, x_mark_enc=None, x_dec=None, x_mark_dec=None, output_attention=False):
        # x: B L_I C
        if self.mask_length_for_comparison != 0:
            # B, L, C = x.shape
            # mean, std = x[:, [self.mask_length_for_comparison]], x[:, self.mask_length_for_comparison:].std(dim=1, keepdim=True) + 1e-8#x[:, self.mask_length_for_comparison:].mean(dim=1, keepdim=True), x[:, self.mask_length_for_comparison:].std(dim=1, keepdim=True) + 1e-8
            # refill_input = ((x[:, self.mask_length_for_comparison:] - mean)/std).permute(0, 2, 1)
            # linear_refill = self.linear_refill_norm(self.linear_refill(refill_input)).permute(0, 2, 1)
            # linear_refill = linear_refill * std + mean
            # x[:, 0:self.mask_length_for_comparison] = linear_refill
            B, L, C = x.shape
            x[:, 0:self.mask_length_for_comparison] = x[:, self.mask_length_for_comparison:].mean(dim=1, keepdim=True).expand(-1, self.mask_length_for_comparison, -1)#x[:, [-1]].expand(-1, self.mask_length_for_comparison, -1)#x[:, [self.mask_length_for_comparison]].expand(-1, self.mask_length_for_comparison, -1)#x[:, [-1]].expand(-1, self.mask_length_for_comparison, -1)#x[:, self.mask_length_for_comparison:].mean(dim=1, keepdim=True).expand(-1, self.mask_length_for_comparison, -1)#x[:, [-1]].expand(-1, self.mask_length_for_comparison, -1)#self.linear_refill(x[:, self.mask_length_for_comparison:].permute(0, 2, 1)).permute(0, 2, 1)
        
        lookback = self.lookback
        future = x_mark_dec.shape[1] - x.shape[1] if x_mark_dec is not None else self.pred_len 
        mean = x[:, [-1], :].detach()
        std = 1

        x = x.permute(0, 2, 1)                           # B C L_I
        output_embedding = self.output_embedding[:, :, 0:future].expand(x.shape[0], x.shape[1], -1) 
        x = torch.cat([x, output_embedding + mean.permute(0, 2, 1)], dim=-1)     # B C L
        
        
        number_of_targets = x.shape[1] if (self.number_of_targets == 0 or x.shape[1] != self.n_channels) else self.number_of_targets

        x_orig = x[:, :, 0:-future].clone()
        if self.training:
            shifting = random.randint(0, self.external_stride)
            if shifting != 0:
                x_orig = x_orig[:, :, 0:-shifting]
        

        B, C, _ = x.shape

        x_target = x[:, -number_of_targets:, -(lookback+future):]

        external_tokenizer = Tokenizer(lookback, future, stride=self.external_stride)
        ex_tokens = external_tokenizer(x_orig) #if self.training else self.tokenizer(x_orig)   # B C_ex N_ex d
        _, _, _, d = ex_tokens.shape

        ex_tokens = ex_tokens.permute(0, 2, 1, 3).reshape(B, -1, d)      # B N_ex*C_ex d

        x_tokens = torch.cat([ex_tokens, x_target], dim=1)    # B N_ex*C_ex+N*C d
        token_mean = x_tokens[:, :, [-(future+1)]].detach()

        token_std = x_tokens[:, :, 0:-future].std(dim=-1, keepdim=True) + 1e-7

        x_tokens = x_tokens - token_mean
        
        x_tokens_orig = x_tokens.clone()
            
        if self.in_context_learning_type == 'concat':
            if self.ICL_embedding:
                x_tokens = torch.cat([self.x_projection[str(lookback)](x_tokens[:, :, 0:-future]), 
                                    self.y_projection[str(future)](x_tokens[:, :, -future:])], dim=-1)                           # B N*C d
            else:
                x_tokens = self.input_projection[str(lookback)+'_'+str(future)](x_tokens)                                        # B N*C d


        channel_discerning_mask = self.channel_discerning_mask[:, -C:, :]
        x_tokens = x_tokens + channel_discerning_mask.repeat(1, x_tokens.shape[1]//C, 1)

        in_context_positional_embedding = self.in_context_positional_embedding[:, -(x_tokens.shape[1]//C):, :, :].expand(-1, -1, C, -1)
        in_context_positional_embedding = in_context_positional_embedding.reshape(in_context_positional_embedding.shape[0], in_context_positional_embedding.shape[1]*in_context_positional_embedding.shape[2], in_context_positional_embedding.shape[3])
        x_tokens = x_tokens + in_context_positional_embedding
        
        if self.linear_warm_up_counter < self.linear_warmup_steps:
            if self.training:
                self.linear_warm_up_counter += 1
            x_output = self.output_projection[str(future)](x_tokens[:, -number_of_targets:, :])
            x_tokens = self.input_norm(x_tokens)
            
            x_tokens = self.output_norm(x_tokens)
            x_output = x_output + token_mean[:, -x_output.shape[1]:, :]
            x_output = x_output.permute(0, 2, 1)
            return x_output
        
        if not self.initialized:
            print('Token Size: ', x_tokens.shape)
            
        x_tokens_orig = x_tokens.clone()
        
        if self.token_retriever_flag:
            x_tokens = self.token_merger(x_tokens, number_of_targets)
            
        x_tokens_merged = x_tokens.clone()
        
        x_tokens = x_tokens[:, -self.in_context_positional_embedding_after.shape[1]:, :] + self.in_context_positional_embedding_after[:, -x_tokens.shape[1]:, :]
        
        if not self.initialized:
            print('Reduced Token Size: ', x_tokens.shape)
            self.initialized = True
            # self.output_projection[str(future)].weight.requires_grad = False
            # self.output_projection[str(future)].bias.requires_grad = False
            

        limit = self.token_limit
        x_tokens = x_tokens[:, -limit:, :]
        mask = None
        if mask is not None:
            mask = mask[:, -limit:, -limit:]

        #x_tokens = x_tokens[:, -number_of_targets:, :]
        
        x_tokens = self.input_norm(x_tokens)

        if output_attention:
            x_tokens, attn = self.transformer_encoder(x_tokens, mask=mask, output_attention=output_attention)
        else:
            x_tokens = self.transformer_encoder(x_tokens, mask=mask)                        # B N*C d

        x_output = x_tokens[:, -number_of_targets:, :]
        x_tokens = self.output_norm(x_tokens)

        x_output = self.output_projection[str(future)](x_output)   # B C L_p

        x_output = x_output + token_mean[:, -x_output.shape[1]:, :]
        
        x_output = x_output.permute(0, 2, 1)

        if output_attention:
            return x_output, attn, x_tokens_orig, x_tokens_merged
        
        return x_output

def generate_indices(n):
    original_indices = list(range(n))
    shuffled_indices = original_indices[:]
    random.shuffle(shuffled_indices)
    
    restore_indices = [0] * n
    for original, shuffled in enumerate(shuffled_indices):
        restore_indices[shuffled] = original

    return shuffled_indices, restore_indices

def random_split_list(lst, max_parts=3):
    if len(lst) <= 1 or max_parts == 1:
        return [lst]

    n = len(lst)
    num_parts = random.randint(1, min(n, max_parts))

    cut_points = sorted(random.sample(range(1, n), num_parts - 1))
    cut_points = [0] + cut_points + [n]
    result = [lst[cut_points[i]:cut_points[i+1]] for i in range(len(cut_points) - 1)]
    return result

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.model = ICTSP(lookback=configs.lookback, 
                              output=configs.pred_len, 
                              depth=configs.e_layers, 
                              heads=configs.n_heads, 
                              mlp_ratio=configs.mlp_ratio, 
                              d_model=configs.d_model, 
                              emb_init=0.01, 
                              output_projection=True, 
                              external_stride=configs.sampling_step, 
                              external_context=True, 
                              task_emb_dim=0, 
                              n_channels=configs.enc_in, 
                              channel_emb_dim=0, 
                              partial_mask=False, 
                              dropout=configs.dropout, 
                              inter_series_latent_dim=0, 
                              time_emb_dim=configs.time_emb_dim, 
                              token_retriever_flag=configs.token_retriever_flag, 
                              linear_warmup_steps=configs.linear_warmup_steps, 
                              token_limit=configs.token_limit,
                              ICL_embedding=configs.ICL_embedding,
                              mask_length_for_comparison=configs.mask_length_for_comparison
                             )
        self.fix_embedding = configs.fix_embedding
        self.independent = configs.sample_independent
    
    def forward(self, x, x_mark_enc=None, x_dec=None, x_mark_dec=None, output_attention=False):
        if self.fix_embedding:
            return self.model(x, None, None, x_mark_dec, output_attention=output_attention)
        else:
            if self.independent:
                output = []
                for c in range(x.shape[-1]):
                    output.append(self.model(x[:, :, [c]], None, None, x_mark_dec))
                return torch.cat(output, dim=-1)
            else:
                if self.training:
                    shuffle_indices, restore_indices = generate_indices(x.shape[-1])
                    split_indices = random_split_list(shuffle_indices, max_parts=1)
                    outputs = []
                    for ind in split_indices:
                        x_input = x[:, :, ind]
                        x_output = self.model(x_input, None, None, x_mark_dec, output_attention=output_attention)
                        outputs.append(x_output)
                    x_output = torch.cat(outputs, dim=-1)
                    output = x_output[:, :, restore_indices]
                else:
                    output = self.model(x, None, None, x_mark_dec, output_attention=output_attention)
                return output