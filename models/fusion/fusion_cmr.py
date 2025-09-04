# models/fusion/fusion_cmr.py
# B버전 (CMR_MFN, TSN) 전용 Fusion Network

import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """CMR_MFN에서 사용하는 어텐션 모듈"""
    def __init__(self,
                 dim,  # input dim
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.5,
                 proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, 128)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim]
        B, N, C = x.shape

        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class FusionCMR(nn.Module):
    """
    B버전 (CMR_MFN, TSN) 전용 Fusion Network
    - attention 기반 fusion
    - concat fusion도 지원
    """

    def __init__(self, input_dim, modality, fusion_type, dropout, num_segments):
        super().__init__()
        self.input_dim = input_dim
        self.num_segments = num_segments
        self.modality = modality
        self.fusion_type = fusion_type
        self.dropout = dropout

        if self.dropout > 0:
            self.dropout_layer = nn.Dropout(p=self.dropout)
            
        if self.fusion_type == 'attention':
            self.selfat = Attention(self.input_dim)

    def forward(self, inputs):
        """
        inputs: dict with modality keys
        returns: dict with 'features' key
        """
        outs = []
        if len(self.modality) > 1:  # Multi modality: Fusion
            if self.fusion_type == 'attention':
                for m in self.modality:
                    out = inputs[m] # [4, 768]
                    out = out.unsqueeze(1) # [4, 1, 768] - sequence 차원 추가
                    outs.append(out)
                    
                # 모든 모달리티를 sequence로 연결
                out = torch.cat(outs, dim=1) # [4, 3, 768] (3개 모달리티)
                
                # self-attention 적용 
                out, attn = self.selfat(out) # [4, 3, 128] (768→128로 projection)
                
                base_out = torch.mean(out, 1) # [4, 128] (3개 모달리티 평균)
                
            elif self.fusion_type == 'concat':
                for m in self.modality:
                    outs.append(inputs[m])  # [4, 768] 각각
                base_out = torch.cat(outs, dim=1) # [4, 768*3] = [4, 2304]
            else:
                raise ValueError(f"Unknown fusion type: {self.fusion_type}")
        else:  # Single modality
            base_out = inputs[list(self.modality)[0]]

        output = {'features': base_out}
        return output
    
    def freeze(self):
        """Freeze all parameters"""
        for p in self.parameters():
            p.requires_grad = False
        self.eval()
        return self