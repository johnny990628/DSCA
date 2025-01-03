from typing import List
import torch
import torch.nn as nn

from .model_utils import *
from .model_utils_extra import GAPool


###########################################################
#  A generic network for WSI with **single magnitude**.
#  A typical case of WSI:
#      level      =   0,   1,   2,   3
#      downsample =   1,   4,  16,  32
###########################################################
class WSIGenericNet(nn.Module):
    def __init__(self, dims:List, emb_backbone:str, args_emb_backbone, 
        tra_backbone:str, args_tra_backbone, dropout:float=0.25, pool:str='max_mean'):
        super(WSIGenericNet, self).__init__()
        assert len(dims) == 4 # [1024, 256, 256, 1]
        assert emb_backbone in ['conv1d', 'avgpool', 'gapool', 'sconv', 'identity']
        assert tra_backbone in ['Nystromformer', 'Transformer', 'Conv1D', 'Conv2D', 'Identity', 'SimTransformer']
        assert pool in ['max', 'mean', 'max_mean', 'gap']
        
        # dims[0] -> dims[1]
        self.patch_embedding_layer = make_embedding_layer(emb_backbone, args_emb_backbone)
        self.dim_hidden = dims[1]
        
        # dims[1] -> dims[2]
        self.patch_encoder_layer = make_transformer_layer(tra_backbone, args_tra_backbone)
        
        if pool == 'gap':
            self.pool = GAPool(dims[2], dims[2])
        else:
            self.pool = pool
        
        # dims[2] -> dims[3]
        if dims[3] == 1: # output is a proportional hazard (coxph model)
            if pool == 'max_mean':
                self.out_layer = nn.Sequential(
                    nn.Linear(2*dims[2], dims[2]),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                    nn.Linear(dims[2], dims[2]//2),
                    nn.ReLU(inplace=True),
                    nn.Linear(dims[2]//2, dims[3]),
                )
            else:
                self.out_layer = nn.Sequential(
                    nn.Linear(dims[2], dims[2]//2),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                    nn.Linear(dims[2]//2, dims[3]),
                )
        else: # output is a hazard function (discrete model)
            self.out_layer = nn.Sequential(nn.Linear(dims[2], dims[3]), nn.Sigmoid())

    def forward(self, x, coord=None):
        """
        x: [B, N, d]
        coord: the coordinates after discretization if not None
        """
        # Patch Embedding
        patch_emb = self.patch_embedding_layer(x)

        # Position Embedding addition
        if coord is not None:
            PE = compute_pe(coord, ndim=self.dim_hidden, device=x.device, dtype=x.dtype)
            patch_emb += PE

        # Patch Transformer
        patch_feat = self.patch_encoder_layer(patch_emb)
        
        # mean_pool/max_pool/global attention pool
        #  [B, L*L, d] -> [B, d]
        if self.pool == 'mean':
            rep = torch.mean(patch_feat, dim=1)
        elif self.pool == 'max':
            rep, _ = torch.max(patch_feat, dim=1)
        elif self.pool == 'max_mean':
            rep_avg = torch.mean(patch_feat, dim=1)
            rep_max, _ = torch.max(patch_feat, dim=1)
            rep = torch.cat([rep_avg, rep_max], dim=1)
        else:
            rep, patch_attn = self.pool(patch_feat)
        
        out = self.out_layer(rep)

        return out


class WSIGenericCAPNet(nn.Module):
    """
    Using CAPool backbone as a embedding layer, we only use a single magnification for prediction.
    """
    def __init__(self, dims:List, emb_backbone:str, args_emb_backbone, 
        tra_backbone:str, args_tra_backbone, dropout:float=0.25, pool:str='max_mean'):
        super(WSIGenericCAPNet, self).__init__()
        assert len(dims) == 4 # [1024, 384, 384, 1]
        assert emb_backbone in ['capool']
        assert tra_backbone in ['Nystromformer', 'Transformer', 'Conv1D', 'Conv2D', 'Identity', 'SimTransformer']
        assert pool in ['max', 'mean', 'max_mean', 'gap']
        
        # dims[0] -> dims[1]
        self.patch_embedding_layer = make_embedding_layer(emb_backbone, args_emb_backbone)
        self.dim_hidden = dims[1]
        
        # dims[1] -> dims[2]
        self.patch_encoder_layer = make_transformer_layer(tra_backbone, args_tra_backbone)
        
        if pool == 'gap':
            self.pool = GAPool(dims[2], dims[2])
        else:
            self.pool = pool
        
        # dims[2] -> dims[3]
        if dims[3] == 1: # output is a proportional hazard (coxph model)
            if pool == 'max_mean':
                self.out_layer = nn.Sequential(
                    nn.Linear(2*dims[2], dims[2]),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                    nn.Linear(dims[2], dims[2]//2),
                    nn.ReLU(inplace=True),
                    nn.Linear(dims[2]//2, dims[3]),
                )
            else:
                self.out_layer = nn.Sequential(
                    nn.Linear(dims[2], dims[2]//2),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                    nn.Linear(dims[2]//2, dims[3]),
                )
        else: # output is a hazard function (discrete model)
            self.out_layer = nn.Sequential(nn.Linear(dims[2], dims[3]), nn.Sigmoid())

    def forward(self, x, x5, x5_coord=None):
        """
        x : [B, 16N, d]
        x5: [B,   N, d]
        x5_coord: the coordinates after discretization if not None
        """
        # Patch Embedding
        patch_emb, cross_attn, _  = self.patch_embedding_layer(x, x5) # [B, 16N, d]->[B, N, d']

        # Position Embedding addition
        if x5_coord is not None:
            PE = compute_pe(x5_coord, ndim=self.dim_hidden, device=x.device, dtype=x.dtype)
            patch_emb += PE

        # Patch Transformer
        patch_feat = self.patch_encoder_layer(patch_emb)
        
        # mean_pool/max_pool/glonal attention pool
        #  [B, L*L, d] -> [B, d]
        if self.pool == 'mean':
            rep = torch.mean(patch_feat, dim=1)
        elif self.pool == 'max':
            rep, _ = torch.max(patch_feat, dim=1)
        elif self.pool == 'max_mean':
            rep_avg = torch.mean(patch_feat, dim=1)
            rep_max, _ = torch.max(patch_feat, dim=1)
            rep = torch.cat([rep_avg, rep_max], dim=1)
        else:
            rep, patch_attn = self.pool(patch_feat)
        
        out = self.out_layer(rep)

        return out

###########################################################
#                  This is DSCA network
###########################################################
class WSIHierNet(nn.Module):
    """
    A hierarchical network for WSI with multiple magnitudes.
    A typical case of WSI:
        level      =   0,   1,   2,   3
        downsample =   1,   4,  16,  32

    Current version utilizes the levels of 1 (20x) and 2 (5x).
    """
    def __init__(self, dims:List, args_x20_emb, args_x5_emb, args_tra_layer, 
        dropout:float=0.25, pool:str='gap', join='post', fusion='cat'):
        super(WSIHierNet, self).__init__()
        assert len(dims) == 4 # [1024, 384, 384, 1]
        assert args_x20_emb.backbone in ['avgpool', 'gapool', 'capool']
        assert args_x5_emb.backbone in ['conv1d'] # equivalent to a FC layer
        assert args_tra_layer.backbone in ['Nystromformer', 'Transformer']
        assert pool in ['max', 'mean', 'max_mean', 'gap']
        assert join in ['pre', 'post'] # concat two embeddings of x5 and x20 by a join way
        assert fusion in ['cat', 'fusion']
        self.x20_emb_backbone = args_x20_emb.backbone
        
        # dims[0] -> dims[1]
        self.patchx20_embedding_layer = make_embedding_layer(args_x20_emb.backbone, args_x20_emb)
        self.patchx5_embedding_layer = make_embedding_layer(args_x5_emb.backbone, args_x5_emb)
        self.dim_hidden = dims[1]
        
        # dims[1] -> dims[2]
        self.join, self.fusion = join, fusion
        if join == 'post':
            args_tra_layer.d_model = dims[1]
            self.patch_encoder_layer = make_transformer_layer(args_tra_layer.backbone, args_tra_layer)
            self.patch_encoder_layer_parallel = make_transformer_layer(args_tra_layer.backbone, args_tra_layer)
            enc_dim = 2 * dims[2] if fusion == 'cat' else dims[2]
        else:
            args_tra_layer.d_model = 2 * dims[1] if fusion == 'cat' else dims[1]
            self.patch_encoder_layer = make_transformer_layer(args_tra_layer.backbone, args_tra_layer)
            enc_dim = args_tra_layer.d_model

        if pool == 'gap':
            self.pool = GAPool(enc_dim, enc_dim)
        else:
            self.pool = pool

        # dims[2] -> dims[3]
        if dims[3] == 1: # output is a proportional hazard (coxph model)
            if pool == 'max_mean':
                self.out_layer = nn.Sequential(
                    nn.Linear(2*enc_dim, enc_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                    nn.Linear(enc_dim, enc_dim//2),
                    nn.ReLU(inplace=True),
                    nn.Linear(enc_dim//2, dims[3]),
                )
            else:
                self.out_layer = nn.Sequential(
                    nn.Linear(enc_dim, enc_dim//2),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                    nn.Linear(enc_dim//2, dims[3]),
                )
        else: # output is a hazard function (discrete model)
            self.out_layer = nn.Sequential(nn.Linear(enc_dim, dims[3]), nn.Sigmoid())

    def forward(self, x20, x5, x5_coord=None, mode=None):
        """
        x5 and x20 must be aligned.

        x20: [B, 16N, d], level = 1, downsample = 4
        x5:  [B,   N, d], level = 2, downsample = 16
        x5_coord: [B, N, 2], the coordinates after discretization for position encoding, used for the stream x20.
        """
        # Patch Embedding
        if self.x20_emb_backbone == 'capool':
            patchx20_emb, x20_x5_cross_attn, _  = self.patchx20_embedding_layer(x20, x5) # [B, 16N, d]->[B, N, d']
        else:
            patchx20_emb = self.patchx20_embedding_layer(x20) # [B, 16N, d]->[B, N, d']
        
        if mode == 'test_ca':
            return x20_x5_cross_attn # [B, L, s*s]

        patchx5_emb = self.patchx5_embedding_layer(x5) # [B, N, d]->[B, N, d']

        # Position Embedding addition
        if x5_coord is not None:
            PEx20 = compute_pe(x5_coord, ndim=self.dim_hidden, device=x20.device, dtype=x20.dtype)
            patchx20_emb = patchx20_emb + PEx20
            patchx5_emb  = patchx5_emb + PEx20.clone()

        # Patch Transformer
        if self.join == 'post':
            patchx20_feat = self.patch_encoder_layer_parallel(patchx20_emb)
            patchx5_feat = self.patch_encoder_layer(patchx5_emb)
            if self.fusion == 'cat':
                patch_feat = torch.cat([patchx20_feat, patchx5_feat], dim=2) # [B, N, 2d']
            else:
                patch_feat = patchx20_feat + patchx5_feat # [B, N, d']
        else:
            if self.fusion == 'cat':
                patch_emb = torch.cat([patchx20_emb, patchx5_emb], dim=2) # [B, N, 2d']
            else:
                patch_emb = patchx20_emb + patchx5_emb # [B, N, d']
            patch_feat = self.patch_encoder_layer(patch_emb) 

        # mean_pool/max_pool/global attention pool
        #  [B, L*L, d] -> [B, d]
        if self.pool == 'mean':
            rep = torch.mean(patch_feat, dim=1)
        elif self.pool == 'max':
            rep, _ = torch.max(patch_feat, dim=1)
        elif self.pool == 'max_mean':
            rep_avg = torch.mean(patch_feat, dim=1)
            rep_max, _ = torch.max(patch_feat, dim=1)
            rep = torch.cat([rep_avg, rep_max], dim=1)
        else:
            rep, patch_attn = self.pool(patch_feat)

        if mode == 'test_gap':
            return patch_attn # [B, 1, L]

        out = self.out_layer(rep)

        return out


"""
Attention Network without Gating (2 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""
class Attn_Net(nn.Module):

    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        super(Attn_Net, self).__init__()
        self.module = [
            nn.Linear(L, D),
            nn.Tanh()]

        if dropout:
            self.module.append(nn.Dropout(0.25))

        self.module.append(nn.Linear(D, n_classes))
        
        self.module = nn.Sequential(*self.module)
    
    def forward(self, x):
        return self.module(x), x # N x n_classes

"""
Attention Network with Sigmoid Gating (3 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""
class Attn_Net_Gated(nn.Module):
    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]
        
        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x

class CLAM_Survival(nn.Module):
    def __init__(self, gate=True, dropout=True, dims=[512,256,256], **kwargs):
        super(CLAM_Survival, self).__init__()
        # self.size_dict = {'xs': [embed_dim, 256, 256], "small": [embed_dim, 512, 256], "big": [embed_dim, 512, 384], 'large': [embed_dim, 1024, 512]}
        # size = self.size_dict[size_arg]
        fc = [nn.Linear(dims[0], dims[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))
        if gate:
            attention_net = Attn_Net_Gated(L=dims[1], D=dims[2], dropout=dropout, n_classes=1)
        else:
            attention_net = Attn_Net(L=dims[1], D=dims[2], dropout=dropout, n_classes=1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.out_layer = nn.Sequential(
                    nn.Linear(dims[1], dims[2]),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.25),
                    nn.Linear(dims[2], dims[3])
        )

    @staticmethod
    def create_positive_targets(length, device):
        return torch.full((length,), 1, device=device).long()

    @staticmethod
    def create_negative_targets(length, device):
        return torch.full((length,), 0, device=device).long()

    def forward(self, h, epoch=0, label=None, instance_eval=False, return_features=False, attention_only=False):
        device = h.device
        h=h.squeeze(0)
        A, h = self.attention_net(h)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        if attention_only:
            return A
        A_raw = A
        A = F.softmax(A, dim=1)  # softmax over N

        M = torch.mm(A, h)  # A: 1 * N h: N * 512 => M: 1 * 512
        # M = torch.cat([M, embed_batch], axis=1)
        risk_score = self.out_layer(M)  
        # Y_hat = torch.topk(risk_score, 1, dim=1)[1]
        result = {
            'risk_score': risk_score,
            'attention_raw': A_raw,
            'M': M
        }
        return risk_score
