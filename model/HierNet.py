from typing import List
import torch
import torch.nn as nn


from .model_utils import *
from .model_utils_extra import GAPool
from .mcat_coattn import *
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast

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
                    nn.Linear(dims[2], 1)
        )

    @staticmethod
    def create_positive_targets(length, device):
        return torch.full((length,), 1, device=device).long()

    @staticmethod
    def create_negative_targets(length, device):
        return torch.full((length,), 0, device=device).long()

    def forward(self, h, epoch=0, label=None, instance_eval=False, return_features=False, attention_only=False):

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




class BilinearFusion(nn.Module):
    r"""
    Late Fusion Block using Bilinear Pooling

    args:
        skip (int): Whether to input features at the end of the layer
        use_bilinear (bool): Whether to use bilinear pooling during information gating
        gate1 (bool): Whether to apply gating to modality 1
        gate2 (bool): Whether to apply gating to modality 2
        dim1 (int): Feature mapping dimension for modality 1
        dim2 (int): Feature mapping dimension for modality 2
        scale_dim1 (int): Scalar value to reduce modality 1 before the linear layer
        scale_dim2 (int): Scalar value to reduce modality 2 before the linear layer
        mmhid (int): Feature mapping dimension after multimodal fusion
        dropout_rate (float): Dropout rate
    """
    def __init__(self, skip=0, use_bilinear=0, gate1=1, gate2=1, dim1=128, dim2=128, scale_dim1=1, scale_dim2=1, mmhid=256, dropout_rate=0.25):
        super(BilinearFusion, self).__init__()
        self.skip = skip
        self.use_bilinear = use_bilinear
        self.gate1 = gate1
        self.gate2 = gate2

        dim1_og, dim2_og, dim1, dim2 = dim1, dim2, dim1//scale_dim1, dim2//scale_dim2
        skip_dim = dim1_og+dim2_og if skip else 0

        self.linear_h1 = nn.Sequential(nn.Linear(dim1_og, dim1), nn.ReLU())
        self.linear_z1 = nn.Bilinear(dim1_og, dim2_og, dim1) if use_bilinear else nn.Sequential(nn.Linear(dim1_og+dim2_og, dim1))
        self.linear_o1 = nn.Sequential(nn.Linear(dim1, dim1), nn.ReLU(), nn.Dropout(p=dropout_rate))

        self.linear_h2 = nn.Sequential(nn.Linear(dim2_og, dim2), nn.ReLU())
        self.linear_z2 = nn.Bilinear(dim1_og, dim2_og, dim2) if use_bilinear else nn.Sequential(nn.Linear(dim1_og+dim2_og, dim2))
        self.linear_o2 = nn.Sequential(nn.Linear(dim2, dim2), nn.ReLU(), nn.Dropout(p=dropout_rate))

        self.post_fusion_dropout = nn.Dropout(p=dropout_rate)
        self.encoder1 = nn.Sequential(nn.Linear((dim1+1)*(dim2+1), 256), nn.ReLU(), nn.Dropout(p=dropout_rate))
        self.encoder2 = nn.Sequential(nn.Linear(256+skip_dim, mmhid), nn.ReLU(), nn.Dropout(p=dropout_rate))

    def forward(self, vec1, vec2):
        ### Gated Multimodal Units
        if self.gate1:
            h1 = self.linear_h1(vec1)
            z1 = self.linear_z1(vec1, vec2) if self.use_bilinear else self.linear_z1(torch.cat((vec1, vec2), dim=1))
            o1 = self.linear_o1(nn.Sigmoid()(z1)*h1)
        else:
            h1 = self.linear_h1(vec1)
            o1 = self.linear_o1(h1)

        if self.gate2:
            h2 = self.linear_h2(vec2)
            z2 = self.linear_z2(vec1, vec2) if self.use_bilinear else self.linear_z2(torch.cat((vec1, vec2), dim=1))
            o2 = self.linear_o2(nn.Sigmoid()(z2)*h2)
        else:
            h2 = self.linear_h2(vec2)
            o2 = self.linear_o2(h2)

        ### Fusion
        o1 = torch.cat((o1, torch.cuda.FloatTensor(o1.shape[0], 1).fill_(1)), 1)
        o2 = torch.cat((o2, torch.cuda.FloatTensor(o2.shape[0], 1).fill_(1)), 1)
        o12 = torch.bmm(o1.unsqueeze(2), o2.unsqueeze(1)).flatten(start_dim=1) # BATCH_SIZE X 1024
        out = self.post_fusion_dropout(o12)
        out = self.encoder1(out)
        if self.skip: out = torch.cat((out, vec1, vec2), 1)
        out = self.encoder2(out)
        return out


class MCAT_Surv(nn.Module):
    def __init__(self, dims, cell_in_dim, top_k=100, fusion='concat', dropout=0.25):
        super(MCAT_Surv, self).__init__()
        self.fusion = fusion
        self.top_k = top_k
        ### FC Layer over WSI bag
        wsi_fc = [
            nn.Linear(dims[0], dims[1]), 
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(dims[1], dims[2]), 
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(dims[2], dims[3]), 
        ]
        self.wsi_net = nn.Sequential(*wsi_fc)
        
        ### Constructing Genomic SNN
        cell_fc = [
            nn.Linear(cell_in_dim, dims[1]), 
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(dims[1], dims[2]),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(dims[2], dims[3]), 
        ]
        self.cell_net = nn.Sequential(*cell_fc)

        ### Multihead Attention
        self.coattn = MultiheadAttention(embed_dim=dims[3], num_heads=1)

        self.gated_attention = Attn_Net_Gated(
            L=dims[3]*2 if fusion=='concat' else dims[3],  # Hidden feature dimension
            D=dims[3]*2 if fusion=='concat' else dims[3],
            dropout=dropout,
            n_classes=1  # Single output: risk score
        )
        
        # Classifier
        classifier_layers = [
            nn.Linear(dims[3]*2, dims[3]) if fusion=='concat' else nn.Linear(dims[3], dims[4]), 
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(dims[3], 1) if fusion=='concat' else nn.Linear(dims[4], 1)
        ]
        self.classifier = nn.Sequential(*classifier_layers)
    

    def forward(self, x_path, x_cell):

        x_path = x_path.squeeze(0)
        x_cell = x_cell.squeeze(0)

        # Bag-Level Representation
        h_path_bag = self.wsi_net(x_path).unsqueeze(1) # path embeddings are fed through a FC layer
        h_cell_bag = self.cell_net(x_cell).unsqueeze(1) # each cell signature goes through it's own FC layer
        
        # Concatenate cellular and pathology features
        if self.fusion=='concat':
            h_path_coattn = torch.cat((h_path_bag, h_cell_bag), dim=-1)  # 在最後一個維度拼接
        else:
            # Apply Gated Attention to cellular features
            A_cell, h_cell = self.gated_attention(h_cell_bag)
            A_cell = torch.transpose(A_cell, 1, 0)
            A_cell = F.softmax(A_cell, dim=1)

            # Select top-k cellular features
            topk_indices = torch.topk(A_cell.squeeze(), self.top_k, dim=0)[1]
            h_cell_topk = h_cell[topk_indices]

            # Co-Attention (Q: cellular features, K/V: pathology features)
            h_path_coattn, A_coattn = self.coattn(h_cell_topk, h_path_bag, h_path_bag) # Q, K, V
        

        # Gated Attention
        A_path, h_path = self.gated_attention(h_path_coattn.squeeze(1))
        A_path = torch.transpose(A_path, 1, 0)  # Adjust dimensions
        h_path = torch.mm(F.softmax(A_path, dim=1), h_path)  # Weighted sum
        
        # Risk Score Prediction
        risk_score = self.classifier(h_path.squeeze())
    
        return risk_score

class FineTuningModel(torch.nn.Module):
    def __init__(self, feature_extractor, survival_model, patch_batch_size=256, num_workers=8):
        """
        Args:
            feature_extractor: foundation model，用於提取 patch-level 特徵，要求提供 forward_no_head 方法。
            survival_model: 用於生存分析的模型（例如 CLAM_Survival)。
            patch_batch_size: 在內部 DataLoader 中每個 batch 處理的 patch 數量。
            num_workers: DataLoader 的工作線程數量。
        """
        super(FineTuningModel, self).__init__()
        self.feature_extractor = feature_extractor
        self.survival_model = survival_model
        self.patch_batch_size = patch_batch_size
        self.num_workers = num_workers

    def forward(self, wsi_bag):
        """
        Args:
            wsi_bag: 一個 WSI 的 patch bag，類型為 Whole_Slide_Bag_FP（或相容類型），
                     該對象實現 __len__ 和 __getitem__，每個元素為 (patch_tensor, ...)

        Returns:
            survival model 的輸出結果
        """
        # 定義 collate 函數：將所有 patch 拼接成一個大 tensor
        def collate_fn(batch):
            # 假設 batch 中的每個元素是個 tuple，patch 為第一個元素
            patches = [item[0] for item in batch]
            # 拼接為 [N, C, H, W]
            return torch.cat(patches, dim=0).half()

        loader = DataLoader(
            wsi_bag,
            batch_size=self.patch_batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_fn
        )
        all_features = []
        # 可選：使用 AMP 進行自動混合精度
        with torch.no_grad(), autocast():
            for imgs in loader:
                # 若 imgs 多出一個維度則 squeeze（根據數據格式調整）
                if imgs.dim() == 5 and imgs.size(1) == 1:
                    imgs = imgs.squeeze(1)
                imgs = imgs.to(next(self.feature_extractor.parameters()).device)
                # 使用 foundation model 提取特徵（使用 forward_no_head 得到中間特徵）
                feats = self.feature_extractor.forward_no_head(imgs, normalize=False)
                all_features.append(feats)
        if len(all_features) > 0:
            features = torch.cat(all_features, dim=0)
        else:
            features = torch.empty(0, device=imgs.device)
        # 將提取的特徵送入 survival model 得到預測結果
        out = self.survival_model(features)
        return out