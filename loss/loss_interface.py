import torch
import torch.nn as nn

class SpearmanSurrogateLoss(nn.Module):
    """
    SpearmanSurrogateLoss 實作了基於可微分排序近似的 Spearman 相關係數損失函數。
    傳統的排序操作 (如 argsort) 是不可微的，無法用於基於梯度的優化。
    此實作透過 sigmoid 函數來近似排序，使其在 PyTorch 中可進行反向傳播。
    """
    def __init__(self, regularization_strength: float = 0.01, k: float = 10.0):
        """
        初始化 SpearmanSurrogateLoss。
        Args:
            regularization_strength (float): 預測排名正規化的強度，有助於穩定訓練。
            k (float): 控制 sigmoid 函數的「陡峭度」，影響排序近似的精確性。
                       k 值越大，近似越接近硬性排序。
        """
        super().__init__()
        self.regularization_strength = regularization_strength
        self.k = k # 控制可微分排序的銳利度

    def differentiable_rank(self, x: torch.Tensor) -> torch.Tensor:
        """
        使用 sigmoid 函數的可微分排序近似。
        對於輸入張量 x 中的每個元素 x_i，其近似排名為所有 j 的 sigmoid(k * (x_i - x_j)) 之和。
        這近似計算了有多少個元素小於或等於 x_i。
        
        Args:
            x (torch.Tensor): 輸入張量，通常為一維的預測值或目標值。
                              要求 x 必須是可追蹤梯度的。

        Returns:
            torch.Tensor: 輸入張量 x 的近似排名。
        """
        # 確保輸入是可追蹤梯度的
        if not x.requires_grad:
            x.requires_grad_(True) # 確保梯度追蹤

        # 將輸入 x 擴展為 (N, 1) 和 (1, N) 進行成對差值計算
        # x.unsqueeze(1) 得到 (N, 1)
        # x.unsqueeze(0) 得到 (1, N)
        # 兩者相減得到 (N, N) 的差值矩陣，其中 diffs[i, j] = x[i] - x[j]
        diffs = x.unsqueeze(1) - x.unsqueeze(0)

        # 應用 sigmoid 函數，並乘以 k 來控制陡峭度
        # F.sigmoid(k * diffs) 會近似一個階梯函數：
        # 如果 x[i] > x[j]，結果接近 1
        # 如果 x[i] < x[j]，結果接近 0
        # 如果 x[i] = x[j]，結果接近 0.5
        # 對每一行求和，得到近似的排名值。每個元素 x_i 的近似排名是所有 j 處於 x_i 前面 (或相等) 的「軟計數」。
        ranks = torch.sigmoid(self.k * diffs).sum(dim=1)

        return ranks

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        計算 Spearman 相關係數的代理損失。
        
        Args:
            pred (torch.Tensor): 模型的預測值。預期為 (batch_size,) 或 (num_samples,)。
            target (torch.Tensor): 真實目標值。預期與 pred 具有相同形狀。

        Returns:
            torch.Tensor: Spearman 代理損失 (1 - 相關係數)。
        """
        # 確保輸入張量是一維的，並移除可能的單維度
        # 例如，從 (batch_size, 1) 變為 (batch_size,)
        pred = pred.squeeze()
        target = target.squeeze()

        # 檢查輸入維度是否適合排名計算
        if pred.dim() != 1 or target.dim() != 1:
            raise ValueError(
                "SpearmanSurrogateLoss 預期輸入為一維張量 (例如 (batch_size,) 或 (num_samples,))。"
                "請確保 pred 和 target 在傳入前已擠壓成一維。"
            )
        
        # 應用可微分的排名近似
        pred_rank = self.differentiable_rank(pred)
        target_rank = self.differentiable_rank(target)

        # 計算兩個排名向量的均值
        pred_rank_mean = pred_rank.mean()
        target_rank_mean = target_rank.mean()

        # 計算偏差 (deviation)
        vx = pred_rank - pred_rank_mean
        vy = target_rank - target_rank_mean

        # 計算 Spearman 相關係數的分子 (偏差乘積之和)
        numerator = (vx * vy).sum()

        # 計算 Spearman 相關係數的分母 (偏差平方和的平方根之積)
        # torch.norm() 計算 L2 範數，即平方和的平方根
        denominator = (torch.norm(vx) * torch.norm(vy) + 1e-8) # 添加一個小值避免除以零

        # 計算相關係數
        corr = numerator / denominator

        # 添加 L2 正規化到預測的排名值，有助於穩定訓練
        # 這是對模型輸出平滑性的一種間接鼓勵
        l2_reg = self.regularization_strength * (pred_rank**2).mean()

        # 我們希望最大化相關係數，所以損失函數是 (1 - 相關係數)
        # 同時加上正規化項
        return 1 - corr + l2_reg

def create_survloss(loss, argv):
    if loss == 'survmle':
        return SurvMLE(**argv)
    elif loss == 'survple':
        return SurvPLE()
    elif loss == 'survnll':
        return NLLLoss()
    elif loss == 'ce':
        return nn.CrossEntropyLoss()
    elif loss == 'bce':
        return nn.BCEWithLogitsLoss()
    elif loss == 'mse':
        return nn.MSELoss()
    elif loss == 'surrogate':
        return SpearmanSurrogateLoss() # 允許傳遞參數
    else:
        raise ValueError(f"Unknown loss function: {loss}")


class SurvMLE(nn.Module):
    """A maximum likelihood estimation function in Survival Analysis.

    As suggested in '10.1109/TPAMI.2020.2979450',
        [*] L = (1 - alpha) * loss_l + alpha * loss_z.
    where loss_l is the negative log-likelihood loss, loss_z is an upweighted term for instances 
    D_uncensored. In discrete model, T = 0 if t in [0, a_1), T = 1 if t in [a_1, a_2) ...

    This implementation is based on https://github.com/mahmoodlab/MCAT/blob/master/utils/utils.py
    """
    def __init__(self, alpha=0.0, eps=1e-7):
        super(SurvMLE, self).__init__()
        self.alpha = alpha
        self.eps = eps
        print('[setup] loss: a MLE loss in discrete SA models with alpha = %.2f' % self.alpha)

    def forward(self, y, hazards_hat, cur_alpha=None):
        """
        y: torch.FloatTensor() with shape of [B, 2] for a discrete model.
        t: torch.LongTensor() with shape of [B, ] or [B, 1]. It's a discrete time label.
        c: torch.FloatTensor() with shape of [B, ] or [B, 1]. 
            c = 0 for uncensored samples (with event), 
            c = 1 for censored samples (without event).
        hazards_hat: torch.FloatTensor() with shape of [B, MAX_T]
        """
        t, c = y[:, 0], y[:, 1]
        batch_size = len(t)
        t = t.view(batch_size, 1).long() # ground truth bin, 0 [0,a_1), 1 [a_1,a_2),...,k-1 [a_k-1,inf)
        c = c.view(batch_size, 1).float() # censorship status, 0 or 1
        S = torch.cumprod(1 - hazards_hat, dim=1) # surival is cumulative product of 1 - hazards
        S_padded = torch.cat([torch.ones_like(c), S], 1) # s[0] = 1.0 to avoid for t = 0
        uncensored_loss = -(1 - c) * (torch.log(torch.gather(S_padded, 1, t).clamp(min=self.eps)) + torch.log(torch.gather(hazards_hat, 1, t).clamp(min=self.eps)))
        censored_loss = - c * torch.log(torch.gather(S_padded, 1, t+1).clamp(min=self.eps))
        neg_l = censored_loss + uncensored_loss
        alpha = self.alpha if cur_alpha is None else cur_alpha
        loss = (1.0 - alpha) * neg_l + alpha * uncensored_loss
        loss = loss.mean()
        
        return loss
        

class SurvPLE(nn.Module):
    """A partial likelihood estimation (called Breslow estimation) function in Survival Analysis.

    This is a pytorch implementation by Huang. See more in https://github.com/huangzhii/SALMON.
    Note that it only suppurts survival data with no ties (i.e., event occurrence at same time).
    
    Args:
        y (Tensor): The absolute value of y indicates the last observed time. The sign of y 
        represents the censor status. Negative value indicates a censored example.
        y_hat (Tensor): Predictions given by the survival prediction model.
    """
    def __init__(self):
        super(SurvPLE, self).__init__()
        print('[setup] loss: a popular PLE loss in coxph')

    def forward(self, y_hat, y):
        device = y_hat.device
        if y.dtype == torch.float16:
            y = y.float()
        if y_hat.dtype == torch.float16:
            y_hat = y_hat.float()
        T = torch.abs(y)
        E = (y > 0).int()

        n_batch = len(T)
        R_matrix_train = torch.zeros([n_batch, n_batch], dtype=torch.int8)
        for i in range(n_batch):
            for j in range(n_batch):
                R_matrix_train[i, j] = T[j] >= T[i]

        train_R = R_matrix_train.float().to(device)
        train_ystatus = E.float().to(device)

        theta = y_hat.reshape(-1)
        exp_theta = torch.exp(theta)

        loss_nn = - torch.mean((theta - torch.log(torch.sum(exp_theta * train_R, dim=1))) * train_ystatus)

        return loss_nn

def loss_reg_l1(coef):
    print('[setup] L1 loss with coef={}'.format(coef))
    coef = .0 if coef is None else coef
    def func(model_params):
        if coef <= 1e-8:
            return 0.0
        else:
            return coef * sum([torch.abs(W).sum() for W in model_params])
    return func

class NLLLoss(nn.Module):
    def __init__(self):
        super(NLLLoss, self).__init__()
    def forward(self, y, y_hat):
         
        T = torch.abs(y).view(-1)
        E = (y > 0).int()
        idx = T.sort(descending=True)[1]
        events = E[idx]
        risk_scores = y_hat[idx]
        events = events.float()
        events = events.view(-1)
        risk_scores = risk_scores.view(-1)
        
        uncensored_likelihood = risk_scores - risk_scores.exp().cumsum(0).log()
        censored_likelihood = uncensored_likelihood * events
        num_observed_events = events.sum()
        neg_likelihood = -censored_likelihood.sum()/num_observed_events
        return neg_likelihood