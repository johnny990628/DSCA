import torch

def handle_clam(model, batch, device):
    fx, _, _, y = batch
    fx = fx.to(device)
    y = y.to(device)
    y_hat = model(fx)
    return y, y_hat

def handle_mcat(model, batch, device):
    fx, fx5, _, y = batch
    fx = fx.to(device)
    fx5 = fx5.to(device)
    y = y.to(device)
    y_hat = model(fx, fx5)
    return y, y_hat

def handle_hiersurv(model, batch, device):
    fx, fx5, cx5, y = batch
    fx = fx.to(device)
    fx5 = fx5.to(device)
    cx5 = cx5.to(device)
    y = y.to(device)
    y_hat = model(fx, fx5, cx5)
    return y, y_hat

def handle_multi_scale_modal(model, batch, device):
    cfx, fx, fx5, cx5, y = batch
    cfx = cfx.to(device)
    fx = fx.to(device)
    fx5 = fx5.to(device)
    cx5 = cx5.to(device)
    y = y.to(device)
    y_hat = model(cfx, fx, fx5, cx5)
    return y, y_hat

def handle_finetune(model, batch, device):
    (wsi_dataset, y) = batch[0]
    y = y.unsqueeze(0).to(device)
    y_hat = model(wsi_dataset)
    y_hat = y_hat.view(-1)
    return y, y_hat