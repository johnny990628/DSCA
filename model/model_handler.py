import os.path as osp
import os
from types import SimpleNamespace
import torch
import numpy as np
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from torch.nn.parallel.scatter_gather import scatter
from .conch import create_model_from_pretrained


from .HierNet import WSIGenericNet, WSIHierNet, CLAM, MCAT, FineTuningModel, Multi_Scale_Modal
from .model_utils import init_weights
from utils import *
from loss import create_survloss, loss_reg_l1
from optim import create_optimizer
from dataset import prepare_dataset
from dataset import WSIDataset
from eval import evaluator
from loss.loss_interface import NLLLoss

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from datasets.dataset_h5 import Whole_Slide_Bag_FP
import time
import datetime
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from torchvision import transforms
from .task_forward import (
    handle_clam, handle_mcat, handle_hiersurv,
    handle_multi_scale_modal, handle_finetune
)

import json


class ModelHandler:
    def __init__(self, cfg):
        self.cfg = cfg
        self.use_deepspeed = cfg['use_deepspeed']
        seed_everything(cfg['seed'])

        self.world_size, self.rank, self.device = self._setup_distributed(cfg)

        self.dims = [int(_) for _ in cfg['dims'].split('-')]

        if cfg['task'] == 'HierSurv':
            cfg_x20_emb, cfg_x5_emb, cfg_tra_backbone = self._multi_scle_backbone()
            self.model = WSIHierNet(
                self.dims, cfg_x20_emb, cfg_x5_emb, cfg_tra_backbone, 
                dropout=cfg['dropout'], pool=cfg['pool'], join=cfg['join'], fusion=cfg['fusion']
            )
        elif cfg['task'] == 'multi_scale_modal':
            cfg_x20_emb, cfg_x5_emb, cfg_tra_backbone= self._multi_scle_backbone()
            self.model = Multi_Scale_Modal(
                self.dims, cfg_x20_emb, cfg_x5_emb, cfg_tra_backbone, 
                dropout=cfg['dropout'], pool=cfg['pool'], join=cfg['join'], fusion=cfg['fusion']
            )
        elif cfg['task'] == 'mcat':
            self.model = MCAT(dims=self.dims, cell_in_dim=int(cfg['cell_in_dim']), top_k=int(cfg['top_k']), fusion=str(cfg['early_fusion']))
        elif cfg['task'] == 'clam':
            self.model = CLAM(dims=self.dims)
        else:
            raise ValueError(f"Expected HierSurv/GenericSurv, but got {cfg['task']}")

        self._wrap_model()

    def _setup_distributed(self, cfg):
        if cfg['multi_gpu']:
            world_size = int(os.environ['WORLD_SIZE'])
            rank = int(os.environ['RANK'])
            if not dist.is_initialized():
                dist.init_process_group(
                    backend='nccl',
                    init_method='env://',
                    world_size=world_size,
                    rank=rank,
                    timeout=datetime.timedelta(hours=5)
                )
            device = torch.device(f'cuda:{rank}')
        else:
            world_size = 1
            rank = 0
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        return world_size, rank, device
        

    def _multi_scle_backbone(self):
        scales = list(map(int, self.cfg['magnification'].split('-')))
        scale = int(scales[1] / scales[0])
        print(f"Scale for magnifications {scales} is {scale}")
        cfg_x20_emb = SimpleNamespace(backbone=self.cfg['emb_x20_backbone'], 
            in_dim=self.dims[0], out_dim=self.dims[1], scale=scale, dropout=self.cfg['dropout'], dw_conv=self.cfg['emb_x20_dw_conv'], ksize=self.cfg['emb_x20_ksize'])
        cfg_x5_emb = SimpleNamespace(backbone=self.cfg['emb_x5_backbone'], 
            in_dim=self.dims[0], out_dim=self.dims[1], scale=1, dropout=self.cfg['dropout'], dw_conv=False, ksize=self.cfg['emb_x5_ksize'])
        cfg_tra_backbone = SimpleNamespace(backbone=self.cfg['tra_backbone'], ksize=self.cfg['tra_ksize'], dw_conv=self.cfg['tra_dw_conv'],
            d_model=self.dims[1], d_out=self.dims[2], nhead=self.cfg['tra_nhead'], dropout=self.cfg['dropout'], num_layers=self.cfg['tra_num_layers'], epsilon=self.cfg['tra_epsilon'])
        return cfg_x20_emb, cfg_x5_emb, cfg_tra_backbone
    def _wrap_model(self):
        self.model = self.model.to(self.device)
        if self.cfg['multi_gpu']:
            if self.use_deepspeed:
                import deepspeed
                ds_config_path = "./config/ds_config.json"
                with open(ds_config_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    print(data)
                self.model, self.optimizer, _, _ = deepspeed.initialize(
                    model=self.model,
                    model_parameters=self.model.parameters(),
                    config_params=ds_config_path
                )
            else:
                self.model = DDP(self.model, device_ids=[self.rank], output_device=self.rank, find_unused_parameters=True)


class MyHandler(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.use_deepspeed = cfg['use_deepspeed']
        self.metrics = self.cfg['metrics']
        self.task_handler = {
            'clam': handle_clam,
            'mcat': handle_mcat,
            'HierSurv': handle_hiersurv,
            'multi_scale_modal': handle_multi_scale_modal,
            'fine_tuning_clam': handle_finetune
        }[self.cfg['task']]

        # set up for path
        self.writer = SummaryWriter(cfg['save_path'])
        self.last_ckpt_path = osp.join(cfg['save_path'], 'model-last.pth')
        self.best_ckpt_path = osp.join(cfg['save_path'], 'model-best.pth')
        self.metrics_path   = osp.join(cfg['save_path'], 'metrics.txt')
        self.config_path    = osp.join(cfg['save_path'], 'print_config.txt')

        model_handler = ModelHandler(cfg)
        self.model, self.world_size, self.rank, self.device = model_handler.model, model_handler.world_size, model_handler.rank, model_handler.device
       
        print_network(self.model)
        self.model_pe = cfg['tra_position_emb']
        print("[model] Transformer Position Embedding: {}".format('Yes' if self.model_pe else 'No'))
        
        # set up for loss, optimizer, and lr scheduler
        pos_weight_val = self.cfg.get('pos_weight', None)
        if pos_weight_val:
            pos_weight = torch.tensor([pos_weight_val], device='cuda:0')
        else:
            pos_weight = torch.tensor([1.0], device='cuda:0')
        print(pos_weight)
        self.loss = create_survloss(cfg['loss'], argv={'alpha': cfg['alpha'], 'pos_weight': pos_weight})
        self.loss = self.loss.to(self.device)
        self.loss_l1 = loss_reg_l1(cfg['reg_l1'])
        cfg_optimizer = SimpleNamespace(opt=cfg['opt'], weight_decay=cfg['weight_decay'], lr=cfg['lr'], 
            opt_eps=cfg['opt_eps'], opt_betas=cfg['opt_betas'], momentum=cfg['opt_momentum'])
        self.optimizer = create_optimizer(cfg_optimizer, self.model)
       
        # 1. Early stopping: patience = 30
        # 2. LR scheduler: lr * 0.5 if val_loss is not decreased in 10 epochs.
        if cfg['es_patience'] is not None:
            if self.cfg['monitor_metrics'] == 'loss':
                self.early_stop = EarlyStopping(warmup=cfg['es_warmup'], patience=cfg['es_patience'], start_epoch=cfg['es_start_epoch'], verbose=cfg['es_verbose'], use_deepspeed=self.use_deepspeed)
            else:
                self.early_stop = Monitor_Metrics(warmup=cfg['es_warmup'], patience=cfg['es_patience'], start_epoch=cfg['es_start_epoch'], verbose=cfg['es_verbose'], monitor_metrics=cfg['monitor_metrics'], use_deepspeed=self.use_deepspeed)  
        else:
            self.early_stop = None
        self.steplr = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=5, verbose=True)
        print_config(cfg, print_to_path=self.config_path)

    def exec(self):
        task = self.cfg['task']
        experiment = self.cfg['experiment']
        print('[exec] start experiment {} on {}.'.format(experiment, task))
        
        path_split = self.cfg['path_data_split'].format(self.cfg['seed_data_split'])
        pids_train, pids_val, pids_test = read_datasplit_npz(path_split)
        print('[exec] read patient IDs from {}'.format(path_split))
        
        # For reporting results
        if experiment == 'sim':
            # Prepare datasets 
            train_set  = prepare_dataset(pids_train, self.cfg, self.cfg['magnification'])
            train_pids = train_set.pids
            val_set    = prepare_dataset(pids_val, self.cfg, self.cfg['magnification'])
            val_pids   = val_set.pids
            test_set    = prepare_dataset(pids_test, self.cfg, self.cfg['magnification'])
            test_pids   = test_set.pids

            if self.cfg['multi_gpu']:
                train_sampler = DistributedSampler(train_set, num_replicas=self.world_size, rank=self.rank, shuffle=True)
            elif not self.cfg['multi_gpu'] and self.cfg['metrics'] == 'classification':
                from torch.utils.data import WeightedRandomSampler
                from collections import Counter
                y_list = train_set.pid2label.values()
                label_count = Counter(y_list)
                print(f'Positive/Negative samples: {label_count}')
                total = sum(label_count.values())
                weight = np.array([total / label_count[i] for i in range(len(label_count))])  # 對應 0, 1
                samples_weight = np.array([weight[int(t)] for t in y_list])
                samples_weight = torch.from_numpy(samples_weight).float()
                train_sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
            else:
                train_sampler = None
            
            train_loader = DataLoader(train_set, batch_size=self.cfg['batch_size'], sampler=train_sampler if train_sampler is not None else None,
            shuffle=(train_sampler is None), generator=seed_generator(self.cfg['seed']),
                pin_memory=True, num_workers=self.cfg['num_workers'])
            val_loader   = DataLoader(val_set,   batch_size=self.cfg['batch_size'], generator=seed_generator(self.cfg['seed']),
                pin_memory=True, num_workers=self.cfg['num_workers'])
            test_loader = DataLoader(test_set,  batch_size=self.cfg['batch_size'], generator=seed_generator(self.cfg['seed']),
                pin_memory=True, num_workers=self.cfg['num_workers'])
            val_name = 'validation'
            val_loaders = {'validation': val_loader, 'test': test_loader}
            self._run_training(train_loader, train_sampler=train_sampler,  val_loaders=val_loaders, val_name=val_name, measure=True, save=False)
            
            # Evals
            if self.rank == 0:
                metrics = {}
                evals_loader = {'test': test_loader}
                for k, loader in evals_loader.items():
                    if loader is None:
                        continue
                    cltor = self._test_model(loader, checkpoint=self.best_ckpt_path)
                    result = evaluator(cltor['y'].cpu().numpy(), cltor['y_hat'].cpu().numpy(), metrics=self.metrics)
                    test_loss = self.loss(cltor['y_hat'], cltor['y'])
                    metric_list = list(result.items()) + [('loss', test_loss)]
                    metrics[k] = metric_list
                    if self.cfg['save_prediction']:
                        path_save_pred = osp.join(self.cfg['save_path'], f"surv_pred_{k}.csv")
                        save_prediction(test_pids, cltor['y'], cltor['y_hat'], path_save_pred, metrics=self.cfg['metrics'])

                print("finish testing")
                print_metrics(metrics, print_to_path=self.metrics_path)

            if self.cfg['multi_gpu']:
                dist.barrier()
                if hasattr(self.model, 'destroy'):
                    self.model.destroy()  # 釋放DeepSpeed資源
                # 或者
                if hasattr(self.model, 'module'):
                    if hasattr(self.model.module, 'destroy'):
                        self.model.module.destroy()
            return metrics if self.rank==0 else None
            

    def _run_training(self, train_loader, train_sampler=None, val_loaders=None, val_name=None, measure=True, save=True, **kws):
        """Traing model.

        Args:
            train_loader ('DataLoader'): DatasetLoader of training set.
            val_loaders (dict): A dict like {'val': loader1, 'test': loader2}, gives the datasets
                to evaluate at each epoch.
            val_name (string): The dataset used to perform early stopping and optimal model saving.
            measure (bool): If measure training set at each epoch.
        """
        epochs = self.cfg['epochs']
        assert self.cfg['bp_every_iters'] % self.cfg['batch_size'] == 0, "Batch size must be divided by bp_every_iters."
        if val_name is not None and self.early_stop is not None:
            assert val_name in val_loaders.keys(), "Not specify the dataloader to perform early stopping."
            print("[training] {} epochs, with early stopping on {}.".format(epochs, val_name))
        else:
            print("[training] {} epochs, without early stopping.".format(epochs))

        last_epoch = -1
        for epoch in range(epochs):
            last_epoch = epoch
            if train_sampler is not None and self.cfg['multi_gpu']:
                train_sampler.set_epoch(epoch)
            
            train_cltor, batch_avg_loss = self._train_epoch(train_loader)
            self.writer.add_scalar('loss/train_batch_avg_loss', batch_avg_loss, epoch+1)
            if measure:
                steplr_monitor_loss = self._log_training_metrics(epoch, batch_avg_loss, train_cltor)

            val_metrics = self._log_validation_metrics(epoch, val_loaders, val_name) if val_loaders is not None else None
            print(f'val_metrics: {val_metrics}')
            if val_metrics is not None and self.early_stop is not None:
                if self._check_early_stop(epoch, val_metrics):
                    break
            self.writer.flush()
           
        if save:
            torch.save(self.model.state_dict(), self.last_ckpt_path)
            self._save_checkpoint(epoch, val_metrics)
            print("[training] last model saved at epoch {}".format(last_epoch))

    def _log_training_metrics(self, epoch, batch_avg_loss, train_cltor):
        train_ci = evaluator(train_cltor['y'].cpu().numpy(), train_cltor['y_hat'].cpu().numpy(), metrics=self.metrics)
        train_loss = self.loss(train_cltor['y_hat'], train_cltor['y'])
        self.writer.add_scalar('loss/train_overall_loss', train_loss, epoch+1)
        for metric_name, value in train_ci.items():
                self.writer.add_scalar(f'{metric_name}/train', value, epoch+1)
        print(f"[training] epoch {epoch+1}, loss: {train_loss:.8f}, " + ", ".join([f"{m}: {v:.5f}" for m, v in train_ci.items()]))
        return train_loss

    def _log_validation_metrics(self, epoch, val_loaders, val_name):
        val_metrics = None
        if val_loaders is None:
            return None
        for k in val_loaders.keys():
            if val_loaders[k] is None:
                continue
            val_cltor = self._test_model(val_loaders[k])
            met_ci = evaluator(val_cltor['y'].cpu().numpy(), val_cltor['y_hat'].cpu().numpy(), metrics=self.metrics)
            met_loss = self.loss(val_cltor['y_hat'], val_cltor['y'])
            self.writer.add_scalar(f'loss/{k}_overall_loss', met_loss, epoch+1)
            for metric_name, value in met_ci.items():
                self.writer.add_scalar(f'{metric_name}/{k}', value, epoch+1)
            print(f"[training] {k} epoch {epoch+1}, loss: {met_loss:.8f}, " + ", ".join([f"{m}: {v:.5f}" for m, v in met_ci.items()]))
            if k == val_name:
                if self.cfg['monitor_metrics'] == 'loss':
                    val_metrics = met_loss
                else:
                    val_metrics = met_ci[self.cfg['monitor_metrics']]
        return val_metrics

    def _check_early_stop(self, epoch, val_metrics):
        self.early_stop(epoch, val_metrics, self.model, ckpt_name=self.best_ckpt_path)
        self.steplr.step(val_metrics)
        should_stop = self.early_stop.if_stop()
        # 使用all_reduce確保所有進程都收到停止信號
        stop_tensor = torch.tensor([1 if should_stop else 0], device=self.device)
        if self.cfg['multi_gpu']:
            dist.all_reduce(stop_tensor, op=dist.ReduceOp.MAX)
            dist.barrier()
        if stop_tensor.item() > 0:
            print(f"[Rank {self.rank}] Early stopping triggered at epoch {epoch+1}")
            return True
        return False


    def _train_epoch(self, train_loader):
        collector, bp_collector, all_loss = {'y': None, 'y_hat': None}, {'y': None, 'y_hat': None}, []
        self.model.train()
        i_batch = 0

        for batch in train_loader:
            i_batch += 1
            y, y_hat = self.task_handler(self.model, batch, self.device)
            y = y.to(self.device)
            y_hat = y_hat.to(torch.float32)
            if y.dim() == 1:
                y = y.unsqueeze(1).float()

            collector = collect_tensor(collector, y.detach().cpu(), y_hat.detach().cpu())
            bp_collector = collect_tensor(bp_collector, y, y_hat)

            if bp_collector['y'].size(0) % self.cfg['bp_every_iters'] == 0:
                loss = self.loss(bp_collector['y_hat'], bp_collector['y'])
                if self.use_deepspeed:
                    self.model.zero_grad()
                    self.model.backward(loss)
                    self.model.step()
                else:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                all_loss.append(loss.item())
                if self.rank==0:
                    print("[training epoch] training batch {}, loss: {:.6f}".format(i_batch, loss.item()))
                bp_collector = {'y': None, 'y_hat': None}

        return collector, sum(all_loss)/len(all_loss)

    def _test_model(self, loader, checkpoint=None):
        if checkpoint is not None:
            self.model.load_state_dict(torch.load(checkpoint))
            print(f"Loaded model from {checkpoint}")
        self.model.eval()
        res = {'y': None, 'y_hat': None}
        with torch.no_grad():
            for batch in loader:
                y, y_hat = self.task_handler(self.model, batch, self.device)
                if y.dim() == 1:
                    y = y.unsqueeze(1).float()
                res = collect_tensor(res, y.cpu(), y_hat.cpu())
        return res
