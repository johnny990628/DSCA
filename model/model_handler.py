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


from .HierNet import WSIGenericNet, WSIHierNet, WSIGenericCAPNet, CLAM_Survival, MCAT_Surv, FineTuningModel
from .model_utils import init_weights
from utils import *
from loss import create_survloss, loss_reg_l1
from optim import create_optimizer
from dataset import prepare_dataset
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


class SurvLabelTransformer(object):
    """
    SurvLabelTransformer: create label of survival data for model training.
    """
    def __init__(self, path_label, column_t='t', column_e='e', verbose=True):
        super(SurvLabelTransformer, self).__init__()
        self.path_label = path_label
        self.column_t = column_t
        self.column_e = column_e
        self.column_label = None
        self.full_data = pd.read_csv(path_label, dtype={'patient_id': str, 'pathology_id': str})
        
        self.pat_data = self.to_patient_data(self.full_data, at_column='patient_id')
        self.min_t = self.pat_data[column_t].min()
        self.max_t = self.pat_data[column_t].max()
        if verbose:
            print('[surv label] at patient level')
            print('\tmin/avg/median/max time = {}/{:.2f}/{}/{}'.format(self.min_t, 
                self.pat_data[column_t].mean(), self.pat_data[column_t].median(), self.max_t))
            print('\tratio of event = {}'.format(self.pat_data[column_e].sum() / len(self.pat_data)))

    def to_patient_data(self, df, at_column='patient_id'):
        df_gps = df.groupby('patient_id').groups
        df_idx = [i[0] for i in df_gps.values()]
        return df.loc[df_idx, :]

    def to_continuous(self, column_label='y'):
        print('[surv label] to continuous')
        self.column_label = [column_label]

        label = []
        for i in self.pat_data.index:
            if self.pat_data.loc[i, self.column_e] == 0:
                label.append(-1 * self.pat_data.loc[i, self.column_t])
            else:
                label.append(self.pat_data.loc[i, self.column_t])
        self.pat_data.loc[:, column_label] = label
        
        return self.pat_data

    def to_discrete(self, bins=4, column_label_t='y_t', column_label_c='y_c'):
        """
        based on the quartiles of survival time values (in months) of uncensored patients.
        see Chen et al. Multimodal Co-Attention Transformer for Survival Prediction in Gigapixel Whole Slide Images
        """
        print('[surv label] to discrete, bins = {}'.format(bins))
        self.column_label = [column_label_t, column_label_c]

        # c = 1 -> censored/no event, c = 0 -> uncensored/event
        self.pat_data.loc[:, column_label_c] = 1 - self.pat_data.loc[:, self.column_e]

        # discrete time labels
        df_events = self.pat_data[self.pat_data[self.column_e] == 1]
        _, qbins = pd.qcut(df_events[self.column_t], q=bins, retbins=True, labels=False)
        qbins[0] = self.min_t - 1e-5
        qbins[-1] = self.max_t + 1e-5

        discrete_labels, qbins = pd.cut(self.pat_data[self.column_t], bins=qbins, retbins=True, labels=False, right=False, include_lowest=True)
        self.pat_data.loc[:, column_label_t] = discrete_labels.values.astype(int)

        return self.pat_data

    def collect_slide_info(self, pids, column_label=None):
        if column_label is None:
            column_label = self.column_label

        sel_pids, pid2sids, pid2label = list(), dict(), dict()
        for pid in pids:
            sel_idxs = self.full_data[self.full_data['patient_id'] == pid].index
            if len(sel_idxs) > 0:
                sel_pids.append(pid)
                pid2sids[pid] = list(self.full_data.loc[sel_idxs, 'pathology_id'])
                
                pat_idx = self.pat_data[self.pat_data['patient_id'] == pid].index[0]
                pid2label[pid] = list(self.pat_data.loc[pat_idx, column_label])

            else:
                print('[warning] patient {} not found!'.format(pid))

        return sel_pids, pid2sids, pid2label

# 定義WSI數據集
class WSIDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, h5_dir, slide_dir, slide_ext='.svs', custom_transforms=None, pids=None):
        self.csv_path = csv_path
        self.h5_dir = h5_dir
        self.slide_dir = slide_dir
        self.slide_ext = slide_ext
        self.custom_transforms = custom_transforms
        
        if isinstance(csv_path, str):
            self.full_data = pd.read_csv(csv_path, dtype={'patient_id': str, 'pathology_id': str})
        elif isinstance(csv_path, pd.DataFrame):
            self.full_data = csv_path.copy()
        else:
            raise ValueError("csv_input 必須為 CSV 檔案路徑或是 pandas DataFrame")
            
        # 收集全部病理切片的信息
        if pids is not None:
            self.full_data = self.full_data[self.full_data["patient_id"].isin(pids)]
        
        self.slide_data = self.load_slide_data()

    def load_slide_data(self):
        """Load slide data and survival information from CSV using SurvLabelTransformer"""
        # 初始化 SurvLabelTransformer
        surv_label = SurvLabelTransformer(self.csv_path, verbose=True)
        
        # 轉換為連續的生存標籤
        patient_data = surv_label.to_continuous(column_label='y')
        
        # 準備返回的數據
        slide_data = []
        for _, row in self.full_data.iterrows():
            pathology_id = row['pathology_id']
            patient_id = row['patient_id']
            
            # 獲取對應病人的標籤
            pat_idx = patient_data[patient_data['patient_id'] == patient_id].index[0]
            label = patient_data.loc[pat_idx, 'y']
            
            slide_data.append((pathology_id, label))
            
        print(f"Loaded {len(slide_data)} slides with survival data")
        return slide_data

    def __len__(self):
        return len(self.slide_data)

    def __getitem__(self, idx):
        slide_id, label = self.slide_data[idx]
        h5_path = os.path.join(self.h5_dir, 'patches', f"{slide_id}.h5")
        slide_path = os.path.join(self.slide_dir, f"{slide_id}{self.slide_ext}")
        
        # 使用Whole_Slide_Bag_FP加載WSI
        # wsi = openslide.open_slide(slide_path)
        wsi_dataset = Whole_Slide_Bag_FP(file_path=h5_path, wsi_path=slide_path, custom_transforms=self.custom_transforms)

        # 確保數據格式正確
        if len(wsi_dataset) > 0:
            first_item = wsi_dataset[0]
            if not isinstance(first_item[0], torch.Tensor):
                raise ValueError(f"Expected torch.Tensor, got {type(first_item[0])}")
            print("Sample tensor shape:", first_item[0].shape)
    
        
        return wsi_dataset, torch.tensor(label, dtype=torch.float32)

class MyHandler(object):
    """Deep Risk Predition Model Handler.
    Handler the model train/val/test for: HierSurv
    """
    def __init__(self, cfg):
        # set up for seed and device
        # self.device = torch.device("cuda:0")
        torch.cuda.set_device(cfg['cuda_id'])
        seed_everything(cfg['seed'])

        self.rank = int(os.environ['RANK'])
        self.world_size = int(os.environ['WORLD_SIZE'])
        self.device = torch.device(f'cuda:{self.rank}')
        # 初始化进程组
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=self.world_size,
            rank=self.rank,
            timeout=datetime.timedelta(seconds=7200)
        )
        # set up for path
        self.writer = SummaryWriter(cfg['save_path'])
        self.last_ckpt_path = osp.join(cfg['save_path'], 'model-last.pth')
        self.best_ckpt_path = osp.join(cfg['save_path'], 'model-best.pth')
        self.metrics_path   = osp.join(cfg['save_path'], 'metrics.txt')
        self.config_path    = osp.join(cfg['save_path'], 'print_config.txt')

        # in_dim / hid1_dim / hid2_dim / out_dim
        dims = [int(_) for _ in cfg['dims'].split('-')] 

        # set up for model
        if cfg['task'] == 'GenericSurv':
            cfg['scale'] = 4 if 'x20' in cfg['magnification'] else 1
            cfg_emb_backbone = SimpleNamespace(in_dim=dims[0], out_dim=dims[1], scale=cfg['scale'], dropout=cfg['dropout'], dw_conv=cfg['emb_dw_conv'], ksize=cfg['emb_ksize'])
            cfg_tra_backbone = SimpleNamespace(d_model=dims[1], d_out=dims[2], nhead=cfg['tra_nhead'], dropout=cfg['dropout'], num_layers=cfg['tra_num_layers'],
                ksize=cfg['tra_ksize'], dw_conv=cfg['tra_dw_conv'], epsilon=cfg['tra_epsilon'])
            self.model = WSIGenericNet(
                dims, cfg['emb_backbone'], cfg_emb_backbone, cfg['tra_backbone'], cfg_tra_backbone, 
                dropout=cfg['dropout'], pool=cfg['pool']
            )
            self.model.apply(init_weights) # model parameter init
            print(self.model)
        elif cfg['task'] == 'GenericCAPSurv':
            cfg['magnification'] = '5x20x'
            cfg_emb_backbone = SimpleNamespace(in_dim=dims[0], out_dim=dims[1], scale=4, dropout=cfg['dropout'], dw_conv=cfg['emb_dw_conv'], ksize=cfg['emb_ksize'])
            cfg_tra_backbone = SimpleNamespace(d_model=dims[1], d_out=dims[2], nhead=cfg['tra_nhead'], dropout=cfg['dropout'], num_layers=cfg['tra_num_layers'],
                ksize=cfg['tra_ksize'], dw_conv=cfg['tra_dw_conv'], epsilon=cfg['tra_epsilon'])
            self.model = WSIGenericCAPNet(
                dims, cfg['emb_backbone'], cfg_emb_backbone, cfg['tra_backbone'], cfg_tra_backbone, 
                dropout=cfg['dropout'], pool=cfg['pool']
            )
            self.model.apply(init_weights) # model parameter init
        elif cfg['task'] == 'HierSurv':
            scales = list(map(int, cfg['magnification'].split('-')))
            scale = int(scales[1] / scales[0])
            print(f"Scale for magnifications {scales} is {scale}")
            cfg_x20_emb = SimpleNamespace(backbone=cfg['emb_x20_backbone'], 
                in_dim=dims[0], out_dim=dims[1], scale=scale, dropout=cfg['dropout'], dw_conv=cfg['emb_x20_dw_conv'], ksize=cfg['emb_x20_ksize'])
            cfg_x5_emb = SimpleNamespace(backbone=cfg['emb_x5_backbone'], 
                in_dim=dims[0], out_dim=dims[1], scale=1, dropout=cfg['dropout'], dw_conv=False, ksize=cfg['emb_x5_ksize'])
            cfg_tra_backbone = SimpleNamespace(backbone=cfg['tra_backbone'], ksize=cfg['tra_ksize'], dw_conv=cfg['tra_dw_conv'],
                d_model=dims[1], d_out=dims[2], nhead=cfg['tra_nhead'], dropout=cfg['dropout'], num_layers=cfg['tra_num_layers'], epsilon=cfg['tra_epsilon'])
            self.model = WSIHierNet(
                dims, cfg_x20_emb, cfg_x5_emb, cfg_tra_backbone, 
                dropout=cfg['dropout'], pool=cfg['pool'], join=cfg['join'], fusion=cfg['fusion']
            )
        elif cfg['task'] == 'clam':
            self.model = CLAM_Survival(dims=dims)
        elif cfg['task'] == 'mcat':
            self.model = MCAT_Surv(dims=dims, cell_in_dim=int(cfg['cell_in_dim']), top_k=int(cfg['top_k']), fusion=str(cfg['early_fusion']))
            self.model = self.model.to(self.device)
            self.model = DDP(self.model, device_ids=[self.rank], output_device=self.rank, find_unused_parameters=True)
            print(f"[Setup] Early Fusion Approch: {str(cfg['early_fusion'])}")
        elif cfg['task'] == 'fine_tuning_clam':
             # ✅ 加載 Foundation Model
            foundation_model, self.preprocess = create_model_from_pretrained(
                "conch_ViT-B-16", 
                checkpoint_path=cfg["ckpt_path"],
                force_image_size=cfg["target_patch_size"]
            )

            if cfg['lora_checkpoint']:
                foundation_model = PeftModel.from_pretrained(
                    foundation_model,
                    cfg['lora_checkpoint']
                )
                print(f"[Checkpoint] Load LoRA Weights from Checkpoint {cfg['lora_checkpoint']}")
            else:
                peft_config = LoraConfig(
                    task_type=TaskType.FEATURE_EXTRACTION, 
                    r=8,
                    lora_alpha=32,
                    lora_dropout=0.1,
                    target_modules=["qkv"],
                    bias="none",
                    inference_mode=False
                )
                foundation_model = get_peft_model(foundation_model, peft_config)
                print(f"[Checkpoint] Not Found LoRA Weights from Checkpoint, Train the Model using Default Setting")

            survival_model = CLAM_Survival(dims=dims)
            self.model = FineTuningModel(foundation_model, survival_model)
            self.model = self.model.to(self.device)
            self.model = DDP(self.model, device_ids=[self.rank], output_device=self.rank, find_unused_parameters=True)
            self.load_checkpoint(self.best_ckpt_path)
            print(f"[Setup] Load All Model Successfully!")
        else:
            raise ValueError(f"Expected HierSurv/GenericSurv, but got {cfg['task']}")
       
        print_network(self.model)
        self.model_pe = cfg['tra_position_emb']
        print("[model] Transformer Position Embedding: {}".format('Yes' if self.model_pe else 'No'))
        
        # set up for loss, optimizer, and lr scheduler
        self.loss = create_survloss(cfg['loss'], argv={'alpha': cfg['alpha']})
        self.loss_l1 = loss_reg_l1(cfg['reg_l1'])
        cfg_optimizer = SimpleNamespace(opt=cfg['opt'], weight_decay=cfg['weight_decay'], lr=cfg['lr'], 
            opt_eps=cfg['opt_eps'], opt_betas=cfg['opt_betas'], momentum=cfg['opt_momentum'])
        self.optimizer = create_optimizer(cfg_optimizer, self.model)
       
        # 1. Early stopping: patience = 30
        # 2. LR scheduler: lr * 0.5 if val_loss is not decreased in 10 epochs.
        if cfg['es_patience'] is not None:
            # self.early_stop = EarlyStopping(warmup=cfg['es_warmup'], patience=cfg['es_patience'], start_epoch=cfg['es_start_epoch'], verbose=cfg['es_verbose'])
            self.early_stop = Monitor_CIndex(warmup=cfg['es_warmup'], patience=cfg['es_patience'], start_epoch=cfg['es_start_epoch'], verbose=cfg['es_verbose'])

        else:
            self.early_stop = None
        self.steplr = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=10, verbose=True)
        self.cfg = cfg
        print_config(cfg, print_to_path=self.config_path)

    def load_checkpoint(self, checkpoint_path):
        if osp.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location="cuda")
            self.model.load_state_dict(checkpoint)
            print(f"[Checkpoint] Loaded from {checkpoint_path}")
        else:
            print(f"[Checkpoint] No checkpoint found at {checkpoint_path}, starting from scratch.")

    def _save_lora_checkpoint(self, epoch):
        if self.rank == 0:
            # 這裡我們假設 lora 模組附加在 feature_extractor 上
            lora_ckpt_dir = osp.join(self.cfg['save_path'], f"lora_checkpoint_epoch_{epoch}")
            os.makedirs(lora_ckpt_dir, exist_ok=True)
            # 由於模型經過 DDP 包裝，因此需要先取得 module
            self.model.module.feature_extractor.save_pretrained(lora_ckpt_dir)
            print(f"[Checkpoint] Saved LoRA checkpoint at {lora_ckpt_dir}")


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

            if task == 'fine_tuning_clam':
                self.surv_label = SurvLabelTransformer(self.cfg["csv_path"])
                self.full_data = self.surv_label.full_data

                train_df = self.full_data[self.full_data["patient_id"].isin(pids_train)]
                val_df = self.full_data[self.full_data["patient_id"].isin(pids_val)]
                test_df = self.full_data[self.full_data["patient_id"].isin(pids_test)]

                print(f"[exec] train_df size: {len(train_df)}, val_df size: {len(val_df)}, test_df size: {len(test_df)}")

                # ✅ 創建 WSIDataset
                train_set = WSIDataset(self.cfg["csv_path"], self.cfg["h5_dir"], self.cfg["slide_dir"], custom_transforms=self.preprocess, pids=train_pids)
                val_set = WSIDataset(self.cfg["csv_path"], self.cfg["h5_dir"], self.cfg["slide_dir"], custom_transforms=self.preprocess, pids=val_pids)
                test_set = WSIDataset(self.cfg["csv_path"], self.cfg["h5_dir"], self.cfg["slide_dir"], custom_transforms=self.preprocess, pids=test_pids)
                
                def wsi_collate_fn(batch):
                    # 这里 batch 是一个 list，每个元素是 (wsi_dataset, label)
                    return batch
                
                train_sampler = DistributedSampler(train_set, num_replicas=self.world_size, rank=self.rank, shuffle=True)
                
                train_loader = DataLoader(
                    train_set, batch_size=self.cfg['batch_size'],sampler=train_sampler,
                    num_workers=self.cfg['num_workers'], pin_memory=True, collate_fn=wsi_collate_fn, worker_init_fn=seed_worker)
                val_loader = DataLoader(
                    val_set, batch_size=self.cfg['batch_size'], 
                    num_workers=self.cfg['num_workers'], pin_memory=True, collate_fn=wsi_collate_fn, worker_init_fn=seed_worker)
                test_loader = DataLoader(
                    test_set, batch_size=self.cfg['batch_size'], 
                    num_workers=self.cfg['num_workers'], pin_memory=True, collate_fn=wsi_collate_fn, worker_init_fn=seed_worker)
                val_name = 'validation'
                val_loaders = {'validation': val_loader, 'test': test_loader}
                self._run_training(train_loader, train_sampler=train_sampler, val_loaders=val_loaders, val_name=val_name, measure=True, save=False)

            else:
                train_sampler = DistributedSampler(train_set, num_replicas=self.world_size, rank=self.rank, shuffle=True)
                
                train_loader = DataLoader(train_set, batch_size=self.cfg['batch_size'], sampler=train_sampler, generator=seed_generator(self.cfg['seed']),
                    pin_memory=True, num_workers=self.cfg['num_workers'])
                val_loader   = DataLoader(val_set,   batch_size=self.cfg['batch_size'], generator=seed_generator(self.cfg['seed']),
                    pin_memory=True, num_workers=self.cfg['num_workers'])
                test_loader = DataLoader(test_set,  batch_size=self.cfg['batch_size'], generator=seed_generator(self.cfg['seed']),
                    pin_memory=True, num_workers=self.cfg['num_workers'])
                val_name = 'validation'
                val_loaders = {'validation': val_loader, 'test': test_loader}
                self._run_training(train_loader, train_sampler=train_sampler, val_loaders=val_loaders, val_name=val_name, measure=True, save=False)

            # Evals
            if self.rank == 0:
                metrics = dict()
                # evals_loader = {'train': train_loader, 'validation': val_loader, 'test': test_loader}
                evals_loader = {'test': test_loader}
                for k, loader in evals_loader.items():
                    if loader is None:
                        continue
                    # cur_pids = [train_pids, val_pids, test_pids][['train', 'validation', 'test'].index(k)]
                    cur_pids = test_pids
                    # cltor is on cpu
                    cltor = self.test_model(self.model, loader, self.cfg['task'], checkpoint=self.best_ckpt_path, model_pe=self.model_pe)
                    ci, loss = evaluator(cltor['y'], cltor['y_hat'], metrics='cindex'), self.loss(cltor['y'], cltor['y_hat'])
                    metrics[k] = [('cindex', ci), ('loss', loss)]

                    if self.cfg['save_prediction']:
                        path_save_pred = osp.join(self.cfg['save_path'], 'surv_pred_{}.csv'.format(k))
                        save_prediction(cur_pids, cltor['y'], cltor['y_hat'], path_save_pred)
                print_metrics(metrics, print_to_path=self.metrics_path)
                return metrics
            else:
                return None

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
            train_sampler.set_epoch(epoch)
            if self.cfg['task'] == 'fine_tuning_clam':
                train_cltor, batch_avg_loss = self._train_finetune_epoch(train_loader)
            else:
                train_cltor, batch_avg_loss = self._train_each_epoch(train_loader)
            
            self.writer.add_scalar('loss/train_batch_avg_loss', batch_avg_loss, epoch+1)
            
            if measure:
                train_ci, train_loss = evaluator(train_cltor['y'], train_cltor['y_hat'], metrics='cindex'), self.loss(train_cltor['y'], train_cltor['y_hat'])
                steplr_monitor_loss = train_loss
                self.writer.add_scalar('loss/train_overall_loss', train_loss, epoch+1)
                self.writer.add_scalar('c_index/train_ci', train_ci, epoch+1)
                print('[training] training epoch {}, avg. batch loss: {:.8f}, loss: {:.8f}, c_index: {:.5f}'.format(epoch+1, batch_avg_loss, train_loss, train_ci))

            val_metrics = None
            if val_loaders is not None:
                for k in val_loaders.keys():
                    if val_loaders[k] is None:
                        continue
                    if self.cfg['task'] == 'fine_tuning_clam':
                        val_cltor = self.test_fine_tune_model(self.model, val_loaders[k], self.cfg['task'], model_pe=self.model_pe)
                    else:
                        val_cltor = self.test_model(self.model, val_loaders[k], self.cfg['task'], model_pe=self.model_pe)
                    
                    # If it is at eval mode, then set alpha in SurvMLE to 0
                    met_ci, met_loss = evaluator(val_cltor['y'], val_cltor['y_hat'], metrics='cindex'), self.loss(val_cltor['y'], val_cltor['y_hat'])
                    self.writer.add_scalar('loss/%s_overall_loss'%k, met_loss, epoch+1)
                    self.writer.add_scalar('c_index/%s_ci'%k, met_ci, epoch+1)
                    print("[training] {} epoch {}, loss: {:.8f}, c_index: {:.5f}".format(k, epoch+1, met_loss, met_ci))

                    if k == val_name:
                        # monitor ci 
                        val_metrics = met_ci if self.cfg['monitor_metrics'] == 'ci' else met_loss
            
            if val_metrics is not None and self.early_stop is not None and self.rank==0:
                self.early_stop(epoch, val_metrics, self.model, ckpt_name=self.best_ckpt_path)
                self.steplr.step(val_metrics)
                if self.early_stop.if_stop():
                    last_epoch = epoch + 1
                    break

            if self.rank == 0 and self.cfg['task']=='fine_tuning_clam':
                self._save_lora_checkpoint(epoch+1)
            
            self.writer.flush()
           
        if save:
            torch.save(self.model.state_dict(), self.last_ckpt_path)
            self._save_checkpoint(epoch, val_metrics)
            print("[training] last model saved at epoch {}".format(last_epoch))

    def _train_each_epoch(self, train_loader):
        bp_every_iters = self.cfg['bp_every_iters']
        collector = {'y': None, 'y_hat': None}
        bp_collector = {'y': None, 'y_hat': None}
        all_loss  = []

        self.model.train()
        i_batch = 0

        for fx, fx5, cx5, y in train_loader:
            i_batch += 1

            fx = fx.cuda()
            fx5 = fx5.cuda() if self.cfg['task']=='mcat' else None
            # cx5 = cx5.cuda() if self.model_pe else None
            y = y.cuda()
            
            if self.cfg['task'] == 'clam':
                y_hat = self.model(fx)
            elif self.cfg['task'] == 'mcat':
                y_hat = self.model(fx,fx5)

            collector = collect_tensor(collector, y.detach().cpu(), y_hat.detach().cpu())
            bp_collector = collect_tensor(bp_collector, y, y_hat)

            if bp_collector['y'].size(0) % bp_every_iters == 0 or bp_collector['y'].size(0)==len(train_loader):
                # 2. backward propagation
                if self.cfg['loss'] == 'survple' and torch.sum(bp_collector['y'] > 0).item() <= 0:
                    print("[warning] batch {}, event count <= 0, skipped.".format(i_batch))
                    bp_collector = {'y': None, 'y_hat': None}
                    continue
                
                # 2.1 zero gradients buffer
                self.optimizer.zero_grad()
                # 2.2 calculate loss
                loss = self.loss(bp_collector['y'], bp_collector['y_hat'])
                # loss += self.loss_l1(self.model.parameters())
                all_loss.append(loss.item())
                print("[training epoch] training batch {}, loss: {:.6f}".format(i_batch, loss.item()))

                # 2.3 backwards gradients and update networks
                loss.backward()
                self.optimizer.step()
                bp_collector = {'y': None, 'y_hat': None}

        return collector, sum(all_loss)/len(all_loss)


    def _train_finetune_epoch(self, train_loader):
        bp_every_iters = self.cfg['bp_every_iters']
        collector = {'y': None, 'y_hat': None}
        bp_collector = {'y': None, 'y_hat': None}
        all_loss  = []

        self.model.train()
        i_batch = 0

        for i, batch in enumerate(train_loader):
            print(f"===[Train Process] {i+1}/{len(train_loader)} WSI===")
            i_batch += 1
            wsi_dataset, y = batch[0]
            y = y.unsqueeze(0).to(self.device)

            start_time = time.time()
            y_hat = self.model(wsi_dataset)
            y_hat = y_hat.view(-1)  
            end_time = time.time()  # 結束計時
            elapsed_time = end_time - start_time
            print(f"Extract features took {elapsed_time:.3f} seconds.")
            
            collector = collect_tensor(collector, y.detach().cpu(), y_hat.detach().cpu())
            bp_collector = collect_tensor(bp_collector, y, y_hat)

            if bp_collector['y'].size(0) % bp_every_iters == 0 or bp_collector['y'].size(0)==len(train_loader):
                # 2. backward propagation
                if self.cfg['loss'] == 'survple' and torch.sum(bp_collector['y'] > 0).item() <= 0:
                    print("[warning] batch {}, event count <= 0, skipped.".format(i_batch))
                    bp_collector = {'y': None, 'y_hat': None}
                    continue
                
                self.optimizer.zero_grad()
                loss = self.loss(bp_collector['y'], bp_collector['y_hat'])
                all_loss.append(loss.item())
                print("[training epoch] training batch {}, loss: {:.6f}".format(i_batch, loss.item()))

                loss.backward()
                self.optimizer.step()
                bp_collector = {'y': None, 'y_hat': None}

        if len(all_loss) == 0:
            batch_avg_loss = 0.0
        else:
            batch_avg_loss = sum(all_loss) / len(all_loss)
        return collector, batch_avg_loss


    @staticmethod
    def test_model(model, loader, task, checkpoint=None, model_pe=False):
        if checkpoint is not None:
            model.load_state_dict(torch.load(checkpoint))

        model.eval()
        res = {'y': None, 'y_hat': None}
        with torch.no_grad():
            for x1, x2, c, y in loader:
                x1 = x1.cuda()
                x2 = x2.cuda()
                # c = c.cuda() if model_pe else None
                y = y.cuda()
                
                if task == 'clam' or task == 'fine_tuning_clam':
                    y_hat = model(x1)
                elif task == 'mcat':
                    y_hat = model(x1, x2)
                res = collect_tensor(res, y.detach().cpu(), y_hat.detach().cpu())
        return res
    

    def test_fine_tune_model(self, model, loader, task, checkpoint=None, model_pe=False):
        if checkpoint is not None:
            model.load_state_dict(torch.load(checkpoint))

        model.eval()
        res = {'y': None, 'y_hat': None}
        with torch.no_grad():
            for i, batch in enumerate(loader):
                print(f"===[Validation Process] {i+1}/{len(loader)} WSI===")
                wsi_dataset, y = batch[0]
                y = y.unsqueeze(0).to(self.device)
                y_hat = model(wsi_dataset)
                y_hat = y_hat.view(-1)
                res = collect_tensor(res, y.detach().cpu(), y_hat.detach().cpu())
        return res
