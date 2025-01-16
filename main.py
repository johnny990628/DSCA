"""
Entry filename: main.py

Code in this file inherts from https://github.com/hugochan/IDGL as it provides a flexible way 
to configure parameters and inspect model performance. Great thanks to the author.
"""
import argparse
import yaml
import numpy as np
from collections import defaultdict, OrderedDict

from model import MyHandler
from utils import print_config
import os
import torch
import torch.distributed as dist


def main(config, local_rank):

    # Initialize distributed environment if needed
    if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(local_rank)
        print(f"[INFO] Distributed training initialized. Local rank: {local_rank}")

    model = MyHandler(config)
    metrics = model.exec()
    print('[INFO] Metrics:', metrics)

def multi_run_main(config):
    hyperparams = []
    for k, v in config.items():
        if isinstance(v, list):
            hyperparams.append(k)

    configs = grid(config)
    for cnf in configs:
        print('\n')
        for k in hyperparams:
            cnf['save_path'] += '-{}_{}'.format(k, cnf[k])
        print(cnf['save_path'])
        model = MyHandler(cnf)
        metrics = model.exec()
        print('[INFO] Metrics:', metrics)

    # print('[INFO] Average C-Index: {}'.format(np.mean(scores['ci'])))
    # print('[INFO] Std C-Index: {}'.format(np.std(scores['ci'])))

    # print('[INFO] Average Loss: {}'.format(np.mean(scores['loss'])))
    # print('[INFO] Std Loss: {}'.format(np.std(scores['loss'])))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-f', required=True, type=str, help='path to the config file')
    parser.add_argument('--multi_run', action='store_true', help='flag: multi run')
    parser.add_argument('--local_rank', type=int, default=0, help='Local rank for distributed training')  # Add local_rank
    args = vars(parser.parse_args())
    return args

def get_config(config_path="config/config.yml"):
    with open(config_path, "r") as setting:
        config = yaml.load(setting, Loader=yaml.FullLoader)
    return config

def grid(kwargs):
    """Builds a mesh grid with given keyword arguments for this Config class.
    If the value is not a list, then it is considered fixed"""

    class MncDc:
        """This is because np.meshgrid does not always work properly..."""

        def __init__(self, a):
            self.a = a  # tuple!

        def __call__(self):
            return self.a

    def merge_dicts(*dicts):
        """
        Merges dictionaries recursively. Accepts also `None` and returns always a (possibly empty) dictionary
        """
        from functools import reduce
        def merge_two_dicts(x, y):
            z = x.copy()  # start with x's keys and values
            z.update(y)  # modifies z with y's keys and values & returns None
            return z

        return reduce(lambda a, nd: merge_two_dicts(a, nd if nd else {}), dicts, {})


    sin = OrderedDict({k: v for k, v in kwargs.items() if isinstance(v, list)})
    for k, v in sin.items():
        copy_v = []
        for e in v:
            copy_v.append(MncDc(e) if isinstance(e, tuple) else e)
        sin[k] = copy_v

    grd = np.array(np.meshgrid(*sin.values()), dtype=object).T.reshape(-1, len(sin.values()))
    return [merge_dicts(
        {k: v for k, v in kwargs.items() if not isinstance(v, list)},
        {k: vv[i]() if isinstance(vv[i], MncDc) else vv[i] for i, k in enumerate(sin)}
    ) for vv in grd]


if __name__ == '__main__':
    cfg = get_args()
    config = get_config(cfg['config'])
    print_config(config)

    local_rank = cfg.get('local_rank', 0)
    if cfg['multi_run']:
        multi_run_main(config)
    else:
        main(config, local_rank)

