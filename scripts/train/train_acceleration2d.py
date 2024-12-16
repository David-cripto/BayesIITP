import argparse
import os
from datetime import datetime
from pathlib import Path

from toysb import SB2D, Logger, RunnerAcceleration2D
from toysb.datasets.dataset2d import get_pair_dataset, load_dataset
from toysb.distributed_utils import init_processes
from torch.multiprocessing import Process
import torch as th
import copy

def create_arguments():
    now = datetime.now().strftime("%y-%m-%d %H:%M:%S")
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default=now, help="experiment ID")
    parser.add_argument("--dataset1", type=str, help="name for initial dataset")
    parser.add_argument("--dataset2", type=str, help="name for terminal dataset")
    parser.add_argument("--n_samples", type=int, default=10**4, help="number of samples for each dataset")
    parser.add_argument("--path_to_save", type=Path, default="", help="path to save data")
    parser.add_argument("--log-dir", type=Path, default=".log", help="path to log std outputs and writer data")
    parser.add_argument("--ckpt_path", type=Path, default="", help="path to save checkpoints")
    parser.add_argument("--load", type=Path, default=None, help="resumed checkpoint name")
    parser.add_argument("--gpu", type=int, default=None, help="choose a particular device")
    parser.add_argument("--n-gpu-per-node", type=int, default=1, help="number of gpu on each node")
    parser.add_argument("--master-address", type=str,   default='localhost', help="address for master")
    parser.add_argument("--num_steps", type=int, default=1000, help="number of steps")
    parser.add_argument("--beta_max", type=float, default=0.3, help="max diffusion")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--microbatch", type=int, default=2, help="accumulate gradient over microbatch until full batch-size")
    parser.add_argument("--microbatch_val", type=int, default=1000, help="number of drawing points or images in val mode")
    parser.add_argument("--num-itr", type=int, default=1000000, help="training iteration")
    parser.add_argument("--val-nfe", type=int, default=100, help="NFEs")
    parser.add_argument("--lr", type=float, default=5e-5, help="learning rate")
    parser.add_argument("--l2-norm", type=float, default=0.0)
    parser.add_argument("--lr-step", type=int, default=1000, help="learning rate decay step size")
    parser.add_argument("--lr-gamma", type=float, default=0.99, help="learning rate decay ratio")
    parser.add_argument("--ema", type=float, default=0.99)
    parser.add_argument("--ot_ode", action="store_true", help="use OT-ODE")
    parser.add_argument("--cond_x1", action="store_true", help="use condition")
    parser.add_argument("--add_x1_noise", action="store_true", help="add noise")
    parser.add_argument("--clip_denoise", action="store_true", help="clip x0 in each iteration")
    parser.add_argument("--port", type=str, default="6020")
    parser.add_argument("--verbose", action="store_true", help="verbosity level (bool)")

    opt = parser.parse_args()
    opt.device='cuda' if opt.gpu is None else f'cuda:{opt.gpu}'
    opt.distributed = opt.n_gpu_per_node > 1

    os.makedirs(opt.log_dir, exist_ok=True)
    (Path(opt.ckpt_path) / opt.name).mkdir(parents=True, exist_ok=True)
    opt.distributed = False
    return opt

def main(opt):
    logger = Logger(opt.log_dir)
    logger.info("toySB training")
    if Path(opt.dataset1).exists() and Path(opt.dataset2).exists():
        train_dataset, dim = load_dataset(opt.dataset1, opt.dataset2, logger, regime="train")
        val_dataset, dim = load_dataset(opt.dataset1, opt.dataset2, logger, regime="val")
    else:
        train_dataset, dim = get_pair_dataset(opt.n_samples, opt.dataset1, opt.dataset2, logger, path_to_save=opt.path_to_save, regime = "train")
        val_dataset, dim = get_pair_dataset(opt.microbatch_val, opt.dataset1, opt.dataset2, logger, path_to_save=opt.path_to_save, regime = "val")
    net = SB2D(x_dim = dim)
    run = RunnerAcceleration2D(opt, logger, net)
    run.train(opt, train_dataset, val_dataset)
    logger.info("Finish!")


if __name__ == '__main__':
    opt = create_arguments()
    if opt.distributed:
        size = opt.n_gpu_per_node

        processes = []
        for rank in range(size):
            opt = copy.deepcopy(opt)
            opt.local_rank = rank
            global_rank = rank + opt.node_rank * opt.n_gpu_per_node
            global_size = opt.num_proc_node * opt.n_gpu_per_node
            opt.global_rank = global_rank
            opt.global_size = global_size
            print('Node rank %d, local proc %d, global proc %d, global_size %d' % (opt.node_rank, rank, global_rank, global_size))
            p = Process(target=init_processes, args=(global_rank, global_size, main, opt))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
    else:
        th.cuda.set_device(0)
        opt.global_rank = 0
        opt.local_rank = 0
        opt.global_size = 1
        init_processes(0, opt.n_gpu_per_node, main, opt)