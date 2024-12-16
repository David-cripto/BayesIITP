from toysb import Logger, SB2D, RunnerAcceleration2D
from toysb.utils import visualize2d_inference, build_ckpt_option
from toysb.datasets.dataset2d import load_dataset
import argparse
from torch.utils.data import DataLoader
from pathlib import Path
import os
import torch as th

def create_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_dataset", type=Path, help="path to terminal dataset")
    parser.add_argument("--path_to_save", type=Path, default="", help="path to save data")
    parser.add_argument("--ckpt_path", type=Path, default="", help="path to load checkpoints")
    parser.add_argument("--gpu", type=int, default=None, help="choose a particular device")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--log-dir", type=Path, default=".log", help="path to log std outputs and writer data")
    parser.add_argument("--nfe", type=int, default=1000, help="number of function evaluations")

    opt = parser.parse_args()

    opt.device='cuda' if opt.gpu is None else f'cuda:{opt.gpu}'
    os.makedirs(opt.log_dir, exist_ok=True)

    return opt

def compute_batch(ckpt_opt, corrupt_img):
    x1 = corrupt_img.to(ckpt_opt.device)

    cond = x1.detach() if ckpt_opt.cond_x1 else None
    if ckpt_opt.add_x1_noise: # only for decolor
        x1 = x1 + th.randn_like(x1)

    return x1, cond

def main(opt):
    logger = Logger(opt.log_dir)
    logger.info("toySB sampling")
    val_dataset, dim = load_dataset(opt.path_to_dataset, opt.path_to_dataset, logger, regime = "val")
    val_loader = DataLoader(val_dataset,
        batch_size=opt.batch_size, shuffle=False, pin_memory=True, num_workers=1, drop_last=False,
    )
    net = SB2D(x_dim = dim)
    ckpt_opt = build_ckpt_option(logger, opt.ckpt_path)
    path_to_save = Path(opt.path_to_save) / Path(opt.ckpt_path).stem
    path_to_save.mkdir(parents=True, exist_ok=True)
    run = RunnerAcceleration2D(ckpt_opt, logger, net, save_opt=False)
    all_x1, all_xs= [], []
    for x0, x1 in val_loader:
        x1, cond = compute_batch(ckpt_opt, x1)
        xs, pred_x0 = run.ddpm_sampling(ckpt_opt, x1, nfe = opt.nfe, cond=cond)
        x1, xs = x1.detach().to("cpu"), xs.detach().to("cpu")
        all_x1.append(x1), all_xs.append(xs)
    all_x1, all_xs = th.vstack(all_x1), th.vstack(all_xs)
    figure = visualize2d_inference(all_x1, all_xs, num_trajectories=20)
    figure.savefig(str(path_to_save / "figure.png"))
    logger.info("Finish!")


if __name__ == '__main__':
    opt = create_arguments()
    main(opt)