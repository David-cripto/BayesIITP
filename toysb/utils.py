import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch as th
import torchvision
from torch.optim import AdamW, lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from prefetch_generator import BackgroundGenerator
from .distributed_utils import all_gather
from torch.utils.data import DataLoader

IMAGE_CONSTANTS = {
    "scale":1,
    "width":0.002,
    "x_range":(-15,15),
    "y_range":(-15,15),
    "figsize":(20,10)
}

class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

def setup_loader(dataset, batch_size, num_workers=4, train=True):
    loader = DataLoaderX(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        pin_memory=True,
        num_workers=num_workers,
        drop_last=True,
    )

    while True:
        yield from loader

def all_cat_cpu(opt, log, t):
    if not opt.distributed: return t.detach().cpu()
    gathered_t = all_gather(t.to(opt.device), log=log)
    return th.cat(gathered_t).detach().cpu()

def unsqueeze_xdim(z, xdim):
    bc_dim = (...,) + (None,) * len(xdim)
    return z[bc_dim]

def make_beta_schedule(n_timestep=1000, linear_start=1e-4, linear_end=2e-2):
    betas = (
        th.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=th.float64) ** 2
    )
    return betas.numpy()

def create_symmetric_beta_schedule(n_timestep, *args, **kwargs):
    betas = make_beta_schedule(n_timestep=n_timestep, *args, **kwargs)
    betas = np.concatenate([betas[:n_timestep//2], np.flip(betas[:n_timestep//2])])
    return betas

def build_optimizer_sched(opt, net, logger):
    optim_dict = {"lr": opt.lr, 'weight_decay': opt.l2_norm}
    optimizer = AdamW(net.parameters(), **optim_dict)
    logger.info(f"[Opt] Built AdamW optimizer {optim_dict=}!")

    if opt.lr_gamma < 1.0:
        sched_dict = {"step_size": opt.lr_step, 'gamma': opt.lr_gamma}
        sched = lr_scheduler.StepLR(optimizer, **sched_dict)
        logger.info(f"[Opt] Built lr step scheduler {sched_dict=}!")
    else:
        sched = None

    if opt.load:
        checkpoint = th.load(opt.load, map_location="cpu")
        if "optimizer" in checkpoint.keys():
            optimizer.load_state_dict(checkpoint["optimizer"])
            logger.info(f"[Opt] Loaded optimizer ckpt {opt.load}!")
        else:
            logger.warning(f"[Opt] Ckpt {opt.load} has no optimizer!")
        if sched is not None and "sched" in checkpoint.keys() and checkpoint["sched"] is not None:
            sched.load_state_dict(checkpoint["sched"])
            logger.info(f"[Opt] Loaded lr sched ckpt {opt.load}!")
        else:
            logger.warning(f"[Opt] Ckpt {opt.load} has no lr sched!")

    return optimizer, sched

def space_indices(num_steps, count):
    assert count <= num_steps

    if count <= 1:
        frac_stride = 1
    else:
        frac_stride = (num_steps - 1) / (count - 1)

    cur_idx = 0.0
    taken_steps = []
    for _ in range(count):
        taken_steps.append(round(cur_idx))
        cur_idx += frac_stride

    return taken_steps

class TensorBoardWriter:
    def __init__(self, opt):
        run_dir = str(opt.log_dir / opt.name)
        os.makedirs(run_dir, exist_ok=True)
        self.writer=SummaryWriter(log_dir=run_dir, flush_secs=20)

    def add_scalar(self, global_step, key, val):
        self.writer.add_scalar(key, val, global_step=global_step)

    def add_figure(self, global_step, key, val):
        self.writer.add_figure(key, val, global_step=global_step)

    def close(self):
        self.writer.close()

def visualize2d(x1, x0, xs, num_trajectories = 100):
    fig = plt.figure(figsize = (10, 10))

    for ind in range(num_trajectories):
        point_trajectory = xs[ind, :, :]
        plt.plot(point_trajectory[:, 0], point_trajectory[:, 1], c = "#880808")

    plt.scatter(x0[:, 0], x0[:, 1], c = "#008000", edgecolors='black', label = "x0")
    plt.scatter(x1[:, 0], x1[:, 1], c = "#89CFF0", edgecolors='black', label = "x1")
    plt.scatter(x1[:num_trajectories, 0], x1[:num_trajectories, 1], c = "#880808", edgecolors='#880808', label = "start trajectory")
    plt.scatter(xs[:num_trajectories, 0, 0],xs[:num_trajectories, 0, 1], c = "#FFBF00", edgecolors='#FFBF00', label = "end trajectory")
    plt.legend()
    return fig

def visualize2d_inference(x1, xs, num_trajectories = 3):
    fig = plt.figure(figsize = (10, 10))

    for ind in range(num_trajectories):
        point_trajectory = xs[ind, :, :]
        plt.plot(point_trajectory[:, 0], point_trajectory[:, 1], c = "#880808")

    plt.scatter(x1[:, 0], x1[:, 1], c = "#89CFF0", edgecolors='black', label = "x1")
    plt.scatter(x1[:num_trajectories, 0], x1[:num_trajectories, 1], c = "#880808", edgecolors='#880808', label = "start trajectory")
    plt.scatter(xs[:, 0, 0],xs[:, 0, 1], c = "#FFBF00", edgecolors='#FFBF00', label = "end trajectory")
    plt.legend()
    return fig

def visualize(xs, x0, log_steps):
    fig, axs = plt.subplots(nrows = xs.shape[0], ncols=xs.shape[1] + 1, squeeze=False, figsize = (30, 20))
    for j, batch in enumerate(xs):
        img = torchvision.transforms.functional.to_pil_image((th.clamp(x0[j], -1., 1.) + 1)/2)
        axs[j, 0].imshow(np.asarray(img))
        axs[j, 0].set_title(f"True image")
        axs[j, 0].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        for i, img in enumerate(batch):
            img = img.detach()
            img = torchvision.transforms.functional.to_pil_image((th.clamp(img, -1., 1.) + 1)/2)
            axs[j, i + 1].imshow(np.asarray(img))
            axs[j, i + 1].set_title(f"Time = {log_steps[i]}")
            axs[j, i + 1].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    return fig

def save_imgs2d(log_steps, path_to_save, draw_dict):
    path_to_save = Path(path_to_save)

    for ind in range(len(log_steps)):
        plt.figure(figsize = IMAGE_CONSTANTS["figsize"])
        for name, vals in draw_dict.items():
            xs = vals["log_steps"]
            points_t = xs[:, ind, :]
            plt.scatter(points_t[:, 0], points_t[:, 1], c = vals["color"], label = name)
            plt.title(f"Points at time {log_steps[ind]}")
            if ind != 0:
                plt.quiver(points_t[:, 0], points_t[:, 1], vals["vel"][:, ind - 1, 0], vals["vel"][:, ind - 1, 1], 
                        angles='xy', scale_units='xy', scale=IMAGE_CONSTANTS["scale"], 
                        label = name, width =IMAGE_CONSTANTS["width"], color = vals["color"],
                        alpha = 0.3)
            
            plt.legend()
            plt.xlim(*IMAGE_CONSTANTS["x_range"])
            plt.ylim(*IMAGE_CONSTANTS["y_range"])
            plt.savefig(str(path_to_save / f"{log_steps[ind]}.png"))
        plt.close()

def save_imgs(xs, log_steps, path_to_save):
    path_to_save = Path(path_to_save)
    for ind, batch in enumerate(xs):
        path_to_dir = path_to_save / f"{ind}"
        path_to_dir.mkdir(exist_ok=True)

        for num_timestep, image in enumerate(batch):
            plt.figure(figsize = IMAGE_CONSTANTS["figsize"])
            plt.imshow(np.asarray(torchvision.transforms.functional.to_pil_image((th.clamp(image, -1., 1.) + 1)/2)))
            plt.axis("off")
            plt.title(f"Time = {log_steps[num_timestep]}")
            plt.savefig(str(path_to_dir / f"{log_steps[num_timestep]}.png"))
            plt.close()

def save_gif(path_to_imgs, path_to_save, range_list):
    import imageio

    path_to_imgs = Path(path_to_imgs)
    images = [imageio.imread(str(path_to_imgs / f"{i}.png")) for i in range_list]
    imageio.mimsave(path_to_save, images)

def build_range_list(path):
    return sorted([int(p.name.split(".")[0]) for p in path.glob("*")], reverse = True)

def expand_to_planes(input, shape):
    return input[..., None, None].repeat([1, 1, shape[2], shape[3]])

def frames_to_video(resulting_video_path, frames_path, fps = 5):
    import cv2
    frames_path = Path(frames_path)
    img_shape = cv2.imread(str(frames_path / '0.png')).shape
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    print("Original fourcc: ", fourcc)
    video = cv2.VideoWriter(resulting_video_path, fourcc, fps, (img_shape[1], img_shape[0]))
    
    for img_name in build_range_list(frames_path):
        img = cv2.imread(str(frames_path / f"{img_name}.png"))
        video.write(img)

    video.release()

def build_ckpt_option(log, ckpt_path):
    import pickle
    ckpt_path = Path(ckpt_path)
    opt_pkl_path = ckpt_path / "options.pkl"
    assert opt_pkl_path.exists()
    with open(opt_pkl_path, "rb") as f:
        ckpt_opt = pickle.load(f)
    log.info(f"Loaded options from {opt_pkl_path=}!")

    ckpt_opt.load = ckpt_path / "latest.pt"
    return ckpt_opt