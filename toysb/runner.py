import pickle
import numpy as np
from .utils import make_beta_schedule, TensorBoardWriter, build_optimizer_sched, setup_loader, all_cat_cpu, space_indices, visualize2d
from .scheduler import Scheduler
import torch as th
from torch_ema import ExponentialMovingAverage
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F
import torchvision.utils as tu
import os
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
import cv2 as cv
import lpips

class Runner(object):
    def __init__(self, opt, log, net, save_opt=True):
        super(Runner,self).__init__()

        if save_opt:
            opt_pkl_path = opt.ckpt_path / "options.pkl"
            with open(opt_pkl_path, "wb") as f:
                pickle.dump(opt, f)
            log.info("Saved options pickle to {}!".format(opt_pkl_path))

        betas = make_beta_schedule(n_timestep=opt.num_steps, linear_end=opt.beta_max / opt.num_steps)
        betas = np.concatenate([betas[:opt.num_steps//2], np.flip(betas[:opt.num_steps//2])])
        self.scheduler = Scheduler(betas, opt.device)
        log.info(f"[Scheduler] Built I2SB scheduler: steps={len(betas)}!")

        print(f"use conditional = {opt.cond_x1}")
        self.net = net
        self.ema = ExponentialMovingAverage(self.net.parameters(), decay=opt.ema)

        if opt.load:
            checkpoint = th.load(opt.load, map_location="cpu")
            self.net.load_state_dict(checkpoint['net'])
            log.info(f"[Net] Loaded network ckpt: {opt.load}!")
            self.ema.load_state_dict(checkpoint["ema"])
            log.info(f"[Ema] Loaded ema ckpt: {opt.load}!")

        self.net.to(opt.device)
        self.ema.to(opt.device)

        self.loss_fn_alex = lpips.LPIPS(net='alex').double().to(opt.device)
        for params in self.loss_fn_alex.parameters():
            params.requires_grad_(False)
        self.loss_fn_alex.eval()

        self.log = log
    
    def compute_label(self, step, x0, xt):
        std_fwd = self.scheduler.get_std_fwd(step, xdim=x0.shape[1:])
        label = (xt - x0) / std_fwd
        return label.detach()
    
    def compute_pred_x0(self, step, xt, net_out, clip_denoise=False):
        std_fwd = self.scheduler.get_std_fwd(step, xdim=xt.shape[1:])
        pred_x0 = xt - std_fwd * net_out
        if clip_denoise: pred_x0.clamp_(-1., 1.)
        return pred_x0

    def sample_batch(self, opt, loader):
        clean_img, corrupt_img = next(loader)

        x0 = clean_img.detach().to(opt.device)
        x1 = corrupt_img.detach().to(opt.device)

        cond = x1.detach() if opt.cond_x1 else None

        if opt.add_x1_noise: # only for decolor
            x1 = x1 + th.randn_like(x1)

        return x0, x1, cond
    
    def train(self, opt, train_dataset, val_dataset):
        self.writer = TensorBoardWriter(opt)
        log = self.log

        net = DDP(self.net, device_ids=[opt.device])
        ema = self.ema
        optimizer, sched = build_optimizer_sched(opt, net, log)

        train_loader = setup_loader(train_dataset, opt.microbatch, train=True)
        val_loader   = setup_loader(val_dataset,   opt.microbatch_val, train=False)

        net.train()
        n_inner_loop = opt.batch_size // (opt.global_size * opt.microbatch)
        for it in range(opt.num_itr):
            optimizer.zero_grad()

            for _ in range(n_inner_loop):
                x0, x1, cond = self.sample_batch(opt, train_loader)

                step = th.randint(0, opt.num_steps, (x0.shape[0],))

                xt = self.scheduler.q_sample(step, x0, x1, ot_ode=opt.ot_ode)
                label = self.compute_label(step, x0, xt)

                pred = net(xt, step, cond=cond)

                loss = F.mse_loss(pred, label)
                loss.backward()

            optimizer.step()
            ema.update()
            if sched is not None: sched.step()

            log.info("train_it {}/{} | lr:{} | loss:{}".format(
                1+it,
                opt.num_itr,
                "{:.2e}".format(optimizer.param_groups[0]['lr']),
                "{:+.4f}".format(loss.item()),
            ))
            if it % 10 == 0:
                self.writer.add_scalar(it, 'loss', loss.detach())

            if it % 5000 == 0:
                if opt.global_rank == 0:
                    th.save({
                        "net": self.net.state_dict(),
                        "ema": ema.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "sched": sched.state_dict() if sched is not None else sched,
                    }, opt.ckpt_path / "latest.pt")
                    log.info(f"Saved latest({it=}) checkpoint to {opt.ckpt_path=}!")
                if opt.distributed:
                    th.distributed.barrier()

            if it == 500 or it % 3000 == 0:
                net.eval()
                self.evaluation(opt, it, val_loader)
                net.train()
        self.writer.close()
    
    @th.no_grad()
    def ddpm_sampling(self, opt, x1, cond=None, clip_denoise=False, nfe=None, verbose=True):
        nfe = nfe or opt.num_steps-1
        steps = space_indices(opt.num_steps, nfe+1)

        x1 = x1.to(opt.device)
        if cond is not None: cond = cond.to(opt.device)

        with self.ema.average_parameters():
            self.net.eval()

            def pred_x0_fn(xt, step):
                step = th.full((xt.shape[0],), step, device=opt.device, dtype=th.long)
                out = self.net(xt, step, cond=cond)
                return self.compute_pred_x0(step, xt, out, clip_denoise=clip_denoise)

            xs, pred_x0 = self.scheduler.ddpm_sampling(
                steps, pred_x0_fn, x1, ot_ode=opt.ot_ode, verbose=verbose,
            )
        return xs, pred_x0

    @th.no_grad()
    def evaluation(self, opt, it, val_loader):
        pass

class RunnerImage(Runner):
    @th.no_grad()
    def evaluation(self, opt, it, val_loader):
        log = self.log
        log.info(f"========== Evaluation started: iter={it} ==========")

        def log_image(tag, img, nrow=10):
            self.writer.add_image(it, tag, tu.make_grid((img + 1) / 2, nrow=nrow))  # [1,1] -> [0,1]

        def create_image_from_tensor(img, nrow=10):
            img_grid = tu.make_grid((img + 1) / 2, nrow=nrow)
            img_grid_numpy = (img_grid.permute((1, 2, 0)) * 255 + 0.5).numpy()
            img_grid_numpy = np.clip(img_grid_numpy, 0.0, 255.0).astype(np.uint8)
            img_grid_numpy = cv.cvtColor(img_grid_numpy, cv.COLOR_BGR2RGB)
            return img_grid_numpy

        def log_lpips(tag, img_clean, img_recon):
            img_clean = img_clean.clip(-1., 1.).to(opt.device)
            img_recon = img_recon.clip(-1., 1.).to(opt.device)
            reconstructed_lpips = self.loss_fn_alex(img_clean, img_recon).cpu().mean()
            print(f"Batch LPIPS on it {it} = {reconstructed_lpips}")
            self.writer.add_scalar(it, tag, reconstructed_lpips)

        def log_ssim(tag, img_clean, img_recon):
            print(f"Rec img with min {th.min(img_recon)} and max {th.max(img_recon)}")
            print(f"Clean img with min {th.min(img_clean)} and max {th.max(img_clean)}")
            img_clean_norm = (img_clean + 1) / 2
            img_recon_norm = (img_recon + 1) / 2
            img_clean_norm = th.clamp(img_clean_norm.detach().cpu().permute((0, 2, 3, 1)), 0.0, 1.0).numpy()
            img_recon_norm = th.clamp(img_recon_norm.detach().cpu().permute((0, 2, 3, 1)), 0.0, 1.0).numpy()
            print(f"Normalization for rec img with min {np.min(img_recon_norm)} and max {np.max(img_recon_norm)}")
            print(f"Normalization for clean img with min {np.min(img_clean_norm)} and max {np.max(img_clean_norm)}")
            num_images_in_batch = img_clean.shape[0]
            ssim_reconstructed_arr = []
            for i in range(num_images_in_batch):
                cur_recon_image = img_recon_norm[i]
                cur_clean_image = img_clean_norm[i]
                ssim_reconstructed_current = structural_similarity(cur_clean_image, cur_recon_image,
                                                                   channel_axis=-1,
                                                                   multichannel=True, data_range=1.0)
                ssim_reconstructed_arr.append(ssim_reconstructed_current)
            ssim_reconstructed_arr = np.array(ssim_reconstructed_arr)
            reconstructed_ssim = ssim_reconstructed_arr.mean()
            print(f"Batch SSIM on it {it} = {reconstructed_ssim}")
            self.writer.add_scalar(it, tag, reconstructed_ssim)

        def log_psnr(tag, img_clean, img_recon):
            print(f"Rec img with min {th.min(img_recon)} and max {th.max(img_recon)}")
            print(f"Clean img with min {th.min(img_clean)} and max {th.max(img_clean)}")
            img_clean_norm = (img_clean + 1) / 2
            img_recon_norm = (img_recon + 1) / 2
            img_clean_norm = th.clamp(img_clean_norm.detach().cpu().permute((0, 2, 3, 1)), 0.0, 1.0).numpy()
            img_recon_norm = th.clamp(img_recon_norm.detach().cpu().permute((0, 2, 3, 1)), 0.0, 1.0).numpy()
            num_images_in_batch = img_clean.shape[0]
            psnr_reconstructed_arr = []
            print(f"Normalization for rec img with min {np.min(img_recon_norm)} and max {np.max(img_recon_norm)}")
            print(f"Normalization for clean img with min {np.min(img_clean_norm)} and max {np.max(img_clean_norm)}")
            for i in range(num_images_in_batch):
                cur_recon_image = img_recon_norm[i]
                cur_clean_image = img_clean_norm[i]
                psnr_reconstructed_current = peak_signal_noise_ratio(cur_clean_image, cur_recon_image, data_range=1.0)
                psnr_reconstructed_arr.append(psnr_reconstructed_current)
            psnr_reconstructed_arr = np.array(psnr_reconstructed_arr)
            reconstructed_psnr = psnr_reconstructed_arr.mean()
            print(f"Batch PSNR on it {it} = {reconstructed_psnr}")
            self.writer.add_scalar(it, tag, reconstructed_psnr)

        img_clean, img_corrupt, cond = self.sample_batch(opt, val_loader)

        x1 = img_corrupt.to(opt.device)

        print(f"evaluation on random val tensor for metrics = {x1.shape}")

        xs, pred_x0s = self.ddpm_sampling(
            opt, x1, cond=cond, clip_denoise=opt.clip_denoise,
            nfe=opt.val_nfe,
            verbose=opt.global_rank == 0
        )

        log.info("Collecting tensors ...")
        img_clean = all_cat_cpu(opt, log, img_clean)
        img_corrupt = all_cat_cpu(opt, log, img_corrupt)

        print(f"img_clean.shape = {img_clean.shape}, img_corrupt = {img_corrupt.shape}")
        xs = all_cat_cpu(opt, log, xs)
        pred_x0s = all_cat_cpu(opt, log, pred_x0s)

        batch, len_t, *xdim = xs.shape
        assert img_clean.shape == img_corrupt.shape == (batch, *xdim)
        assert xs.shape == pred_x0s.shape
        log.info(f"Generated recon trajectories: size={xs.shape}")

        img_recon = xs[:, 0, ...]

        log.info("Logging LPIPS ...")
        log_lpips("lpips_val", img_clean, img_recon)
        log.info("Logging SSIM ...")
        log_ssim("ssim_val", img_clean, img_recon)
        log.info("Logging PSNR ...")
        log_psnr("psnr_val", img_clean, img_recon)

        path_to_log_exp = os.path.join(opt.log_dir, opt.name)
        clean_img_path = os.path.join(path_to_log_exp, f"clean_val_img_{it}.png")
        recon_img_path = os.path.join(path_to_log_exp, f"recon_val_img_{it}.png")

        img_clean_val = create_image_from_tensor(img_clean, nrow=10)
        img_recon_val = create_image_from_tensor(img_recon, nrow=10)
        cv.imwrite(clean_img_path, img_clean_val)
        print(f"saving {clean_img_path}")
        cv.imwrite(recon_img_path, img_recon_val)
        print(f"saving {recon_img_path}")

        log_image("image/clean", img_clean)
        log_image("image/corrupt", img_corrupt)
        log_image("image/recon", img_recon)

        log.info(f"========== Evaluation finished: iter={it} ==========")
        th.cuda.empty_cache()

class Runner2D(Runner):
    @th.no_grad()
    def evaluation(self, opt, it, val_loader):
        log = self.log
        log.info(f"========== Evaluation started: iter={it} ==========")
        img_clean, img_corrupt, cond = self.sample_batch(opt, val_loader)
        x1 = img_corrupt.to(opt.device)

        print(f"evaluation on random val tensor for metrics = {x1.shape}")

        xs, pred_x0s = self.ddpm_sampling(
            opt, x1, cond=cond, clip_denoise=opt.clip_denoise,
            nfe=opt.val_nfe,
            verbose=opt.global_rank == 0
        )
        img_corrupt, img_clean, xs = img_corrupt.detach().to("cpu"), img_clean.detach().to("cpu"), xs.detach().to("cpu")
        figure = visualize2d(img_corrupt, img_clean, xs)
        self.writer.add_figure(it, "log images", figure)
        log.info(f"========== Evaluation finished: iter={it} ==========")
        th.cuda.empty_cache()

class RunnerHorizon2D(Runner2D):   
    def train(self, opt, train_dataset, val_dataset):
        self.writer = TensorBoardWriter(opt)
        log = self.log

        net = DDP(self.net, device_ids=[opt.device])
        ema = self.ema
        optimizer, sched = build_optimizer_sched(opt, net, log)

        train_loader = setup_loader(train_dataset, opt.microbatch, train=True)
        val_loader   = setup_loader(val_dataset,   opt.microbatch_val, train=False)

        net.train()
        n_inner_loop = opt.batch_size // (opt.global_size * opt.microbatch)
        for it in range(opt.num_itr):
            optimizer.zero_grad()

            for _ in range(n_inner_loop):
                x0, x1, cond = self.sample_batch(opt, train_loader)

                t_start_np = np.random.randint(low=opt.horizon, high=opt.num_steps)
                horizon_np = np.random.randint(low=2, high=opt.horizon+1)
                t_end_np = t_start_np - horizon_np
                t_start = th.full((x0.shape[0],), t_start_np)
                horizon = th.full((x0.shape[0],), horizon_np)

                x_start = self.scheduler.q_sample(t_start, x0, x1, ot_ode=opt.ot_ode)
                t_end = t_start - horizon
                x_end = self.scheduler.q_sample(t_end, x0, x1, ot_ode=opt.ot_ode)
                steps = [i for i in range(t_end_np, t_start_np + 1)]
                with self.ema.average_parameters():
                    def pred_x0_fn(xt, step, out_network):
                        step = th.full((xt.shape[0],), step, device=opt.device, dtype=th.long)
                        out = self.net(xt, step, cond=cond)
                        if out_network is None:
                            out_network = out
                        else:
                            out_network = 0.99*out_network + 0.01*out
                        return self.compute_pred_x0(step, xt, out_network, clip_denoise=opt.clip_denoise)
                    xs = self.scheduler.ddpm_sampling_training(steps, pred_x0_fn, x_start)

                loss = F.mse_loss(xs, x_end)
                loss.backward()

            optimizer.step()
            ema.update()
            if sched is not None: sched.step()

            log.info("train_it {}/{} | lr:{} | loss:{}".format(
                1+it,
                opt.num_itr,
                "{:.2e}".format(optimizer.param_groups[0]['lr']),
                "{:+.4f}".format(loss.item()),
            ))
            if it % 10 == 0:
                self.writer.add_scalar(it, 'loss', loss.detach())

            if it % 5000 == 0:
                if opt.global_rank == 0:
                    th.save({
                        "net": self.net.state_dict(),
                        "ema": ema.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "sched": sched.state_dict() if sched is not None else sched,
                    }, opt.ckpt_path / "latest.pt")
                    log.info(f"Saved latest({it=}) checkpoint to {opt.ckpt_path=}!")
                if opt.distributed:
                    th.distributed.barrier()

            if it == 500 or it % 3000 == 0:
                net.eval()
                self.evaluation(opt, it, val_loader)
                net.train()
        self.writer.close()
    
class RunnerAcceleration2D(Runner2D):   
    def train(self, opt, train_dataset, val_dataset):
        self.writer = TensorBoardWriter(opt)
        log = self.log

        net = DDP(self.net, device_ids=[opt.device])
        ema = self.ema
        optimizer, sched = build_optimizer_sched(opt, net, log)

        train_loader = setup_loader(train_dataset, opt.microbatch, train=True)
        val_loader   = setup_loader(val_dataset,   opt.microbatch_val, train=False)

        net.train()
        n_inner_loop = opt.batch_size // (opt.global_size * opt.microbatch)
        for it in range(opt.num_itr):
            optimizer.zero_grad()

            for _ in range(n_inner_loop):
                x0, x1, cond = self.sample_batch(opt, train_loader)

                t_start_np = np.random.randint(2, high=opt.num_steps)
                t_start = th.full((x0.shape[0],), t_start_np)

                x_start = self.scheduler.q_sample(t_start, x0, x1, ot_ode=opt.ot_ode)
                label = self.compute_label(t_start, x0, x_start)
                timestep = th.full((x0.shape[0],), 0) if np.random.binomial(1, 0.7) else t_start
                pred = net(x_start, timestep, cond=cond)
                loss = F.mse_loss(pred, label)

                x0_hat = self.compute_pred_x0(t_start, x_start, pred.detach(), clip_denoise=opt.clip_denoise)
                xtm1 = self.scheduler.p_posterior(t_start_np - 1, t_start_np, x_start, x0_hat, ot_ode=opt.ot_ode)

                timestep = th.full((x0.shape[0],), 0) if np.random.binomial(1, 0.7) else t_start - 1
                pred1 = net(xtm1, timestep, cond=cond)
                x0_hat = self.compute_pred_x0(t_start - 1, xtm1, pred1, clip_denoise=opt.clip_denoise)
                xtm2 = self.scheduler.p_posterior(t_start_np - 2, t_start_np - 1, xtm1, x0_hat, ot_ode=opt.ot_ode)
                loss = loss + 1000*F.mse_loss(xtm2 - xtm1, xtm1 - x_start)
                
                loss.backward()

            optimizer.step()
            ema.update()
            if sched is not None: sched.step()

            log.info("train_it {}/{} | lr:{} | loss:{}".format(
                1+it,
                opt.num_itr,
                "{:.2e}".format(optimizer.param_groups[0]['lr']),
                "{:+.4f}".format(loss.item()),
            ))
            if it % 10 == 0:
                self.writer.add_scalar(it, 'loss', loss.detach())

            if it % 5000 == 0:
                if opt.global_rank == 0:
                    th.save({
                        "net": self.net.state_dict(),
                        "ema": ema.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "sched": sched.state_dict() if sched is not None else sched,
                    }, opt.ckpt_path / "latest.pt")
                    log.info(f"Saved latest({it=}) checkpoint to {opt.ckpt_path=}!")
                if opt.distributed:
                    th.distributed.barrier()

            if it == 500 or it % 3000 == 0:
                net.eval()
                self.evaluation(opt, it, val_loader)
                net.train()
        self.writer.close()

class RunnerAngle2D(Runner2D):   
    def train(self, opt, train_dataset, val_dataset):
        self.writer = TensorBoardWriter(opt)
        log = self.log

        net = DDP(self.net, device_ids=[opt.device])
        ema = self.ema
        optimizer, sched = build_optimizer_sched(opt, net, log)

        train_loader = setup_loader(train_dataset, opt.microbatch, train=True)
        val_loader   = setup_loader(val_dataset,   opt.microbatch_val, train=False)

        net.train()
        n_inner_loop = opt.batch_size // (opt.global_size * opt.microbatch)
        weight = 2
        for it in range(opt.num_itr):
            optimizer.zero_grad()

            for _ in range(n_inner_loop):
                x0, x1, cond = self.sample_batch(opt, train_loader)

                t_start_np = np.random.randint(2, high=opt.num_steps)
                t_start = th.full((x0.shape[0],), t_start_np)

                x_start = self.scheduler.q_sample(t_start, x0, x1, ot_ode=opt.ot_ode)
                label = self.compute_label(t_start, x0, x_start)
                pred = net(x_start, t_start, cond=cond)
                loss = F.mse_loss(pred, label)

                x0_hat = self.compute_pred_x0(t_start, x_start, pred.detach(), clip_denoise=opt.clip_denoise)
                xtm1 = self.scheduler.p_posterior(t_start_np - 1, t_start_np, x_start, x0_hat, ot_ode=opt.ot_ode)
                v1 = xtm1 - x_start

                pred1 = net(xtm1, t_start - 1, cond=cond)
                x0_hat = self.compute_pred_x0(t_start - 1, xtm1, pred1, clip_denoise=opt.clip_denoise)
                xtm2 = self.scheduler.p_posterior(t_start_np - 2, t_start_np - 1, xtm1, x0_hat, ot_ode=opt.ot_ode)
                v2 = xtm2 - xtm1
                cos_angle = th.clamp((v2 * v1).sum(1)/(th.norm(v1, dim = 1)*th.norm(v2, dim = 1)), min = -1, max= 1)
                
                loss = loss + th.acos(cos_angle).mean()
                loss.backward()

            optimizer.step()
            ema.update()
            if sched is not None: sched.step()

            log.info("train_it {}/{} | lr:{} | loss:{}".format(
                1+it,
                opt.num_itr,
                "{:.2e}".format(optimizer.param_groups[0]['lr']),
                "{:+.4f}".format(loss.item()),
            ))
            if it % 10 == 0:
                self.writer.add_scalar(it, 'loss', loss.detach())

            if it % 5000 == 0:
                if opt.global_rank == 0:
                    th.save({
                        "net": self.net.state_dict(),
                        "ema": ema.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "sched": sched.state_dict() if sched is not None else sched,
                    }, opt.ckpt_path / "latest.pt")
                    log.info(f"Saved latest({it=}) checkpoint to {opt.ckpt_path=}!")
                if opt.distributed:
                    th.distributed.barrier()

            if it == 500 or it % 3000 == 0:
                net.eval()
                self.evaluation(opt, it, val_loader)
                net.train()
                if it < 10000:
                    weight *= 2
            
        self.writer.close()
    
class RunnerTime2D(Runner2D):   
    def train(self, opt, train_dataset, val_dataset):
        self.writer = TensorBoardWriter(opt)
        log = self.log

        net = DDP(self.net, device_ids=[opt.device])
        ema = self.ema
        optimizer, sched = build_optimizer_sched(opt, net, log)

        train_loader = setup_loader(train_dataset, opt.microbatch, train=True)
        val_loader   = setup_loader(val_dataset,   opt.microbatch_val, train=False)

        net.train()
        n_inner_loop = opt.batch_size // (opt.global_size * opt.microbatch)
        for it in range(opt.num_itr):
            optimizer.zero_grad()

            for _ in range(n_inner_loop):
                x0, x1, cond = self.sample_batch(opt, train_loader)

                t_start_np = np.random.randint(2, high=opt.num_steps)
                t_start = th.full((x0.shape[0],), t_start_np)

                x_start = self.scheduler.q_sample(t_start, x0, x1, ot_ode=opt.ot_ode)
                label = self.compute_label(t_start, x0, x_start)
                timestep = th.full((x0.shape[0],), 0) 
                pred = net(x_start, timestep, cond=cond)
                loss = F.mse_loss(pred, label)
                
                loss.backward()

            optimizer.step()
            ema.update()
            if sched is not None: sched.step()

            log.info("train_it {}/{} | lr:{} | loss:{}".format(
                1+it,
                opt.num_itr,
                "{:.2e}".format(optimizer.param_groups[0]['lr']),
                "{:+.4f}".format(loss.item()),
            ))
            if it % 10 == 0:
                self.writer.add_scalar(it, 'loss', loss.detach())

            if it % 5000 == 0:
                if opt.global_rank == 0:
                    th.save({
                        "net": self.net.state_dict(),
                        "ema": ema.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "sched": sched.state_dict() if sched is not None else sched,
                    }, opt.ckpt_path / "latest.pt")
                    log.info(f"Saved latest({it=}) checkpoint to {opt.ckpt_path=}!")
                if opt.distributed:
                    th.distributed.barrier()

            if it == 500 or it % 3000 == 0:
                net.eval()
                self.evaluation(opt, it, val_loader)
                net.train()
        self.writer.close()

    @th.no_grad()
    def ddpm_sampling(self, opt, x1, cond=None, clip_denoise=False, nfe=None, verbose=True):
        nfe = nfe or opt.num_steps-1
        steps = space_indices(opt.num_steps, nfe+1)

        x1 = x1.to(opt.device)
        if cond is not None: cond = cond.to(opt.device)

        with self.ema.average_parameters():
            self.net.eval()

            def pred_x0_fn(xt, step):
                step = th.full((xt.shape[0],), 0, device=opt.device, dtype=th.long)
                out = self.net(xt, step, cond=cond)
                return self.compute_pred_x0(step, xt, out, clip_denoise=clip_denoise)

            xs, pred_x0 = self.scheduler.ddpm_sampling(
                steps, pred_x0_fn, x1, ot_ode=opt.ot_ode, verbose=verbose,
            )
        return xs, pred_x0
    
class RunnerODE(object):
    def __init__(self, opt, log, net_f, net_b, save_opt=True):
        super(RunnerODE,self).__init__()

        if save_opt:
            opt_pkl_path = opt.ckpt_path / "options.pkl"
            with open(opt_pkl_path, "wb") as f:
                pickle.dump(opt, f)
            log.info("Saved options pickle to {}!".format(opt_pkl_path))

        betas = make_beta_schedule(n_timestep=opt.num_steps, linear_end=opt.beta_max / opt.num_steps)
        betas = np.concatenate([betas[:opt.num_steps//2], np.flip(betas[:opt.num_steps//2])])
        self.scheduler = Scheduler(betas, opt.device)
        log.info(f"[Scheduler] Built I2SB scheduler: steps={len(betas)}!")

        print(f"use conditional = {opt.cond_x1}")
        self.net_f = net_f
        self.ema_f = ExponentialMovingAverage(self.net_f.parameters(), decay=opt.ema)

        self.net_b = net_b
        self.ema_b = ExponentialMovingAverage(self.net_b.parameters(), decay=opt.ema)

        if opt.load:
            checkpoint = th.load(opt.load, map_location="cpu")
            self.net_f.load_state_dict(checkpoint['net_f'])
            log.info(f"[Net] Loaded forward network ckpt: {opt.load}!")
            self.ema_f.load_state_dict(checkpoint["ema_f"])
            log.info(f"[Ema] Loaded forward ema ckpt: {opt.load}!")

            self.net_b.load_state_dict(checkpoint['net_b'])
            log.info(f"[Net] Loaded backward network ckpt: {opt.load}!")
            self.ema_b.load_state_dict(checkpoint["ema_b"])
            log.info(f"[Ema] Loaded backward ema ckpt: {opt.load}!")

        self.net_f.to(opt.device)
        self.ema_f.to(opt.device)

        self.net_b.to(opt.device)
        self.ema_b.to(opt.device)

        self.log = log

    def sample_batch(self, opt, loader):
        clean_img, corrupt_img = next(loader)

        x0 = clean_img.detach().to(opt.device)
        x1 = corrupt_img.detach().to(opt.device)

        cond = x1.detach() if opt.cond_x1 else None

        if opt.add_x1_noise: # only for decolor
            x1 = x1 + th.randn_like(x1)

        return x0, x1, cond

    def compute_label(self, step, x0, x1, xt):
        std_fwd = self.scheduler.get_std_fwd(step, xdim=x0.shape[1:])
        std_bwd = self.scheduler.get_std_bwd(step, xdim=x1.shape[1:])
        label_b = (xt - x0) / (std_fwd**2)
        label_f = (x1 - xt) / (std_bwd**2)
        return label_b.detach(), label_f.detach()
    
    def train(self, opt, train_dataset, val_dataset):
        self.writer = TensorBoardWriter(opt)
        log = self.log

        net_f = DDP(self.net_f, device_ids=[opt.device])
        ema_f = self.ema_f
        net_b = DDP(self.net_b, device_ids=[opt.device])
        ema_b = self.ema_b
        optimizer_f, sched_f = build_optimizer_sched(opt, net_f, log)
        optimizer_b, sched_b = build_optimizer_sched(opt, net_b, log)

        train_loader = setup_loader(train_dataset, opt.microbatch, train=True)
        val_loader   = setup_loader(val_dataset,   opt.microbatch_val, train=False)

        net_f.train(), net_b.train()
        n_inner_loop = opt.batch_size // (opt.global_size * opt.microbatch)
        for it in range(opt.num_itr):
            optimizer_f.zero_grad(), optimizer_b.zero_grad()

            for _ in range(n_inner_loop):
                x0, x1, cond = self.sample_batch(opt, train_loader)

                step = th.randint(0, opt.num_steps, (x0.shape[0],))

                xt = self.scheduler.q_sample(step, x0, x1, ot_ode=opt.ot_ode)
                label_b, label_f = self.compute_label(step, x0, x1, xt)

                pred_f = net_f(xt, step, cond=cond)
                pred_b = net_b(xt, step, cond=cond)
                sigma = self.scheduler.get_std_sb(step, xdim=x1.shape[1:])**2
                loss_f = F.mse_loss(pred_f, label_f * sigma)
                loss_b = F.mse_loss(pred_b, label_b * sigma)
                loss_consistency = F.mse_loss(pred_b + pred_f, (label_b + label_f) * sigma)
                loss = loss_f + loss_b + loss_consistency
                loss.backward()

            optimizer_f.step(), optimizer_b.step()
            ema_f.update(), ema_b.update()
            if sched_f is not None: sched_f.step()
            if sched_b is not None: sched_b.step()

            log.info("train_it {}/{} | lr:{} | loss:{}".format(
                1+it,
                opt.num_itr,
                "{:.2e}".format(optimizer_f.param_groups[0]['lr']),
                "{:+.4f}".format(loss.item()),
            ))
            if it % 10 == 0:
                self.writer.add_scalar(it, 'loss', loss.detach())
                self.writer.add_scalar(it, 'loss_f', loss_f.detach())
                self.writer.add_scalar(it, 'loss_b', loss_b.detach())
                self.writer.add_scalar(it, 'loss_consistency', loss_consistency.detach())

            if it % 5000 == 0:
                if opt.global_rank == 0:
                    th.save({
                        "net_f": self.net_f.state_dict(),
                        "ema_f": ema_f.state_dict(),
                        "optimizer_f": optimizer_f.state_dict(),
                        "sched_f": sched_f.state_dict() if sched_f is not None else sched_f,
                        "net_b": self.net_b.state_dict(),
                        "ema_b": ema_b.state_dict(),
                        "optimizer_b": optimizer_b.state_dict(),
                        "sched_b": sched_b.state_dict() if sched_b is not None else sched_b
                    }, opt.ckpt_path / "latest.pt")
                    log.info(f"Saved latest({it=}) checkpoint to {opt.ckpt_path=}!")
                    net_f.eval()
                    net_b.eval()
                    self.evaluation(opt, it, val_loader)
                    net_f.train()
                    net_b.eval()
                if opt.distributed:
                    th.distributed.barrier()

            
        self.writer.close()
    
    @th.no_grad()
    def velocity_sampling(self, opt, x0, cond=None, nfe=None, verbose=True):
        nfe = nfe or opt.num_steps-1
        steps = space_indices(opt.num_steps, nfe+1)

        x0 = x0.to(opt.device)
        if cond is not None: cond = cond.to(opt.device)

        with self.ema_f.average_parameters():
            with self.ema_b.average_parameters():
                self.net_f.eval(), self.net_b.eval()

                def pred_velocity(xt, step):
                    step = th.full((xt.shape[0],), step, device=opt.device, dtype=th.long)
                    out = (self.net_f(xt, step, cond=cond) + self.net_b(xt, step, cond=cond))/2
                    return out

                xs = self.scheduler.velocity_sampling(
                    steps, pred_velocity, x0, verbose=verbose,
                )
        return xs

    @th.no_grad()
    def evaluation(self, opt, it, val_loader):
        log = self.log
        log.info(f"========== Evaluation started: iter={it} ==========")
        img_clean, img_corrupt, cond = self.sample_batch(opt, val_loader)
        x0 = img_clean.to(opt.device)

        print(f"evaluation on random val tensor for metrics = {x0.shape}")

        xs = self.velocity_sampling(
            opt, x0, cond=cond,
            nfe=opt.val_nfe,
            verbose=opt.global_rank == 0
        )
        img_corrupt, img_clean, xs = img_corrupt.detach().to("cpu"), img_clean.detach().to("cpu"), xs.detach().to("cpu")
        figure = visualize2d(img_corrupt, img_clean, xs)
        self.writer.add_figure(it, "log images", figure)
        log.info(f"========== Evaluation finished: iter={it} ==========")
        th.cuda.empty_cache()