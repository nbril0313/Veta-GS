#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from datetime import datetime
import os
from pathlib import Path
import time
import torch
from random import randint
import math

from utils.loss_utils import compute_monossim_loss, l1_loss, ssim
from gaussian_renderer import render, network_gui, GaussianRasterizationSettings, GaussianRasterizer
import sys
from scene import Scene, GaussianModel, DeformModel, EFMModel, FFMModel, IFMModel
from utils.general_utils import safe_state, get_linear_noise_func
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from itertools import islice, count

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def training(dataset, opt, pipe, testing_iterations, saving_iterations, device):
    start_time = time.time()
    tb_writer, model_path = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, device)

    EFM = EFMModel(device)
    EFM.train_setting(opt)
    IFM = IFMModel(device)
    IFM.train_setting(opt)
    FFM = FFMModel(device)
    FFM.train_setting(opt)
    deform = DeformModel(device, dataset.is_blender, dataset.is_6dof)
    deform.train_setting(opt)

    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt, device)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32).to(device)

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    best_psnr = 0.0
    best_iteration = 0
    progress_bar = tqdm(total=opt.iterations, desc="Training progress")
    smooth_term = get_linear_noise_func(lr_init=0.1, lr_final=1e-15, lr_delay_mult=0.01, max_steps=20000)
    # for iteration in range(1, opt.iterations + 1):
    for iteration in islice(count(1), opt.iterations):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.do_shs_python, pipe.do_cov_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, device, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2,
                                                                                                               0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()

        total_frame = len(viewpoint_stack)
        time_interval = 1 / total_frame

        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
        if dataset.load2gpu_on_the_fly:
            viewpoint_cam.load2device()
        fid = viewpoint_cam.fid
        if iteration < opt.warm_up:
            d_xyz_full, d_rotation_full, d_scaling_full = 0.0, 0.0, 0.0
        else:
            R = viewpoint_cam.R
            T = viewpoint_cam.T
            R_torch = torch.from_numpy(R).float().to(device)  
            T_torch = torch.from_numpy(T).float().to(device)

            Rt = torch.eye(4, device=device)
            Rt[:3, :3] = R_torch.t()  
            Rt[:3, 3] = -R_torch.t() @ T_torch       

            c2w = Rt
            view_dir = -c2w[:3, 2]       

            with torch.no_grad():
                # Compute visibility_filter using rasterizer directly
                tanfovx = math.tan(viewpoint_cam.FoVx * 0.5)
                tanfovy = math.tan(viewpoint_cam.FoVy * 0.5)
                raster_settings = GaussianRasterizationSettings(
                    image_height=viewpoint_cam.image_height,
                    image_width=viewpoint_cam.image_width,
                    tanfovx=tanfovx,
                    tanfovy=tanfovy,
                    bg=background,
                    scale_modifier=1.0,
                    viewmatrix=viewpoint_cam.world_view_transform,
                    projmatrix=viewpoint_cam.full_proj_transform,
                    sh_degree=gaussians.active_sh_degree,
                    campos=viewpoint_cam.camera_center,
                    prefiltered=False,
                    debug=pipe.debug
                )
                screenspace_points = torch.zeros_like(gaussians.get_xyz, dtype=gaussians.get_xyz.dtype, device=device)
                screenspace_points_densify = torch.zeros_like(gaussians.get_xyz, dtype=gaussians.get_xyz.dtype, device=device)
                opacity = gaussians.get_opacity
                scales = gaussians.get_scaling
                rotations = gaussians.get_rotation
                shs = gaussians.get_features
            
                rasterizer = GaussianRasterizer(raster_settings)
                _, radii, _ = rasterizer(
                    means3D=gaussians.get_xyz,
                    means2D=screenspace_points,
                    means2D_densify=screenspace_points_densify,
                    shs=shs,
                    colors_precomp=None,
                    opacities=opacity,
                    scales=scales,
                    rotations=rotations,
                    cov3D_precomp=None
                )

                visibility_filter = radii > 0
                visible_indices = visibility_filter.nonzero(as_tuple=True)[0]
                xyz_visible = gaussians.get_xyz[visible_indices]
            time_input = fid.unsqueeze(0).expand(xyz_visible.shape[0], -1)
            ast_noise = 0 if dataset.is_blender else torch.randn(1, 1, device=device).expand(xyz_visible.shape[0], -1) * time_interval * smooth_term(iteration)
            view_dir = view_dir.unsqueeze(0).expand(xyz_visible.shape[0], -1)  # -> (N, 3)
            cam_pos  = viewpoint_cam.camera_center.unsqueeze(0).expand(xyz_visible.shape[0], -1)   # -> (N, 3)

            d_xyz, d_rotation, d_scaling = deform.step(xyz_visible.detach(), time_input + ast_noise, cam_pos, view_dir)

            d_xyz_full = torch.zeros_like(gaussians.get_xyz)  # shape (N, 3)
            d_xyz_full[visible_indices] = d_xyz
            d_rotation_full = torch.zeros_like(gaussians.get_rotation)  # shape (N, 3)
            d_rotation_full[visible_indices] = d_rotation
            d_scaling_full = torch.zeros_like(gaussians.get_scaling)  # shape (N, 3)
            d_scaling_full[visible_indices] = d_scaling

        # Render
        render_pkg_re = render(viewpoint_cam, gaussians, pipe, background, d_xyz_full, d_rotation_full, d_scaling_full, device, dataset.is_6dof)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg_re["render"], render_pkg_re[
            "viewspace_points"], render_pkg_re["visibility_filter"], render_pkg_re["radii"]

        gt_image = viewpoint_cam.original_image.to(device)
        
        # Loss
        Ll1 = l1_loss(image, gt_image)
        if iteration <= 20000:
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        else:
            monossim_loss = compute_monossim_loss(image, gt_image, EFM, FFM, IFM, device)

            loss = (1.0 - opt.lambda_dssim - 0.2) * (Ll1) + opt.lambda_dssim * (1.0 - ssim(image, gt_image)) + 0.2 * monossim_loss
        
        loss.backward()

        iter_end.record()

        if dataset.load2gpu_on_the_fly:
            viewpoint_cam.load2device('cpu')

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Keep track of max radii in image-space for pruning
            gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
                                                                 radii[visibility_filter])

            # Log and save
            cur_psnr = training_report(dataset.source_path, model_path, tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end),
                                       testing_iterations, scene, render, (pipe, background), deform,
                                       dataset.load2gpu_on_the_fly, device, dataset.is_6dof, start_time=start_time)
            if iteration in testing_iterations:
                if cur_psnr.item() > best_psnr:
                    best_psnr = cur_psnr.item()
                    best_iteration = iteration

            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                deform.save_weights(model_path, iteration)
                EFM.save_weights(model_path, iteration)
                FFM.save_weights(model_path, iteration)
                IFM.save_weights(model_path, iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                viewspace_point_tensor_densify = render_pkg_re["viewspace_points_densify"]
                gaussians.add_densification_stats(viewspace_point_tensor_densify, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, device)

                if iteration % opt.opacity_reset_interval == 0 or (
                        dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.update_learning_rate(iteration)
                deform.optimizer.step()
                EFM.optimizer.step()
                FFM.optimizer.step()
                IFM.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
                deform.optimizer.zero_grad()
                deform.update_learning_rate(iteration)
                EFM.optimizer.zero_grad()
                FFM.optimizer.zero_grad()
                IFM.optimizer.zero_grad()
                EFM.update_learning_rate(iteration)
                FFM.update_learning_rate(iteration)
                IFM.update_learning_rate(iteration)

    print("Best PSNR = {} in Iteration {}".format(best_psnr, best_iteration))

    final_num_gaussians = scene.gaussians.get_xyz.shape[0]
    print(f"\n\033[1;35mFinal number of Gaussians: {final_num_gaussians}\033[0m")

    result_file_path = os.path.join(model_path, "result.txt")
    with open(result_file_path, "a") as result_file:
        result_file.write(f"Final number of Gaussians: {final_num_gaussians}\n")
        result_file.write(f"Best PSNR = {best_psnr} in Iteration {best_iteration}\n")


def prepare_output_and_logger(args):
    dataset_path = Path(args.source_path)  
    parts = dataset_path.parts
    try:
        dataset_name = parts[parts.index('TI-NSD') + 1]
    except ValueError:
        dataset_name = "default_dataset"
        
    if not args.model_path:
        current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        args.model_path = os.path.join("./output/", f"{dataset_name}_{current_time}/all")

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer, args.model_path


def training_report(source_path, model_path, tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene: Scene, renderFunc,
                    renderArgs, deform, load2gpu_on_the_fly, device, is_6dof=False, scale_factor=1, start_time=None):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    test_psnr = 0.0
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()},
                              {'name': 'train',
                               'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in
                                           range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                images = torch.tensor([], device=device)
                gts = torch.tensor([], device=device)
                for idx, viewpoint in enumerate(config['cameras']):
                    if load2gpu_on_the_fly:
                        viewpoint.load2device()
                    fid = viewpoint.fid

                    R = viewpoint.R
                    T = viewpoint.T
                    R_torch = torch.from_numpy(R).float().to(device)
                    T_torch = torch.from_numpy(T).float().to(device)

                    Rt = torch.eye(4, device=device)
                    Rt[:3, :3] = R_torch.t()  
                    Rt[:3, 3] = T_torch       

                    c2w = Rt
                    view_dir = -c2w[:3, 2]       

                    with torch.no_grad():
                        render_pkg = render(viewpoint, scene.gaussians, *renderArgs, 0, 0, 0, device, is_6dof)
                        visibility_filter = render_pkg["visibility_filter"]
                        visible_indices = visibility_filter.nonzero(as_tuple=True)[0]  
                        xyz_visible = scene.gaussians.get_xyz[visible_indices]
                    time_input = fid.unsqueeze(0).expand(xyz_visible.shape[0], -1)
                    view_dir = view_dir.unsqueeze(0).expand(xyz_visible.shape[0], -1)  
                    cam_pos  = viewpoint.camera_center.unsqueeze(0).expand(xyz_visible.shape[0], -1)   

                    d_xyz, d_rotation, d_scaling = deform.step(xyz_visible.detach(), time_input, cam_pos, view_dir)

                    d_xyz_full = torch.zeros_like(scene.gaussians.get_xyz)  
                    d_xyz_full[visible_indices] = d_xyz
                    d_rotation_full = torch.zeros_like(scene.gaussians.get_rotation)  
                    d_rotation_full[visible_indices] = d_rotation
                    d_scaling_full = torch.zeros_like(scene.gaussians.get_scaling)  
                    d_scaling_full[visible_indices] = d_scaling

                    image = torch.clamp(
                        renderFunc(viewpoint, scene.gaussians, *renderArgs, d_xyz_full, d_rotation_full, d_scaling_full, device, is_6dof)["render"],
                        0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to(device), 0.0, 1.0)
   
                    images = torch.cat((images, image.unsqueeze(0)), dim=0)
                    gts = torch.cat((gts, gt_image.unsqueeze(0)), dim=0)

                    if load2gpu_on_the_fly:
                        viewpoint.load2device('cpu')
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name),
                                             image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name),
                                                 gt_image[None], global_step=iteration)

                l1_test = l1_loss(images, gts)
                psnr_test = psnr(images, gts).mean()
                if config['name'] == 'test' or len(validation_configs[0]['cameras']) == 0:
                    test_psnr = psnr_test
                result_file_path = os.path.join(model_path, "result.txt")
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                with open(result_file_path, "a") as result_file:
                    result_file.write("[ITER {}] Evaluating {}: L1 {} PSNR {}\n".format(iteration, config['name'], l1_test, psnr_test))

                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

    if start_time and iteration == testing_iterations[-1]:
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"\nTraining completed in: {elapsed_time:.2f} seconds")
        result_file_path = os.path.join(model_path, "result.txt")
        with open(result_file_path, "a") as result_file:
            result_file.write(f"Training completed in: {elapsed_time:.2f} seconds\n")
            
    return test_psnr


if __name__ == "__main__":
    # Set up command line argument parser
    # Use CUDA_VISIBLE_DEVICES environment variable to control GPU selection
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser, device)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int,
                        default=[5000, 6000, 7_000, 8000, 9000] + list(range(10000, 40001, 1000)))
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 10_000, 20_000, 30_000, 40000])
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet, device)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, device)

    # All done
    print("\nTraining complete.")
