import os
import torch
import numpy as np
from configs_loader import initial
from evaluators import img2mse, mse2psnr
from data_loader.loader_lego import load_blender_data
from nerf_constructor import get_embedder, NeRF
from data_processor import get_rays_batch_per_image, z_val_sample
from render import nerf_main, render_test

np.random.seed(0)
torch.cuda.manual_seed(3)

def train():
    model_fine.train()
    model_coarse.train()
    N_iters = args.N_iters+1
    args.perturb = 1. # stratified sampling. check here

    # main process
    for i in range(0, N_iters):
        img_i = np.random.choice(i_train)
        gt_rgb = imgs[img_i].to(args.device)
        pose = poses[img_i, :3, :4].to(args.device)
        mg2c = torch.Tensor(mg2c_np).to(args.device)

        # get random sampled rays batch
        gt_rgb_batch, rays_batch = get_rays_batch_per_image(gt_rgb, focal, pose, args.N_train, mg2c)

        # network inference

        # coarse sampling
        # no prior information about in what depth the ray will terminate on the surface. sampling 
        # in the whole space from near to far.
        z_vals_coarse = z_val_sample(args.N_train, args.near, args.far, args.N_samples) # z value, ranges from near to far
        all_info = nerf_main(rays_batch, position_embedder, view_embedder, model_coarse, model_fine, z_vals_coarse, args)

        # coarse losses
        rgb_loss_coarse = img2mse(all_info['rgb_coarse'], gt_rgb_batch)

        # fine losses
        rgb_loss_fine = img2mse(all_info['rgb_fine'], gt_rgb_batch)
        psnr_fine = mse2psnr(rgb_loss_fine)

        # total loss
        total_loss = rgb_loss_fine + rgb_loss_coarse

        # optimizing
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # losses decay
        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** ((i) / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
        ###################################

        if i % args.i_print == 0:
            print(f"[TRAIN] Iter: {i} PSNR: {psnr_fine.item()} Total_Loss: {total_loss.item()}")

        if i % args.i_save == 0:
            path = os.path.join(args.basedir, args.expname, args.log_time, '{:06d}.tar'.format(i))
            save_model = {
                'iteration': i,
                'network_coarse_state_dict': model_coarse.state_dict(),
                'network_fine_state_dict': model_fine.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(save_model, path)

        if i % args.i_test == 0:
            model_coarse.eval()
            model_fine.eval()
            args.is_train = False
            selected_indices = np.random.choice(len(i_test), size=[10], replace=False)
            selected_i_test = i_test[selected_indices]
            testsavedir = os.path.join(args.basedir, args.expname, args.log_time, 'testset_{:06d}'.format(i))
            os.makedirs(testsavedir, exist_ok=True)
            with torch.no_grad():
                test_poses = torch.Tensor(poses[selected_i_test].to(args.device))
                test_imgs = imgs[selected_i_test]
                render_test(position_embedder, view_embedder, model_coarse, model_fine, test_poses, hwf, mg2c, args,
                            gt_imgs=test_imgs, savedir=testsavedir)
            print('Training model saved!')
            args.is_train = True
            model_coarse.train()
            model_fine.train()




if __name__ == '__main__':

    # load arguments
    args = initial()

    # load data
    imgs, poses, render_poses, hwf, i_split = load_blender_data(args.datadir, args.resize_factor, args.testskip, args.white_bkgd)
    H,W,focal = hwf
    H,W = int(H), int(W)
    i_train, i_val, i_test = i_split
    # the radius of the camera pose ==4.03
    # the obj is enclosed with a shpere with r=4
    args.near = 2.
    args.far = 6.

    # create nerf models
    position_embedder, input_ch_pos = get_embedder(args.multires, args.i_embed)
    view_embedder, input_ch_view = get_embedder(args.multires_views, args.i_embed)
    model_coarse = \
        NeRF(args.netdepth, args.netwidth, input_ch_pos, input_ch_view, [4]).to(args.device)

    model_fine = \
        NeRF(args.netdepth, args.netwidth, input_ch_pos, input_ch_view, [4]).to(args.device)

    # create optimizer
    grad_vars = list(model_coarse.parameters()) + list(model_fine.parameters())
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    # meshgrid to camera
    mg2c_np = np.array([[1,0,0],[0,-1,0],[0,0,-1]],dtype=float)

    # move data to gpu
    imgs = torch.Tensor(imgs).cpu()
    poses = torch.Tensor(poses).cpu()

    train()
    