import os
import torch
import numpy as np
from configs.configs_loader import initial
from tools.evaluators import img2mse, mse2psnr
from data_loader.loader_blender import load_blender_data
from nerf.nerf_constructor import get_embedder, NeRF
from tools.data_processor import get_rays_batch_per_image, z_val_sample
from nerf.render import nerf_main, render_test

#np.random.seed(0)
#torch.cuda.manual_seed(3)

def train():
    model_fine.train()
    model_coarse.train()
    N_iters = args.N_iters+1
    args.perturb = 1. # stratified sampling. check here


    # main process
    for i in range(0 if int(checkpoint)==0 else int(checkpoint), N_iters):
        img_i = np.random.choice(i_train)
        gt_rgb = imgs[img_i].to(args.device)
        pose = poses[img_i, :3, :4].to(args.device)

        # get random sampled rays batch
        if args.pre_crop and i < args.pre_crop_iters:
            gt_rgb_batch, rays_batch = get_rays_batch_per_image(gt_rgb, p2c, pose, args.N_train,True,args.crop_factor)
        else:
            gt_rgb_batch, rays_batch = get_rays_batch_per_image(gt_rgb, p2c, pose, args.N_train)

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
        decay_steps = N_iters
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

        if i % args.i_test == 0 and i>0:
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
                render_test(position_embedder, view_embedder, model_coarse, model_fine, test_poses, hwK, p2c, args,
                            gt_imgs=test_imgs, savedir=testsavedir)
            print('Training model saved!')
            args.is_train = True
            model_coarse.train()
            model_fine.train()




if __name__ == '__main__':

    # load arguments
    args, logdir, checkpoint = initial()

    # load data
    imgs, poses, render_poses, hwK, i_split,_,_ = load_blender_data(args.datadir, args.resize_factor, args.testskip, args.white_bkgd)
    H,W,K = hwK
    print("h,w,k:{},{},{}".format(H,W,K))
    H,W = int(H), int(W)
    i_train, i_val, i_test = i_split
    print("train set: {} imgs; test set: {} imgs".format(i_train,i_test))
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

    # checkpoint resume
    if int(checkpoint) > 0:
        print('Resume from checkpoint:{}/{}.tar'.format(logdir,checkpoint))
        ckpt = torch.load(os.path.join(logdir,'{}.tar'.format(checkpoint)))
        model_coarse.load_state_dict(ckpt['network_coarse_state_dict'])
        model_fine.load_state_dict(ckpt['network_fine_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    else:
        print('Start training from iter=0')


    # move data to gpu
    imgs = torch.Tensor(imgs).cpu()
    poses = torch.Tensor(poses).cpu()
    K = torch.Tensor(K).to(args.device)
    p2c = torch.linalg.inv(K).to(args.device)

    train()
    