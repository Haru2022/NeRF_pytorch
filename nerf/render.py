import torch
import time
import imageio
import numpy as np
import cv2
import os
import lpips
import torch.nn.functional as F
from tools.evaluators import to8b
from tools.data_processor import sample_pdf, z_val_sample, get_rays
from skimage import metrics

def render_train(raw, z_vals, rays_d, white_bkgd):

    # get alpha value of a single sampling point
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1. - torch.exp(-act_fn(raw) * dists)
    
    #distance between adjacent samples
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[..., :1].shape)], -1)  # The 'distance' from the last integration time is infinity.

    # the raw distance is along the z direction, not the real distance from the origin to the point
    # thus, based on the similar triangle, the dists should be multiplied with the norm of the corresponding length of the rays o->pt.
    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

    # extract the predicted rgb and density info from the raw data
    rgb = torch.sigmoid(raw[..., :3])  # [N_rays, N_samples, 3] #why not put sigmoid into the nerf?
    # why does the density not be normalized by sigmiod?
    density = raw[..., 3]

    # get alpha data based on the volume density along the rays, represent the 
    alpha = raw2alpha(density, dists)  # [N_rays, N_samples]

    # get the accumulated transmittance T(t_n) along the ray
    # that denotes the probability that the ray travels from t0 to t_{n-1} without hitting any other particle.
    T_accum = torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1. - alpha + 1e-10], -1), -1)[:, :-1]

    # get the weight to for the depth estimation and resampling
    # represent the contribution of each sampled point to the final render
    # ideal situation: no obstacle before and the point is 100% non-transparent
    weights = alpha*T_accum

    # the color of a ray is determined by 1) the probability of the ray travels
    # from t0 to t_{n-1} without hitting any other particle; and 2) the alpha value
    # that denotes the transparency of the sampled point; 3) the colors of each sampled point
    rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]

    # To composite onto a white background, use the accumulated alpha map.
    if white_bkgd:
        # Sum of weights along each ray. This value is in [0, 1] up to numerical error.
        acc_map = torch.sum(weights,-1)
        rgb_map = rgb_map + (1.-acc_map[..., None])

     # Estimated depth map is expected distance.
    depth_map = torch.sum(weights * z_vals, -1)

    return rgb_map, weights, depth_map
    


def nerf_main(rays, position_embedder, view_embedder, model_coarse, model_fine, z_vals_coarse, args):


    rays_o, rays_d = rays
    
    # unit dir vector
    viewdirs = rays_d/torch.norm(rays_d,dim=-1, keepdim=True)
    viewdirs = torch.reshape(viewdirs, [-1, 3]).float()


    # stratified samples in those intervals if perturb >0
    if args.perturb > 0.:
        interval = (args.far-args.near)/args.N_samples
        t_rand = torch.rand(z_vals_coarse.shape)
        lower = z_vals_coarse[...,:-1]
        z_vals_coarse = lower + interval*t_rand[...,:-1]


    # [N_rays, 1, 3] + [N_rays, 1, 3] * [N_rays, N_samples, 1], [1,3] * [N_samples, 1] will expand to [N_samples, 3] * [N_samples, 3] by replicating the elems
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals_coarse[..., :, None]  # [N_rays, N_samples, 3], sampling pts along the rays in the world coord
    
    pts_flat = torch.reshape(pts, [-1, pts.shape[-1]]) # [N_rays * N_samples, 3]

    # coarse prediction part
    embedded_pos = position_embedder.embed(pts_flat)
    input_dirs = viewdirs[:, None].expand(pts.shape)
    input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
    embedded_dirs = view_embedder.embed(input_dirs_flat)
    embedded = torch.cat([embedded_pos, embedded_dirs], -1)

    raw_coarse = model_coarse(embedded) #[N_rays * N_samples, 4]
    raw_coarse = torch.reshape(raw_coarse, list(pts.shape[:-1]) + [raw_coarse.shape[-1]]) # reshape [N_rays * N_samples, 4] to [N_rays, N_samples, rgb + density]
    
    #render via coarse network
    rgb_coarse, weights_coarse, depth_coarse = render_train(raw_coarse, z_vals_coarse, rays_d, args.white_bkgd)


    ## why to fine samping
    # free space and occluded regions that do not contribute to the rendered image are still sampled repeatedly in the coarse stage

    #fine point sampling based on the weights
    z_vals_mid = .5 * (z_vals_coarse[..., 1:] + z_vals_coarse[..., :-1])
    z_samples = sample_pdf(z_vals_mid, weights_coarse[..., 1:-1], args.N_importance, det=(args.perturb==0.))
    z_samples = z_samples.detach() # stop gradient propagation

    z_vals_fine, _ = torch.sort(torch.cat([z_vals_coarse, z_samples], -1), -1)
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals_fine[..., :, None]  # [N_rays, N_samples + N_importance, 3]

    pts_flat = torch.reshape(pts, [-1, pts.shape[-1]])

    # fine prediction part
    embedded = position_embedder.embed(pts_flat)
    input_dirs = viewdirs[:, None].expand(pts.shape)
    input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
    embedded_dirs = view_embedder.embed(input_dirs_flat)
    embedded = torch.cat([embedded, embedded_dirs], -1)

    raw_fine = model_fine(embedded)
    raw_fine = torch.reshape(raw_fine, list(pts.shape[:-1]) + [raw_fine.shape[-1]])

    # fine render part
    rgb_fine, weights_fine, depth_fine = render_train(raw_fine, z_vals_fine, rays_d, args.white_bkgd)

    all_info = {'rgb_fine': rgb_fine, 'z_vals_fine': z_vals_fine,
                'rgb_coarse': rgb_coarse, 'z_vals_coarse': z_vals_coarse,
                'raw_fine': raw_fine, 'raw_coarse': raw_coarse,
                'depth_fine': depth_fine, 'depth_coarse': depth_coarse}

    return all_info



def render_test(position_embedder, view_embedder, model_coarse, model_fine, render_poses, hwk, args,
                gt_imgs=None, savedir=None, obj_recon = False):
    
    H, W, K = hwk
    psnrs, ssims, lpipses = [], [], []
    if gt_imgs is not None:
        gt_imgs_cpu = gt_imgs.cpu().numpy()
        gt_imgs_gpu = gt_imgs.to(args.device)
        lpips_vgg = lpips.LPIPS(net="vgg").to(args.device)

    pcl_rgb_valid_total = None
    rgbs = None

    for i, c2w in enumerate(render_poses):
        print('=' * 50, i, '=' * 50)
        t = time.time()
        z_val_coarse = z_val_sample(args.N_test, args.near, args.far, args.N_samples)
        rays_o, rays_d = get_rays(H,W,K,torch.Tensor(c2w))
        rays_o = torch.reshape(rays_o, [-1, 3]).float()
        rays_d = torch.reshape(rays_d, [-1, 3]).float()
        full_rgb, full_pcl= None, None
        for step in range(0, H * W, args.N_test):
            N_test = args.N_test
            if step + N_test > H * W:
                N_test = H * W - step
                z_val_coarse = z_val_sample(N_test, args.near, args.far, args.N_samples)
            rays_io = rays_o[step:step + N_test]  # (chuck, 3)
            rays_id = rays_d[step:step + N_test]  # (chuck, 3)
            batch_rays = torch.stack([rays_io, rays_id], dim=0)
            all_info = nerf_main(batch_rays, position_embedder, view_embedder,
                            model_coarse, model_fine, z_val_coarse, args)
            if full_rgb is None:
                full_rgb = all_info['rgb_fine']
            else:
                full_rgb = torch.cat((full_rgb, all_info['rgb_fine']), dim=0)
            
            if obj_recon:
                depth = all_info['depth_fine'][...,None]
                zeros = torch.zeros_like(depth)
                pts = rays_io + rays_id * depth.expand(rays_id.shape)
                pts[...,2] = torch.where(depth[...,0]>args.near,pts[...,2],zeros[...,0])
                pts[...,2] = torch.where(depth[...,0]<args.far,pts[...,2],zeros[...,0])
                if full_pcl is None:
                    full_pcl = pts
                else:
                    full_pcl = torch.cat((full_pcl,pts),dim=0)

        rgb = full_rgb.reshape([H, W, full_rgb.shape[-1]])

        if rgbs is None:
            rgbs = rgb[None,...] 
        else:
            rgbs = torch.cat([rgbs,rgb],0) #[imgs_num,H,W,3]

        if obj_recon:
            pcl_rgb = torch.cat([full_pcl,full_rgb],dim=1)
            pcl_rgb_valid = pcl_rgb[pcl_rgb[...,2]>0.] # remove invalid points as background
            if pcl_rgb_valid_total is None:
                pcl_rgb_valid_total =  pcl_rgb_valid
            else:
                pcl_rgb_valid_total = torch.cat([pcl_rgb_valid_total,pcl_rgb_valid],0)
        

        if gt_imgs is not None:
            # rgb image evaluation part
            psnr = metrics.peak_signal_noise_ratio(rgb.cpu().numpy(), gt_imgs_cpu[i], data_range=1)
            ssim = metrics.structural_similarity(rgb.cpu().numpy(), gt_imgs_cpu[i], multichannel=True, data_range=1)
            lpips_i = lpips_vgg(rgb.permute(2, 0, 1).unsqueeze(0), gt_imgs_gpu[i].permute(2, 0, 1).unsqueeze(0))
            psnrs.append(psnr)
            ssims.append(ssim)
            lpipses.append(lpips_i.item())
            print(f"PSNR: {psnr} SSIM: {ssim} LPIPS: {lpips_i.item()}")

        if savedir is not None:
            rgb8 = to8b(rgb.cpu().numpy())
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)
            

    if gt_imgs is not None:

        output = np.stack([psnrs, ssims, lpipses])
        output = output.transpose([1, 0])
        mean_output = np.array([np.mean(psnrs), np.mean(ssims), np.mean(lpipses)])
        mean_output = mean_output.reshape([1, 3])
        output = np.concatenate([output, mean_output], 0)
        test_result_file = os.path.join(savedir, 'test_results.txt')
        np.savetxt(fname=test_result_file, X=output, fmt='%.6f', delimiter=' ')
        print('=' * 49, 'Avg', '=' * 49)
        print('PSNR: {:.4f}, SSIM: {:.4f},  LPIPS: {:.4f} '.format(np.mean(psnrs), np.mean(ssims), np.mean(lpipses)))

    if obj_recon:
        return pcl_rgb_valid_total.cpu().numpy(), rgbs.cpu().numpy()
    else:
        return rgbs.cpu().numpy()
    
