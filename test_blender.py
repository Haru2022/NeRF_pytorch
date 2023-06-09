import os
import torch
import numpy as np
from tools.evaluators import to8b
from tools.visualizer_ import pcd_gen
from configs.configs_loader import initial
from data_loader.loader_blender import load_blender_data
from nerf.nerf_constructor import get_embedder, NeRF
from nerf.render import render_test
from tools.data_processor import get_rays_np, get_rays_decomposed_np
import imageio


def test():
    model_coarse.eval()
    model_fine.eval()
    args.is_train = False
    pcl_rgb_normal_gt = None
    pcl_rgb_valids, rgbs = None, None
    visual_dir = os.path.join(args.basedir, args.expname, args.log_time,'visualization')
    os.makedirs(visual_dir,exist_ok=True)
    
    # get ground truth pcl
    if reload_gt:
        count = len(i_train) + len(i_val)
        #print(poses.shape,imgs.shape, count)
        for idx in range(0,len(i_test)):
            print('Ground Truth Collecting: Loading img {}'.format(i_test[idx]))
            depth = depth_imgs[idx]
            normal = normal_imgs[idx]
            pose_test = poses[count+idx]
            rays_o, rays_d = get_rays_np(H,W,p2c,pose_test) #(H,W,3)
            
            # reshape data
            rays_o = np.reshape(rays_o, [-1,3]) #(HxW,3)
            rays_d = np.reshape(rays_d, [-1,3]) #(HxW,3)
            depth = np.reshape(depth, [-1,1]) #(HxW,1)
            normal = np.reshape(normal,[-1,3]) #(HxW,3)
            rgb = np.broadcast_to(np.array([0.5,0.5,0.5], dtype=float), np.shape(rays_d))
            
            # pcl_rgb concatenate
            pts = rays_o + rays_d * np.broadcast_to(depth,np.shape(rays_d)) #(HxW,3)
            pcl_rgb_normal_gt_local = np.concatenate((pts, rgb, normal), -1)
            valid = np.where(depth>0)
            pcl_rgb_normal_gt_local = pcl_rgb_normal_gt_local[pcl_rgb_normal_gt_local[valid[...,0],...]]

            if pcl_rgb_normal_gt is None:
                pcl_rgb_normal_gt = pcl_rgb_normal_gt_local
            else:
                pcl_rgb_normal_gt = np.concatenate((pcl_rgb_normal_gt,pcl_rgb_normal_gt_local),0)
    
    np.save(os.path.join(visual_dir,'{}_gt.npy'.format(args.expname)),pcl_rgb_normal_gt,'wb')
    print('Save the ground truth pointcloud with normal info.')

    # main process

    # move data to gpu
    #imgs = torch.Tensor(imgs).cpu()
    #poses = torch.Tensor(poses).cpu()
    #render_poses = torch.Tensor(render_poses).to(args.device)

    with torch.no_grad():
        print('Rendering......')
        #print(render_poses.shape)
        for idx, pose in enumerate(render_poses):
            #print(pose)
            print('Process: {}/{}'.format(idx,render_poses.shape[0]))
            pcl_rgb_valid, rgb = render_test(position_embedder, view_embedder, model_coarse, model_fine, pose[None,...], hwK, p2c, args, obj_recon=True)
            if pcl_rgb_valids is None:
                pcl_rgb_valids = pcl_rgb_valid
                rgbs = rgb
            else:
                pcl_rgb_valids = np.concatenate([pcl_rgb_valids,pcl_rgb_valid],0)
                rgbs = np.concatenate([rgbs,rgb],0)
            print('Add {} valid 3d points, total:{}'.format(pcl_rgb_valid.shape[0],pcl_rgb_valids.shape[0]))
            img_name = visual_dir+'/test_{}.jpg'.format(idx)
            imageio.imwrite(img_name,to8b(np.squeeze(rgb,0)))

        
        print('Gererating video...')
        
        imageio.mimwrite(visual_dir + '/rgb.mp4',
                             to8b(rgbs), fps=30, quality=8)
        print('video saved!')

        # visualizer test
        #pcd_gen(pcl_rgb_valids[:,:3],pcl_rgb_valids[:,3:])

        np.save(os.path.join(visual_dir,'obj_resize_{}.npy'.format(args.resize_factor)),pcl_rgb_valids,'wb')






if __name__ == '__main__':

    # load arguments
    args, logdir, checkpoint = initial()

    # load data
    imgs, poses, render_poses, hwK, i_split, depth_imgs, normal_imgs = load_blender_data(args.datadir, args.resize_factor, args.testskip, args.white_bkgd)
    H,W,K = hwK
    p2c = np.linalg.inv(K)
    print("h,w,k:{},{},{}".format(H,W,K))
    H,W = int(H), int(W)
    i_train, i_val, i_test = i_split
    #print(i_val, i_test)
    reload_gt = True
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


    # load model
    ckpt = torch.load(os.path.join(logdir,'{}.tar'.format(checkpoint)))
    model_coarse.load_state_dict(ckpt['network_coarse_state_dict'])
    model_fine.load_state_dict(ckpt['network_fine_state_dict'])

    # move data to gpu
    #imgs = torch.Tensor(imgs).cpu()
    #poses = torch.Tensor(poses).cpu()
    #render_poses = torch.Tensor(render_poses).to(args.device)
    #K = torch.Tensor(K).to(args.device)

    test()
    