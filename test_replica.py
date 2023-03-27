import os
import torch
import numpy as np
from tools.evaluators import to8b
from tools.visualizer_ import pcd_gen
from configs.configs_loader import initial
from data_loader.loader_replica import load_replica_data
from nerf.nerf_constructor import get_embedder, NeRF
from nerf.render import render_test
from tools.data_processor import get_rays_np
from tools.render_pose_gen import render_pose_marching
from tools.coord_trans_np import rot_x, rotation_along_axis, trans_t
import imageio


def test():
    model_coarse.eval()
    model_fine.eval()
    args.is_train = False
    pcl_rgb_valids, pcl_rgb_gt, pcl_rgb_pred, rgbs = None, None, None,None
    visual_dir = os.path.join(args.basedir, args.expname, args.log_time,'visualization')
    os.makedirs(visual_dir,exist_ok=True)

    # get ground truth pcl
    #bds = np.array([[-5.4025796,1.24342659],[-3.0360394,2.68781675],[-1.40633904,1.34493595]], dtype=float)
    if reload_gt:
        for idx in range(0,len(train_ids)):
            print('load data from img_{}'.format(train_ids[idx]))
            img = imgs[idx] #(H,W,3)
            depth = depths[idx] #(H,W,1)
            pose = poses[idx]
            rays_o, rays_d = get_rays_np(H,W,K,pose) #(H,W,3)

            # reshape data
            rays_o = np.reshape(rays_o, [-1,3]) #(HxW,3)
            rays_d = np.reshape(rays_d, [-1,3]) #(HxW,3)
            depth = np.reshape(depth, [-1,1]) #(HxW,1)
            rgb = np.reshape(img,[-1,3]) #(HxW,3)
            #print( rays_o.shape,rays_d.shape,depth.shape,rgb.shape)

            # pcl_rgb concatenate
            pts = rays_o + rays_d * np.broadcast_to(depth,np.shape(rays_d)) #(HxW,3)
            pcl_rgb_gt_local = np.concatenate((pts, rgb), -1)
            valid = np.where(depth>0)
            pcl_rgb_gt_local = pcl_rgb_gt_local[pcl_rgb_gt_local[valid[...,0],...]]

            if pcl_rgb_gt is None:
                pcl_rgb_gt = pcl_rgb_gt_local
            else:
                pcl_rgb_gt = np.concatenate((pcl_rgb_gt,pcl_rgb_gt_local),0)

        bds = np.stack((np.min(pcl_rgb_gt[...,:3],axis=0), np.max(pcl_rgb_gt[...,:3],axis=0)), 0)
        bds = np.transpose(bds)
        [np.min(pcl_rgb_gt[...,:3],axis=0), np.max(pcl_rgb_gt[...,:3],axis=0)]
        np.save(os.path.join(visual_dir,'{}_gt.npy'.format(args.expname)),pcl_rgb_gt,'wb')


    # generate rendering poses based on the boundary of the scene from ground truth pcl
    print("boundary of the scene in x,y,z:{}".format(bds))
    mid_z = (bds[2,0]+bds[2,1])/2.
    mid_x = (bds[0,0]+bds[0,1])/2.
    mid_y = (bds[1,0]+bds[1,1])/2.
    bds = bds * .6
    #bds[0,:] = bds[0,[1,0]]
    seq = 6
    render_poses = None
    st_pt = [bds[0,0],bds[1,0],bds[2,1]]
    end_pt = [bds[0,1],bds[1,1],bds[2,1]]
    render_poses = render_pose_marching(st_pt,end_pt,seq,type='opencv')
    render_poses = np.concatenate((render_poses,render_pose_marching(end_pt,st_pt,seq,type='opencv')),0)
    st_pt = [bds[0,1],bds[1,0],bds[2,1]]
    end_pt = [bds[0,0],bds[1,1],bds[2,1]]
    render_poses = np.concatenate((render_poses,render_pose_marching(st_pt,end_pt,seq,type='opencv')),0)
    render_poses = np.concatenate((render_poses,render_pose_marching(end_pt,st_pt,seq,type='opencv')),0)
    render_poses[:,...] = render_poses[:,...] @ rot_x(-np.pi/2)
    st_pt = [bds[0,0],bds[1,0],bds[2,1]]
    end_pt = [bds[0,1],bds[1,1],bds[2,0]]
    render_poses = np.concatenate((render_poses,render_pose_marching(st_pt,end_pt,seq,type='opencv')),0)
    render_poses = np.concatenate((render_poses,render_pose_marching(end_pt,st_pt,seq,type='opencv')),0)
    st_pt = [bds[0,1],bds[1,0],bds[2,1]]
    end_pt = [bds[0,0],bds[1,1],bds[2,0]]
    render_poses = np.concatenate((render_poses,render_pose_marching(st_pt,end_pt,seq,type='opencv')),0)
    render_poses = np.concatenate((render_poses,render_pose_marching(end_pt,st_pt,seq,type='opencv')),0)
    st_pt = [bds[0,0],bds[1,1],bds[2,1]]
    end_pt = [bds[0,1],bds[1,0],bds[2,0]]
    render_poses = np.concatenate((render_poses,render_pose_marching(st_pt,end_pt,seq,type='opencv')),0)
    render_poses = np.concatenate((render_poses,render_pose_marching(end_pt,st_pt,seq,type='opencv')),0)
    st_pt = [bds[0,1],bds[1,1],bds[2,1]]
    end_pt = [bds[0,0],bds[1,0],bds[2,0]]
    render_poses = np.concatenate((render_poses,render_pose_marching(st_pt,end_pt,seq,type='opencv')),0)
    render_poses = np.concatenate((render_poses,render_pose_marching(end_pt,st_pt,seq,type='opencv')),0)
    #T_rot = rotation_along_axis([0,0,1],np.pi*(26/180.))
    #trans_ = trans_t([0.5,0,0])
    #render_poses[:,...] = trans_ @ render_poses[:,...]

    # main process
    render_poses_torch = torch.Tensor(render_poses).to(args.device)
    with torch.no_grad():
        print('Rendering......')
        #print(render_poses.shape)
        for idx, pose in enumerate(render_poses_torch):
            #print(pose)
            print('Process: {}/{}'.format(idx,render_poses_torch.shape[0]))
            pcl_rgb_valid, rgb = render_test(position_embedder, view_embedder, model_coarse, model_fine, pose[None,...], hwk, args, obj_recon=True)
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
                             to8b(rgbs), fps=10, quality=8)
        print('video saved!')

        np.save(os.path.join(visual_dir,'obj_resize_{}.npy'.format(args.resize_factor)),pcl_rgb_valids,'wb')



if __name__ == '__main__':
    args, logdir, checkpoint = initial()

    # load data
    total_num = 900
    step = 5
    train_ids = list(range(0, total_num, step))
    test_ids = []
    reload_gt = True
    imgs, poses, hwk, i_split = load_replica_data(args,train_ids,test_ids,'rgb')
    depths, _, _, _ = load_replica_data(args,train_ids,test_ids,'depth') # in meters
    print('Load data from', args.datadir)

    H, W, K = hwk
    print("h,w,k:{},{},{}".format(H,W,K))
    H,W = int(H), int(W)
    i_train, i_test = i_split


    # Create nerf model
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

        
    test()