import os
import torch
import numpy as np
import imageio 
import json
import cv2
from tools.data_processor import central_resize_batch
from tools.coord_trans_np import gen_intrinsics, coord_trans_mtx_gen_w2p
from tools.render_pose_gen import render_pose_circle


trans_t = lambda t: torch.Tensor([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, t],
    [0, 0, 0, 1]]).float()

# rot along x axis in current coord
# https://en.wikipedia.org/wiki/Rotation_matrix
rot_x = lambda phi: torch.Tensor([
    [1, 0, 0, 0],
    [0, np.cos(phi), -np.sin(phi), 0],
    [0, np.sin(phi), np.cos(phi), 0],
    [0, 0, 0, 1]]).float()

# rot along -y axis in current coord.
# no influence on the final result. Only decide the 
# rotation is in clock/counter-clock.
rot_y = lambda th: torch.Tensor([
    [np.cos(th), 0, -np.sin(th), 0],
    [0, 1, 0, 0],
    [np.sin(th), 0, np.cos(th), 0],
    [0, 0, 0, 1]]).float()

# P_w =  T_c * R_neg_y * R_x * T_t * P_c = T_c2w * P_c
# P_c = T_t^-1 * R_x^-1 * R_neg_y^-1 * T_c^-1 * P_w
# https://en.wikipedia.org/wiki/Spherical_coordinate_system

def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_x(phi/180.*np.pi) @ c2w
    c2w = rot_y(theta/180.*np.pi) @ c2w
    c2w = torch.tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])).float() @ c2w
    #print(c2w[:3,3])
    #print(torch.norm(c2w[:3,3])) ==4 the radius of the new generated view is always equals to 4. 
    return c2w

def load_blender_data(basedir, resize_factor, testskip=1, white_bkgd=False):
    splits = ['train', 'val', 'test']
    metas = {}
    for s in splits:
        with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
            metas[s] = json.load(fp)

    all_imgs = []
    all_poses = []
    depth_imgs = []
    normal_imgs = []
    counts = [0]
    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []
        if s=='train' or testskip==0:
            skip = 1
        else:
            skip = testskip
            
        for frame in meta['frames'][::skip]:
            fname = os.path.join(basedir, frame['file_path'] + '.png')
            #print(fname)
            imgs.append(imageio.imread(fname))
            poses.append(np.array(frame['transform_matrix']))
            #print(np.array(frame['transform_matrix']))
            if s == 'test':
                depth_file = os.path.join(basedir, frame['file_path'] + '_depth_0001' + '.png')
                normal_file = os.path.join(basedir, frame['file_path'] + '_normal_0001' + '.png')
                #print(fname,depth_file,normal_file)
                depth_img = imageio.imread(depth_file)
                normal_img = imageio.imread(normal_file)
                depth_imgs.append(depth_img[...,0])
                normal_imgs.append(normal_img[...,:3])

        imgs = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
        poses = np.array(poses).astype(np.float32)
        if s == 'test':
            depth_imgs = (((255-np.array(depth_imgs))/255.)*7.2).astype(np.float32) # this depth is incorrect
            normal_imgs = (np.array(normal_imgs) / 255.).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_poses.append(poses)
    
    i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]
    
    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)
    #poses_norm = np.linalg.norm(poses[:,:3,3],axis=-1)
    #poses_avg = np.sum(poses,axis=0)/poses.shape[0]
    #print(poses_norm) ==4.03
    

    #print(poses_avg) # the center of the object is at z=2
    #[[-2.1076227e-02  1.1426907e-02 -3.6099937e-02 -1.4552325e-01]
    #[-3.6769193e-02 -1.1398306e-02  5.3406791e-03  2.1528799e-02]
    #[ 3.1722819e-11  7.9818541e-01  5.0783080e-01  2.0471294e+00]
    #[ 0.0000000e+00  0.0000000e+00  0.0000000e+00  1.0000000e+00]]
    
    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta['camera_angle_x']) # angle between ox_l and ox_r
    focal = .5 * W / np.tan(.5 * camera_angle_x)

    #render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]],0)

    render_poses = torch.tensor(render_pose_circle([0,0,0],4,np.pi/6,40,'opengl')).float()
    
    if resize_factor != 1.:
        #cv2.imshow('raw',imgs[0])
        imgs, H, W, focal = central_resize_batch(imgs, resize_factor,focal)

    if white_bkgd:
        print('white bkgd')
        imgs = imgs[..., :3]*imgs[..., -1:] + (1.-imgs[..., -1:])
    else:
        print('no white bkgd')
        imgs = imgs[..., :3]

    #K = gen_intrinsics(focal=focal,H=H,W=W,type='opengl')
    K = coord_trans_mtx_gen_w2p(H=H,W=W,focal=focal,cam_coord_type='opengl',output_type='c2p')
    K = K[:3,:3]
    
        
    return imgs, poses, render_poses, [H, W, K], i_split, depth_imgs, normal_imgs


#basedir = "H:\DM-NeRF-main\data/nerf_synthetic/lego"
#load_blender_data(basedir=basedir)