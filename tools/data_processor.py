import os
import torch
import numpy as np
import imageio 
import json
import cv2
from tools.coord_trans_np import coord_trans_cam2img, coord_trans_img2pix



# transform matrix tool
trans_t = lambda t:np.array([
    [1,0,0,t[0]],
    [0,1,0,t[1]],
    [0,1,0,t[2]],
    [0,1,0,1]],dtype=float)
r_x = lambda roll: np.array([
    [1, 0, 0, 0],
    [0, np.cos(roll), -np.sin(roll), 0],
    [0, np.sin(roll), np.cos(roll), 0],
    [0, 0, 0, 1]])
r_y = lambda pitch: np.array([
    [np.cos(pitch), 0, np.sin(pitch), 0],
    [0, 1, 0, 0],
    [-np.sin(pitch), 0, np.cos(pitch), 0],
    [0, 0, 0, 1]])
r_z = lambda yaw: np.array([
    [np.cos(yaw), -np.sin(yaw), 0, 0],
    [np.sin(yaw), np.cos(yaw), 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]])




# Ray generation tool

### About the coordianate transformation ###
    # the dim of dirs[..., np.newaxis, :]: (H,W,1,3).
    # the dim of c2w[:3,:3]: (3,3)
    # When c2w left multiplies the expanded dirs, the dirs will be further
    # expanded to (H,W,3,3) by replicate the last dim along the 3rd axis. 
    # but this mult is actually dot product between this two mtxs. Note that
    # the dir is a 1x3 row vector instead of col vector that is usually used 
    # in the field of robotics.
    # this mult is like:
    # [dir]     [r11 r12 r13]     [dir]*[r11 r12 r13]  
    # [dir]  *  [r21 r22 r23] =   [dir]*[r21 r22 r23] 
    # [dir]     [r31 r32 r33]     [dir]*[r31 r32 r33]
    # If the result is summed along the col axis, this whole process equals to
    # R*[dir]' where [dir]' is the transpose vector of [dir] (col vector)
    # Then it's very clear that the corresponding line in get_ray function
    # tries to reproject the direction represented in the camera coordinate back 
    # to the world coordinate.
###

###
    # the transforms of the point and the coordinate are in the opposite direction.
    # e.g., move the point from 0 to 4 along +z dir means the coord itself moves -4 unit
    # along -z dir. similar to rotation opeartion.
    # a more general example is like: 1) when fixing the coordinate and transforming the point,
    # the transformation matrix from world to camera is T_w2c = R_x(alpha) * R_y(beta) \
    # * R_z(gamma) * trans(t); 2) when fixing the point and transforming the coordinate,
    # the transformation matrix is inversely R_x(-alpha) * R_y(-beta) * R_z(-gamma) * trans(-t).
    # these two transformations are the same.
    # this can be helpful for understanding the process of rendering_poses generation of lego data.
###

def get_rays(H, W, K, c2w=None):
    i, j = torch.meshgrid(torch.linspace(0, W - 1, W),
                          torch.linspace(0, H - 1, H))  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()

    # depth is always equals to 1, (H,W,3)
    dirs = torch.stack([(i - K[0, 2]) / K[0, 0], (j - K[1, 2]) / K[1, 1], K[2, 2] * torch.ones_like(i)], -1)

    #dirs = dirs[..., np.newaxis] # (H,W,3,1)
    #rays_d = torch.matmul(c2w[:3, :3],dirs).squeeze(-1) # (H,W,3,1) to (H,W,3)
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)

    # Rotate ray directions from camera frame to the world frame
    # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    #rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    return rays_o, rays_d

def get_rays_decomposed_np(H=0, W=0, dx=1.,dy=1.,u0=None,v0=None, type=None, focal=0., c2w=None):
    
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    
    
    T_p2i = np.linalg.inv(coord_trans_img2pix(dx,dy,u0,v0,H,W,type)) # pixel to image plane
    T_i2c = np.linalg.inv(coord_trans_cam2img(focal,focal)) # image plane to camera coordiates
    K = coord_trans_img2pix(dx,dy,u0,v0,H,W,type) @ coord_trans_cam2img(focal,focal)
    print(K)
    K = K[:3,:3]
    
    
    # Pixel plane < In Homogenuous Form>
    pt_coords_pix = np.stack([i,j,np.ones_like(i)],-1) # (H,W,3)
    
    # Image plane
    pt_coords_img = np.sum(pt_coords_pix[...,np.newaxis,:] * T_p2i[:3,:3],-1)
    #print(pt_coords_pix[0,0,:],pt_coords_img[0,0,:])
    
    # Camera coordinates
    rays_o_cam = np.stack([np.zeros_like(i),np.zeros_like(i),np.zeros_like(i)],-1) #(H,W,3)
    pt_coords_cam = np.sum(pt_coords_img[...,np.newaxis,:] * T_i2c[:3,:3],-1)
    # here comes the direction with depth==1, will be rescaled according to
    # the predicted depth image for 3d location prediction in the world coordinates.
    rays_d_cam = pt_coords_cam-rays_o_cam
    dirs = np.stack([(i - K[0, 2]) / K[0, 0], (j - K[1, 2]) / K[1, 1], K[2, 2] * np.ones_like(i)], -1)
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)
    
    print(dirs[0,0,:],rays_d_cam[0,0,:])
    print(np.mean(dirs-rays_d_cam))
    print(np.concatenate([dirs,rays_d_cam],axis=-1))
    
    # World coordinates 
    rays_o_cam_homo = np.concatenate([rays_o_cam,np.ones((H,W,1),dtype=float)],axis=-1) # (H,W,4)
    rays_o_w = np.sum(rays_o_cam_homo[..., np.newaxis, :] * c2w, -1)
    rays_o_w = np.matmul(np.broadcast_to(c2w,(H,W,4,4)),rays_o_cam_homo[..., np.newaxis]).squeeze(-1)
    scalar_o = rays_o_w[...,2]
    scalar_o = np.broadcast_to(scalar_o[...,np.newaxis],np.shape(rays_o_w))
    rays_o_w = rays_o_w/scalar_o
    
    #print(rays_d_cam-dirs)
    rays_d_w = np.sum(rays_d_cam[..., np.newaxis, :] * c2w[:3, :3], -1)
    print(rays_d[0,0,:],rays_d_w[0,0,:])
    #print(np.concatenate([rays_d,rays_d_w],axis=-1))

    
    # for visualization test
    return rays_o_w,rays_d_w
    


def get_rays_np(H, W, K, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    # the camera coords is right-up-backward. However, the generated meshgrid is right-down-forward.
    # Therefore the direction should be transformed by (1,-1,-1).
    dirs = np.stack([(i - K[0, 2]) / K[0, 0], (j - K[1, 2]) / K[1, 1], K[2, 2] * np.ones_like(i)], -1) # depth is always equals to 1, (H,W,3)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d)) # (H,W,3)
    return rays_o, rays_d

def get_rays_batch_per_image(rgb, intrinsic, c2w, N_train):
    H, W, C = rgb.shape
    rays_o, rays_d = get_rays(H, W, intrinsic, c2w) #(H,W,3)

    #coords = torch.stack(torch.meshgrid(torch.linspace(0, H - 1, H), torch.linspace(0, W - 1, W)),-1) #(W,H,2)
    #coords = torch.reshape(coords,[-1,2])
    #select_inds = np.random.choice(coords.shape[0], size=[N_train], replace=False)

    coords_h, coords_w = torch.meshgrid(torch.linspace(0, H - 1, H), torch.linspace(0, W - 1, W))
    coords_h, coords_w = torch.reshape(coords_h, [-1]).long(), torch.reshape(coords_w, [-1]).long()

    select_inds = np.random.choice(coords_h.shape[0], size=[N_train], replace=False)
    select_h, select_w = coords_h[select_inds], coords_w[select_inds]

    # sample the rays and corresponding rgbs
    rays_o, rays_d, batch_rgb = rays_o[select_h, select_w],  rays_d[select_h, select_w], rgb[select_h, select_w] # (N_train,3)
    batch_rays = torch.stack([rays_o, rays_d], 0)  # (2, N_train, 3)

    return batch_rgb, batch_rays



#TODO
def get_ray_batch_full_set(rgbs, focal, poses, N_train, random=True):
    1
    

def z_val_sample(N_rays, near, far, N_samples):
    near, far = near * torch.ones(size=(N_rays, 1)), far * torch.ones(size=(N_rays, 1))
    t_vals = torch.linspace(0., 1., steps=N_samples)
    z_vals_coarse = near + t_vals * (far - near) # depth along the z direction in the camera coord
    z_vals_coarse = z_vals_coarse.expand([N_rays, N_samples])
    return z_vals_coarse


# Hierarchical sampling https://en.wikipedia.org/wiki/Inverse_transform_sampling
# check here: det is always false?
def sample_pdf(bins, weights, N_samples, det=False):
    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True) # \in (0,N_coarse-2)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_fine, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]] #(batch, N_fine, N_coarse-2)
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    #print(matched_shape,inds_g.shape,cdf_g.shape) #[1024, 128, 62] torch.Size([1024, 128, 2]) torch.Size([1024, 128, 2])
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g) # map the bins with equal interval to new bins concentrating on the regions with principal probability distribution.

    denom = (cdf_g[..., 1] - cdf_g[..., 0]) # probability density in every interval, like diffrentiating the cdf to get pdf (batch, N_fine)
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples

# data resize
def central_resize_batch(imgs, factor, K):
    # (weight, hight)
    h, w = imgs[0].shape[:2]
    h_, w_, K_ = int(h*factor), int(w*factor), K*factor

    if np.size(np.array(K_))>1:
        K_[2,2] = 1
        
    print("Before resize:{}".format(imgs.shape))

    imgs_resize = []
    for img in imgs:
        imgs_resize.append(cv2.resize(img,(w_,h_),interpolation=cv2.INTER_NEAREST))   #size=[w,h]  

    imgs_resize = np.array(imgs_resize,dtype=float)

    #for i in range(100):
    #    print(i)
    #    cv2.imwrite(os.path.join('./data/nerf_synthetic/debug','{}_resize.png'.format(i)),
    #                imgs_resize[i][...,:3]*255)
    #    cv2.imwrite(os.path.join('./data/nerf_synthetic/debug','{}_raw.png'.format(i)),
    #                imgs[i][...,:3]*255)
        

    print("After resize:{}".format(imgs_resize.shape))
    
    return imgs_resize, h_, w_, K_

# meshgrid to camera coordinate transformation
def meshgrid2cam(trans=[1,1,1]):

    trans = np.array(trans, dtype=float)
    mg2c = np.zeros((3,3),dtype=float)
    mg2c[0,0] = trans[0]
    mg2c[1,1] = trans[1]
    mg2c[2,2] = trans[2]

    return mg2c

# create intrinsic matrix K_3x3 by an isotropic focal length
def focal2intrinsic(focal, H, W):

    K = np.zeros((3,3), dtype=float)
    K[0,0] = focal
    K[1,1] = focal
    K[2,2] = 1.
    K[0,2] = (W - 1) * .5
    K[1,2] = (H - 1) * .5

    return K

##TODO
# central crop imgs
def central_crop(img, factor):
    1
