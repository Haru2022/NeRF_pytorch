import torch
import numpy as np

trans_t = lambda t:torch.Tensor([
    [1,0,0,t[0]],
    [0,1,0,t[1]],
    [0,1,0,t[2]],
    [0,1,0,1]]).float()
rot_x = lambda roll: torch.Tensor([
    [1, 0, 0, 0],
    [0, np.cos(roll), -np.sin(roll), 0],
    [0, np.sin(roll), np.cos(roll), 0],
    [0, 0, 0, 1]]).float()
rot_y = lambda pitch: torch.Tensor([
    [np.cos(pitch), 0, np.sin(pitch), 0],
    [0, 1, 0, 0],
    [-np.sin(pitch), 0, np.cos(pitch), 0],
    [0, 0, 0, 1]]).float()
rot_z = lambda yaw: torch.Tensor([
    [np.cos(yaw), -np.sin(yaw), 0, 0],
    [np.sin(yaw), np.cos(yaw), 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]]).float()

# transformation from image coordinate to pixel coordinate
def coord_trans_img2pix(dx=1.,dy=1.,u0=None,v0=None, H=0, W=0 ,type='opencv', other_type=None):
    """transformation from image coordinate to pixel coordinate

        Args:
            dx: scaling ratio along the x/u direcition. 1 unit / 1 pixel
            dx: scaling ratio along the y/v direcition. 1 unit / 1 pixel
            u0: offset of the origin along u axis from img to pix plane.
            v0: offset of the origin along v axis from img to pix plane.
            *Note: (u0,v0) will be set defaultly to (W/2, H/2)
            type: define the coordinate rotation from img plane to pix plane, 'opencv' or 'opengl'
                  or 'other'. If it's set to 'opengl', the z and y axes will be in the opposite 
                  directions. If type='other', please specify the rotation matrix like 
                  [[-1,0,0],[0,1,0],[0,0,-1]] and pass it to this fun via arg 'other_type'.
    """

    # the transformation matrix is [1/dx, 0, W/2; 0, 1/dy, H/2; 0,0,1] if u0 and v0 are not defined.
    mtx = np.eye(3,dtype=float)
    mtx[0,0] = 1./dx
    mtx[1,1] = 1./dy
    if u0 is None and v0 is None:
        u0 = (W - 1) * .5
        v0 = (H - 1) * .5
    mtx[0,2] = u0
    mtx[1,2] = v0


    if type == 'opengl':
        mtx[1,:] = -mtx[1,:]
        mtx[2,:] = -mtx[2,:]
    elif type == 'other':
        other_type = np.array(other_type,dtype=float)
        mtx = other_type @ mtx
    elif type != 'opencv':
        print('Error: The image coordinate type "{}" is wrong. please specify one type from "opencv", "opengl" or your self-defined type "other" and pass it by the arg \'other_type\''.format(type))
        raise Exception('coord_trans_img2pix')
    
    return torch.Tensor(mtx)


# transformation matrix from camera coord to image coord
def coord_trans_cam2img(focal):
    """transformation from camera coordinate to image coordinate.\n
        z_cam * p_img_norm = trans_cam2img * p_cam
    """ 
    mtx = np.eye(3,dtype=float)
    mtx[0,0] = focal
    mtx[1,1] = focal
    return torch.Tensor(mtx)


# get the intrinsics
def gen_intrinsics(focal, dx=1.,dy=1.,u0=None,v0=None, H=0, W=0 ,type='opencv', other_type=None):
    """create cam intrinsics by K = img2pix @ cam2img = cam2pix
    
        Args:
            dx: scaling ratio along the x/u direcition. 1 unit / 1 pixel
            dx: scaling ratio along the y/v direcition. 1 unit / 1 pixel
            u0: offset of the origin along u axis from img to pix plane.
            v0: offset of the origin along v axis from img to pix plane.
            *Note: (u0,v0) will be set defaultly to (W/2, H/2)
            type: define the coordinate rotation from img plane to pix plane, 'opencv' or 'opengl'
                  or 'other'. If it's set to 'opengl', the z and y axes will be in the opposite 
                  directions. If type='other', please specify the rotation matrix like 
                  [[-1,0,0],[0,1,0],[0,0,-1]] and pass it to this fun via arg 'other_type'
            focal: the focal length of the camera
    """

    cam2img = coord_trans_cam2img(focal)
    img2pix = coord_trans_img2pix(dx, dy, u0, v0, H, W, type, other_type)

    intrinsics = img2pix @ cam2img

    return intrinsics

#print(gen_intrinsics(focal=200.,H=500,W=700,type='opengl'))

def extrinsic_decompose(T_w2c):
    """ input: 4x4 or 3x4 transformation matrix from world to camera coordinate
        output: R, c. 3x3 rotation matrix and 3x1 camera center w.r.f the world coordinate
    """
    # T_w2c = [R,-RC;0,1]
    R = T_w2c[:3,:3]
    t = T_w2c[:3,3]
    cam_pose = torch.transpose(R, 0, 1) @ (-t)
    cam_pose = cam_pose[...,None]

    return R, cam_pose

#print(extrinsic_decompose(torch.Tensor([[-1,0,0,2],[0,1,0,1],[0,0,-1,1],[0,0,0,1]])))

def world2pix_decompose(intrinsic, T_w2p):
    """ input: 4x4 or 3x4 transformation matrix from world to pixel coordinate
        output: R, c. 3x3 rotation matrix and 3x1 camera center w.r.f the world coordinate
    """
    if T_w2p.shape[0] == 4:
        buffer = torch.eye(4,dtype=float)
        buffer[:3,:3] = intrinsic
        intrinsic = buffer
    T_w2c = torch.linalg.inv(intrinsic).float() @ T_w2p
    R, cam_pose = extrinsic_decompose(T_w2c)

    return R, cam_pose

#print(world2pix_decompose(torch.eye(3),torch.Tensor([[-1,0,0,2],[0,1,0,1],[0,0,-1,1],[0,0,0,1]])))


def world2cam_lookat(cam_pose, up, target):
    """ generate the extrinsic by look-at camera
        http://ksimek.github.io/2012/08/22/extrinsic/

        Args:
            inputs:
                cam_pose: the camerea pos in the world coordinate
                up: the +y direction of the camera coord
                target: the target point that the camera is looking at
            outputs:
                R_3x3: camera totation matrix
                t_3x1: translation part from world to camera coord
                T_w2c: transformation matrix from world to camera coord = [R,t;0,1]
    """

    neg_z_axis = target - cam_pose
    neg_z_axis = neg_z_axis/torch.linalg.norm(neg_z_axis).float()
    x_axis = torch.cross(neg_z_axis, up)
    x_axis = x_axis/torch.linalg.norm(x_axis).float()
    y_axis = torch.cross(x_axis,neg_z_axis)
    z_axis = -neg_z_axis

    R = torch.stack((x_axis,y_axis,z_axis),0)
    t = -R @ cam_pose
    T_w2c = torch.eye(4,dtype=float)
    T_w2c[:3,:3] = R
    T_w2c[:3,3] = t

    return R, torch.reshape(t,(3,1)), T_w2c

#R,t,T = world2cam_lookat(torch.Tensor([1.,1.,1.]), torch.Tensor([0.,0.,-1.]), torch.Tensor([0.,0.,0.]))
#print(R, t, torch.linalg.det(R), T @ torch.Tensor([0,0,0,1]))