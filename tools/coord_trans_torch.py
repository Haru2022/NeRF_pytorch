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
                or 'other' see https://stackoverflow.com/questions/44375149/opencv-to-opengl-coordinate-system-transform 
                for details. If it's set to 'opengl', the z and y axes will be in the opposite 
                directions. If type='other', please specify the rotation matrix like 
                [1,-1,1] and pass it to this fun via arg 'other_type'. No further operation if 
                type='opencv' cause the image coord is the same as pixel plane.
    """

    # the transformation matrix is [1/dx, 0, W/2; 0, 1/dy, H/2; 0,0,1] if u0 and v0 are not defined.
    mtx = torch.eye(4,dtype=float)
    mtx[0,0] = 1./dx
    mtx[1,1] = 1./dy
    if u0 is None and v0 is None:
        u0 = (W - 1) * .5
        v0 = (H - 1) * .5
    mtx[0,2] = u0
    mtx[1,2] = v0
    
    # coordinate totation
    rot_coords = torch.eye(4,dtype=float)
    if type == 'opengl':
        rot_coords[1,1] = -rot_coords[1,1]
        rot_coords[2,2] = -rot_coords[2,2]
    elif type == 'other':
        other_type = torch.Tensor(other_type,dtype=float)
        for i in range(0,3):
            rot_coords[i,i] = rot_coords[i,i]*other_type[i]
    elif type != 'opencv':
        print('Error: The image coordinate type "{}" is wrong. please specify one type from "opencv", "opengl" or your self-defined type "other" and pass it by the arg \'other_type\''.format(type))
        raise Exception('coord_trans_img2pix')
    
    mtx = mtx @ rot_coords
    
    return mtx


# transformation matrix from camera coord to image coord
def coord_trans_cam2img(focal_x, focal_y):
    """transformation from camera coordinate to image coordinate.\n
        z_cam * p_img_norm = trans_cam2img * p_cam
    """ 
    mtx = torch.eye(4,dtype=float)
    mtx[0,0] = focal_x
    mtx[1,1] = focal_y
    
    return mtx

# transformation from world coordinates to the camera coordinates
def coord_trans_world2cam(R_cam,C_cam,type=None):
    """create world-to-camera or camera-to-world transformation matrix by
       the rotation and center of the camera in world coordinates.

       Args:
            R_cam: rotation matrix describing the cam's orientation w.r.t the world coords
            C_cam: the camera cneter in the world coords
            type: (w2c or c2w) return the transformation from world to camera coords or inversely.

    """
    T = torch.eye(4,dtype=float)
    R_cam = torch.Tensor(R_cam,dtype=float)
    C_cam = torch.Tensor(C_cam,dtype=float)

    if type == 'w2c':# world to camera
        T[:3,:3] = R_cam.transpose()
        T[:3,3] = -R_cam.transpose() @ C_cam
    elif type == 'c2w':# camera to world
        T[:3,:3] = R_cam
        T[:3,3] = C_cam
    else:
        print("Error: please specify the T type \'w2c\' or \'c2w\'")
        raise Exception()

    return T


# get the intrinsics
def gen_intrinsics(focal, dx=1.,dy=1.,u0=None,v0=None, H=0, W=0 ,type=None, other_type=None):
    """create cam intrinsics by K = img2pix @ cam2img = cam2pix
    
        Args:
            dx: scaling ratio along the x/u direcition. 1 unit / 1 pixel
            dx: scaling ratio along the y/v direcition. 1 unit / 1 pixel
            u0: offset of the origin along u axis from img to pix plane.
            v0: offset of the origin along v axis from img to pix plane.
            *Note: (u0,v0) will be set defaultly to (W/2, H/2)
            type: define the coordinate rotation from img plane to pix plane, 'opencv' or 'opengl'
                or 'other' see https://stackoverflow.com/questions/44375149/opencv-to-opengl-coordinate-system-transform 
                for details. If it's set to 'opengl', the z and y axes will be in the opposite 
                directions. If type='other', please specify the rotation matrix like 
                [[-1,0,0],[0,1,0],[0,0,-1]] and pass it to this fun via arg 'other_type'. No further operation if 
                type='opencv' cause the image coord is the same as pixel plane.
            focal: the focal length of the camera
    """

    cam2img = coord_trans_cam2img(focal,focal)
    img2pix = coord_trans_img2pix(dx, dy, u0, v0, H, W, type, other_type)

    intrinsics = img2pix @ cam2img

    return intrinsics

#print(gen_intrinsics(focal=200.,H=500,W=700,type='opengl'))

def extrinsic_decompose(T_w2c):
    """ input: 4x4 or 3x4 transformation matrix from world to camera coordinate
        output: R, c. 3x3 rotation matrix and 3x1 camera center w.r.f the world coordinate
    """
    # T_w2c = [R,-RC;0,1]
    T_w2c = torch.Tensor(T_w2c)
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
    T_w2p = torch.array(T_w2p,dtype=float)
    T_w2c = torch.linalg.inv(intrinsic) @ T_w2p
    R, cam_pose = extrinsic_decompose(T_w2c)

    return R, cam_pose

#print(world2pix_decompose(torch.eye(3),torch.Tensor([[-1,0,0,2],[0,1,0,1],[0,0,-1,1],[0,0,0,1]])))


def T_lookat(cam_pose, up, target,type=None, cam_coord=None):
    """ generate the extrinsic by look-at camera 
        http://ksimek.github.io/2012/08/22/extrinsic/
        the default camera coords is opengl type, which means the camera coordinate
        is left-up-backward; for opencv type: left-down-forward

        Args:
            inputs:
                cam_pose: the camerea pos in the world coordinate
                up: the +y direction of the camera coord in the world coordiante
                target: the target point that the camera is looking at in the world coordiante
                cam_coord: the type of the camera coordinate, 'opencv' or 'opengl'
                type: the output type, 'w2c' means the transformation from world to camera, 'c2w' means inverse transformation
            outputs:
                T: the transformation from world to camera or inverse. (defined by the arg 'type')
    """

    if type is None:
        print("Error: please specify the transformation type \'w2c\' or \'c2w\' into the arg 'type'")
        raise Exception()
    if cam_coord is None:
        print('please specify the type of the camera coordinate, \'opencv\' or \'opengl\'')
        raise Exception()
    
    neg_z_axis = target - cam_pose
    neg_z_axis = neg_z_axis/torch.linalg.norm(neg_z_axis)
    x_axis = torch.cross(neg_z_axis, up)
    x_axis = x_axis/torch.linalg.norm(x_axis)
    y_axis = torch.cross(x_axis,neg_z_axis)
    z_axis = -neg_z_axis

    R_cam = torch.stack((x_axis,y_axis,z_axis),0).transpose()
    T = coord_trans_world2cam(R_cam,cam_pose,type)
     
    if cam_coord=='opencv':
        if type == 'c2w':
            T = T @ torch.Tensor([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
        elif type == 'w2c':
            T = torch.Tensor([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]]) @ T
    elif cam_coord=='opengl':
        pass
    else:
        print('Incorrect camera coordinate form. Please specify the type as \'opencv\' or \'opengl\'')
        raise Exception()

    #t = -R @ cam_pose
    #T_w2c = np.eye(4,dtype=float)
    #T_w2c[:3,:3] = R
    #T_w2c[:3,3] = t

    return T

#print(world2cam_lookat(np.array([4.,3.,0]), np.array([0.,0.,1.]), np.array([0.,0.,0.])))
#print(R, t, np.linalg.det(R), T @ np.array([0,0,0,1]))



def rotation_mtx_compose(rad_x,rad_y,rad_z, order=[0,1,2]):
    """create rotation matrix by 3 rotation rads along three axes

        Args:
            rad_x: rotation angle (in rads) along current x-axis
            rad_y: rotation angle (in rads) along current y-axis
            rad_z: rotation angle (in rads) along current z-axis
            order: the rotation order. e,g, [2,0,1] means the order is y-z-x
    """
    r_x = rot_x(rad_x)
    r_y = rot_y(rad_y)
    r_z = rot_z(rad_z)

    r_stack = torch.stack([r_x,r_y,r_z],0)
    order = torch.array(order)
    buffer = torch.zeros_like(r_stack)
    for i in range(0,3):
        buffer[i,:] = r_stack[order[i],:]

    rotation = buffer[2,:] @ buffer[1,:] @ buffer[0,:]

    return rotation[:3,:3]


#print(rotation_mtx_compose(np.pi/3,np.pi/4,np.pi/6,[2,1,0]))


def T_convert(T,type=None):
    
    R = T[:3,:3]
    t = T[:3,3]

    if type =='wc2cw':
        R_cam = R.transpose()
        C_cam = - R.transpose() @ t
        T[:3,:3] = R_cam
        T[:3,3] = C_cam
    elif type == 'cw2wc':
        R_cam = R
        C_cam = t
        T[:3,:3] = R_cam.transpose()
        T[:3,3] = - R_cam.transpose() @ C_cam
    else:
        print("Error: please specify the convert type \'wc2cw\' or \'cw2wc\'")
        raise Exception()

    return T


def rotation_along_axis(axis, rad):

    axis = torch.Tensor(axis,dtype=float)
    norm = axis/torch.linalg.norm(axis)
    n_x = norm[0]
    n_y = norm[1]
    n_z = norm[2]

    c = torch.cos(rad)
    s = torch.sin(rad)

    col_1 = torch.Tensor([n_x*n_x*(1-c)+c, n_x*n_y*(1-c)+n_z*s, n_x*n_z*(1-c)-n_y*s], dtype=float)
    col_2 = torch.Tensor([n_x*n_y*(1-c)-n_z*s, n_y*n_y*(1-c)+c, n_y*n_z*(1-c)+n_x*s], dtype=float)
    col_3 = torch.Tensor([n_x*n_z*(1-c)+n_y*s, n_y*n_z*(1-c)-n_z*s, n_z*n_z*(1-c)+c], dtype=float)

    R = torch.stack([col_1,col_2,col_3],axis=-1)

    T = torch.eye(4,dtype=float)
    T[:3,:3] = R

    #print(np.stack([[1,0,0],[1,2,0],[1,0,3]],axis=-1))

    return T

#print(Rotation_along_axis([1,1,1],np.pi))

def coord_trans_mtx_gen_p2w(u0=None, v0=None, dx=1., dy=1., H=None, W=None, cam_coord_type=None, focal=0., R_cam=None, C_cam=None, pose_c2w=None, output_type=None):
    """_summary_

    Args:
        u0: the pixel offset along x axis, default = W/2
        v0: the pixel offset along x axis, default = H/2
        dx: the scalar to convert unit in the image plane to pixel unit along x axis, default = 1.
        dy the scalar to convert unit in the image plane to pixel unit along y axis, default = 1.
        cam_coord_type: choose from 'opencv' or 'opengl'
        output_type: the outout transformation mtx type. choose from ['p2i','p2c','p2w','i2c','i2w','c2w']

    Returns:
        T: the transformation matrix from the original coords to the target coords.
    """
    
    output_type_list = ['p2i','p2c','p2w','i2c','i2w','c2w']
    # param checking
    if cam_coord_type is None:
        print("Please specify the camera coordiante type: 'opengl' or 'opencv'.")
        raise Exception()
    elif cam_coord_type != 'opencv' and cam_coord_type != 'opengl':
        print("Error type of the camera coordinate: {}".format(cam_coord_type))
        raise Exception()
    else:
        pass
    
    if output_type_list.count(output_type):
        print("output type = {}".format(output_type))
    else:
        print("Error output type:{}, please choose among the following output types:{}".format(output_type,output_type_list))
        raise Exception()
    
    
    #############Pixel plane to Image plane#############
    if output_type in ['p2i','p2c','p2w']:
        if u0 is None and v0 is None:
            u0 = (W - 1) * .5
            v0 = (H - 1) * .5
        T_p2i = torch.eye(4,dtype=float)
        T_p2i[0,0] = dx
        T_p2i[0,2] = -dx * u0
        T_p2i[1,1] = dy
        T_p2i[1,2] = -dy * v0
        if cam_coord_type == 'opengl':
            T_p2i = torch.Tensor([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]]) @ T_p2i
            
        if output_type == 'p2i':
            return T_p2i
    
    
    #############Image plane to Camera coordiante#############
    if output_type in ['i2c','i2w','p2c','p2w']:
        # the default depth of the point in the camera coordinate is 1. Additional depth information is need acting as scalar to recover the 3D info.
        T_i2c = torch.Tensor([[1./focal, 0, 0,0],[0, 1./focal, 0,0], [0,0,1,0],[0,0,0,1]])
        
        if output_type == 'p2c':
            return T_i2c @ T_p2i
        elif output_type == 'i2c':
            return T_i2c
    
    
    #############Camera coordinate to World coordinate#############
    if output_type in ['c2w','i2w','p2w']:
        T_c2w = torch.eye(4,dtype=float)
        if pose_c2w is not None:
            T_c2w = torch.Tensor(pose_c2w)
        else:
            T_c2w[:3,:3] = torch.Tensor(R_cam)
            T_c2w[:3,3] = torch.Tensor(C_cam)
        
        if output_type == 'p2w':
            return T_c2w @ T_i2c @ T_p2i
        elif output_type == 'i2w':
            return T_c2w @ T_i2c
        elif output_type == 'c2w':
            return T_c2w
    
    


def coord_trans_mtx_gen_w2p(u0=None, v0=None, dx=1., dy=1., H=None, W=None, cam_coord_type=None, focal=0., R_cam=None, C_cam=None, pose_w2c=None, output_type=None):
    """_summary_

    Args:
        u0: the pixel offset along x axis, default = W/2
        v0: the pixel offset along x axis, default = H/2
        dx: the scalar to convert unit in the image plane to pixel unit along x axis, default = 1.
        dy the scalar to convert unit in the image plane to pixel unit along y axis, default = 1.
        cam_coord_type: choose from 'opencv' or 'opengl'
        output_type: the outout transformation mtx type. choose from ['w2c','w2i','w2p','c2i','c2p','i2p']

    Returns:
        T: the transformation matrix from the original coords to the target coords.
    """
    
    output_type_list = ['w2c','w2i','w2p','c2i','c2p','i2p']
    # param checking
    if cam_coord_type is None:
        print("Please specify the camera coordiante type: 'opengl' or 'opencv'.")
        raise Exception()
    elif cam_coord_type != 'opencv' and cam_coord_type != 'opengl':
        print("Error type of the camera coordinate: {}".format(cam_coord_type))
        raise Exception()
    else:
        pass
    
    if output_type_list.count(output_type):
        print("output type = {}".format(output_type))
    else:
        print("Error output type:{}, please choose among the following output types:{}".format(output_type,output_type_list))
        raise Exception()
    
    
    #############World coordinate to Camera coordinate#############
    if output_type in ['w2c','w2i','w2p']:
        T_w2c = torch.eye(4,dtype=float)
        if pose_w2c is not None:
            T_w2c = torch.Tensor(pose_w2c)
        elif output_type == 'w2c':
            R_cam = torch.Tensor(R_cam)
            C_cam = torch.Tensor(C_cam)
            T_w2c[:3,:3] = R_cam.transpose()
            T_w2c[:3,3] = - R_cam.transpose() @ C_cam
        
        if output_type == 'w2c':
            return T_w2c
    
    #############Camera coordinate to Image plane#############
    if output_type in ['c2i', 'c2p','w2i','w2p']:
        T_c2i = torch.eye(4,dtype=float)
        T_c2i[0,0] = focal
        T_c2i[1,1] = focal
        
        if output_type == 'w2i':
            return T_c2i @ T_w2c
        elif output_type == 'c2i':
            return T_c2i
    
    #############Image plane to Pixel plane#############
    if output_type in ['w2p','c2p','i2p']:
        if u0 is None and v0 is None:
            u0 = (W - 1) * .5
            v0 = (H - 1) * .5
        T_i2p = torch.eye(4,dtype=float)
        T_i2p[0,0] = 1./dx
        T_i2p[1,1] = 1./dy
        T_i2p[0,2] = u0
        T_i2p[1,2] = v0
        if cam_coord_type == 'opengl':
            T_i2p = T_i2p @ torch.Tensor([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
            
        if output_type == 'w2p':
            return T_i2p @ T_c2i @ T_w2c
        elif output_type == 'c2p': # the intrinsic.
            return T_i2p @ T_c2i
        elif output_type == 'i2p':
            return T_i2p
    
def coords_trans_toolbox(trans_mtx, trans_type, pts, depths, z_axis_type=None):
    """_summary_

    Args:
        trans_mtx: _description_
        trans_type: the start and end coord sys, choose from '['w2c','w2i','w2p','c2i','c2p','i2p','p2i','p2c','p2w','i2c','i2w','c2w']'
        pts: the input points, (N,3)
        depths: the depth of the input points in the camera coordinates. (N,1)
        z_axis_type: +1/-1 for forward/backward z axis type of the camera coordinates


    Returns:
       pts_out: the corresponding point coordinates defined in the target coordinate system
    """
    
    type_list = ['w2c','w2i','w2p','c2i','c2p','i2p','p2i','p2c','p2w','i2c','i2w','c2w']
    if type_list.count(trans_type):
        print("output type = {}".format(trans_type))
    else:
        print("Error output type:{}, please choose among the following output types:{}".format(trans_type,type_list))
        raise Exception()
    
    # expand to homogeneous form 4x4 or 4x1
    if pts.shape[1]<4:
        pts = torch.concatenate((pts,torch.ones(pts.shape[0],1)),0)
    if trans_mtx.shape[1]<4:
        T = torch.eye(4)
        T[:3,:3] = trans_mtx
        trans_mtx = T
    # if the z axis of the camera coordinate system is backwawrd, the third elem of the image cooridnate should be set to -1 for transformation consistency
    if trans_type in ['i2c','i2w','i2p']:
        pts[...,2] = z_axis_type * torch.abs(pts[...,2])
    pts_out = torch.sum(pts[...,np.newaxis,:] * trans_mtx,-1)
    
    if trans_type in ['w2p','c2p']:
        # need to normalize the 3rd elem of the homogeneous coordinate to 1 to get the final pixel coordinate
        scalars = pts_out[...,2]
        pts_out = pts_out/torch.broadcast_to(scalars[...,np.newaxis],pts_out.shape)
    elif trans_type in ['w2i','c2i']:
        # need to normalize the 3rd elem of the homogeneous coordinate to +1/-1 to get the final image coordinate
        scalars = torch.abs(pts_out[...,2])
        pts_out = pts_out/torch.broadcast_to(scalars[...,np.newaxis],pts_out.shape)
    elif trans_type in ['p2w','p2c','i2c','i2w']:
        # add depth info as scalars to recover the 3D coordinate
        depths = torch.reshape(depths,(-1,1))
        pts_out = pts_out * torch.broadcast_to(depths[...,np.newaxis],pts_out.shape)
    elif trans_type in ['w2c','c2w','i2p','p2i']:
        pass
    
    return pts_out[...,:3]