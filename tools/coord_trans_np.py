import numpy as np


trans_t = lambda t:np.array([
    [1,0,0,t[0]],
    [0,1,0,t[1]],
    [0,1,0,t[2]],
    [0,1,0,1]],dtype=float)
rot_x = lambda roll: np.array([
    [1, 0, 0, 0],
    [0, np.cos(roll), -np.sin(roll), 0],
    [0, np.sin(roll), np.cos(roll), 0],
    [0, 0, 0, 1]])
rot_y = lambda pitch: np.array([
    [np.cos(pitch), 0, np.sin(pitch), 0],
    [0, 1, 0, 0],
    [-np.sin(pitch), 0, np.cos(pitch), 0],
    [0, 0, 0, 1]])
rot_z = lambda yaw: np.array([
    [np.cos(yaw), -np.sin(yaw), 0, 0],
    [np.sin(yaw), np.cos(yaw), 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]])

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
    mtx = np.eye(3,dtype=float)
    mtx[0,0] = 1./dx
    mtx[1,1] = 1./dy
    if u0 is None and v0 is None:
        u0 = (W - 1) * .5
        v0 = (H - 1) * .5
    mtx[0,2] = u0
    mtx[1,2] = v0


    if type == 'opengl':
        mtx[1,1] = -mtx[1,1]
        mtx[2,2] = -mtx[2,2]
    elif type == 'other':
        other_type = np.array(other_type,dtype=float)
        for i in range(0,3):
            mtx[i,i] = mtx[i,i]*other_type[i]
    elif type != 'opencv':
        print('Error: The image coordinate type "{}" is wrong. please specify one type from "opencv", "opengl" or your self-defined type "other" and pass it by the arg \'other_type\''.format(type))
        raise Exception('coord_trans_img2pix')
    
    return mtx

#print(coord_trans_img2pix(H=500,W=600,type='other',other_type=[[-1,0,0],[0,1,0],[0,0,-1]]))
#print(coord_trans_img2pix(H=500,W=600,type='openg'))

# transformation matrix from camera coord to image coord
def coord_trans_cam2img(focal):
    """transformation from camera coordinate to image coordinate.\n
        z_cam * p_img_norm = trans_cam2img * p_cam
    """ 
    mtx = np.eye(3,dtype=float)
    mtx[0,0] = focal
    mtx[1,1] = focal
    return mtx

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
                or 'other' see https://stackoverflow.com/questions/44375149/opencv-to-opengl-coordinate-system-transform 
                for details. If it's set to 'opengl', the z and y axes will be in the opposite 
                directions. If type='other', please specify the rotation matrix like 
                [[-1,0,0],[0,1,0],[0,0,-1]] and pass it to this fun via arg 'other_type'. No further operation if 
                type='opencv' cause the image coord is the same as pixel plane.
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
    T_w2c = np.array(T_w2c,dtype=float)
    R = T_w2c[:3,:3]
    t = T_w2c[:3,3]
    cam_pose = np.transpose(R) @ (-t)
    cam_pose = cam_pose[...,None]

    return R, cam_pose

#print(extrinsic_decompose([[-1,0,0,2],[0,1,0,1],[0,0,-1,1],[0,0,0,1]]))

def world2pix_decompose(intrinsic, T_w2p):
    """ input: 4x4 or 3x4 transformation matrix from world to pixel coordinate
        output: R, c. 3x3 rotation matrix and 3x1 camera center w.r.f the world coordinate
    """
    T_w2p = np.array(T_w2p,dtype=float)
    if T_w2p.shape[0] == 4:
        buffer = np.eye(4,dtype=float)
        buffer[:3,:3] = intrinsic
        intrinsic = buffer
    T_w2c = np.linalg.inv(intrinsic) @ T_w2p
    R, cam_pose = extrinsic_decompose(T_w2c)

    return R, cam_pose

#print(world2pix_decompose(np.eye(3,dtype=float),[[-1,0,0,2],[0,1,0,1],[0,0,-1,1],[0,0,0,1]]))

def RC2T(R_cam,C_cam,type=None):
    """create world-to-camera or camera-to-world transformation matrix by
       the rotation and center of the camera in world coordinates.

       Args:
            R_cam: rotation matrix describing the cam's orientation w.r.t the world coords
            C_cam: the camera cneter in the world coords
            trans_w2c: return the transformation from world to camera coords or inversely.

    """
    T = np.eye(4,dtype=float)
    R_cam = np.array(R_cam,dtype=float)
    C_cam = np.array(C_cam,dtype=float)

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

def T_lookat(cam_pose, up, target,type=None):
    """ generate the extrinsic by look-at camera 
        http://ksimek.github.io/2012/08/22/extrinsic/
            the camera coords is opengl type

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
    neg_z_axis = neg_z_axis/np.linalg.norm(neg_z_axis)
    x_axis = np.cross(neg_z_axis, up)
    x_axis = x_axis/np.linalg.norm(x_axis)
    y_axis = np.cross(x_axis,neg_z_axis)
    z_axis = -neg_z_axis

    R_cam = np.stack((x_axis,y_axis,z_axis),0).transpose()
    T = RC2T(R_cam,cam_pose,type)
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

    r_stack = np.stack([r_x,r_y,r_z],0)
    order = np.array(order)
    buffer = np.zeros_like(r_stack)
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
        C_cam = - R @ t
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


def Rotation_along_axis(axis, rad):

    axis = np.array(axis,dtype=float)
    norm = axis/np.linalg.norm(axis)
    n_x = norm[0]
    n_y = norm[1]
    n_z = norm[2]

    c = np.cos(rad)
    s = np.sin(rad)

    col_1 = np.array([n_x*n_x*(1-c)+c, n_x*n_y*(1-c)+n_z*s, n_x*n_z*(1-c)-n_y*s], dtype=float)
    col_2 = np.array([n_x*n_y*(1-c)-n_z*s, n_y*n_y*(1-c)+c, n_y*n_z*(1-c)+n_x*s], dtype=float)
    col_3 = np.array([n_x*n_z*(1-c)+n_y*s, n_y*n_z*(1-c)-n_z*s, n_z*n_z*(1-c)+c], dtype=float)

    T = np.stack([col_1,col_2,col_3],axis=-1)

    #print(np.stack([[1,0,0],[1,2,0],[1,0,3]],axis=-1))

    return T

#print(Rotation_along_axis([1,1,1],np.pi))

