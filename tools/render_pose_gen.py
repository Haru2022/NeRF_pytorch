import numpy as np
from coord_trans_np import world2cam_lookat, T_convert

# for blender dataset test
def render_pose_circle(target, radius, rad_xy2z, seq):
    z = radius * np.sin(rad_xy2z)
    proj_xy = radius * np.cos(rad_xy2z)
    render_poses = []
    interval = 2*np.pi/seq
    up = np.array([0,0,1])

    for idx in range(0,seq):
        x = proj_xy * np.cos(interval*idx)
        y = proj_xy * np.sin(interval*idx)
        T_w2c = world2cam_lookat([x,y,z],up,target)
        T_c2w = T_convert(T_w2c)
        render_poses.append(T_c2w)
    
    return np.array(render_poses,dtype=float)


    

# for replica dataset test
def render_pose_marching(central_ray, seq, circle_count):
    1
