import numpy as np
from tools.coord_trans_np import T_lookat

# for blender dataset test
def render_pose_circle(target, radius, rad_xy2z, seq):
    target = np.array(target,dtype=float)
    z = radius * np.sin(rad_xy2z)
    proj_xy = radius * np.cos(rad_xy2z)
    render_poses = []
    interval = 2*np.pi/seq
    up = np.array([0,0,1])

    for idx in range(0,seq):
        x = proj_xy * np.cos(interval*idx)
        y = proj_xy * np.sin(interval*idx)
        T_c2w = T_lookat([x,y,z],up,target,type='c2w')
        render_poses.append(T_c2w)
    
    return np.array(render_poses,dtype=float)


    

# for replica dataset test
def render_pose_marching(start, end, seq):
    
    interval_center = end-start

    up = [0,0,1]
    render_poses = []

    for idx in range(seq):

        C_cam = start + interval_center*idx/seq
        T_c2w = T_lookat(C_cam, up, end,type='c2w')
        render_poses.append(T_c2w)
        
    return np.array(render_poses,dtype=float)
    
    
