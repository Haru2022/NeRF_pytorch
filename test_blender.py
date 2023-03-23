import os
import torch
import numpy as np
from tools.evaluators import to8b
from tools.visualizer import pcd_gen
from configs.configs_loader import initial
from data_loader.loader_blender import load_blender_data
from nerf.nerf_constructor import get_embedder, NeRF
from nerf.render import render_test
import imageio


def test():
    model_coarse.eval()
    model_fine.eval()
    args.is_train = False

    pcl_rgb_valids, rgbs = None, None
    visual_dir = os.path.join(args.basedir, args.expname, args.log_time,'visualization')
    os.makedirs(visual_dir,exist_ok=True)
    # main process
    with torch.no_grad():
        print('Rendering......')
        #print(render_poses.shape)
        for idx, pose in enumerate(render_poses):
            #print(pose)
            print('Process: {}/{}'.format(idx,render_poses.shape[0]))
            pcl_rgb_valid, rgb = render_test(position_embedder, view_embedder, model_coarse, model_fine, pose[None,...], hwK, args, obj_recon=True)
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
    imgs, poses, render_poses, hwK, i_split = load_blender_data(args.datadir, args.resize_factor, args.testskip, args.white_bkgd)
    H,W,K = hwK
    print("h,w,k:{},{},{}".format(H,W,K))
    H,W = int(H), int(W)
    i_train, i_val, i_test = i_split
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
    imgs = torch.Tensor(imgs).cpu()
    poses = torch.Tensor(poses).cpu()
    render_poses = torch.Tensor(render_poses).to(args.device)
    K = torch.Tensor(K).to(args.device)

    test()
    