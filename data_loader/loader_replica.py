import os
import h5py
import json
import torch
import imageio
import numpy as np
from data_processor import central_resize_batch
from tools.coord_trans_np import gen_intrinsics

np.random.seed(0)


class rgb_processor:
    def __init__(self, basedir, train_ids, test_ids, testskip=1):
        super(rgb_processor, self).__init__()
        self.basedir = basedir
        self.testskip = testskip
        self.train_ids = train_ids
        self.test_ids = test_ids

    def load_rgb(self):
        # testskip operation
        skip_idx = np.arange(0, len(self.test_ids), self.testskip)

        # load poses
        traj_file = os.path.join(self.basedir, 'traj_w_c.txt')
        Ts_full = np.loadtxt(traj_file, delimiter=" ").reshape(-1, 4, 4)
        train_poses = Ts_full[self.train_ids]
        test_poses = Ts_full[self.test_ids]
        test_poses = test_poses[skip_idx]
        poses = np.concatenate([train_poses, test_poses], axis=0)

        # load rgbs
        rgb_basedir = os.path.join(self.basedir, 'rgb')
        train_rgbs = [imageio.imread(os.path.join(rgb_basedir, f'rgb_{idx}.png')) for idx in self.train_ids]
        test_rgbs = [imageio.imread(os.path.join(rgb_basedir, f'rgb_{idx}.png')) for idx in self.test_ids]
        train_rgbs = np.array(train_rgbs)
        test_rgbs = np.array(test_rgbs)[skip_idx]
        rgbs = np.concatenate([train_rgbs, test_rgbs], axis=0)
        rgbs = (rgbs / 255.).astype(np.float32)

        i_train = np.arange(0, len(self.train_ids), 1)
        i_test = np.arange(len(self.train_ids), len(self.train_ids) + len(skip_idx), 1)
        i_split = [i_train, i_test]

        return rgbs, poses, i_split


def load_replica_data(args):
    # load color image RGB
    total_num = 900
    step = 5
    train_ids = list(range(0, total_num, step))
    test_ids = [x + step // 2 for x in train_ids]

    imgs, poses, i_split = rgb_processor(args.datadir, train_ids, test_ids, testskip=args.testskip).load_rgb()


    H, W = imgs[0].shape[:2]

    focal = W / 2.0
    K = gen_intrinsics(focal=focal,H=H,W=W,type='opencv')
    #K = np.array([[focal, 0, (W - 1) * 0.5], [0, focal, (H - 1) * 0.5], [0, 0, 1]])

    if args.resize_factor != 1.:
        #cv2.imshow('raw',imgs[0])
        imgs, H, W, K = central_resize_batch(imgs, args.resize_factor,K)

    hwk = [int(H), int(W), K]


    return imgs, poses, hwk, i_split