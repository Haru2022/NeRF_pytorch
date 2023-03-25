import os
import h5py
import json
import torch
import imageio
import numpy as np
from tools.data_processor import central_resize_batch
from tools.coord_trans_np import gen_intrinsics

np.random.seed(0)


class data_processor:
    def __init__(self, basedir, train_ids, test_ids, testskip=1):
        super(data_processor, self).__init__()
        self.basedir = basedir
        self.testskip = testskip
        self.train_ids = train_ids
        self.test_ids = test_ids
        self.traj_file = os.path.join(self.basedir, 'traj_w_c.txt')
        self.Ts_full = np.loadtxt(self.traj_file, delimiter=" ").reshape(-1, 4, 4)

    def load_data(self, type):
        # testskip operation
        skip_idx = np.arange(0, len(self.test_ids), self.testskip)

        # load poses
        train_poses = self.Ts_full[self.train_ids]
        test_poses = self.Ts_full[self.test_ids]
        test_poses = test_poses[skip_idx]
        poses = np.concatenate([train_poses, test_poses], axis=0)

        # load data
        data_basedir = os.path.join(self.basedir, type)
        train_data = [imageio.imread(os.path.join(data_basedir, f'{type}_{idx}.png')) for idx in self.train_ids]
        test_data = [imageio.imread(os.path.join(data_basedir, f'{type}_{idx}.png')) for idx in self.test_ids]
        train_data = np.array(train_data)
        test_data = np.array(test_data)[skip_idx]
        if test_data.shape[0]>0 and train_data.shape[0]>0:
            data = np.concatenate([train_data, test_data], axis=0)
        elif test_data.shape[0]==0 and train_data.shape[0]>0:
            data = train_data
        elif test_data.shape[0]>0 and train_data.shape[0]==0:
            data = test_data
        else:
            print('Error: no data is loaded.')
            raise Exception()
        

        if type=='rgb':
            data = (data / 255.).astype(np.float32) # 0-255 to 0-1
        elif type=='depth':
            data = (data / 1000.).astype(np.float32) # metric from millimeter to meter
        elif type=='ins_seg':
            #TODO
            1

        i_train = np.arange(0, len(self.train_ids), 1)
        i_test = np.arange(len(self.train_ids), len(self.train_ids) + len(skip_idx), 1)
        i_split = [i_train, i_test]

        return data, poses, i_split
    



def load_replica_data(args, train_ids, test_ids, data_type='rgb'):
    # load data

    data_loader = data_processor(args.datadir, train_ids, test_ids, args.testskip)

    imgs, poses, i_split =  data_loader.load_data(data_type)

    H, W = imgs[0].shape[:2]

    focal = W / 2.0
    K = gen_intrinsics(focal=focal,H=H,W=W,type='opencv')
    #K = np.array([[focal, 0, (W - 1) * 0.5], [0, focal, (H - 1) * 0.5], [0, 0, 1]])

    if args.resize_factor != 1.:
        #cv2.imshow('raw',imgs[0])
        imgs, H, W, K = central_resize_batch(imgs, args.resize_factor,K)

    hwk = [int(H), int(W), K]


    return imgs, poses, hwk, i_split