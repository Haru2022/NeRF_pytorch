Rewrite the NeRF code with some slight modifications.

1. Different datasets have different definitions of the transformation matrix from the meshgrid (used in get_rays function for ray generation) to the camera coordinates. In this code, the transformation matrix should be explicitly defined as mg2c. (e.g., in lego dataset, the camera coord is right-up-backward, while the meshgrid coord is right-down-forward. Thus, a mg2c of (1,-1,-1) should be defined in the train_lego.py. Similar to replica dataset with (1,1,1).)

2. There are two ways to define the camera intrinsics: 1) just an isotropic focal length with no origin translation; 2) a 3x3 intrinsics matrix containing params $f_x$, $f_y$, $c_x$, $c_y$. The get_ray function is modified to be able to handle both of these two params f or K. Besides, I think use torch.matmul() is more intuitive than the way the nerf used for coordinate transformation:(.

3. (For author himself :) Please see the comments about coordinate transformation in data_processor.py. 



I refer to the [NeRF](https://github.com/bmild/nerf) and [DM-NeRF](https://github.com/vLAR-group/DM-NeRF) for code rewriting. the datasets I test the code are [lego](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1) (synthetic data) and [replica](https://github.com/Harry-Zhi/semantic_nerf).