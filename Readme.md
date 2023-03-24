Rewrite the NeRF code with some slight modifications.

1. Different datasets have different definitions of the transformation matrix from the meshgrid (used in get_rays function for ray generation) to the camera coordinates. In this code, the transformation matrix should be explicitly defined as mg2c. (e.g., in lego dataset, the camera coord is right-up-backward, while the meshgrid coord is right-down-forward. Thus, a mg2c of (1,-1,-1) should be defined in the train_lego.py. Similar to replica dataset with (1,1,1).)

2. There are two ways to define the camera intrinsics: 1) just an isotropic focal length with no origin translation; 2) a 3x3 intrinsics matrix containing params $f_x$, $f_y$, $c_x$, $c_y$. The get_ray function is modified to be able to handle both of these two params f or K. Besides, I think use torch.matmul() is more intuitive than the way the nerf used for coordinate transformation:(.

3. (For author himself :) Please see the comments about coordinate transformation in data_processor.py. 



I refer to the [NeRF](https://github.com/bmild/nerf) and [DM-NeRF](https://github.com/vLAR-group/DM-NeRF) for code rewriting. the datasets I test the code are [lego](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1) (synthetic data) and [replica](https://github.com/Harry-Zhi/semantic_nerf). 



21 Mar 2023 v1.1

Add tools for coordinate transformations (see ./tools) based on [here](http://ksimek.github.io/2012/08/22/extrinsic/).
The mg2c in the last version is integrated into the process of intrinsics generation. It's actually the coordinate transformation from image/camera coordinates with different definitions (e.g., opencv, opengl or others) to the pixel plane.

23 Mar 2023 v1.1

- Add visualizer for generating pointcloud and mesh of the object by the perdicted depth+rgb map.
- Test the coordinate transformation tool. It works correctly now.
- Train the network with blender datset (lego and hotdot). The results on 3 metrics (PSNR, SSIM and LPIPS) are all similar to that in the paper.
- Adjust the code structre with higher readibility.
- Add render_pose generation tool for algo. test. Now it can only provide circle-around path and marching-like path, both of which looking at the target.

24 Mar 2023 v1.1
- The evaluation metrics compared with that in the paper
    | dataset_code  | PSNR  | SSIM   | LPIPS  |
    |---|---|---|---|
    | synthetic_lego_haru  |  30.509233 |0.955127 |0.039615 |
    | synthetic_hotdog_haru  |  34.816590 |0.966425 |0.047743 |
    | synthetic_avg_nerf  |  31.01 |0.947 |0.081 |
    ||||

- Visualization about the syhthetic video:

<iframe src='//gifs.com/embed/111-EqVQv4' frameborder='0' scrolling='no' width='1920px' height='1080px' style='-webkit-backface-visibility: hidden;-webkit-transform: scale(1);' ></iframe>
