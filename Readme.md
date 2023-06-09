**3 April 2023 v1.2**
- Add crop opearation for accelarating the convergence of the network. In some synthetic datasets, most of the area of an image is filled with white background. If too many rays are sampled from such meaningless area at the early stage, the network may be guided towards wrong gradient direction and the final network performance varies in a wide region. Sometimes it may be impossible to turn it back to the right direction. Therefore, for accelerating the training, the images will be central-cropped at the begining. Below is an example about the influence of crop operation on the final network performance (Resizing the image to be small can act as similar effect to central crop because the possibility that the rays sampled from the meaningless region decreases when the total number of sampling rays is the same.):

    | dataset_code_is-croppd  | PSNR  | SSIM   | LPIPS  |
    |---|---|---|---|
    | synthetic_drum_haru_False_1     |  divergent |divergent |divergent |
    | synthetic_drum_haru_False_2     |  21.870 |0.843 |0.201 |
    | synthetic_drum_haru_False_3     |  23.090 |0.869 |0.173 |
    | synthetic_drum_haru_True     |  25.382 | 0.927 | 0.083 |
    | synthetic_drum_haru_False(resize:50%)     |  24.929 |0.923 |0.067 |
    | synthetic_drum_nerf_unknown     |  25.01 |0.925 |0.091 |
- The version 1.2 is fixed.

**2 April 2023 v1.2**
- Correct the construction of the transformation from image coordinates to pixel coordinates.
- Add the training result on blender_drums. See **24 Mar 2023 v1.1**.
- Visualization about the coordinate transformation from pixel plane to the world coordiantes.
    <div align=center>

    <img src="./readme_visual/coord_trans_visual.png" width="50%"><img src="./readme_visual/coord_trans_visual1.png" width="50%"><center>Coordinate Transformation Visualization</center>
    </div>

    The images with red, green and blue background are the projections of the scene in the world coordinates on the pixel plane, image plane and in the coordinate coordinates, respectively. The details about these three transformations among these four coordinate system will be posted soon.


**28 Mar 2023 v1.2**
- Update the evaluation metrics on the blender_lego dataset after 500K iters. See **24 Mar 2023 v1.1**

**27 Mar 2023 v1.2**

- The stable version v1.1 is fixed.
- update the coordinate transformation tool. All the input and output of the functions are uniformly defined in $4\times4$ homogeneous transformation form.
- Remove the blackpoint that comes from the zero-depth pixels with all rgbs=[0,0,0] in the grund truch pointcloud.

     <div align=center><img src="./readme_visual/room0/gt_pcl_blackpoint_removal.png" width="60%"><center>Ground Truth Pointcloud after zero-depth points removal</center></div>
    

**27 Mar 2023 v1.1**
- Visualization on room_0

    <div align=center><img src="./readme_visual/room0/rgb_room0.gif" width="60%"><center>Synthetic Views</center></div>

    <div align=center><img src="./readme_visual/room0/gt_pcl2.png" width="50%"><img src="./readme_visual/room0/gt_pcl1.png" width="50%"><center>Ground Truth Pointcloud</center></div>
    
    <div align=center><img src="./readme_visual/room0/pred_pcl1.png" width="50%"><img src="./readme_visual/room0/pred_pcl2.png" width="50%"><center>Predicted Pointcloud</center></div>



**25 Mar 2023 v1.1**
- The camera coordinate type should be **explicitly claimed** like 'opencv' or 'opengl' when using all the functions in coord_trans_np.py to avoid coordinate inconsistency.
- Testing the network on Replica dataset room_1 and room_0. The depth and rgb imgs are obtained from [Semantic-NeRF](https://github.com/Harry-Zhi/semantic_nerf)

    | dataset_code_iters  | PSNR  | SSIM   | LPIPS  |
    |---|---|---|---|
    | replica_room1_haru_200K  |  32.87 |0.899 |0.173 |
    | replica_room1_[DM-Nerf](https://github.com/vLAR-group/DM-NeRF)_500K  |  34.72 |0.931 | 0.134 |
    | replica_room0_haru_200K     |  33.24 |0.915 |0.167 |
    | replica_room0_DM-Nerf_500K  |  34.97 |0.940 | 0.127 |
    ||||


*the following visualization is implemented with resize factor=0.5. But there is **no resize operation** when training.
- Visualization about the synthetic views:
    <div align=center><img src="./readme_visual/room1/rgb_room1.gif" width="50%"></div>
    *Note that this path will be optimized after. Now it's just a combination of several marching pathes looking at a fix point.*

- Ground-Truth and predicted pointcloud of dataset Replica_room1:
    <div align=center><img src="./readme_visual/room1/gt_pcl2.png" width="50%"><img src="./readme_visual/room1/gt_pcl1.png" width="50%"><center>Ground Truth Pointcloud</center></div>
   
    <div align=center><img src="./readme_visual/room1/pred_pcl1.png" width="50%"><img src="./readme_visual/room1/pred_pcl2.png" width="50%"> <center>Predicted Pointcloud</center></div>
   
    <div align=center><img src="./readme_visual/room1/gt_voxel2.png" width="50%"><img src="./readme_visual/room1/gt_voxel1.png" width="50%"><center>Ground Truth Voxel</center></div>
    
    <div align=center><img src="./readme_visual/room1/pred_voxel1.png" width="50%"><img src="./readme_visual/room1/pred_voxel2.png" width="50%"><center>Predicted Voxel (after outlier removal)</center></div>
    
    <div align=center><img src="./readme_visual/room1/pred_mesh1.png" width="50%"><img src="./readme_visual/room1/pred_mesh2.png" width="50%"><center>Predicted Mesh</center></div>


**24 Mar 2023 v1.1**
- The evaluation metrics compared with that in the paper
    | dataset_code_iters  | PSNR  | SSIM   | LPIPS  |
    |---|---|---|---|
    | synthetic_lego_haru_200K (resize:50%)  |  30.43 |0.957 |0.039 |
    | synthetic_lego_haru_500K (full size)  |  32.35 |0.965 |0.045 |
    | synthetic_lego_nerf_500K  |  32.54 |0.961 |0.050 |
    | synthetic_hotdog_haru_200K (full size)  |  34.82 |0.966 |0.048 |
    | synthetic_hotdog_nerf_500K  |  36.18 |0.974 | 0.121 |
    | synthetic_drums_haru_500K (resize:50%)  |  24.93 |0.923 |0.067 |
    | synthetic_drums_nerf_500K  |  25.01 |0.925 | 0.091 |


- Visualization about the syhthetic views:
    <div align=center><img src="./readme_visual/rgb_lego.gif" width="50%"><img src="./readme_visual/rgb_hotdog.gif" width="50%"></div>

- Visualization about the pointcloud and mesh
    <div align=center>

    <img src="./readme_visual/lego/haru_lego_pcl.png" width="60%"><center>Predicted pointcloud with simple outlier removal</center>

    <img src="./readme_visual/lego/gt_lego_mesh.jpg" width="60%"><center>Ground truth 3D structure</center>

    <img src="./readme_visual/lego/haru_lego_mesh_gray.png" width="60%"><center>Predicted mesh (gray version)</center>
    
    <img src="./readme_visual/lego/haru_lego_mesh_color.png" width="60%"><center>Predicted mesh (color version)</center>
    </div>
    
- Videos about [Pointcloud](https://youtu.be/Vi1iftw7FQQ), [voxel](https://youtu.be/irh28e_FcYI) and [mesh](https://youtu.be/D5L9xWYBkY8) of the lego dataset


**23 Mar 2023 v1.1**

- Add visualizer for generating pointcloud and mesh of the object by the perdicted depth+rgb map.
- Test the coordinate transformation tool. It works correctly now.
- Train the network with blender datset (lego and hotdot). The results on 3 metrics (PSNR, SSIM and LPIPS) are all similar to that in the paper.
- Adjust the code structre with higher readibility.
- Add render_pose generation tool for algo. test. Now it can only provide circle-around path and marching-like path, both of which looking at the target.


**21 Mar 2023 v1.1**

Add tools for coordinate transformations (see ./tools) based on [here](http://ksimek.github.io/2012/08/22/extrinsic/).
The mg2c in the last version is integrated into the process of intrinsics generation. It's actually the coordinate transformation from image/camera coordinates with different definitions (e.g., opencv, opengl or others) to the pixel plane.


**13 Mar 2023 v1.0**

Rewrite the NeRF code with some slight modifications.

1. Different datasets have different definitions of the transformation matrix from the meshgrid (used in get_rays function for ray generation) to the camera coordinates. In this code, the transformation matrix should be explicitly defined as mg2c. (e.g., in lego dataset, the camera coord is right-up-backward, while the meshgrid coord is right-down-forward. Thus, a mg2c of (1,-1,-1) should be defined in the train_lego.py. Similar to replica dataset with (1,1,1).)

2. There are two ways to define the camera intrinsics: 1) just an isotropic focal length with no origin translation; 2) a 3x3 intrinsics matrix containing params $f_x$, $f_y$, $c_x$, $c_y$. The get_ray function is modified to be able to handle both of these two params f or K. Besides, I think use torch.matmul() is more intuitive than the way the nerf used for coordinate transformation:(.

3. (For author himself :) Please see the comments about coordinate transformation in data_processor.py. 



I refer to the [NeRF](https://github.com/bmild/nerf) and [DM-NeRF](https://github.com/vLAR-group/DM-NeRF) for code rewriting. the datasets I test the code are [lego](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1) (synthetic data) and [replica](https://github.com/Harry-Zhi/semantic_nerf). 
