---
layout: cs180
---
# CS280A Project 4, Kushal Nimkar
{% include mathjax.html %}
# Part 0
## 0.1: Calibration
For the first part, for each scene, I took 50 pictures of the 2 rows, 3 column aruco board (with horizontal spacing of 90 mm, vertical spacing of 75.67, and width of aruco marker of 60).
<div class="image-row3a">
  <figure>
    <img src="/assets/images/project4/image_coordinate_example.png" alt="calib_1">
    <figcaption>Figure: Example image of corners found by aruco_detection for calibration. Colors represent oriented corners. The annotated id in red is the marker id it was identified as </figcaption>
  </figure>
</div>

<div class="image-row3a">
  <figure>
    <img src="/assets/images/project4/world_coordinate_example.png" alt="calib2">
    <figcaption>Figure: Corresponding world coordinates (using aruco dimensions). Colors represent oriented corners, and anntoated id is the marker id it was identified as.</figcaption>
  </figure>
</div>
I used this to calibrate my camera by passing the pixel coordinates associated with corners with the matched 3d object coordinates (with z=0) and got an average reprojection error of 0.85. This got me my intrinsics matrix and distortion coefficients for pose sstimation.

## 0.2: Capturing a 3D object scan.

Now I printed a bigger since marker (30 mm wide) and took 35 photos of my duck (two examples shown below) for pose estimation. 

<div class="image-row3a">
  <figure>
    <img src="/assets/images/project4/duck1.JPG" alt="duck1">
    <img src="/assets/images/project4/duck2.JPG" alt="duck2">
    <figcaption>Figure: Two example photos I took (I took roughly 35 that had markers in them).</figcaption>
  </figure>
</div>

## 0.3: Estimating Camera Pose

Now we aruco detect the marker in each of the previous images, use the intrinsics matrix and distortion coefficients from calibration into the solvePnp. This gives us the rvec and tvec which we can construct into a $w2c$ matrix by convert rvec to a 3d rotation matrix via cv2.Rodrigues, and concatentation. Then we can calculate the rotation and translation matrix/vector of c2w as $R_{w2c} = R_{c2w}^T$ and $t_{c2w} = -R_{w2c}^Tt_{c2w}$. Finally, we concatenate to get the final $c2w$ matrix. Below are some vizer visualizations of the images in their poses. Like in the first part, these images are the ones that actually have the tag detected (rest were removed)

<div class="image-row3a">
  <figure>
    <img src="/assets/images/project4/duck_vizer_1.png" alt="dv1">
    <img src="/assets/images/project4/duck_vizer_2.png" alt="dv22">
    <figcaption>Figure: Two example screenshots of the vizer pose estimation. Note that the z coordinate is inverted (e.g negative z values). First image is from the side and second is from the top.</figcaption>
  </figure>
</div>

## 0.4: Undistorting images
I created undistorted images from the intrinsics/distortion coefficients, then cropped to the ROI via cv2.getOptimalNewCameraMatrix and used the new camera matrix as my intrinsics. I left aside 0.1 of the total pose estimations (4 images) and the rest as training data.

# Part 1

Here I used exactly the model architecture specified. It used positional encoding with L= 10, 3 linear layers with ReLU (and last layer being sigmoid). It took normalized coordinates and outputed rgb pixels at each coordinate. I used a learning rate of 1e-2 and batch ize of 10k for 2000 iterations. I tested it on two images below.

<div class="image-row3a">
  <figure>
    <img src="/assets/images/project4/fox_10_256_imgs.jpg" alt="foxpic">
    <img src="/assets/images/project4/fox_10_256_training_curve.jpg" alt="foxval">
    <figcaption>Figure: Top: Memorizing the fox image. Bottom: Training psnr. </figcaption>
  </figure>
</div>
I also tested it on one of the images of my old dog
<div class="image-row3a">
  <figure>
    <img src="/assets/images/project4/Georgie_10_256_imgs.jpg" alt="foxpic">
    <img src="/assets/images/project4/Georgie_10_256_training_curve.jpg" alt="foxval">
    <figcaption>Figure: Top: Memorizing the Georgie image. Bottom: Training psnr.</figcaption>
  </figure>
</div>

Finally I did a 2x2 hyperparameter sweep on the fox image to check how changing the number of positional encoding dimensions (L) and 

<div class="image-row3a">
  <figure>
    <img src="/assets/images/project4/fox_grid_final_imgs.png" alt="foxpic">
    <figcaption>Figure: Showing how adjusting L and batch size affects memorization ability</figcaption>
  </figure>
</div>

Lower values of L and network_width hurt memorization ability, with L seemingly be more important.


# Part 2
## 2.1-2.3
I implemented all functions as specified. In 2.1, I wrote the machinery to turn pixel coordinates into rays based off the extrinsics matrix (c2w) and intrinsics matrix. e.g. first make the transformation matrices to flip between coordinate systems

\begin{align}
M_{c2w} x_c &= x_w
\end{align}
and
\begin{align}
sK^{-1}u &= x_c
\end{align}
then use that to get the rays.

I made a RaysData class that inherts from torch.Dataset which takes the images and corresponding c2w matrices and the intrinsics matrix to create origin and direction rays. Finally, my dataloader simply returns the origin rays, distance rays, and corresponding rgb pixels from a random set of images. Below are a few visualizations of rays (with samples generated from 2.4). 

I also allowed a random perturbation so that when sampling along rays during training, each point is slightly shifted within a window. For example, I subdivide (n=64 or n=32 samples) between far and near distances provided along the direction ray, then randomly perturb the sample slightly within the subinterval along the length.

<div class="image-row3a">
  <figure>
    <img src="/assets/images/project4/2.3_100_samples_all_images.png" alt="foxpic">
    <figcaption>Figure: Showing 100 rays from random cameras</figcaption>
  </figure>
</div>

<div class="image-row3a">
  <figure>
    <img src="/assets/images/project4/2.3_many_samples_1_img.png" alt="foxpic">
    <figcaption>Figure: Showing 100 rays from one camera</figcaption>
  </figure>
</div>

## 2.4
Here I implemented the exact network specified. For positional encoding, I used $L_{world coordinates}=10$, $L_{direction}=4$. Note that when I concatenated the embeddings via a skip connection, I made the inputs into the next layer bigger (e.g. 256 + L*6 + 3) and then had the output return 256.

<div class="image-row3a">
  <figure>
    <img src="/assets/images/project4/architecture.png" alt="foxpic">
    <figcaption>Figure: Neural network architecture</figcaption>
  </figure>
</div>

## 2.5

I implemented the volume rendering equation:
\begin{align}
\hat{C}(r) = \sum_{i=1}^Nw_ic_i
\end{align}
where
\begin{align}
 w_i= (1-\exp(-\sigma_i\delta_i))\exp\left(-\sum_{j=1}^{i-1}\sigma_j\delta_j\right)
\end{align}
and $\delta_i$ was the step size, and $\sigma_i$ are the densities outputted by the network. I let $\delta_i=\delta$ to simplify things (e.g. the random perturbation doesnt change $\delta_i$ much during training, and keep it constant).

For the Bells and Whistles, I also implemented a naive distance measurent $D$

\begin{align}
 \hat{D}(r)= \sum_{i=1}^Nw_it_i
\end{align}
where $t_i$ is the distance along the ray. Essentially, this says the distance to the nearest object is where the density peaks (e.g. averaged across the ray).

My volume render function passes the assert statement. 

Finally, here are some images generated by using a btach size of 10k, learning rate of 5e-4, sampling 64 points over near=2.0 and far=6.0 with network width =256, and positional encoding with lengths 10 and 4 for coordinates and direction respectively.

<div class="image-row3a">
  <figure>
    <img src="/assets/images/project4/Example_over_epochs.png" alt="foxpic">
    <figcaption>Figure: Showing a validation image over training (epochs: 1,3,5,7, each is roughly 432 iterations).</figcaption>
  </figure>
</div>

<div class="image-row3a">
  <figure>
    <img src="/assets/images/project4/PSNR_valid.png" alt="foxpic">
    <figcaption>Figure: Showing PSNRs over validation images over training.</figcaption>
  </figure>
</div>

Finally I producted gifs of both the forklift and its depth map below. Note that dark image in depth map = farther away. I created it by using the previous depth rendering function, and taking 1- that (and normalizing it to 0-1 range).

<div class="image-row3a">
  <figure>
    <img src="/assets/images/project4/forklift_novel_v2.gif" alt="foxpic">
    <img src="/assets/images/project4/depth_novel.gif" alt="foxpic">
    <figcaption>Figure: GIF of forklift reconstruction on test set.</figcaption>
  </figure>
</div>

## 2.6
(This didn't turn out the best...)

Here, are the following chnages I made when running it on my data
* I changed my near and far parameters (which are on mm scale to 150 and 200) 
* I reran my calibration on images 10x downsampled (e.g. 300 by 400), and made sure to normalize inputs to the network were in (0,1) range or so by normalizing the origin rays before they went into the network. 
* I tried both removing the black borders and not when saving the dataset after cv2.undistort. 
* I used a learning rate of 1e-4

<div class="image-row3a">
  <figure>
    <img src="/assets/images/project4/duck_v2_MSE_train.png" alt="foxpic">
      <figcaption>Figure: Training loss goes down</figcaption>
  </figure>
</div>

<div class="image-row3a">
  <figure>
      <img src="/assets/images/project4/duck_v2_PSNR_valid.png" alt="foxpic">
      <figcaption>Figure: Validation PSNR plateaus</figcaption>
  </figure>
</div>

<div class="image-row3a">
  <figure>
      <img src="/assets/images/project4/duck_v2_over_epochs.png" alt="foxpic">
      <figcaption>Figure: Example validation image</figcaption>
  </figure>
</div>

<div class="image-row3a">
  <figure>
      <img src="/assets/images/project4/duck_v2_train_0_start_pos_z.gif" alt="foxpic">
      <figcaption>Figure: The gif.</figcaption>
  </figure>
</div>
Possible issues:
* I know one issue is that some angles my aruco tag was blocked from a lot of angles (e.g. I probably have 180 degree coverage), so some angles look pretty bad. 
* I think the sampling probably has to be dynamic: e.g. the tag is getting reconstructed more evenly than the duck.

Overall, it is possible my issues come from needing to try different architectures, taking more photos, and trying different sampling approaches.






















