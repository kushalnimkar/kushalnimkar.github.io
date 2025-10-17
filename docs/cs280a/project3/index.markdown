---
layout: cs180
---
# CS280A Project 3, Kushal Nimkar
{% include mathjax.html %}
## A.1: Shoot the Pictures

For the first part, for each scene, I took two pictures with my iPhone camera, where I attempted to keep the center of projection still by pivoting around the point where the camera was (e.g. like a door hinge). The focal length is not changed between images.
<div class="image-row3a">
  <figure>
    <img src="/assets/images/project3/A1_i.png" alt="A1_i">
    <figcaption>Figure: Living room </figcaption>
  </figure>
</div>

<div class="image-row3a">
  <figure>
    <img src="/assets/images/project3/A1_church.png" alt="A1_church">
    <figcaption>Figure: Religious school on Euclid Avenue</figcaption>
  </figure>
</div>

<div class="image-row3a">
  <figure>
    <img src="/assets/images/project3/A1_street.png" alt="A1_street">
    <figcaption>Figure: Oxford and Vine Street  </figcaption>
  </figure>
</div>



## A.2: Recover Homographies

First, we must figure out how to calculate $H\in \mathbb R^{3 \times 3}$ which relates $(x,y)$ in perspective 1 to $(x',y')$ in perspective 2. We first write the general equation in homogeneous coordinates

\begin{align}
\begin{bmatrix}wx' \newline wy' \newline w\end{bmatrix} = \begin{bmatrix}h_{11} & h_{12} & h_{13} \newline h_{21} & h_{22} & h_{23} \newline h_{31} & h_{32} & h_{33} \end{bmatrix} = \begin{bmatrix}x \newline y \newline 1\end{bmatrix}
\end{align}

We realize that any solution $H$ can be scaled by any factor and obtain the exact same output $(x',y')$ for this form of equation. Thus, we fix $h_{33}=1$ to fix the scaling factor. Doing this we can write out the 3 equations dictated by this matrix.
\begin{align}
w&=h_{31}x + h_{32}y + 1 \newline
wx'&=h_{11}x + h_{12}y + h_{13} \newline
wy'&=h_{21}x + h_{22}y + h_{23} \newline
\end{align}

Replacing $w$ into both $x'$ and $y'$ equations we obtain

\begin{align}
(h_{31}x + h_{32}y + 1)x'&=h_{11}x + h_{12}y + h_{13} \newline
(h_{31}x + h_{32}y + 1)y'&=h_{21}x + h_{22}y + h_{23} \newline
\end{align}

This can be written in terms of matrices again
\begin{align}
\begin{bmatrix} x & y & 1 & 0 & 0 & 0 & -xx' & -yx'\newline 0 & 0 & 0 & x & y & 1 & -xy' & -yy' \end{bmatrix}\begin{bmatrix}h_{11} \newline h_{12} \newline h_{13} \newline h_{21} \newline h_{22} \newline h_{23} \newline h_{31} \newline h_{32} \newline\end{bmatrix} = \begin{bmatrix}x' \newline y' \end{bmatrix}
\end{align}
which is of the form
\begin{align}
Ah = b
\end{align}


Now, given a set of keypoints $(x_i,y_i)$ with correspondences $(x'_i,y'_i)$ we can repeat A (two rows for each datapoint) and b (two rows for each datapoint) to set this up as a regression problem. I used np.linalg.lstsq(A,b) to actually find $h$.

Below are the manually labeled correspondence points and the calculated Homography matrices. I used matplotlib to create an interactive figure, then zoomed in a lot and manually picked points (which I typed the coordinates in myself...)

<div class="image-row3a">
  <figure>
    <img src="/assets/images/project3/A2_correspondences_i.png" alt="A1_corri">
    <figcaption>Figure: Correspondence points (red) for living room </figcaption>
  </figure>
</div>

\begin{align}
H_{living\;room} = \begin{bmatrix}  1.56557185e^0  & -3.41995921e^{-2} & -6.68544928e^{2}]\newline
  2.45456822e^{-1} &  1.34609181e^0 & -2.57104097e^{2}\newline
  3.51558085e^{-4} & 4.05706558e^{-6} & 1.00000000e^{0} \end{bmatrix}
\end{align}
<div class="image-row3a">
  <figure>
    <img src="/assets/images/project3/A2_correspondences_church.png" alt="A1_corrc">
    <figcaption>Figure: Correspondence points (red) for religious school </figcaption>
  </figure>
</div>
\begin{align}
H_{religious\;school}   =\begin{bmatrix}7.16645991e^{-1} & 4.10356844e^{-3} &  3.22535054e^2 \newline
 -1.17582687e^{-1} &  8.79899412e^{-1} & 8.94382948e^{1} \newline
 -1.76562945e^{-4} & 1.47537236e^{-6} & 1.00000000e^{0} \end{bmatrix}
\end{align}
<div class="image-row3a">
  <figure>
    <img src="/assets/images/project3/A2_correspondences_street.png" alt="A1_corrs">
    <figcaption>Figure: Correspondence points (red) for Oxford & Vine </figcaption>
  </figure>
</div>
\begin{align}
H_{Oxford\;and\;Vine}   =\begin{bmatrix} 1.48076445e^{0} & -7.61857945e^{-3} & -5.66846471e^{2} \newline
  1.92838911e^{-1} &  1.29985088e^{0} & -2.07688655e^2 \newline
  2.94224501e^{-4} & 1.68245832e^{-5} & 1.00000000e^0\end{bmatrix}
\end{align}

## A.3: Warp the Images

Here, I first implemented the warp function. I did this by applying the learned Homography $H$ to a meshgrid of coordinates $(x,y)$ (e.g. each pixel in the image has an x,y coordinate). Then I found the boundaries by taking the minimum and maximum of the x' and y' of the transformed coordinates in each image (indexed as 1 and 2) e.g. 

\begin{align}
x_{min}'= \textrm{min}(x_{min,1}',x_{min,2}') \newline
y_{min}' = \textrm{min}(y_{min,1}',y_{min,2}') \newline
x_{max}' = \textrm{max}(x_{max,1}',x_{max,2}') \newline
y_{max}' = \textrm{max}(y_{max,1}',y_{max,2}') \newline
\end{align}
With the new minimum and maximum bounds, I create an output image with dimensions $|x_{max}'-x_{min}'|$ and $|y_{max}'-y_{min}'|$ (both rounded up). Note that implicitly this step also changes the coordinate systems by 
\begin{align}
H_{shift} = \begin{bmatrix} 1 & 0 & -x_{min} \newline 0 & 1 & -y_{min} \newline 0 & 0 & 1\end{bmatrix}
\end{align} 

Then I can find the overall $H^{-1}$ from the final output image to the very original coordinates through

\begin{align}
H_{overall}^{-1}= (H_{shift}H)^{-1} = H^{-1}H_{shift}^{-1}
\end{align}
Using this inverse transformation, and creating a meshgrid of the output image, we can do inverse warping to access the original. Note that if the inverse map leads to a coordinate outside the bounds of the original image, we set $\alpha=0$ and $RGB=  [0,0,0]$ at that pixel to make it transparent and also not show up in the final image. 

I implemented both nearest neighbor interpolation (e.g. take RGB pixel of nearest neighbor after applying $H_{overall}^{-1}$) or do bilinear interpolation (take weighted average of four pixels enclosing that mapped point, see Wikipedia [here](https://en.wikipedia.org/wiki/Bilinear_interpolation#/media/File:Bilinear_interpolation_visualisation.svg) for visualization).

I tested my warping and inverse mapping on two sample rectifications, where I denote anchor points in the image, then map them to manually annotated (arbitrary) rectangular coordinates.

<div class="image-row3a">
  <figure>
    <img src="/assets/images/project3/A2_correspondences_r_cropped.png" alt="monitor_rect_bilinear">
    <img src="/assets/images/project3/A3_rectification_r_NN.png" alt="monitor_rect_bilinear">
    <img src="/assets/images/project3/A3_rectification_r_bilinear.png" alt="monitor_rect_bilinear">
    <figcaption>Figure: Monitor rectification. Top: Original image with keypoints in red. Middle: Warping and using nearest-neighbor interpolation for color. Bottom: Warping and using bilinear interpolation for color </figcaption>
  </figure>
</div>
Another example is shown below
<div class="image-row3a">
  <figure>
    <img src="/assets/images/project3/A2_correspondences_s_cropped.png" alt="floor_rect_bilinear">
    <img src="/assets/images/project3/A3_rectification_s_NN.png" alt="floor_rect_bilinear">
    <img src="/assets/images/project3/A3_rectification_s_bilinear.png" alt="floor_rect_bilinear">
    <figcaption>Figure: Floor tile rectification. Top: Original image with keypoints in red. Middle: Warping and using nearest-neighbor interpolation for color. Bottom: Warping and using bilinear interpolation for color </figcaption>
  </figure>
</div>
It doesn't really seem like there is a whole visible difference between NN and bilinear interpolation, probably because my images are kind of bland to begin with here. However, bilinear interpolation should generally be more pleasing (since it does a better smoothing so to speak). I will note that bilinear interpolation takes longer to run (e.g. 0.6 s for NN, 0.9s for bilinear) which makes sense since it requires more queries to the original image.

## A.4: Blend the Images into a Mosaic

Now, I use the second image as the final output canvas coordinate system. So after warping (as shown previously), we can simply shift the second image by the new zero denoted by $H_{shift}$ above, and paste it onto the final canvas. Two of my images had minimal edge effects, but for the third I had some. To fix this, I used feathering. For both images, I calculated the distance for each pixel from pixels with $\alpha = 0$ (call this $D_1$ and $D_2$ for image 1 and image 2, respectively). Then I calculated a new alpha channel for each image as

\begin{align}
\alpha_1 &= \frac{D_1}{D_1+D_2}\newline
\alpha_2 &= \frac{D_2}{D_1+D_2}\newline
\end{align}

Finally, on the final coordinate system, I can calculate the pixel intensity of the final image $I_{final}$ as a weighted average of the two original (aligned) images ($I_{1},I_2$)
\begin{align}
I_{final} = \alpha_1 \cdot I_1 + \alpha_2\cdot I_2
\end{align}
where $\cdot$ is elementwise multiplication. This is visualized below

<div class="image-row">
  <figure>
    <img src="/assets/images/project3/A4_distances_i_1.png" alt="floor_rect_bilinear">
    <img src="/assets/images/project3/A4_distances_i_2.png" alt="floor_rect_bilinear">
    <img src="/assets/images/project3/A4_post_blend_i.png" alt="floor_rect_bilinear">
    <figcaption>Figure: Visualizing the weighted alphas. Row 1: Distance map from $\alpha=0$ for image 1. Row 2: Distance map from $\alpha =0$ for image 2. Final blended image using these weighted distance maps (with normalization).</figcaption>
  </figure>
</div>


**Final outputs:** Below are my original images (rows **1** and **2**), naive cut and paste post alignment images where image2 just overwrites image 1 pixels on the canvas (**3**), and feathered images (**4**).

**Living room:**
<div class="image-row3a">
  <figure>
    <img src="/assets/images/project3/i1_resized.jpg" alt="floor_rect_bilinear">
    <img src="/assets/images/project3/i2_resized.jpg" alt="floor_rect_bilinear">
    <img src="/assets/images/project3/A4_pre_blend_i.png" alt="floor_rect_bilinear">
    <img src="/assets/images/project3/A4_post_blend_i.png" alt="floor_rect_bilinear">
    <figcaption>Figure: Living room blending. Row 1: Image 1. Row 2: Image 2. Row 3: Warping and alignment, (Image 2 overwrites image 1). Row 4: Blending the previous image using feathering.</figcaption>
  </figure>
</div>

**Religious school:**

<div class="image-row3a">
  <figure>
    <img src="/assets/images/project3/church1_resized.jpg" alt="floor_rect_bilinear">
    <img src="/assets/images/project3/church2_resized.jpg" alt="floor_rect_bilinear">
    <img src="/assets/images/project3/A4_pre_blend_church.png" alt="floor_rect_bilinear">
    <img src="/assets/images/project3/A4_post_blend_church.png" alt="floor_rect_bilinear">
    <figcaption>Figure: Religious school blending. Row 1: Image 1. Row 2: Image 2. Row 3: Warping and alignment, (Image 2 overwrites image 1). Row 4: Blending the previous image using feathering.</figcaption>
  </figure>
</div>

**Rose & Oxford**

<div class="image-row3a">
  <figure>
    <img src="/assets/images/project3/street1_resized.jpg" alt="floor_rect_bilinear">
    <img src="/assets/images/project3/street2_resized.jpg" alt="floor_rect_bilinear">
    <img src="/assets/images/project3/A4_pre_blend_street.png" alt="floor_rect_bilinear">
    <img src="/assets/images/project3/A4_post_blend_street.png" alt="floor_rect_bilinear">
    <figcaption>Figure: Rose & Oxford street. Row 1: Image 1. Row 2: Image 2. Row 3: Warping and alignment, (Image 2 overwrites image 1). Note the slight seam by the car on the left. Row 4: Blending the previous image using feathering.</figcaption>
  </figure>
</div>

## Bells and Whistles
I chose to try to do cylindrical mapping. What I first did was map the original image coordinates through this transformation $(x,y)\rightarrow (x',y')$ given in the Computer Vision textbook by Richard Szeliski. Let $x_c,y_c$ be the center coordinates of an image (e.g. width and height both divided by two). First find
\begin{align}
\theta &= \textrm{tan}^{-1}\left(\frac{x-x_c}{f}\right)\newline
h &= \frac{(y-y_c)}{\sqrt{(x-x_c)^2+f^2}}
\end{align}
Then defining the focal length $f$ we find
\begin{align}
x' &= f\theta\newline
y' &= fh
\end{align}

Similarly, we can define an inverse transformation (which we need for the inverse warp) by inverting the above equations

\begin{align}
x &= f\tan(x'/f) + x_c \newline
y &= \frac{y'\sqrt{x^2+f^2}}{f} + y_c
\end{align}

First we show what the warping to the cylinder looks like for the living room as an example (f=1200) with the keypoints also transferred in red. I chose 

<div class="image-row3a">
  <figure>
    <img src="/assets/images/project3/BW_i_1200_1.png" alt="floor_rect_bilinear">
    <img src="/assets/images/project3/BW_i_1200_2_png.png" alt="floor_rect_bilinear">
    <figcaption>Figure: Cylindrical warp example. Top: Image 1 with keypoints transferred to cylinder (f=1200),Bottom: Image 2 with keypoints transferred to cylinder (f=1200)</figcaption>
  </figure>
</div>
We can also compare what $f=1200$ and $f=600$ look like.
<div class="image-row3a">
  <figure>
    <img src="/assets/images/project3/BW_i_1200_1.png" alt="floor_rect_bilinear">
    <img src="/assets/images/project3/BW_i_600_1.png" alt="floor_rect_bilinear">
    <figcaption>Figure: Cylindrical warp example. Top: Image 1 with keypoints transferred to cylinder (f=1200). Bottom: Same image with (f=600).</figcaption>
  </figure>
</div>

In theory, the correct $f$ is the same for all images (since I used the same camera and didn't use zoom). I thought $f=1200$ looked good and justified it as follows: in theory two images projected to the cylinder around the same center of projection (COP) should be related by translation only. Therefore certain objects like the piano in both images at the correct $f$ should have the same orientation (e.g. lines are at similar angles).

In practice, I found the COP is probably not perfectly maintained by me, so I still use a homography to "fix" it a bit (when I tried translation only the alignment isn't great). I simply stitched the two images using the same approach we did in the non-Bells and whistles part (but with the new cylindrically warped images and maintaining the alpha masks). The results are below

<div class="image-row3a">
  <figure>
    <img src="/assets/images/project3/BW_i_final.png" alt="floor_rect_bilinear">
    <img src="/assets/images/project3/BW_church_final.png" alt="floor_rect_bilinear">
    <img src="/assets/images/project3/BW_street_final.png" alt="floor_rect_bilinear">
    <figcaption>Figure: Final stitched photos for 3 mosaics using cylindrical coordinates</figcaption>
  </figure>
</div>
It honestly does not look that different (other than the bounding shape), probably because I have a lot of overlap between the images and didn't take a huge mosaic.

# Part 2: Feature matching for autostitching
## B.1: Detecting corner features in an image

First, we detect corners through the harris detector code in harris.py. Briefly, this approach uses a Taylor approximation to calculate the change in pixel intensities when you shift a window around a point. As lightweight preprocessing, we used min_distance=3 (instead of min_distance=1) for skimage.feature.peak_local_mask to generate roughly 18k corners.

In order to narrow this down in a way that mantains a roughly uniform distribution of points across the image, we used the adaptive nonmaximal suppression suggested in the MOPS paper. In practice, the way I implemented this is by first sorting the harris corners by their strength in descending order. Then, for each point indexed by $i$, I found the set of all all points indexed by j that has $f(x_i) < cf(x_j)$ with $c=0.9$ being a robustness constant. Note that we only search $j<i$ since it is sorted. The closest point sets the suppression radius $r_i$. If no suppressing point is found (which is possible due to the robustness constant $c$ or it is the highest value point) the suppression radius is infinity. Finally, we sort the suppression radii in descending order and take the points with the top 500 $r_i$.

<div class="image-row2">
  <figure>
    <img src="/assets/images/project3/B1_i.png" alt="floor_rect_bilinear">
    <figcaption>Figure: Rows are two different images of my living room (related by perspective projection). Columns are original image, harris points in blue, and post adaptive nonmaximal suppression filtering</figcaption>
  </figure>
</div>

## B.2: Feature Extraction

To do this, we first create a blurred version of the WHOLE grayscale image using sk.filters.gaussian, with $\sigma = 5$. Then, we center a 40 x 40 window around each corner (post adaptive nonmaximal suppresion filtering), and sample every 5th pixel to extract a 8 x 8 feature descriptor. We also scale each pixel in grayscale ($z_i = \frac{x_i - \mu}{\sigma}$) for each window separately.

<div class="image-row2">
  <figure>
      <img src="/assets/images/project3/B2_with_annotations_i.png" alt="floor_rect_bilinear">
    <img src="/assets/images/project3/B2_feature_descriptors_i.png" alt="floor_rect_bilinear">
    <figcaption>Figure: Top is the original image, with red boxes and numbering highlighting a subset of selected 40 x 40 patches centered around previously detected points. Bottom shows those patches converted to features post blurring, downsampling to 8x8, and scaling. Note that "red" is brighter intensity, and blue is lower intensity, and image is clipped to 0 to 1 range.</figcaption>
  </figure>
</div>

## B.3: Feature Matching
To match features in both images, we pick one image as the reference, and build a nearest neighbor graph on the extracted features (each of which is a 64-dimensional vector) using sklearn.neighbors.NearestNeighbors. Then, for every the second image, we query each feature's distance to the previously built graph.

We then calculate the ratio of the 1st nearest neighbor distance over the second nearest neighbor distance. We assume that a good correspondence point has one good match in the other image (and thus the second and lower matches should have much higher euclidean distance than the first one). If this ratio is small (say < 0.5) we can say that correspondence is likely to be good. Thus, we take the 1st nearest neighbors for points below the cutoff to be good matches. 
<div class="image-row3a">
  <figure>
      <img src="/assets/images/project3/B3_thresholding_i.png" alt="floor_rect_bilinear">
    <figcaption>Figure: Comparing the ratio of the euclidean distances of 1st nearest neighbor over 2nd nearest neighbor for each point in an image. A threshold of 0.5 is chosen (all points below are considered to have their 1st nearest neighbor being a correspondence).</figcaption>
  </figure>
</div>
<div class="image-row2">
  <figure>
      <img src="/assets/images/project3/B3_i.png" alt="floor_rect_bilinear">
    <figcaption>Figure: Showing correspondence points found by feature matching in both images that are related by perspective projection.</figcaption>
  </figure>
</div>

Using this approach, we filtered 500 candidate points to ~76 matches. But there could still be outliers.

## B.4: RANSAC for Robust Homography

We can further filter the points above by using RANSAC. Simply put, we randomly take 4 (pairs) of correspondence points randomly and calculate $H$ the homography matrix (exactly) to go from image 1 to image 2. We then apply $H$ to all the correspondence points in image 1, and count how many of their mapped correspondences in image 2 are within $\epsilon =2$ pixels of the actually detected correspondences (in euclidean distance). We will call these points inliers. We repeatedly take a random 4 (pairs) and repeat this procedure n=10000 times. Finally, we take the iteration with the highest number of inlier points, and use those inliers to calculate the final $H$. 

<div class="image-row2">
  <figure>
      <img src="/assets/images/project3/B3_post_ransac_features_i.png" alt="floor_rect_bilinear">
    <figcaption>Figure: Showing inliner correspondence points found post RANSAC filtering</figcaption>
  </figure>
</div>

I end up using roughly 47 points as inliners from this procedure (down from 76 from the previous step).

We now can show the comparison between the images generated from the manual and autostitched mosaics. The top image is manually stitched (part A)


<div class="figure-split">
  <figure>
      <img src="/assets/images/project3/A4_post_blend_i.png" alt="floor_rect_bilinear">
  </figure>
  <figure>
      <img src="/assets/images/project3/B4_post_blend_i.png" alt="floor_rect_bilinear">
  </figure>
  <figcaption>Figure: Living room Left: Using manual correspondences to stitch mosaic. Right: Using automatic correspondences to stitch mosaic</figcaption>
</div>


<div class="figure-split">
  <figure>
      <img src="/assets/images/project3/A4_post_blend_church.png" alt="floor_rect_bilinear">
  </figure>
  <figure>
      <img src="/assets/images/project3/B4_post_blend_church.png" alt="floor_rect_bilinear">
  </figure>
  <figcaption>Figure: Religious school Left: Using manual correspondences to stitch mosaic. Right: Using automatic correspondences to stitch mosaic</figcaption>
</div>

<div class="figure-split">
  <figure>
      <img src="/assets/images/project3/A4_post_blend_street.png" alt="floor_rect_bilinear">
  </figure>
  <figure>
      <img src="/assets/images/project3/B4_post_blend_street.png" alt="floor_rect_bilinear">
  </figure>
  <figcaption>Figure: Oxford and Vine street Left: Using manual correspondences to stitch mosaic. Right: Using automatic correspondences to stitch mosaic</figcaption>
</div>
In general there isn't a huge difference between the two approaches. I did notice the religious school autostitching did a much better job of reducing the blurring artifacts in the brown grass/leaves (probably because I didn't put manual correspondences there). In general the manual correspondence were a huge pain to annotate. So I was much happier with part B :).

## B.5: Bells and Whistles (Rotational Invariance of Feature Descriptors)
First, I calculated the gradient orientation at every pixel location. I reused code from Bells and Whistles for the cameraman in Project 2 to do this: Essentially you blur ($\sigma = 3$) the image, convolve with both Dx and Dy filters, then calculate the gradient orientation via arctan.
<div class="image-row2">
  <figure>
      <img src="/assets/images/project3/B_BW_gradient_viz.png" alt="floor_rect_bilinear">
    <figcaption>Figure: Left: Blurred image of living room. Right: Calculating the gradient orientation at each location in the image by convolving the blurred image with Dx and Dy filters like in Project 2, then calculating the orientation using arctan. I reused code from Bells and Whistles for the cameraman in Project 2.</figcaption>
  </figure>
</div>

I realize I need to sample more than a 40 x 40 window. Specifically a 58 by 58 window to be safe. This ensures that regardless of how you rotate the patch, a 40 x 40 patch can't exceeds the bounds of the window and hit any padding. This value can be calculated by noting that the half-diagonal length of a 40 x 40 window is $\sqrt{20^2 + 20^2} = 28.28$. So $\text{ceil}(28.28)*2 = 58$ is a safe width/height to take the original patch at.

Then, I restrict my correspondences points accordingly (e.g. nothing within 58 pixels of the edge of the image). Then, for each extracted patch, you get an orientation $\theta$. This orientation is gotten from sampling the correspondence point exactly in the gradient image calculated previously, with the hopes that the heavily blurring smooths the gradient out. I then rotate the image $-\theta$ so that all gradient vectors point in the $\theta = 0$ direction. Then I crop to 40 x 40, and do the blurring/downsampling as before. I used the rotation function implemented using sk.transform.rotate for simplicity, though in theory I could adapt the inverse waring code in part A...

<div class="image-row2">
  <figure>
      <img src="/assets/images/project3/B_BW_patches_viz.png" alt="floor_rect_bilinear">
    <figcaption>Figure: Example of transforming a patch. The gradient vector is black. A window is sampled, reoriented, and then cropped/blurred/transform</figcaption>
  </figure>
</div>

This approach tries to have it so every patch is oriented in exactly the same direction (as defined by the local gradient). With this, the rest of my pipeline is the same. Results are shown below:


<div class="image-row2">
  <figure>
      <img src="/assets/images/project3/B_BW_rotation_invariance_post_blend_i.png" alt="floor_rect_bilinear">
      <img src="/assets/images/project3/B_BW_rotation_invariance_post_blend_church.png" alt="floor_rect_bilinear">
      <img src="/assets/images/project3/B_BW_rotation_invariance_post_blend_street.png" alt="floor_rect_bilinear">
    <figcaption>Figure: The end products using rotation invariant patches for my pipeline.</figcaption>
  </figure>
</div>

It doesn't change things too much. Interestingly, I can think of scenarios where rotation invariance may not always be good to implement. Like a feature rotated 60 degrees might have a different meaning...so perhaps it should be invariant to small rotations instead.








