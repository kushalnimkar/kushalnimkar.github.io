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


Now, given a set of keypoints $(x_i,y_i)$ with correspondences $(x'_i,y'_i)$ we can repeate A (two rows for each datapoint) and b (two rows for each datapoint) to set this up as a regression problem. I used np.linalg.lstsq(A,b) to actually find $h$.

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
    <figcaption>Figure: Monitor rectification. Top: Original image with keypoints in red. Middle: Warping and using nearest-neighbor interpolation for color. Bottom: Warping and using nearest-neighbor bilinear interpolation for color </figcaption>
  </figure>
</div>
Another example is shown below
<div class="image-row3a">
  <figure>
    <img src="/assets/images/project3/A2_correspondences_s_cropped.png" alt="floor_rect_bilinear">
    <img src="/assets/images/project3/A3_rectification_s_NN.png" alt="floor_rect_bilinear">
    <img src="/assets/images/project3/A3_rectification_s_bilinear.png" alt="floor_rect_bilinear">
    <figcaption>Figure: Floor tile rectification. Top: Original image with keypoints in red. Middle: Warping and using nearest-neighbor interpolation for color. Bottom: Warping and using nearest-neighbor bilinear interpolation for color </figcaption>
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

In theory, the correct $f$ is the same for all images (since I used the same camera and didn't use zoom). I thought $f=1200$ looked good and justified it as follows: in theory two images projected to the cylinder aroudn the same center of projection (COP) should be related by translation only. Therefore certain objects like the piano in both images at the correct $f$ should have the same orientation (e.g. lines are at similar angles).

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

