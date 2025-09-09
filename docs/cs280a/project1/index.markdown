---
layout: cs180
---
## Project 1 Approach


In this project we aligned 3 plates, each corresponding to R,G,B, for a set of images from the Prokudin-Gorskii photo collection. 

The .jpg images aligned using either L2 or NCC metric using exhaustive search of -15 to +15 pixels and a predefined crop at the margins (10% of pixels from each border of the image). Note: Without the crop, things fell apart. 

All the .tif images (including two I downloaded from the collection) aligned using a pyramidal search with a max of 5 recursive calls (and an additional stopping condition if any image dimensions were < 100), where at each iteration I did a 2x2 average pooling. Note that if the image had an odd number for a dimension I clipped that dimension by 1 pixel. I used the NCC metric. The run time was roughly 40 seconds for each image.

I also implemented all bells and whistles and show the results for all images (+2 additional .tif images I pulled from the online photo collection: "Group of 11 children" and "In Iasnaia Poliana") at the bottom. Note that the emir image aligned using edge detection (see Bells and Whistles).

**Final images are at bottom**


## Bells and Whistles

### Alignment using edges
**Using edges as features** I needed to use edge-based alignment to get emir to line up using the NCC metric. I used the Roberts filter.
<div class="image-row">
  <figure>
    <img src="/assets/images/project1/bells_and_whistles/emir_blue.jpg" alt="ZG close">
    <figcaption>Blue channel only (shown as greyscale)</figcaption>
  </figure>
  <figure>
    <img src="/assets/images/project1/bells_and_whistles/EDGES_emir_blue.jpg" alt="A cat in grayscale">
    <figcaption>Edges from Robert Filter (faint but there)</figcaption>
  </figure>
</div>

### Other required bells and whistles

**Automated cropping**: This was down post alignment. I used the Canny edge detector to find edges, then the probabilistic Hough transform to try to identify separate contours segments. Then, I filtered for (nearly) horizontal line segments near the top or bottom of the image (within 10% of the borders) and nearly vertical line segments near the left or right of the image. Finally, I took the maximum x coordinate of the left hand side set, minimum x coordinate of the right hand side set, minimum y coordinate of the top segments, maximum y coordinate of the bottom segments, and used those coordinates to crop the image. This was an imperfect approach, and in a few cases (see emir.tif) slightly overcropped the image. I could not get this to work perfectly and did try alternatives like drawing contours but that failed. This part almost took me than the actual exhaustive search/pyramidal search assignment...I hope this isn't a trend for future projects :(

**Automatic contrasting** I took the 2nd and 98th percentiles of my image across all channels. Then I rescaled the intensities by clipping to that range (e.g. 2nd percentile is now 0... and 98th is now 1) and then all values inbetween are rescaled linearly to be in the same part of the range as originally. This was implemented using skimage.exposure.rescale_intensity. I might have overdone it with this one...

**Automated white balance** I tried to white balance the image by first finding candidate white parts of the image by taking the greyscale image intensity, taking the 99th percentile, and taking the RGB values for pixel locations that were above this cutoff (e.g. candidate white pixels). Then I used the mean across each channel to calculate a scaling factor for R,G,and B, and applied it to each channel.


**Better color mapping** I assumed that the channels were not perfectly R,G,B. My assumption was that each channel intensity probably has some bleed through into other channels. Accordingly, I used a handpicked linear transformation to transform each RGB pixel value (e.g. rather than multiply RGB by the identitiy matrix, I had each pixel contribute some intensity to the other channels). This worked worse than white balance so I didn't use it in the final images.

**Better features** See above section robert's filter


<div class="image-row">
  <figure>
    <img src="/assets/images/project1/bells_and_whistles/aligned_three_generations.jpg" alt="ZG close">
    <figcaption>Aligned (with edges)</figcaption>
  </figure>
  <figure>
    <img src="/assets/images/project1/bells_and_whistles/three_generations.jpg" alt="A cat in grayscale">
    <figcaption>Automatic Cropping</figcaption>
  </figure>
  <figure>
    <img src="/assets/images/project1/bells_and_whistles/remapped_three_generations.jpg" alt="A cat with a sepia filter">
    <figcaption>Color map remapping</figcaption>
  </figure>
  <figure>
    <img src="/assets/images/project1/bells_and_whistles/white_balance_three_generations.jpg" alt="A cat with inverted colors">
    <figcaption>White balance</figcaption>
  </figure>
  <figure>
    <img src="/assets/images/project1/final_outputs/final_three_generations.jpg" alt="A heavily blurred cat">
    <figcaption>Final (no color map remapping used; scaling added)</figcaption>
  </figure>
</div>

<div class="image-row">
  <figure>
    <img src="/assets/images/project1/bells_and_whistles/aligned_emir.jpg" alt="ZG close">
    <figcaption>Aligned (with edges)</figcaption>
  </figure>
  <figure>
    <img src="/assets/images/project1/bells_and_whistles/emir.jpg" alt="A cat in grayscale">
    <figcaption>Automatic Cropping</figcaption>
  </figure>
  <figure>
    <img src="/assets/images/project1/bells_and_whistles/remapped_emir.jpg" alt="A cat with a sepia filter">
    <figcaption>Color map remapping</figcaption>
  </figure>
  <figure>
    <img src="/assets/images/project1/bells_and_whistles/white_balance_emir.jpg" alt="A cat with inverted colors">
    <figcaption>White balance</figcaption>
  </figure>
  <figure>
    <img src="/assets/images/project1/final_outputs/final_emir.jpg" alt="A heavily blurred cat">
    <figcaption>Final (no color map remapping used; scaling added)</figcaption>
  </figure>
</div>

## Final outputs

All displacements are written in terms of how much the channel needs to be shifted to align with blue and in the format (dx,dy) (dx = column shift, dy= row shift).
<div class="image-row">
  <figure>
    <img src="/assets/images/project1/final_outputs/final_tobolsk.jpg" alt="ZG close">
    <figcaption>Tobolsk, G: [3, 3], R: [3, 6]</figcaption>
  </figure>
  <figure>
    <img src="/assets/images/project1/final_outputs/final_monastery.jpg" alt="ZG close">
    <figcaption>Monastery, G: [2, -3], R: [2, 3]</figcaption>
  </figure>
  <figure>
    <img src="/assets/images/project1/final_outputs/final_cathedral.jpg" alt="ZG close">
    <figcaption>Cathedral, G: [2, 5], R: [3, 12]</figcaption>
  </figure>
  <figure>
    <img src="/assets/images/project1/final_outputs/final_church.jpg" alt="ZG close">
    <figcaption>Church, G: [4, 25], R: [-4, 58]</figcaption>
  </figure>
  <figure>
    <img src="/assets/images/project1/final_outputs/final_emir.jpg" alt="ZG close">
    <figcaption>Emir, G: [24, 49], R: [40, 107]</figcaption>
  </figure>
  <figure>
    <img src="/assets/images/project1/final_outputs/final_group_of_11_children.jpg" alt="ZG close">
    <figcaption>Group of 11 children, G: [36,60]; R: [49,129]</figcaption>
  </figure>
  <figure>
    <img src="/assets/images/project1/final_outputs/final_harvesters.jpg" alt="ZG close">
    <figcaption>Harvesters, G: [18, 60], R: [13, 124]</figcaption>
  </figure>
  <figure>
    <img src="/assets/images/project1/final_outputs/final_icon.jpg" alt="ZG close">
    <figcaption>Icon, G: [17, 41], R: [23, 90]</figcaption>
  </figure>
  <figure>
    <img src="/assets/images/project1/final_outputs/final_In Iasnaia Poliana.jpg" alt="ZG close">
    <figcaption>In Iasnaia Poliana, G: [-2, 43]; R: [9,97]</figcaption>
  </figure>
  <figure>
    <img src="/assets/images/project1/final_outputs/final_italil.jpg" alt="ZG close">
    <figcaption>Italil, G: [22, 38], R: [36, 77]</figcaption>
  </figure>
  <figure>
    <img src="/assets/images/project1/final_outputs/final_lastochikino.jpg" alt="ZG close">
    <figcaption>Lastochikino, G: [-1, -3], R: [-8, 76]</figcaption>
  </figure>
  <figure>
    <img src="/assets/images/project1/final_outputs/final_lugano.jpg" alt="ZG close">
    <figcaption>Lugano, G: [-17, 41], R: [-29, 92]</figcaption>
  </figure>
  <figure>
    <img src="/assets/images/project1/final_outputs/final_melons.jpg" alt="ZG close">
    <figcaption>Melons, G: [10, 80], R: [13, 177]</figcaption>
  </figure>
  <figure>
    <img src="/assets/images/project1/final_outputs/final_self_portrait.jpg" alt="ZG close">
    <figcaption>Self portrait, G: [29, 78], R: [37, 175]</figcaption>
  </figure>
  <figure>
    <img src="/assets/images/project1/final_outputs/final_siren.jpg" alt="ZG close">
    <figcaption>Siren, G: [-6, 49], R: [-24, 96]</figcaption>
  </figure>
  <figure>
    <img src="/assets/images/project1/final_outputs/final_three_generations.jpg" alt="ZG close">
    <figcaption>Three generations, G: [12, 54], R: [8, 111]</figcaption>
  </figure>
</div>