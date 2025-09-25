---
layout: cs180
---
# CS280A Project 2, Kushal Nimkar
## 1.1
{% include mathjax.html %}

I implemented convolution for both padding settings ('full' and 'same'). In this case boundaries are just symmetrically padded with zeros, in a way that in the 'full' case allows for all possible combinations (that are not ALL padding entries) of the image and filter, while the 'same' case (symmetrically) pads in a way so that the output image size is the same as the input size.
<div class="image-row2">
  <figure>
    <img src="/assets/images/project2/1.1_code.png" alt="1.1_code">
    <figcaption>Figure: Code snippet for convolution. Both are identical to scipy.signal.convolve2d for the mode='full' and mode='same' settings </figcaption>
  </figure>
</div>

The blur is more noticeable when reading the text at the bottom left of my shirt (but it is there). Notice that the derivative filter are signed quantities since they represent the direction where pixel intensities increase. Interestingly, for my image (which was pretty small) the 4 for loop version is faster than the 2 for loop version. This is probably because my image is small enough that maybe the overhead of creating a view in the 2 loop case offsets things? For larger images when I tested the 2 loop version is faster. Both are slower than scipy regardless (likely because it is compiled to use C++ for the for loops).
<div class="image-row2">
  <figure>
    <img src="/assets/images/project2/1_1_box_filt.png" alt="1.1_box">
    <figcaption>Figure: Applying the box filter convolution. </figcaption>
  </figure>
  <figure>
    <img src="/assets/images/project2/1_1_box_dx.png" alt="1.1_dx">
    <figcaption>Figure: Applying the dx filter. </figcaption>
  </figure>
  <figure>
    <img src="/assets/images/project2/1_1_box_dy.png" alt="1.1_dy">
    <figcaption>Figure: Applying the dy filter. Notice the eyebrows are captured. </figcaption>
  </figure>
</div>

## 1.2 
I convolved the filters with the cameraman image, calculated the magnitude of the gradient, and picked a threshold to call a "real edge". This is not foolproof: you either miss some aspects of the cameraman, or capture some of the background. I prioritized getting the cameraman overall correct.
<div class="image-row2">
  <figure>
    <img src="/assets/images/project2/1.2_edge.png" alt="1.2_camera">
    <figcaption>Figure: Applying derivative convolutions. </figcaption>
  </figure>
</div>

## 1.3
Here I define $\circledast$ to be the convolution operator, $G$ to be a gaussian blur filter with $k=7, \sigma = 3$, $f $ to be the original image, and $df $ to be either the dx or dy filters (this is a bit of abuse of notation...) which you can take the magnitude of to get gradients. 

In theory, the convolution operator is commutative so the order of filters does not matter. In practice the images are very slightly different because I end up using the mode="same" scipy.signal.convolve2d operator for the final images which doesn't perfectly preserve this property.

I first show the filters.

<div class="image-row2">
  <figure>
    <img src="/assets/images/project2/1.3_filts.png" alt="1.3_filt">
    <figcaption>Figure: Gaussian blur filter, and what it looks like when convolved with the Dx and Dy filters (mode='full')</figcaption>
  </figure>
</div>

Then show the results
<div class="image-row2">
  <figure>
    <img src="/assets/images/project2/1.3_convolution_order.png" alt="1.3_convo">
    <figcaption>Figure: Comparing results from flipping the order of the filters. </figcaption>
  </figure>
</div>

Note that the edge detection looks much better than the finite difference method since the blur helps remove the noise that causes the gradient intensity to get too whacky at a local scale.

## 1.3 Bells and Whistles
I used np.arctan2 to convert the images filtered with dy and dx to an image with pixels representing gradient vector angles $[-\pi,\pi]$. I converted to $[0,2\pi]$ and created a HSV image format where $H=\theta$, $S = 1$, and $V= |\nabla f|$. I then created a cyclic colorwheel with 0 and 2 pi as the end points (very annoying) and added it as a colorbar. Note that for $\theta$ because "up" is technically in the negative direction, I flipped the y coordinate sign when inputting it to arctan2 to make sure "up" is $\pi/2$. 
<div class="image-row2">
  <figure>
    <img src="/assets/images/project2/1.3_bells_and_whistles.png" alt="1.3_convo">
    <figcaption>Figure: Visualizing gradient direction and magnitude using the HSV colorspace (color/hue = direction, value = magnitude) </figcaption>
  </figure>
</div>

## 2.1
 Let $I$ be the image and $G$ be a gaussian blur. We first show you can subtract out low frequencies with a single filter operation.
\begin{align}
 I_{sharpened} &\equiv I - (I \circledast G)\newline
 &= I \circledast (E_{mm}- G)\newline
 &\equiv I \circledast F_{highfrequency}
\end{align} 
where $F_{highfrequency} \equiv (E_{mm}-G)$, and $E_{mm}$ is the 2D matrix with 0 everywhere but the center point, which is 1. The first line is the definition, second line is the distributive property, and last line is by our definition.

Similarly, we can add these high frequencies back to the image (as a single filter operation) as follows. Let $\alpha$ be some parameter to dictate how much high frequency component you want to add. then

\begin{align}
I_{unsharpmask} &\equiv I + \alpha I_{sharpened}\newline
&= I + \alpha (I - (I \circledast G))\newline
&= I + \alpha I -  I \circledast \alpha G\newline
&= I \circledast ((1+\alpha) E_{mm} - \alpha G)\newline
&\equiv I \circledast F_{unsharpmask}
\end{align}
where $F_{unsharpmask} \equiv (1+\alpha) E_{mm} - \alpha G$. Again, we used the distributive property and commutative property in the proof above.

We apply both filters to the Taj Mahal picture below ($ksize=7, \sigma= 2$)

<div class="image-row2">
  <figure>
    <img src="/assets/images/project2/2.1_example.png" alt="2.1">
    <figcaption>Figure: Visualizing the Taj Mahal. Original (left), high frequency only (middle), and adding the high frequency back the image (right). $\alpha=1$</figcaption>
  </figure>
</div>
We can increase $\alpha$ to add even more of the high frequency content $\alpha=2$.
<div class="image-row2">
  <figure>
    <img src="/assets/images/project2/2.1_example_taj_alpha2.png" alt="2.1_2">
    <figcaption>Figure: Visualizing the Taj Mahal. Original (left), high frequency only (middle), and adding the high frequency back the image (right).$\alpha=2$</figcaption>
  </figure>
</div>
We can also reduce the effect a bit by $\alpha=0.5$.
<div class="image-row2">
  <figure>
    <img src="/assets/images/project2/2.1_example_taj_alpha0.5.png" alt="2.1_1/2">
    <figcaption>Figure: Visualizing the Taj Mahal. Original (left), high frequency only (middle), and adding the high frequency back the image (right). $\alpha=0.5$</figcaption>
  </figure>
</div>

I can also test this on a new image.

<div class="image-row2">
  <figure>
    <img src="/assets/images/project2/2.1_example2.png" alt="pet">
    <figcaption>Figure: Visualizing the supposed "World's Ugliest Dog" of 2025, Petunia. Original (left), high frequency only (middle), and adding the high frequency back the image (right). $\alpha=1,ksize=10,\sigma=3$. Notice the skin folds are more pronounced. Also this dog is not that ugly...Image source: NYT</figcaption>
  </figure>
</div>

We can try another example

<div class="image-row2">
  <figure>
    <img src="/assets/images/project2/2.1_example3.png" alt="chicken">
    <figcaption>Figure: Chicken dinner. Original (left),high frequency only (middle), adding high frequency back. Note the meat texture is sharper (but not the best effect) </figcaption>
  </figure>
</div>

Finally, we can try to blur and resharpen our previous dog photo. Note that because unsharp mask is not the mathematical inverse to gaussian blur, it makes sense we do not recover the original image exactly. But it does a decent job of getting some of the sharpness back.

<div class="image-row2">
  <figure>
    <img src="/assets/images/project2/2.1_blur_resharpen.png" alt="pet_resharpen">
    <figcaption>Figure: Blurring and resharpening Petunia. Original (left), blurred ($ksize=10,\sigma=3$ for both blur and sharpening operations) </figcaption>
  </figure>
</div>

## 2.2
For Derek and Nutmeg, we show the original images
<div class="image-row">
  <figure>
    <img src="/assets/images/project2/2.2_original_derek_nutmeg.jpg" alt="align_derek_nutmeg">
    <figcaption>Figure: Derek and Nutmeg originally. </figcaption>
  </figure>
</div>
Then align them
<div class="image-row">
  <figure>
    <img src="/assets/images/project2/2.2_aligned_derek_nutmeg.jpg" alt="align_derek_nutmeg">
    <figcaption>Figure: Derek and Nutmeg postalignment </figcaption>
  </figure>
</div>
Then we low pass Derek with Gaussian blur ($\sigma=7, ksize=43$) and high pass Nutmeg with our previously defined unsharp mask filter ($\sigma = 9, ksize=55$)
<div class="image-row">
  <figure>
    <img src="/assets/images/project2/2.2_high_and_low.png" alt="hl">
    <figcaption>Figure: Derek and Nutmeg post filtering </figcaption>
  </figure>
</div>

We can also compare the original frequency domains to the post transform domains.
<div class="image-row">
  <figure>
    <img src="/assets/images/project2/2.2_fourier_highpass.png" alt="hp">
    <figcaption>Figure: Original and high pass frequency domains for Nutmeg </figcaption>
  </figure>
</div>
<div class="image-row">
  <figure>
    <img src="/assets/images/project2/2.2_fourier_lowpass.png" alt="ld">
    <figcaption>Figure: Figure: Original and low pass frequency domains for Derek </figcaption>
  </figure>
</div>

And finally we show the two images we get when we average the low pass and high pass images, and the associated frequency domain (which has both high and low now)

<div class="image-row">
  <figure>
    <img src="/assets/images/project2/2.2_fourier_hybrid.png" alt="fh">
    <figcaption>Figure: Hybrid image frequency domain. </figcaption>
  </figure>
</div>
<div class="image-row2">
  <figure>
    <img src="/assets/images/project2/2.2_color1False_color2False.png" alt="no_color_hybrid">
    <figcaption>Figure: Hybrid Derek and Nutmeg (both gray)</figcaption>
  </figure>
</div>

I also tried this effect with a few other images. Below is making hybrid images of two former presidents. This one is good overall, but their faces aren't quite the same size which makes the far away image not as clean.
<div class="image-row2">
  <figure>
    <img src="/assets/images/project2/2.2_obama_wbush.png" alt="obama_bush">
    <figcaption>Figure: Obama (low frequency, grey) + George W Bush (high, colored). Color strategy is discussed in next B&W section after this.</figcaption>
  </figure>
</div>
I tried this with my partner. Like Professor Efros says, it works better if the hair aligns (which it does NOT in this case in many ways). 
<div class="image-row2">
  <figure>
    <img src="/assets/images/project2/2.2_zg_kn.png" alt="kn_zg">
    <figcaption>Figure: Me (low frequency) and partner (high frequency). Next time I will shave and get a haircut</figcaption>
  </figure>
</div>

## 2.2 Bells and Whistles

We can try coloring either Derek or Nutmeg (or both). I do this by repeating the filtering for each color channel individually in the case of color. I noticed it is better when the high frequency image only is colored (because it makes the effect more dramatic). I think it is because you can see colors more easily when you are close, which is also when you can see high frequency. But color in low frequency is not as noticeable. I also tried this on Obama + George W Bush in the previous section.
<div class="image-row2">
  <figure>
    <img src="/assets/images/project2/2.2_color1True_color2False.png" alt="no_color_hybrid">
    <figcaption>Figure: Hybrid Derek and Nutmeg (Left and Middle are originals, Right: Derek in color, Nutmeg in gray)</figcaption>
  </figure>
</div>

<div class="image-row2">
  <figure>
    <img src="/assets/images/project2/2.2_color1False_color2True.png" alt="no_color_hybrid">
    <figcaption>Figure: Hybrid Derek and Nutmeg (Left and Middle are originals, Right: Derek in gray, Nutmeg in color)</figcaption>
  </figure>
</div>

<div class="image-row2">
  <figure>
    <img src="/assets/images/project2/2.2_color1True_color2True.png" alt="no_color_hybrid">
    <figcaption>Figure: Hybrid Derek and Nutmeg (Left and Middle are originals,Right: both colored)</figcaption>
  </figure>
</div>


## 2.3 + 2.4
First we create the gaussian stacks for both the apple and orange. I blurred each color channel independently.
<div class="image-row2">
  <figure>
    <img src="/assets/images/project2/2.3_apple_gstack.png" alt="apple_gausian">
    <figcaption>Figure: From left to right, increasing gaussian blur on the apple. I used 9 levels with $\sigma_{arr} = [3,4,5,6,7,8,9,11,13]$ (different for each level) and $ksize=6\sigma + 1$ </figcaption>
  </figure>
</div>

<div class="image-row2">
  <figure>
    <img src="/assets/images/project2/2.3_orange_gstack.png" alt="orange_gaussian">
    <figcaption>Figure: From left to right, increasing gaussian blur on the orange. I used 9 levels with $\sigma_{arr} = [3,4,5,6,7,8,9,11,13]$ (different for each level) and $ksize=6\sigma + 1$</figcaption>
  </figure>
</div>

Then we create the laplacian stacks by taking the difference between consecutive levels in the gaussian stack (e.g. subtracting (layer[i] - layer[i+1])

<div class="image-row2">
  <figure>
    <img src="/assets/images/project2/2.3_apple_lstack.png" alt="apple_laplac">
    <figcaption>Figure: From left to right, levels of the laplacian stack for the apple</figcaption>
  </figure>
</div>

<div class="image-row2">
  <figure>
    <img src="/assets/images/project2/2.3_orange_lstack.png" alt="orange_laplac">
    <figcaption>Figure: From left to right, levels of the laplacian stack for the orange</figcaption>
  </figure>
</div>

We can recreate the outcomes of Figure 3.42, incorporating an initially binary mask, which is blurred increasingly at each level $\sigma_{arr} = (6,8,10,12,14,16,18,20,22)$ and $ksize=6\sigma + 1$

<div class="image-row2">
  <figure>
    <img src="/assets/images/project2/2.3_342.png" alt="342">
    <figcaption>Figure: The first 3 rows are the laplacian stack with the blurred mask applied, sampled from levels 1,3, and 5 (indexed starting at 1). The final row is adding all the levels together (with the final gaussian blurred low frequency image as well). The first column is for the apple, the second column is for the orange, and the last column is from adding the orange and apple columns together.</figcaption>
  </figure>
</div>
Finally, we can show the results.
<div class="image-row2">
  <figure>
    <img src="/assets/images/project2/2.3_final_orapple.png" alt="orapple">
    <figcaption>Figure: Left: Original apple, Middle: Original orange, Right: Blended to get orapple</figcaption>
  </figure>
</div>

We can also create blends for a few irregular masks created by the lasso tool in Adobe Photoshop.
<div class="image-row2">
  <figure>
    <img src="/assets/images/project2/2.4_georgie_thomas.png" alt="gt">
    <figcaption>Figure: Blending a dog and toy</figcaption>
    <img src="/assets/images/project2/2.4_bottle_sticker.png" alt="bs">
    <figcaption>Figure: Blending a water bottle sticker onto a thermos</figcaption>
  </figure>
</div>

## Bells and Whistles
Here I experimented with trying to make the colors in the orapple more distinct. I tried to make the apple image more "red" by increasing the intensity of the R color channel by a factor of 2. However, the blend then creates a noticeable seam. To try to fix this, I also changed the mask to not be a blurred step function, but a blurred window that has a linear change in intensity starting at some threshold.
<div class="image-row2">
  <figure>
    <img src="/assets/images/project2/2.4_bells_and_whistles.png" alt="2.4bw">
    <figcaption>Figure: Left: Original, Middle: Increasing R color channel in apple, Right: Increasing R color channel in apple and changing blur to a linear window between 17%-83% percentile of the image columns.</figcaption>
  </figure>
</div>

I think the colors are a bit distinct but I make the orange a bit red as well...