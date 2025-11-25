---
layout: cs180
---
# CS280A Project 5, Kushal Nimkar
# Part 0
{% include mathjax.html %}

I tested the stage 1 and stage 2 DeepFloyd Models on HuggingFace, using num_inference_steps=20 and 30 for my own custom prompts. I used a random seed of 100. The folowing is a list of prompts I generated:

['a high quality picture','an oil painting of a lush mountain village','a photo of the California coast','a photo of a woman','a photo of an Italian chef','a photo of a cat','an oil painting of people at a concert', 'an oil painting of a baby boy','a lithograph of a glacier','a lithograph of a white horse','a man wearing a hat','a high quality photo','a cargo ship','a pen','']

Stage 1 model results (for two num_inference_steps):
<div class="image-row3a">
  <figure>
    <img src="/assets/images/project5/5a/0_n20_stage1.png" alt="0">
    <figcaption>Figure: Generated images given text prompts to DeepFloyd  (stage1 model, num_inference_steps=20) </figcaption>
  </figure>
  <figure>
    <img src="/assets/images/project5/5a/0_n30_stage1.png" alt="0">
    <figcaption>Figure: Generated images given text prompts to DeepFloyd (stage1 model, num_inference_steps=30) </figcaption>
  </figure>
</div>
Stage 2 model results (for two num_inference_steps):
<div class="image-row3a">
  <figure>
    <img src="/assets/images/project5/5a/0_n20_stage2.png" alt="0">
    <figcaption>Figure: Generated images given text prompts to DeepFloyd  (stage2 model, num_inference_steps=20) </figcaption>
  </figure>
  <figure>
    <img src="/assets/images/project5/5a/0_n30_stage2.png" alt="0">
    <figcaption>Figure: Generated images given text prompts to DeepFloyd (stage2 model, num_inference_steps=30) </figcaption>
  </figure>
</div>

In this case, it seems both num_inference_steps produce reasonable looking images. However, I am sure if I picked some small num_inference_steps (like 1 or 2), the image will look bad, and increasing the num_inference_steps will increase the quality of the images.

# Part 1
## 1.1 Implementing the forward process

The image at $x_t$ is defined as adding noise to the original image $x_0$ by following a schedule dictated by $\bar{\alpha}_t$
\begin{equation}
    x_t = \sqrt{\bar{\alpha_t}}x_0 + \sqrt{1-\bar{\alpha_t}}\epsilon
\end{equation}
where $\epsilon \sim N(0,1)$. This is implemented as below:

<div class="image-row3a">
  <figure>
    <img src="/assets/images/project5/5a/1.1_forward.png" alt="0">
    <figcaption>Figure: Forward noising implementation in Python</figcaption>
  </figure>
</div>

Below is the campanile with noise added to it for various stages

<div class="image-row3a">
  <figure>
    <img src="/assets/images/project5/5a/1_1.png" alt="0">
    <figcaption>Figure: Forward noising implementation on the campanile for t =250,500, and 750 </figcaption>
  </figure>
</div>

## 1.2 Classical Denoising
I attempted to denoise the noisy campanile images with Gaussian Blur (kernel_size=1,, sigma =2) as implemented in torchvision.transforms.functional
<div class="image-row3a">
  <figure>
    <img src="/assets/images/project5/5a/1_1.png" alt="0">
    <img src="/assets/images/project5/5a/1_2.png" alt="0">
    <figcaption>Figure: Top: Forward noising implementation on the campanile for t =250,500, and 750; Bottom: attempting to denoise the noisy images </figcaption>
  </figure>
</div>

Blurring does not work!

## 1.3 One-step Denoising

We can add noise to our images using the forward function we implemented, then pass it to the stage1 U-net denoiser to get the estimated noise in the image, $\epsilon$. Then, use the forward equation in part 1.1 to solve for $x_0$. Specifically, the equation is

\begin{equation}
x_0 = \frac{x_t-\sqrt{1-\bar{\alpha_t}}\epsilon}{\sqrt{\bar{\alpha_t}}}
\end{equation}

where $x_t$ is the noisy image at timestep $t$, $\epsilon$ is the noise predicted by the network, and $\bar{\alpha}_t$ is the relevant alpha value at that timesteps.

Note that the inputs of the model have pixels in the range of [-1,1], and I transform the outputs of the model back to [0,1] space for visualization in matplotlib.


<div class="image-row3a">
  <figure>
    <img src="/assets/images/project5/5a/1_3.png" alt="0">
    <figcaption>Figure: One stop denoising of various levels of noise added to the campanile.</figcaption>
  </figure>
</div>

The denoiser does much better than gaussian blur (which makes sense as it was trained on noise of this form). Note that as more noise is added, the denoiser creates a slightly different looking campanile...


## 1.4 Iterative Denoising

Here, we created strided timesteps of the form [900,870,840...,60,30,0], and fed that to the stage1_scheduler. We then implement the iterative denoise function 
<div class="image-row3a">
  <figure>
    <img src="/assets/images/project5/5a/1.3_strided_timesteps.png" alt="0">
    <figcaption>Figure: Creating strided timesteps</figcaption>
  </figure>
</div>

<div class="image-row3a">
  <figure>
    <img src="/assets/images/project5/5a/1.3_iterative_denoise.png" alt="0">
    <figcaption>Figure: Implementing the iterative denoising</figcaption>
  </figure>
</div>

whose mathematical equation is
<div class="image-row3a">
  <figure>
    <img src="/assets/images/project5/5a/iterative_denoise_eqn.png" alt="0">
    <figcaption>Figure: Implementing the iterative denoising</figcaption>
  </figure>
</div>
with $x_t$ being the noisy image at timestep t, $x_{t'}$ being the noisy image at timestep t', $\bar{\alpha_t}$ is the relevant timestep alpha from alphas_cumprod, $\alpha_t=\frac{\bar{\alpha_t}}{\bar{\alpha_{t'}}}$, $\beta = 1- \alpha$, and  $v_{\sigma}$ is the variance in the noise (also outputted by the denoiser).

We then visualize the iterative denoising every 5 iterations, where each iteration is 30 timesteps, starting at i_start = 10 (so t=690, 540...)

<div class="side-by-side">
  <figure>
    <img src="/assets/images/project5/5a/1_4_clean_690.png" alt="0">
    <img src="/assets/images/project5/5a/1_4_clean_540.png" alt="0">
    <img src="/assets/images/project5/5a/1_4_clean_390.png" alt="0">
    <img src="/assets/images/project5/5a/1_4_clean_240.png" alt="0">
    <img src="/assets/images/project5/5a/1_4_clean_90.png" alt="0">
    <figcaption>Figure: Implementing the iterative denoising</figcaption>
  </figure>
</div>

The original campanile:
<div class="image-row3a">
  <figure>
    <img src="/assets/images/project5/5a/campanile_downsample.png" alt="0">
    <figcaption>Figure: Original campanile </figcaption>
  </figure>
</div>

The final results comparing the iterative, one step, gaussian blur approaches on the original are below

<div class="side-by-side">
  <figure>
    <img src="/assets/images/project5/5a/1_4_onestep.png" alt="0">
    <img src="/assets/images/project5/5a/1_4_iterative.png" alt="0">
    <img src="/assets/images/project5/5a/1_4_blurred.png" alt="0">
    <figcaption>Figure: Comparing denoising approaches on the original campanile.</figcaption>
  </figure>
</div>



## 1.5 Diffusion Model Sampling

We can sample images from the diffusion model by creating random noise from a gaussian normal random, and then denoising them (assuming i_start=0 aka t=990).

<div class="side-by-side">
  <figure>
    <img src="/assets/images/project5/5a/1_5_n0.png" alt="0">
    <img src="/assets/images/project5/5a/1_5_n1.png" alt="0">
    <img src="/assets/images/project5/5a/1_5_n2.png" alt="0">
    <img src="/assets/images/project5/5a/1_5_n3.png" alt="0">
    <img src="/assets/images/project5/5a/1_5_n4.png" alt="0">
    <figcaption>Figure: 5 images sampled from denoising random noise generated from a standard normal gaussian</figcaption>
  </figure>
</div>

Some images are a bit weird...

## 1.6 Classifier-Free Guidance

We can create better looking images by using classifier free guidance to create a noise $\epsilon$ from noise $\epsilon_c$ and $\epsilon_u$ generated from a conditional and unconditional prompts "a high quality photo" and "",respectively.


\begin{equation}
\epsilon = \epsilon_u + \gamma(\epsilon_c - \epsilon_u)
\end{equation}
where $\gamma = 7$. This requires a small update to our function 

<div class="image-row3a">
  <figure>
    <img src="/assets/images/project5/5a/1_6_code.png" alt="0">
    <figcaption>Figure: Implementing classifier-free guidance</figcaption>
  </figure>
</div>

Below are some samples from denoising standard gaussian noise

<div class="side-by-side">
  <figure>
    <img src="/assets/images/project5/5a/1_6_n0.png" alt="0">
    <img src="/assets/images/project5/5a/1_6_n1.png" alt="0">
    <img src="/assets/images/project5/5a/1_6_n2.png" alt="0">
    <img src="/assets/images/project5/5a/1_6_n3.png" alt="0">
    <img src="/assets/images/project5/5a/1_6_n4.png" alt="0">
    <figcaption>Figure: 5 images sampled from standard gaussian random noise using classifier free guidance and denoising</figcaption>
  </figure>
</div>

The image are better, but the diversity is lower (a lot of women/people)

## 1.7 Image-to-image translation
Here we edited the campanile and a few other of my photos by adding noise and denoising at different noise levels (i_start values) of [1,3,5,7,10,20]. We show results below:

Campanile:
<div class="image-row3a">
  <figure>
    <img src="/assets/images/project5/5a/downsample_campanile.jpg" alt="0">
    <figcaption>Figure: Original campanile</figcaption>
  </figure>
</div>


<div class="side-by-side">
  <figure>
    <img src="/assets/images/project5/5a/1_7_start_1_campanile.jpg" alt="0">
    <img src="/assets/images/project5/5a/1_7_start_3_campanile.jpg" alt="0">
    <img src="/assets/images/project5/5a/1_7_start_5_campanile.jpg" alt="0">
    <img src="/assets/images/project5/5a/1_7_start_7_campanile.jpg" alt="0">
    <img src="/assets/images/project5/5a/1_7_start_10_campanile.jpg" alt="0">
    <img src="/assets/images/project5/5a/1_7_start_20_campanile.jpg" alt="0">
    <figcaption>Figure: Editing the campanile by adding various noise levels. The first level (1) is the top left corner, and the levels increase as you go left to right, and then up to down (e.g. top right corner is level 3, bottom right corner is level=20))</figcaption>
  </figure>
</div>

Mountain:
<div class="image-row3a">
  <figure>
    <img src="/assets/images/project5/5a/downsample_mountain.jpg" alt="0">
    <figcaption>Figure: Original mountain</figcaption>
  </figure>
</div>

<div class="side-by-side">
  <figure>
    <img src="/assets/images/project5/5a/1_7_start_1_mountain.jpg" alt="0">
    <img src="/assets/images/project5/5a/1_7_start_3_mountain.jpg" alt="0">
    <img src="/assets/images/project5/5a/1_7_start_5_mountain.jpg" alt="0">
    <img src="/assets/images/project5/5a/1_7_start_7_mountain.jpg" alt="0">
    <img src="/assets/images/project5/5a/1_7_start_10_mountain.jpg" alt="0">
    <img src="/assets/images/project5/5a/1_7_start_20_mountain.jpg" alt="0">
    <figcaption>Figure: Editing a mountain photo by adding various noise levels. The first level (1) is the top left corner, and the levels increase as you go left to right, and then up to down (e.g. top right corner is level 3, bottom right corner is level=20)</figcaption>
  </figure>
</div>
Stanley Hall:

<div class="image-row3a">
  <figure>
    <img src="/assets/images/project5/5a/downsample_stanley_outside.jpg" alt="0">
    <figcaption>Figure: Original stanley</figcaption>
  </figure>
</div>
<div class="side-by-side">
  <figure>
    <img src="/assets/images/project5/5a/1_7_start_1_stanley_outside.jpg" alt="0">
    <img src="/assets/images/project5/5a/1_7_start_3_stanley_outside.jpg" alt="0">
    <img src="/assets/images/project5/5a/1_7_start_5_stanley_outside.jpg" alt="0">
    <img src="/assets/images/project5/5a/1_7_start_7_stanley_outside.jpg" alt="0">
    <img src="/assets/images/project5/5a/1_7_start_10_stanley_outside.jpg" alt="0">
    <img src="/assets/images/project5/5a/1_7_start_20_stanley_outside.jpg" alt="0">
    <figcaption>Figure: Editing a photo of Stanley hall by adding various noise levels. The first level (1) is the top left corner, and the levels increase as you go left to right, and then up to down (e.g. top right corner is level 3, bottom right corner is level=20)</figcaption>
  </figure>
</div>


## 1.7.1 Editing Hand-Drawn and Web Images
Here, we experiment with an image from the web, and two hand-drawn images, and edit it in the same procedure as before.

<div class="image-row3a">
  <figure>
    <img src="/assets/images/project5/5a/downsample_gemini_stick.jpg" alt="0">
    <figcaption>Figure: Original stick figure generated by Google Gemini</figcaption>
  </figure>
</div>
<div class="image-row2">
  <figure>
    <img src="/assets/images/project5/5a/1_7_1gemini_stick.jpg" alt="0">
    <figcaption>Editing the stick figure at different noise levels (notice the hat and facial features change)</figcaption>
  </figure>
</div>
Two hand-drawn images are shown below.

<div class="image-row3a">
  <figure>
    <img src="/assets/images/project5/5a/downsample_house_v2.jpg" alt="0">
    <figcaption>Figure: Original house drawn by me</figcaption>
  </figure>
</div>
<div class="image-row2">
  <figure>
    <img src="/assets/images/project5/5a/1_7_1house_v2.jpg" alt="0">
    <figcaption>Editing the house at different noise levels (notice the window changes)</figcaption>
  </figure>
</div>


<div class="image-row3a">
  <figure>
    <img src="/assets/images/project5/5a/downsample_pizza.jpg" alt="0">
    <figcaption>Figure: Original pizza slice drawn by me</figcaption>
  </figure>
</div>
<div class="image-row2">
  <figure>
    <img src="/assets/images/project5/5a/1_7_1pizza.jpg" alt="0">
    <figcaption>Editing the pizza at different noise levels (notice the crust color changes)</figcaption>
  </figure>
</div>



At some of the higher levels, you can see small edits being made to the details of the images. At the lower levels, it looks completely different.


## 1.7.2 Inpainting

We implement inpainting a select ROI by reusing our code previously with slight modification. Specifically, at the end of each iterative_denoise_cfg iteration we force the inside of the ROI to be the current denoising step but the rest to be the forward noise function on the original image. We test it on the campanile and two photos of dogs.

<div class="image-row3a">
  <figure>
    <img src="/assets/images/project5/5a/1_7_3_inpaint.png" alt="0">
    <figcaption>Figure: Important code for inpaint. The last line is the key step in masking</figcaption>
  </figure>
</div>

Campanile:
<div class="image-row3a">
  <figure>
    <img src="/assets/images/project5/5a/1.7.2_campanile_process.jpg" alt="0">
    <figcaption>Figure: Masking the campanile</figcaption>
  </figure>
</div>

<div class="single-image-centered">
  <figure>
    <img src="/assets/images/project5/5a/1_7_3_inpaint_campanile.png" alt="0">
    <figcaption>Figure: Replacing the top of the campanile with something a bit different via inpainting the above</figcaption>
  </figure>
</div>

Georgie:
<div class="image-row3a">
  <figure>
    <img src="/assets/images/project5/5a/1.7.2_Georgie_process.jpg" alt="0">
    <figcaption>Figure: Masking the Georgie image.</figcaption>
  </figure>
</div>

<div class="single-image-centered">
  <figure>
    <img src="/assets/images/project5/5a/1_7_3_inpaint_Georgie.png" alt="0">
    <figcaption>Figure: Replacing an ornament with a window via inpainting the ROI above</figcaption>
  </figure>
</div>
Puppy:
<div class="image-row3a">
  <figure>
    <img src="/assets/images/project5/5a/1.7.2_puppy_process.jpg" alt="0">
    <figcaption>Figure: Masking the puppy image.</figcaption>
  </figure>
</div>

<div class="single-image-centered">
  <figure>
    <img src="/assets/images/project5/5a/1_7_3_puppy.png" alt="0">
    <figcaption>Figure: Replacing the face of a puppy with a more hound-like snout/mouth via inpainting the above</figcaption>
  </figure>
</div>


### 1.7.3 Text-Conditional Image-to-image Translation

Now we do the same as above, but use a prompt to guide it. We do 3 (one of the campanile, two of my chosen images)

Prompt: "a rocket ship", Original image: Campanile

<div class="side-by-side">
  <figure>
    <img src="/assets/images/project5/5a/1_7_text_conditioned_1_campanile.jpg" alt="0">
    <img src="/assets/images/project5/5a/1_7_text_conditioned_3_campanile.jpg" alt="0">
    <img src="/assets/images/project5/5a/1_7_text_conditioned_5_campanile.jpg" alt="0">
    <img src="/assets/images/project5/5a/1_7_text_conditioned_7_campanile.jpg" alt="0">
    <img src="/assets/images/project5/5a/1_7_text_conditioned_10_campanile.jpg" alt="0">
    <img src="/assets/images/project5/5a/1_7_text_conditioned_20_campanile.jpg" alt="0">
    <img src="/assets/images/project5/5a/downsample_campanile.jpg" alt="0">
    <figcaption>Figure: Editing the campanile by adding various noise levels and denoising with the prompt: "a rocket ship". The first level (1) is the top left corner, and the levels increase as you go left to right, and then up to down (e.g. top right corner is level 3, bottom right corner is original image (t=0))</figcaption>
  </figure>
</div>

Prompt: "a fat cat", Original image: Georgie

<div class="side-by-side">
  <figure>
    <img src="/assets/images/project5/5a/1_7_text_conditioned_1_cat_Georgie.jpg" alt="0">
    <img src="/assets/images/project5/5a/1_7_text_conditioned_3_cat_Georgie.jpg" alt="0">
    <img src="/assets/images/project5/5a/1_7_text_conditioned_5_cat_Georgie.jpg" alt="0">
    <img src="/assets/images/project5/5a/1_7_text_conditioned_7_cat_Georgie.jpg" alt="0">
    <img src="/assets/images/project5/5a/1_7_text_conditioned_10_cat_Georgie.jpg" alt="0">
    <img src="/assets/images/project5/5a/1_7_text_conditioned_20_cat_Georgie.jpg" alt="0">
    <img src="/assets/images/project5/5a/downsample_Georgie.jpg" alt="0">
    <figcaption>Figure: Editing the Georgie photo by adding various noise levels and denoising with the prompt: "a fat cat". The first level (1) is the top left corner, and the levels increase as you go left to right, and then up to down (e.g. top right corner is level 3, bottom right corner is original image (t=0))</figcaption>
  </figure>
</div>
This does a good job of creating a cat in the same pose as the original image.

Prompt: "a fat cat", Original image: Stanley Hall

<div class="side-by-side">
  <figure>
    <img src="/assets/images/project5/5a/1_7_text_conditioned_1_stanley_outside.jpg" alt="0">
    <img src="/assets/images/project5/5a/1_7_text_conditioned_3_stanley_outside.jpg" alt="0">
    <img src="/assets/images/project5/5a/1_7_text_conditioned_5_stanley_outside.jpg" alt="0">
    <img src="/assets/images/project5/5a/1_7_text_conditioned_7_stanley_outside.jpg" alt="0">
    <img src="/assets/images/project5/5a/1_7_text_conditioned_10_stanley_outside.jpg" alt="0">
    <img src="/assets/images/project5/5a/1_7_text_conditioned_20_stanley_outside.jpg" alt="0">
    <img src="/assets/images/project5/5a/downsample_stanley_outside.jpg" alt="0">
    <figcaption>Figure: Editing the Stanley Hall photo by adding various noise levels and denoising with the prompt: "a fat cat". The first level (1) is the top left corner, and the levels increase as you go left to right, and then up to down (e.g. top right corner is level 3, bottom right corner is original image (t=0))</figcaption>
  </figure>
</div>

The second to last image edit is funny.

## 1.8 Visual Anagrams

Here, we create visual anagrams by creating two noises per iteration: One from the current image and using prompt 1 and the other from flipping the current image and using prompt2. Then, the flipped image's noise is flipped again and averaged with the current image noise. Some code snippets that are most relevant are shown below (rest is same as the previous iterative_denoise_cfg)


Example visual anagrams:
<div class="image-row3a">
  <figure>
    <img src="/assets/images/project5/5a/1.8_code1.png" alt="0">
    <img src="/assets/images/project5/5a/1.8_code2.png" alt="0">
    <figcaption>Figure: Important code for implementing the visual anagrams</figcaption>
  </figure>
</div>

<div class="image-row3a">
  <figure>
    <img src="/assets/images/project5/5a/1_8_oldman_campfire.png" alt="0">
    <figcaption>Figure: Old man and campfire visual anagram</figcaption>
  </figure>
</div>

<div class="image-row3a">
  <figure>
    <img src="/assets/images/project5/5a/1_8_mountain_concert_v2.png" alt="0">
    <figcaption>Figure: A glacier and a horse</figcaption>
  </figure>
</div>

<div class="image-row3a">
  <figure>
    <img src="/assets/images/project5/5a/1_8_glacier_horse_v2.png" alt="0">
    <figcaption>Figure: A glacier and a horse (This one isn't perfect)</figcaption>
  </figure>
</div>

## 1.9: Hybrid Images

Here, we created hybrid images: We get two noises by denoising the image with two separat prompts. Then we blur one, and sharpen the other (by taking the image and subtracting the blurred image). The final noise is the average of both noises. Note I just used the first image variance for the purpose of adding variance in the next iteration image.  Some code snippets that are most relevant are shown below (rest is same as the previous section). I used the recommended kernel size and sigma values, 32 and 2, respectively.


<div class="image-row3a">
  <figure>
    <img src="/assets/images/project5/5a/1.9_code.png" alt="0">
    <figcaption>Figure: Important code for implementing the Hybrid Images</figcaption>
  </figure>
</div>

<div class="single-image-centered">
  <figure>
    <img src="/assets/images/project5/5a/1_9_skullwaterfall_v3.png" alt="0">
    <figcaption>Figure: A skull (low frequency) and waterfall (high frequncy) hybrid image</figcaption>
  </figure>
</div>

<div class="single-image-centered">
  <figure>
    <img src="/assets/images/project5/5a/1_9_glacier_horse_v2.png" alt="0">
    <figcaption>Figure: A glacier (low frequency) and a horse (high frequency) hybrid image</figcaption>
  </figure>
</div>

<div class="single-image-centered">
  <figure>
    <img src="/assets/images/project5/5a/1_9_eagle_laptop.png" alt="0">
    <figcaption>Figure: An eagle (low frequency) and a laptop (high frequency) hybrid image </figcaption>
  </figure>
</div>

## Bells and Whistles

I implemented two additional transformations for visual anagrams as in the original paper. The first one is simple: I create a new view by inverting the colors in the image (multiply by -1 since pixels are roughly symmetric around 0 during processing). Then do the same thing to the modified noise and average the two noises. The second is the skew: This is a bit more complicated. To implement this, I create a new view by rolling each pixel in the image a different amount down depending on its index (e.g. column 0 gets rolled down 0, column 1 gets rolled down 2, column 3 gets rolled down 6...). Then reverse the rolling on the modified noise. Results are shown below. The prompts I used are examples from the paper.

<div class="single-image-centered">
  <figure>
    <img src="/assets/images/project5/5a/2_tudor_skull_v3.png" alt="0">
    <figcaption>Figure: Creating a visual anagram by skewing (rolling pixels). </figcaption>
  </figure>
</div>

<div class="single-image-centered">
  <figure>
    <img src="/assets/images/project5/5a/2_houseplants_mountain_v4.png" alt="0">
    <figcaption>Figure: Creating a visual anagram by color inversion of pixels</figcaption>
  </figure>
</div>

I also tried to create a course logo. I used a prompt to guide the original Berkeley seal image below.

<div class="single-image-centered">
  <figure>
    <img src="/assets/images/project5/5a/downsample_ucb_logo.jpg" alt="0">
    <figcaption>Figure: Original image (ucb logo)</figcaption>
  </figure>
</div>

I tried a few prompts (e.g. 'a programmer in front of his computer celebrating that his code works', 'a programmer in front of his computer frustrated with a bug', 'a digital camera lens'...) but ironically the prompt with 'a programmer in front of his computer frustrated with a bug' with edit level 7 created the cutest looking photo.

<div class="side-by-side">
  <figure>
    <img src="/assets/images/project5/5a/2_programmer_mad_v2_1_ucb_logo.jpg" alt="0">
    <img src="/assets/images/project5/5a/2_programmer_mad_v2_3_ucb_logo.jpg" alt="0">
    <img src="/assets/images/project5/5a/2_programmer_mad_v2_5_ucb_logo.jpg" alt="0">
    <img src="/assets/images/project5/5a/2_programmer_mad_v2_7_ucb_logo.jpg" alt="0">
    <img src="/assets/images/project5/5a/2_programmer_mad_v2_10_ucb_logo.jpg" alt="0">
    <img src="/assets/images/project5/5a/2_programmer_mad_v2_20_ucb_logo.jpg" alt="0">
    <img src="/assets/images/project5/5a/downsample_ucb_logo.jpg" alt="0">
    <figcaption>Figure: Editing the berkeley seal by adding various noise levels and denoising with the prompt: "a programmer in front of his computer frustrated with a bug". The first level (1) is the top left corner, and the levels increase as you go left to right, and then up to down (e.g. top right corner is level 3, bottom right corner is original image (t=0))</figcaption>
  </figure>
</div>

I end up choosing the image at level 7


<div class="single-image-centered">
  <figure>
    <img src="/assets/images/project5/5a/2_programmer_mad_v2_7_ucb_logo.jpg" alt="0">
    <figcaption>Figure: Final CS280a logo. I have no idea why this image is created given the prompt ‘a programmer in front of his computer frustrated with a bug’, but it looks cute.</figcaption>
  </figure>
</div>




