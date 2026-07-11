# Optical Flow
optical flow is computer vision technique that determines the apparent motion of the pixels in consecutive frames in a video sequence, it assigns a velocity vector (u, v) to each of the pixels that represents their apparent 2D motion. While the optical can be used to determine the global motion of the camera it does not directly calculate that, it calculates the apparent motion of the pixels between the frames that be caused by the motion of the object, motion of the camera, and lightning change. The key requirements for the proper working of the optical flow include: 

* **No lightning changes** : as if the sphere is static, and the light source moves around the sphere, we will see motion in spite of the sphere being static. as the brightness of the pixels on the sphere would change.
* **Insufficient Texture** : if we move the camera infront of a textureless wall, there would be no detection of motion as the brightness of the pixels would remain same and not change with motion.

## Mathematical Foundation 
_Note : i have used point and pixel interchangeably even tho they are not exactly the same_

For a given consecutive frames of the video, two images L1  and  L2, the optical flow indicates the relative position of each pixel in L1 and the corresponding pixel in L2. Assume the intensity of the pixel at the location x, y at time t is represented by $I(x,y,t)$. After the apparent motion of the image the pixel in the scene moves by: $dx$ and $dy$ in time $dt$ then the optical flow assumes that : 
$$I(x,y,t)=I(x+dx,y+dy,t+dt)$$
meaning the same physical point keeps approximately the same brightness when it moved from one location to another. This assumes there is not lightning change in then environment, this is called the **Brightness constancy assumption**.
#### Dense Optical Flow Representation
Dense optical flow representation determines the change in position of each and every pixel in the frame whereas in the sparse optical flow representation we choose distinctive points such as corners and feature points and then track only those across the frames using feature matching.
Suppose a pixel in video frame at time t is an image: $I(x,y,t)$, Dense Optical flow assigns a velocity vector to this pixel as:
$V(x,y,t)=(u(x,y,t),v(x,y,t))$
where, $u(x,y,t)$ is horizontal velocity and $v(x,y,t)$ is vertical velocity of the pixel. The **Lucas–Kanade (LK) algorithm** is one of the most influential methods in computer vision. It estimates the motion of image patches between two consecutive frames. It works on the idea that all the pixel in a small neighbourhood would have the same velocity. Using the brightness constancy assumption :

$$ I(x,y,t)=I(x+u,y+v,t+Δt)$$
Using a first-order Taylor expansion:

$$I(x+u,y+v,t+Δt)≈I+Ixu+Iyv+It$$​
Substituting into the brightness constancy equation:

$$I=I+Ixu+Iyv+It$$
$$Ix​u+Iy​v+It​=0$$Now, the equation contains two unknowns (u,v) and only one equation, known as the **aperture problem**. Both Lucas–Kanade and Gunner Farneback solve this problem by introducing additional assumptions.

#### **Gunner Farneback algorithm** 
The **Gunner Farnebäck algorithm** is a widely used method in computer vision to compute **dense optical flow**. Unlike sparse methods (like Lucas-Kanade) which track only a few key features, dense methods calculate the motion vector for _every single pixel_ between two consecutive frames.

The Farnebäck algorithm assumes that the brightness pattern within a small neighborhood can be represented by a smooth quadratic surface. It fits such a surface to the neighborhood in the first frame and another to the corresponding neighborhood in the second frame. If the neighborhood has undergone a small translation, the shape of the surface remains nearly the same, while its position changes. By analyzing how the polynomial coefficients change between the two frames, the algorithm estimates the horizontal and vertical displacement of that neighborhood. This displacement becomes the optical flow vector for the pixel.
Assume a region with brightness as : 
12 14 16
15 20 22
17 24 26
These numbers represent the brightness of the pixels.

Farneback assumes that within this small region, the brightness varies smoothly and can be approximated by:
$$f(x,y)=ax2+bxy+cy2+dx+ey+f$$
this turns the neighbourhood into a curved surface. Because real images contain noise and motion is not perfectly uniform, Farnebäck does not rely on a single pixel. Instead, it combines information from neighboring pixels using weighted averaging and least-squares optimization.

The algorithm also employs a **Gaussian pyramid**, which is a set of progressively smaller versions of the image. The algorithm also employs a **Gaussian pyramid**, which is a set of progressively smaller versions of the image. In OpenCV it is implemented as : 

````
flow = cv2.calcOpticalFlowFarneback(
    prev,
    next,
    None,
    pyr_scale,
    levels,
    winsize,
    iterations,
    poly_n,
)
```` 
* The parameter **`prev`** is the previous grayscale frame, and **`next`** is the next grayscale frame.
* The third parameter, often set to **`None`**, specifies an initial flow estimate. If it is None it starts with the assumption that the initial motion is zero.
* The parameter **`pyr_scale`** determines how much each level of the image pyramid is reduced.
* The parameter **`levels`** specifies the number of pyramid levels.
* The parameter **`winsize`** controls the size of the averaging window used when estimating motion.
* The parameter **`iterations`** specifies how many refinement iterations are performed at each pyramid level
* The parameter **`poly_n`** determines the size of the neighborhood used for fitting the quadratic polynomial. Common values are `5` and `7`.
---
## Optical Flow For Global Camera Motion
**To Determine** :
When the camera it has rotation $R$ about its center and translation $T$ of its center, Optical flow cannot directly determine the motion of the camera but can used along with IMU and depth sensor to calculate the value of $R$ and $T$ to determine the motion of the camera. The motion of the point in space depends upon its distance/depth from the camera as the points nearby the camera moves faster than the point far away. 

Assume a 3D point: 
$P= \begin{bmatrix} X\\Y\\Z \end{bmatrix}​​$
It is projected on the camera screen as : 
$y=f\frac{Y}{Z} , x=f\frac{X}{Z}$
where, $f$ is the focal length of the camera (intrinsic property of the camera) and $Z$ is the depth. Therefore the projection of the point on the camera screen depend on depth and focal point of the camera.

