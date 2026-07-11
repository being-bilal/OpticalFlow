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
where, $u(x,y,t)$ is horizontal velocity and $v(x,y,t)$ is vertical velocity of the pixel. The **Lucas–Kanade (LK) algorithm** is one of the most influential methods in computer vision. It estimates the motion of image patches between two consecutive frames. It works on the idea that all the pixel in a small neighbourhood would have the same velocity. 

## Optical Flow For Global Camera Motion
**To Determine** :
When the camera it has rotation $R$ about its center and translation $T$ of its center, Optical flow cannot directly determine the motion of the camera but can used along with IMU and depth sensor to calculate the value of $R$ and $T$ to determine the motion of the camera. The motion of the point in space depends upon its distance/depth from the camera as the points nearby the camera moves faster than the point far away. 

Assume a 3D point: 
$P= \begin{bmatrix} X\\Y\\Z \end{bmatrix}​​$
It is projected on the camera screen as : 
$y=f\frac{Y}{Z} , x=f\frac{X}{Z}$
where, $f$ is the focal length of the camera (intrinsic property of the camera) and $Z$ is the depth. Therefore the projection of the point on the camera screen depend on depth and focal point of the camera.
