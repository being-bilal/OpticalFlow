# Localisation
### Localisation using Optical flow as the temporary DVL along with IMU and Depth sensor that fused together using EKF.
The structure would be a **state-estimation pipeline** with three sensor streams feeding one EKF, where **optical flow acts like a short-range velocity sensor**, the **IMU provides attitude and motion-rate information**, and the **depth sensor constrains vertical position**.

At the highest level, the system has four parts: **sensor preprocessing**, **state prediction**, **measurement updates**, and **state output**. The optical flow node first estimates the AUV’s motion in the image plane and converts it into an estimate of vehicle velocity, usually in the body frame or a frame that can be transformed into it. The IMU provides orientation and angular rates, which are critical because optical flow velocities must be interpreted relative to the vehicle’s attitude. The depth sensor provides the z-position, or at least a strong constraint on it. All of these measurements then go into the EKF, which keeps one consistent estimate of the AUV’s full motion state.

A good EKF state for this system would usually contain position, orientation, linear velocity, angular velocity, and possibly acceleration. Conceptually, it looks like:

$$
x =
\begin{bmatrix}
x & y & z & roll & pitch & yaw & v_x & v_y & v_z & \omega_x & \omega_y & \omega_z
\end{bmatrix}^T
$$

Sometimes acceleration is included too, but the exact choice depends on how your implementation is configured. The important point is that the EKF is not just storing “where the AUV is.” It is storing a belief about where it is, how it is oriented, how fast it is moving, and how fast it is rotating.

The **prediction step** uses the motion model. This is the EKF’s internal model of how the AUV should evolve if no new measurements arrived. It typically assumes that velocity and angular velocity continue for a short time, so position is updated by integrating velocity, and orientation is updated by integrating angular velocity. This prediction does not come from the sensors directly; it comes from the filter’s own kinematic model. The IMU does not act as the prediction engine. Instead, the IMU is a measurement source that corrects the predicted state.

Then each sensor performs a different job during the **update step**. The IMU usually corrects roll, pitch, yaw, and angular velocity. This matters because underwater vehicles tilt and rotate, and the optical flow velocity estimate depends on that attitude. If the vehicle pitches or rolls, the image motion changes even if the vehicle’s translational velocity has not changed. So the IMU helps the filter understand whether a measured flow pattern comes from translation, rotation, or both. The depth sensor corrects the z state, anchoring the vehicle vertically. The optical flow update corrects horizontal velocity, usually (v_x) and (v_y), and in some systems it may also help indirectly with x and y position through integration over time.

The optical flow part is the one that acts like a temporary DVL. A DVL measures velocity relative to the seabed. Optical flow can provide a similar kind of estimate by tracking image motion against the ground texture, then converting pixel motion into a metric velocity estimate using altitude and camera geometry. So the flow system is not giving position directly. It is giving **velocity**, which the EKF then integrates over time. That is why it behaves like a temporary DVL: it gives a velocity constraint that limits drift, but it does not fully eliminate long-term position error by itself.

The EKF then fuses these measurements in a very specific way. The IMU update reduces uncertainty in attitude and rotation rate. That improved attitude estimate helps interpret the optical flow correctly. The optical flow update reduces uncertainty in horizontal velocity. The depth update reduces uncertainty in vertical position. Because the filter integrates these corrected states forward in time, the better the velocity estimate is, the less x and y position drift accumulates. Because the depth is directly observed, z remains bounded. And because orientation is continuously corrected by the IMU, the vehicle’s frame stays well aligned with the world frame.

So the structure is really a **closed loop of mutual support**. The IMU makes the optical flow velocity estimate physically meaningful. The optical flow estimate keeps horizontal dead reckoning from drifting too fast. The depth sensor locks the vehicle vertically. The EKF acts as the central fusion engine that combines them into one coherent estimate.
