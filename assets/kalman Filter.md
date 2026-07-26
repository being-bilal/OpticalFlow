# KALMAN FILTER
A Kalman filter combines multiple available uncertain sources of information about the state of a system to give the best estimate.  These uncertain sources are:

1. A prediction from the system's mathematical model.
2. A measurement from one or more sensors.

Both are uncertain, and the Kalman filter uses their uncertainties to decide how much to trust each one.

---
Since both the measurement from the sensor and mathematical model contain a level of uncertainty (noise). thus, they can be model as a Gaussian distribution as Gaussian distribution (also called a normal distribution) is a mathematical way of representing uncertainty. Instead of assuming that a measured value is exact, it describes how likely different values are to be the true value 
* when a linear function is added to the gaussian distribution produces another gaussian distribution but same is not true in the case of non-linear function as they does not necessarily produce the gaussian distribution as the output. 
$$z∼N(μ,σ^2)$$
where, $z$ is the reading from any sensor with mean reading of $μ$ and standard deviation of $σ$. Thus, both the mathematical model as well as the sensor readings gets modelled as the gaussian function .The Kalman filter is fundamentally a **recursive Gaussian fusion** in which after both are modelled with their respective mean estimate and the standard deviation, they are fused together. Fusing two sources of information mathematically means **multiplying their Gaussian probability density functions**

---
### Gaussian Fusion
To combine two independent Gaussian estimates, Prediction $N(x1​,σ1^2​)$ and Measurement $N(x2​,σ2^2​)$ we multiply their probability density functions together (The result is also a gaussian distribution whose standard deviation is less than both). 
*  Fused Mean

$$
\hat{x}
=
\left(
\frac{\sigma_2^2}{\sigma_1^2 + \sigma_2^2}
\right)x_1
+
\left(
\frac{\sigma_1^2}{\sigma_1^2 + \sigma_2^2}
\right)x_2
$$
* Fused Variance : 
$$
\sigma_{\text{fused}}^2
=
\frac{\sigma_1^2 \sigma_2^2}
{\sigma_1^2 + \sigma_2^2}
$$
Then, if we define the kalman gain as : 
$$K = \frac{\sigma_1^2}
{\sigma_1^2 + \sigma_2^2}$$
we get, 
$$\hat{x}
= x_1 + K(x_2 - x_1)
$$

### Core loop of the Kalman Filter
- **Start** with your best current estimate (mean and variance).
    
- **Predict**: Use your physics model to move the mean forward in time, and add process noise Q to update the variance ($σ^2​=σold^2​+Q$).
    
- **Measure**: Take a reading from the sensor (which has its own mean and known sensor variance σm2​). 
    
- **Fuse**: Multiply the two Gaussian curves together using the **Kalman Gain** to find the narrowest, most accurate fused mean and variance.
    
- **Repeat**: Pass this new fused estimate right back into Step 2 for the next time step.

---
## Extended Kalman Filter (EKF)
The extended kalman filter is the version of the normal kalman filter that is used when the data we are working with is non-linear. The normal Kalman filter assumes that both the motion model and the measurement model are linear that means that the new state is obtained by multiplying the current state by a matrix such that there are no sines, cosines and exponentials. For example : 
$$new(position)=old(position)+velocity×Δt$$
Therefore, the normal kalman filter assumes the linear motion model which is given as :
$$
x_{k+1} = F x_k + B u_k + w_k
$$
The same way it assumes a linear measurement model, given as :
$$z=Hx+v$$
But in real life scenarios, the linear measurement and motion model are not true as they contain non-linearities in real life such as in the case of a robot turning. In the normal Kalman filter, the state prediction looks like this: the next state is found by multiplying the current state by a matrix and adding noise. Because the equations are linear and the noises are Gaussian, the output is also a gaussian distribution whose mean and co-variance can be easily determined but in the case of extended kalman filters where we have a nonlinear equations, the motion and measurement model transforms into : 
$$x_{k+1}​=f(x_k​,u_k​)+w_k​$$
$$z_k​=h(x_k​)+v_k​$$
where $f$ and $h$ are non-linear functions that are added to gaussian noise. Since addition of nonlinear functions do not preserve the simple Gaussian form exactly EKF linearises the non-linear function using the first order Taylor expansion :
$$f(x)≈f(x0​)+f′(x0​)(x−x0​).$$
which converts the non-linear function $f(x)$ into a linear equation. if the function is a multi variable function then the EKF uses Jacobian :$$f(x)≈f(x0​)+J(x0​)(x−x0​)$$​​​So in the EKF, the nonlinear motion model is replaced by a local linear model for the prediction step, and the nonlinear measurement model is also replaced by a local linear model for the update step. After that, the EKF uses the same familiar Kalman filter equations, but applied to these local linear approximations. The main difference, then, is this: the normal Kalman filter is exact for linear systems while the EKF is an approximation for nonlinear systems. The normal Kalman filter assumes the system equations are already linear. The EKF assumes they are nonlinear,