# KALMAN FILTER
A Kalman filter combines multiple available uncertain sources of information about the state of a system to give a more precise output. These uncertain sources are:

1. A prediction from the system's mathematical model.
2. A measurement from one or more sensors.

Both are uncertain, and the Kalman filter uses their uncertainties to decide how much to trust each one.

---
Since both the measurement from the sensor and mathematical model contain a level of uncertainty (noise), they can be model as a Gaussian function : 
$$z∼N(μ,σ^2)$$
where, $z$ is the reading from any sensor with mean reading of $μ$ and standard deviation of $σ$. Thus, both the mathematical model as well as the sensor readings gets modelled as the gaussian function .The Kalman filter is fundamentally a **recursive Gaussian fusion** in which after both are modelled with their respective mean estimate and the standard deviation, they are fused together. Fusing two sources of information mathematically means **multiplying their Gaussian probability density functions**

---
### Gaussian Fusion
To combine two independent Gaussian estimates, Prediction $N(x1​,σ1^2​)$ and Measurement $N(x2​,σ2^2​)$ we multiply their probability density functions together. 
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
