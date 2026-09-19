# OpticalFlow: Underwater Visual-Inertial Odometry

A visual-inertial odometry pipeline for AUVs using stereo optical flow, IMU fusion, and depth sensing. Implements both classical (Farneback) and deep learning (RAFT) optical flow approaches, evaluated against ORB-SLAM3 and VINS-Fusion on the NTNU underwater dataset.

## Project Structure

```
OpticalFlow/
├── OpticalFlow/          # Main VIO pipeline
│   ├── main.py           # Entry point - runs full pipeline on NTNU dataset
│   ├── stereo_depth.py   # Stereo rectification & depth from disparity
│   ├── velocity_calculation.py  # Camera velocity from flow + IMU + depth (RANSAC)
│   ├── ekf.py            # Extended Kalman Filter for sensor fusion
│   ├── GT_vis.py         # Ground truth visualization
│   ├── Gunner_Franeback/ # Classical dense optical flow
│   └── RAFT/             # Deep learning optical flow (RAFT)
│       ├── core/         # Model components (encoder, correlation, GRU update)
│       ├── evaluate.py   # Benchmark evaluation (Sintel/KITTI/Chairs)
│       └── frames-demo.py # Demo on image sequences
├── Simulation/           # VIO/SLAM benchmarking
│   ├── src/              # Evaluation & conversion utilities
│   ├── orb_slam3/        # ORB-SLAM3 integration
│   └── results/          # Benchmark results (ORB-SLAM3, VINS-Fusion)
└── Theory/               # Technical documentation
    ├── Optical Flow.md   # Farneback, RAFT, camera motion from flow
    ├── kalman Filter.md  # KF/EKF theory
    └── localisation.md   # AUV state estimation architecture
```

## Installation

```bash
git clone https://github.com/being-bilal/OpticalFlow.git
cd OpticalFlow
python3 -m venv venv && source venv/bin/activate
pip install numpy scipy matplotlib opencv-python pyyaml rosbags
# For RAFT: pip install torch torchvision (see PyTorch site for CUDA/MPS)
```

Download the NTNU dataset from Hugging Face (ntnu-arl/underwater-datasets) and extract to `OpticalFlow/dataset/` with:
- `mclab.bag` - ROS bag with stereo images + IMU
- `mclab.tum` - Ground truth trajectory
- Calibration YAML files for intrinsics/extrinsics

## Usage

**Main pipeline (Farneback):**
```bash
cd OpticalFlow && python3 main.py
```
Key parameters in `main.py`: `START_FRAME`, `END_FRAME`, `SUBSAMPLE`, `Z_MIN`/`Z_MAX`, `CAM_TILT_DEG`. Outputs `vo_result.png` with trajectory comparison and ATE metrics.

**RAFT deep learning flow:**
```bash
cd OpticalFlow/RAFT
python3 frames-demo.py --model models/raft-things.pth --path /path/to/images
python3 evaluate.py --model models/raft-things.pth --dataset sintel
```

**Simulation benchmarking:**
```bash
cd Simulation
# ORB-SLAM3, VINS-Fusion - see Simulation/README.md
python3 src/align_and_evaluate.py data/gt.tum results/est.tum --align-scale --plot out.png
```

## Pipeline Overview

1. **Stereo Depth** - Fisheye rectification + StereoSGBM disparity → metric depth Z = f·b/d
2. **Optical Flow** - Farneback (CPU) or RAFT (GPU) dense flow between rectified frames
3. **Rotational Flow Removal** - Predict rotational component from IMU angular velocity
4. **Velocity Solve** - RANSAC + least-squares on translational flow residuals + depth
5. **Trajectory Integration** - Dead reckoning with camera tilt correction, gyro-bias optimization
6. **Evaluation** - Umeyama alignment (Sim(3)) to ground truth, ATE metrics (RMSE, mean, max)

## Key Results

| Method | ATE RMSE | Notes |
|--------|----------|-------|
| OpticalFlow (Farneback + IMU) | ~0.08-0.15m | Metric scale via stereo, gyro-bias corrected |
| ORB-SLAM3 Monocular-Inertial | ~0.1-0.2m | Metric scale, SE(3) aligned |
| VINS-Fusion | ~0.08-0.15m | Tightly-coupled VIO |

## Theory

See `Theory/` for detailed documentation on:
- Optical flow mathematics (brightness constancy, Farneback polynomial expansion, RAFT architecture)
- Kalman Filter / EKF (Gaussian fusion, Jacobian linearization)
- AUV localization architecture (optical flow as virtual DVL, IMU for attitude, depth for Z)
