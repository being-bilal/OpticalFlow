# VIO Benchmark

Runs visual-inertial odometry pipelines on a Simulation dataset and compares trajectories

## Dataset
(https://drive.google.com/drive/folders/18MMPc_2qB2OM7NqWjYV7AMxNkOy_e5Ol?usp=sharing)

Extract to `data/` directory.

## Setup

### ORB-SLAM Monocular
```bash
cd orb_slam3
./build.sh
./Examples/Monocular/mono_euroc Vocabulary/ORBvoc.txt Examples/Monocular/EuRoC.yaml ../data/dataset trajectory.tum
```

### ORB-SLAM3 Monocular-Inertial
```bash
cd orb_slam3
./build.sh
./Examples/Monocular-Inertial/mono_inertial_euroc Vocabulary/ORBvoc.txt Examples/Monocular-Inertial/EuRoC.yaml ../data/dataset trajectory.tum
```

### VINS-Fusion
```bash
cd VINS-Fusion
catkin build
roslaunch vins_estimator euroc.launch
# In another terminal:
rosbag play ../data/dataset.bag
```
Results saved to `~/.ros/vins_result/`.

## Results

All trajectories saved in `results/`:
- `results/ORB-SLAM`
- `results/VINS-FUSION`
- `results/eval`

## Evaluate

```bash
python3 align_and_evaluate.py data/gt.tum results/vins_fusion.tum --align-scale --plot
```