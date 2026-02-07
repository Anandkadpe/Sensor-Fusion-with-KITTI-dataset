# Sensor Fusion Project

A multi-sensor fusion framework that combines **Visual Odometry (VO)**, **GPS/IMU data**, and **Extended Kalman Filter (EKF)** for robust vehicle localization and trajectory estimation using the KITTI dataset.

## Overview

This project implements sensor fusion for autonomous driving applications by integrating:

- **Visual Odometry**: Estimates motion from stereo camera pairs using feature matching and 3D triangulation
- **GPS/IMU Measurements**: Provides absolute position and orientation data from the OXTS system
- **Extended Kalman Filter**: Fuses multiple sensor modalities to produce optimal state estimates (position, velocity, orientation)

The system compares the performance of Visual Odometry alone vs. sensor-fused EKF estimates against GPS ground truth.

## Features

### 1. **Visual Odometry Pipeline**
- ORB feature detection and matching across stereo frames
- Disparity computation from stereo pairs
- 3D point triangulation from disparity maps
- PnP RANSAC-based motion estimation
- Robust outlier rejection

### 2. **Sensor Integration**
- GPS/IMU data loading and preprocessing
- GPS coordinates converted to local Cartesian coordinates
- IMU acceleration and gyroscope measurements for motion prediction
- Camera intrinsics and baseline calibration from KITTI format

### 3. **Extended Kalman Filter**
- **State Vector**: [x, y, yaw, vx, vy, omega]
- **Prediction**: IMU-based motion model with acceleration and angular velocity
- **Updates**: GPS position measurements and VO pose estimates
- **Adaptive fusion**: Handles GPS dropouts and VO failures gracefully

### 4. **Performance Visualization**
- Trajectory comparison (GPS-only, VO-only, EKF-fused)
- Localization error tracking over time
- Statistical performance metrics (mean and max error)

## Dataset

This project uses the **KITTI dataset** (drive sequence 2011_09_26_drive_0060):

```
2011_09_26_drive_0060_sync/
├── image_00/        # Left camera images
├── image_01/        # Right camera images
├── image_02/        # Left camera (alternate)
├── image_03/        # Right camera (alternate)
├── oxts/            # GPS/IMU measurements
│   └── data/        # Raw IMU/GPS data files
└── velodyne_points/ # LiDAR scans (optional)

2011_09_26_calib/
└── calib_cam_to_cam.txt  # Camera calibration parameters
```

## Project Structure

```
.
├── visuall_odometry.py       # Main sensor fusion pipeline
├── data_loading.ipynb        # Jupyter notebook for data exploration
├── Data structure.txt        # Dataset organization documentation
└── README.md                 # This file
```

## Dependencies

- **Python 3.8+**
- **NumPy**: Numerical computations
- **OpenCV (cv2)**: Computer vision (ORB detection, stereo matching, PnP solving)
- **Matplotlib**: Visualization
- **Pathlib**: File handling

### Install Dependencies

```bash
pip install numpy opencv-python matplotlib
```

## Usage

### Running the Sensor Fusion Pipeline

```bash
python visuall_odometry.py
```

This will:
1. Load GPS/IMU data from OXTS files
2. Load stereo image pairs
3. Load camera calibration parameters
4. Run the Visual Odometry algorithm on 100 frames
5. Fuse sensor data using Extended Kalman Filter
6. Display trajectory comparison and error metrics
7. Show visualization plots

### Expected Output

```
Loaded 100 OXTS measurements
GPS trajectory: 100 points
Camera calibration loaded
Baseline: 0.54 m
Loaded 100 stereo image pairs

Starting sensor fusion on 100 frames...
Processing frame 10/100
Processing frame 20/100
...

[OK] Sensor fusion complete!
EKF trajectory shape: (99, 2)
VO trajectory shape: (99, 2)

=== PERFORMANCE METRICS ===
VO Mean Error: 2.145 m
EKF Mean Error: 0.823 m
VO Max Error: 5.234 m
EKF Max Error: 1.567 m
```

## Algorithm Details

### Visual Odometry

1. **Feature Detection**: Uses ORB detector with 3000 features per frame
2. **Stereo Matching**: Computes disparity map using stereo block matching
3. **Triangulation**: Converts disparity to 3D points using camera intrinsics
4. **Feature Tracking**: Matches features between consecutive frames
5. **Motion Estimation**: 
   - Solves PnP problem with RANSAC
   - Recovers rotation matrix (R) and translation vector (t)
   - Filters outliers with reprojection error threshold

### Extended Kalman Filter

**Prediction Step** (IMU):
```
x' = f(x, u, dt)
P' = F·P·F^T + Q
```
Where u includes acceleration and gyro measurements

**Update Steps**:
- **GPS Update**: Observes x, y positions (every 50 frames to simulate dropout)
- **VO Update**: Observes x, y, yaw from visual odometry

**Sensor Noise Characteristics**:
- GPS: σ = 2.0 m (horizontal)
- VO: σ = 0.1 m (position), σ = 0.01 rad (orientation)
- IMU: σ² for acceleration (0.5), angular velocity (0.1)

## Configuration Parameters

Edit `visuall_odometry.py` to customize:

| Parameter | Location | Purpose |
|-----------|----------|---------|
| `max_frames=100` | Line 125 | Process first N frames |
| `nfeatures=3000` | Line 140 | ORB features per frame |
| `numDisparities=64` | Line 148 | Stereo matching resolution |
| `blockSize=15` | Line 148 | Stereo block size |
| `confidence=0.99` | Line 204 | RANSAC confidence level |
| `dt=0.1` | Line 305 | Time step between frames |
| `Q` matrix | Line 285 | Process noise covariance |
| `R_gps` matrix | Line 286 | GPS measurement noise |
| `R_vo` matrix | Line 287 | VO measurement noise |

## Key Results

- **EKF Fusion Improvement**: Combines complementary strengths:
  - GPS provides long-term absolute accuracy
  - VO provides high-frequency relative motion estimates
  - EKF optimally weights both sources based on noise characteristics

- **Robustness**: 
  - Handles GPS dropouts (every 50 frames)
  - Recovers from VO failures with GPS correction
  - Smooth trajectory filtering with Kalman prediction

## Data Processing Flow

```
KITTI Raw Data
    ├─→ Load OXTS (GPS/IMU) ──→ Convert to local coordinates
    ├─→ Load Stereo Images ────→ ORB Feature Detection
    ├─→ Load Calibration ──────→ Camera internals & baseline
    │
    └─→ Visual Odometry
        ├─ Compute stereo disparity
        ├─ Triangulate 3D points
        ├─ Match frame-to-frame features
        └─ Estimate motion (R, t)
    
    └─→ Extended Kalman Filter
        ├─ Predict: IMU acceleration & gyro
        ├─ Update: GPS position
        ├─ Update: VO pose estimates
        └─ Fused state (x, y, yaw, vx, vy, ω)

Output: Trajectories & Error Metrics
```

## Visualization Outputs

1. **Trajectory Comparison Plot**:
   - Blue line: GPS-only (ground truth)
   - Green line: Visual Odometry-only
   - Red line: EKF Fused estimate

2. **Error Plot**:
   - Localization error over time
   - Shows how fusion reduces drift

## Future Improvements

- [ ] Real-time processing optimization
- [ ] Loop closure detection
- [ ] Place recognition using image descriptors
- [ ] Multi-hypothesis tracking
- [ ] 3D mapping from LiDAR/stereo
- [ ] Deep learning-based feature extraction
- [ ] Support for longer sequences
- [ ] Uncertainty quantification

## Known Limitations

- Processes only 100 frames by default (edit for longer sequences)
- Assumes constant velocity model between IMU impulses
- VO degradation in low-texture or repetitive environments
- No loop closure or global optimization
- Single-thread processing (not real-time)

## References

- KITTI Dataset: [geom.io/datasets/kitti](http://www.cvlibs.net/datasets/kitti/)
- OpenCV Stereo Matching: [docs.opencv.org](https://docs.opencv.org)
- EKF for Navigation: Durrant-Whyte & Bailey (2006)

## License

This project uses KITTI dataset. See KITTI terms for dataset usage restrictions.

## Contact & Contributing

For questions or improvements, please refer to the code documentation and inline comments.

---

**Last Updated**: February 2026
**Status**: Active Development
