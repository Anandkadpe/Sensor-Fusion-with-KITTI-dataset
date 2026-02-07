import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import cv2

# ============================================================================
# 1. LOAD OXTS DATA (GPS/IMU)
# ============================================================================

def load_oxts_data(oxts_dir):
    """Load all GPS/IMU data from oxts directory"""
    oxts_files = sorted(Path(oxts_dir).glob('*.txt'))
    data = []
    
    for file in oxts_files:
        with open(file, 'r') as f:
            values = [float(x) for x in f.readline().split()]
            data.append(values)
    
    return np.array(data)

# Load OXTS data
oxts_dir = r'E:\Sensor Fusion project\2011_09_26_drive_0060_sync\2011_09_26\2011_09_26_drive_0060_sync\oxts\data'
oxts_data = load_oxts_data(oxts_dir)

# Extract measurements - USE DESCRIPTIVE NAMES
lat = oxts_data[:, 0]
lon = oxts_data[:, 1]
alt = oxts_data[:, 2]
roll = oxts_data[:, 3]
pitch = oxts_data[:, 4]
yaw = oxts_data[:, 5]
accel_x = oxts_data[:, 11]  # CHANGED FROM ax
accel_y = oxts_data[:, 12]  # CHANGED FROM ay
gyro_z = oxts_data[:, 19]   # CHANGED FROM wz

print(f"Loaded {len(oxts_data)} OXTS measurements")

# ============================================================================
# 2. CONVERT GPS TO LOCAL COORDINATES
# ============================================================================

def latlon_to_xy(lat, lon, lat0, lon0):
    """Convert lat/lon to local XY coordinates (meters)"""
    R = 6378137.0
    
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    lat0_rad = np.radians(lat0)
    lon0_rad = np.radians(lon0)
    
    x = R * (lon_rad - lon0_rad) * np.cos(lat0_rad)
    y = R * (lat_rad - lat0_rad)
    
    return x, y

# Use first position as origin
lat0, lon0 = lat[0], lon[0]
gps_x, gps_y = latlon_to_xy(lat, lon, lat0, lon0)

print(f"GPS trajectory: {len(gps_x)} points")

# ============================================================================
# 3. LOAD CALIBRATION
# ============================================================================

def load_calibration(calib_file):
    """Load camera calibration from KITTI calib file"""
    calib = {}
    
    with open(calib_file, 'r') as f:
        for line in f.readlines():
            if ':' not in line:
                continue
            
            key, value = line.split(':', 1)
            key = key.strip()
            
            try:
                calib[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                continue
    
    P0 = calib['P_rect_00'].reshape(3, 4)
    P1 = calib['P_rect_01'].reshape(3, 4)
    
    K = P0[:3, :3]
    baseline = (P1[0, 3] - P0[0, 3]) / (-P1[0, 0])
    
    return K, baseline

calib_file = r'E:\Sensor Fusion project\2011_09_26_calib\2011_09_26\calib_cam_to_cam.txt'
K, baseline = load_calibration(calib_file)

print(f"Camera calibration loaded")
print(f"Baseline: {baseline:.4f} m")

# ============================================================================
# 4. LOAD IMAGES
# ============================================================================

def load_images(image_dir, max_frames=None):
    """Load images"""
    image_files = sorted(Path(image_dir).glob('*.png'))
    if max_frames:
        image_files = image_files[:max_frames]
    
    images = []
    for file in image_files:
        img = cv2.imread(str(file), cv2.IMREAD_GRAYSCALE)
        images.append(img)
    
    return images

# Load stereo images (use a subset for testing - 100 frames)
left_dir = r'E:\Sensor Fusion project\2011_09_26_drive_0060_sync\2011_09_26\2011_09_26_drive_0060_sync\image_00\data'
right_dir = r'E:\Sensor Fusion project\2011_09_26_drive_0060_sync\2011_09_26\2011_09_26_drive_0060_sync\image_01\data'

left_images = load_images(left_dir, max_frames=100)
right_images = load_images(right_dir, max_frames=100)

print(f"Loaded {len(left_images)} stereo image pairs")

# ============================================================================
# 5. VISUAL ODOMETRY CLASS
# ============================================================================

class VisualOdometry:
    def __init__(self, K, baseline):
        self.K = K
        self.baseline = baseline
        self.detector = cv2.ORB_create(nfeatures=3000)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.prev_keypoints = None
        self.prev_descriptors = None
        self.prev_3d_points = None
        
    def compute_disparity(self, left_img, right_img):
        stereo = cv2.StereoBM_create(numDisparities=64, blockSize=15)
        disparity = stereo.compute(left_img, right_img)
        disparity = disparity.astype(np.float32) / 16.0
        disparity[disparity <= 0] = 0.1
        return disparity
    
    def triangulate_points(self, keypoints, disparity):
        fx = self.K[0, 0]
        fy = self.K[1, 1]
        cx = self.K[0, 2]
        cy = self.K[1, 2]
        
        points_3d = []
        valid_kp = []
        
        for kp in keypoints:
            u, v = int(kp.pt[0]), int(kp.pt[1])
            
            if u < 0 or u >= disparity.shape[1] or v < 0 or v >= disparity.shape[0]:
                continue
            
            d = disparity[v, u]
            
            if d > 0:
                Z = (fx * self.baseline) / d
                X = (u - cx) * Z / fx
                Y = (v - cy) * Z / fy
                
                if 0.5 < Z < 50:
                    points_3d.append([X, Y, Z])
                    valid_kp.append(kp)
        
        return np.array(points_3d), valid_kp
    
    def estimate_motion(self, left_img_curr, right_img_curr):
        keypoints_curr, descriptors_curr = self.detector.detectAndCompute(left_img_curr, None)
        
        if self.prev_keypoints is None:
            disparity = self.compute_disparity(left_img_curr, right_img_curr)
            points_3d, valid_kp = self.triangulate_points(keypoints_curr, disparity)
            
            self.prev_keypoints = valid_kp
            self.prev_descriptors = descriptors_curr
            self.prev_3d_points = points_3d
            
            return np.eye(3), np.zeros((3, 1))
        
        matches = self.matcher.match(self.prev_descriptors, descriptors_curr)
        matches = sorted(matches, key=lambda x: x.distance)
        matches = matches[:int(len(matches) * 0.8)]
        
        if len(matches) < 10:
            return None, None
        
        points_3d = []
        points_2d = []
        
        for match in matches:
            if match.queryIdx < len(self.prev_3d_points):
                points_3d.append(self.prev_3d_points[match.queryIdx])
                points_2d.append(keypoints_curr[match.trainIdx].pt)
        
        points_3d = np.array(points_3d, dtype=np.float32)
        points_2d = np.array(points_2d, dtype=np.float32)
        
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            points_3d, points_2d, self.K, None,
            iterationsCount=200,
            reprojectionError=2.0,
            confidence=0.99
        )
        
        if not success or inliers is None or len(inliers) < 10:
            return None, None
        
        R, _ = cv2.Rodrigues(rvec)
        t = tvec
        
        disparity = self.compute_disparity(left_img_curr, right_img_curr)
        points_3d_curr, valid_kp = self.triangulate_points(keypoints_curr, disparity)
        
        self.prev_keypoints = valid_kp
        self.prev_descriptors = descriptors_curr
        self.prev_3d_points = points_3d_curr
        
        return R, t

# ============================================================================
# 6. EKF CLASS
# ============================================================================

class EKF_with_VO:
    def __init__(self):
        self.x = np.zeros(6)
        self.P = np.eye(6) * 1.0
        self.Q = np.diag([0.1, 0.1, 0.01, 0.5, 0.5, 0.1])
        self.R_gps = np.diag([4.0, 4.0])
        self.R_vo = np.diag([0.1, 0.1, 0.01])
        self.vo_x = 0.0
        self.vo_y = 0.0
        self.vo_yaw = 0.0
        
    def predict(self, dt, acc_x, acc_y, gyro_rate):  # RENAMED PARAMETERS
        x, y, yaw_angle, vx, vy, omega = self.x
        
        x_new = x + (vx * np.cos(yaw_angle) - vy * np.sin(yaw_angle)) * dt
        y_new = y + (vx * np.sin(yaw_angle) + vy * np.cos(yaw_angle)) * dt
        yaw_new = yaw_angle + omega * dt
        vx_new = vx + acc_x * dt
        vy_new = vy + acc_y * dt
        omega_new = gyro_rate
        
        self.x = np.array([x_new, y_new, yaw_new, vx_new, vy_new, omega_new])
        
        F = np.eye(6)
        F[0, 2] = (-vx * np.sin(yaw_angle) - vy * np.cos(yaw_angle)) * dt
        F[0, 3] = np.cos(yaw_angle) * dt
        F[0, 4] = -np.sin(yaw_angle) * dt
        F[1, 2] = (vx * np.cos(yaw_angle) - vy * np.sin(yaw_angle)) * dt
        F[1, 3] = np.sin(yaw_angle) * dt
        F[1, 4] = np.cos(yaw_angle) * dt
        F[2, 5] = dt
        
        self.P = F @ self.P @ F.T + self.Q
        
    def update_gps(self, gps_x_meas, gps_y_meas):
        H = np.zeros((2, 6))
        H[0, 0] = 1
        H[1, 1] = 1
        
        z = np.array([gps_x_meas, gps_y_meas])
        z_pred = H @ self.x
        y_innov = z - z_pred
        
        S = H @ self.P @ H.T + self.R_gps
        K = self.P @ H.T @ np.linalg.inv(S)
        
        self.x = self.x + K @ y_innov
        self.P = (np.eye(6) - K @ H) @ self.P
        
    def update_vo(self, R, t):
        if R is None or t is None:
            return
        
        delta_yaw = np.arctan2(R[1, 0], R[0, 0])
        delta_x = t[0, 0]
        delta_z = t[2, 0]
        
        cos_yaw = np.cos(self.vo_yaw)
        sin_yaw = np.sin(self.vo_yaw)
        
        self.vo_x += delta_z * cos_yaw - delta_x * sin_yaw
        self.vo_y += delta_z * sin_yaw + delta_x * cos_yaw
        self.vo_yaw += delta_yaw
        self.vo_yaw = np.arctan2(np.sin(self.vo_yaw), np.cos(self.vo_yaw))
        
        H = np.zeros((3, 6))
        H[0, 0] = 1
        H[1, 1] = 1
        H[2, 2] = 1
        
        z = np.array([self.vo_x, self.vo_y, self.vo_yaw])
        z_pred = H @ self.x
        
        y_innov = z - z_pred
        y_innov[2] = np.arctan2(np.sin(y_innov[2]), np.cos(y_innov[2]))
        
        S = H @ self.P @ H.T + self.R_vo
        K = self.P @ H.T @ np.linalg.inv(S)
        
        self.x = self.x + K @ y_innov
        self.P = (np.eye(6) - K @ H) @ self.P

# ============================================================================
# 7. RUN SENSOR FUSION
# ============================================================================

# Initialize
ekf = EKF_with_VO()
vo = VisualOdometry(K, baseline)

# Set initial state
ekf.x[0] = gps_x[0]
ekf.x[1] = gps_y[0]
ekf.x[2] = yaw[0]
ekf.vo_x = gps_x[0]
ekf.vo_y = gps_y[0]
ekf.vo_yaw = yaw[0]

# Storage
ekf_trajectory = []
vo_trajectory = []

dt = 0.1

# Main loop
num_frames = min(len(left_images), len(oxts_data))
print(f"\nStarting sensor fusion on {num_frames} frames...")

for frame_idx in range(1, num_frames):  # RENAMED FROM i
    if frame_idx % 10 == 0:
        print(f"Processing frame {frame_idx}/{num_frames}")
    
    # 1. PREDICT with IMU - USING RENAMED VARIABLES
    ekf.predict(dt, accel_x[frame_idx], accel_y[frame_idx], gyro_z[frame_idx])
    
    # 2. UPDATE with Visual Odometry
    R, t = vo.estimate_motion(left_images[frame_idx], right_images[frame_idx])
    if R is not None:
        ekf.update_vo(R, t)
    
    # 3. UPDATE with GPS (simulate dropout every 50 frames)
    if frame_idx % 50 != 0:
        ekf.update_gps(gps_x[frame_idx], gps_y[frame_idx])
    
    # Store results
    ekf_trajectory.append([ekf.x[0], ekf.x[1]])
    vo_trajectory.append([ekf.vo_x, ekf.vo_y])

ekf_trajectory = np.array(ekf_trajectory)
vo_trajectory = np.array(vo_trajectory)

print(f"\n[OK] Sensor fusion complete!")
print(f"EKF trajectory shape: {ekf_trajectory.shape}")
print(f"VO trajectory shape: {vo_trajectory.shape}")

# ============================================================================
# 8. PLOT RESULTS
# ============================================================================

fig, (axis_traj, axis_err) = plt.subplots(1, 2, figsize=(16, 6))

# Align GPS
gps_aligned_x = gps_x[1:len(ekf_trajectory)+1]
gps_aligned_y = gps_y[1:len(ekf_trajectory)+1]

# Plot trajectories
axis_traj.plot(gps_aligned_x, gps_aligned_y, 'b-', alpha=0.5, linewidth=2, label='GPS Only')
axis_traj.plot(vo_trajectory[:, 0], vo_trajectory[:, 1], 'g-', alpha=0.5, linewidth=2, label='VO Only')
axis_traj.plot(ekf_trajectory[:, 0], ekf_trajectory[:, 1], 'r-', linewidth=2, label='EKF Fused')
axis_traj.set_xlabel('X (meters)')
axis_traj.set_ylabel('Y (meters)')
axis_traj.set_title('Trajectory Comparison')
axis_traj.legend()
axis_traj.axis('equal')
axis_traj.grid(True)

# Plot errors
error_vo = np.sqrt((vo_trajectory[:, 0] - gps_aligned_x)**2 + (vo_trajectory[:, 1] - gps_aligned_y)**2)
error_ekf = np.sqrt((ekf_trajectory[:, 0] - gps_aligned_x)**2 + (ekf_trajectory[:, 1] - gps_aligned_y)**2)

axis_err.plot(error_vo, 'g-', label='VO Error', alpha=0.7)
axis_err.plot(error_ekf, 'r-', label='EKF Fused Error', alpha=0.7)
axis_err.set_xlabel('Frame')
axis_err.set_ylabel('Position Error (meters)')
axis_err.set_title('Localization Error Over Time')
axis_err.legend()
axis_err.grid(True)

plt.tight_layout()
plt.show()

# Statistics
print(f"\n=== PERFORMANCE METRICS ===")
print(f"VO Mean Error: {np.mean(error_vo):.3f} m")
print(f"EKF Mean Error: {np.mean(error_ekf):.3f} m")
print(f"VO Max Error: {np.max(error_vo):.3f} m")
print(f"EKF Max Error: {np.max(error_ekf):.3f} m")