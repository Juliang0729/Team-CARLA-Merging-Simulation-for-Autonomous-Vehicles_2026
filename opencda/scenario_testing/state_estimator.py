# -*- coding: utf-8 -*-
"""
State Estimator Module for OpenCDA
Adapted from phase9_live_carla_perception.py

Features:
- 8-camera surround view with YOLO object detection
- Position error tracking for 2-3 nearby vehicles
- Drift rate calculation with velocity smoothing (EMA)
- Lightweight LiDAR clustering for detection validation (4Hz)
- Blue box highlighting for tracked vehicles in mosaic view
- Merge-gated tracking (only tracks after ego merges onto mainline)
"""

import numpy as np
import cv2
import carla
import torch
import threading
import time
import warnings
import traci
from collections import defaultdict
from opencda.co_simulation.sumo_integration.bridge_helper import BridgeHelper

# Suppress all FutureWarnings (YOLOv5 torch.cuda.amp deprecation warnings)
warnings.filterwarnings('ignore', category=FutureWarning)

# Configuration constants
CAMERA_W = 640
CAMERA_H = 384
CAMERA_FOV = 100
CAMERA_POS_X = 1.35
CAMERA_POS_Y = 0.0
CAMERA_POS_Z = 2.4
CAMERA_FRONT_CORNER_X = 1.45
CAMERA_REAR_CORNER_X = -1.45
CAMERA_SIDE_X = -0.25
CAMERA_SIDE_Y = 1.08
CAMERA_CORNER_Y = 0.95
LIDAR_RANGE_M = 80.0
LIDAR_MIN_RANGE_M = 2.5
LIDAR_MIN_Z = -1.3
LIDAR_MAX_Z = 2.8
MAX_TRACK_AGE = 20
DET_EVERY_N = 1
DET_SCALE = 0.75
SHOW_3D_BOX = True
SHOW_HEADING_VECTOR = True
BOX_LENGTH = 4.0
BOX_WIDTH = 2.0
BOX_HEIGHT = 1.5


class StateEstimator:
    """
    State estimation module that processes sensor data from a CARLA vehicle.
    Provides object detection, tracking, and visualization.
    """
    
    def __init__(self, world, ego_vehicle, yolo_model_path=None, ego_id=None):
        """
        Initialize the state estimator.
        
        Parameters
        ----------
        world : carla.World
            CARLA world instance
        ego_vehicle : carla.Vehicle
            The ego vehicle to attach sensors to
        yolo_model_path : str, optional
            Path to YOLO model weights
        ego_id : str, optional
            SUMO vehicle ID for ego vehicle (for co-simulation)
        """
        self.world = world
        self.ego_vehicle = ego_vehicle
        self.world_map = world.get_map()
        self.ego_id = ego_id  # Store SUMO vehicle ID
        
        # Sensor data storage
        self.camera_data = {}  # Dict of camera name -> image data
        self.lidar_data = None
        self.gnss_data = None
        self.imu_data = None
        
        # Camera mosaic
        self.camera_names = ['cam0', 'cam1', 'cam2', 'cam3', 'cam4', 'cam5', 'cam6', 'cam7']
        
        # Object tracking
        self.tracks = {}
        self.next_id = 0
        self.detected_positions = []  # Store detected vehicle positions
        self.tracked_vehicles = []  # Store IDs of 2-3 vehicles to track for position error
        self.position_errors = {}  # Dict: vehicle_id -> list of errors
        self.position_error_times = {}  # Dict: vehicle_id -> list of timestamps when errors were recorded
        self.drift_rates = {}  # Dict: vehicle_id -> list of drift rates (m/s)
        self.tracking_frames = {}  # Dict: vehicle_id -> frame count (for min duration filter)
        self.lost_tracking_frames = {}  # Dict: vehicle_id -> consecutive frames without detection
        self.MIN_TRACKING_FRAMES = 60  # ~3 seconds at 20 Hz
        self.MAX_LOST_FRAMES = 50  # Remove vehicle after 50 consecutive frames without detection (~2.5s)
        
        # Position error tracking gate: only start after merge confirmation
        self.merge_started = False
        
        # Store which detections correspond to tracked vehicles (for blue box highlighting)
        self.tracked_vehicle_detections = set()  # Set of (camera_name, detection_index)
        
        # Lightweight LiDAR processing (runs every N frames for efficiency)
        self.lidar_clusters = []
        self.frame_count = 0
        self.LIDAR_PROCESS_EVERY_N = 5  # Process LiDAR every 5 frames (4Hz instead of 20Hz)
        
        # Velocity smoothing for more robust drift rate (simple exponential moving average)
        self.smoothed_velocities = {}  # Dict: vehicle_id -> (vx_smooth, vy_smooth)
        self.velocity_alpha = 0.3  # EMA smoothing factor (0.3 = 30% new, 70% old)
        
        # Threading locks
        self.camera_lock = threading.Lock()
        self.lidar_lock = threading.Lock()
        
        # SUMO-CARLA coordinate transformation offset
        # BridgeHelper.offset is set by the co-simulation bridge
        self.sumo_carla_offset = BridgeHelper.offset
        print(f"[StateEstimator] Using SUMO-CARLA offset: {self.sumo_carla_offset}")
        
        # Visualization
        self.display = None
        self.display_width = CAMERA_W
        self.display_height = CAMERA_H
        
        # YOLO model (using torch.hub for YOLOv5 compatibility)
        if yolo_model_path:
            try:
                # Use torch.hub.load for YOLOv5 models
                self.yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', 
                                                 path=yolo_model_path, force_reload=False)
                self.yolo_model.conf = 0.5  # Confidence threshold
                self.yolo_model.iou = 0.45  # NMS IoU threshold
                print(f"[StateEstimator] Loaded YOLO model from {yolo_model_path}")
            except Exception as e:
                print(f"[StateEstimator] Failed to load YOLO model: {e}")
                self.yolo_model = None
        else:
            self.yolo_model = None
        
        # Spawn sensors
        self.sensors = []
        self._spawn_sensors()
        
        print("[StateEstimator] Initialized successfully")
    
    def _build_camera_mounts(self):
        """Build 8-camera surround view configuration."""
        return {
            "cam0": carla.Transform(
                carla.Location(x=CAMERA_POS_X, y=0.0, z=CAMERA_POS_Z),
                carla.Rotation(yaw=0.0),
            ),
            "cam1": carla.Transform(
                carla.Location(x=CAMERA_FRONT_CORNER_X, y=CAMERA_CORNER_Y, z=CAMERA_POS_Z),
                carla.Rotation(yaw=48.0),
            ),
            "cam2": carla.Transform(
                carla.Location(x=CAMERA_SIDE_X, y=CAMERA_SIDE_Y, z=CAMERA_POS_Z),
                carla.Rotation(yaw=102.0),
            ),
            "cam3": carla.Transform(
                carla.Location(x=CAMERA_REAR_CORNER_X, y=CAMERA_CORNER_Y, z=CAMERA_POS_Z),
                carla.Rotation(yaw=148.0),
            ),
            "cam4": carla.Transform(
                carla.Location(x=CAMERA_REAR_CORNER_X, y=0.0, z=CAMERA_POS_Z),
                carla.Rotation(yaw=180.0),
            ),
            "cam5": carla.Transform(
                carla.Location(x=CAMERA_REAR_CORNER_X, y=-CAMERA_CORNER_Y, z=CAMERA_POS_Z),
                carla.Rotation(yaw=-148.0),
            ),
            "cam6": carla.Transform(
                carla.Location(x=CAMERA_SIDE_X, y=-CAMERA_SIDE_Y, z=CAMERA_POS_Z),
                carla.Rotation(yaw=-102.0),
            ),
            "cam7": carla.Transform(
                carla.Location(x=CAMERA_FRONT_CORNER_X, y=-CAMERA_CORNER_Y, z=CAMERA_POS_Z),
                carla.Rotation(yaw=-48.0),
            ),
        }
    
    def _spawn_sensors(self):
        """Spawn 8-camera mosaic, lidar, GNSS, and IMU sensors on the ego vehicle."""
        bp_lib = self.world.get_blueprint_library()
        
        # 8-camera surround view mosaic
        camera_mounts = self._build_camera_mounts()
        cam_bp = bp_lib.find('sensor.camera.rgb')
        cam_bp.set_attribute('image_size_x', str(CAMERA_W))
        cam_bp.set_attribute('image_size_y', str(CAMERA_H))
        cam_bp.set_attribute('fov', str(CAMERA_FOV))
        
        for cam_name, mount_tf in camera_mounts.items():
            cam = self.world.spawn_actor(cam_bp, mount_tf, attach_to=self.ego_vehicle)
            cam.listen(lambda img, name=cam_name: self._on_camera_data(img, name))
            self.sensors.append(cam)
        
        print(f"[StateEstimator] Spawned {len(camera_mounts)} surround cameras")
        
        # Lidar
        lidar_bp = bp_lib.find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('range', str(LIDAR_RANGE_M))
        lidar_bp.set_attribute('rotation_frequency', '20')
        lidar_bp.set_attribute('channels', '32')
        lidar_bp.set_attribute('points_per_second', '100000')
        lidar_bp.set_attribute('upper_fov', '10.0')
        lidar_bp.set_attribute('lower_fov', '-30.0')
        
        lidar_transform = carla.Transform(
            carla.Location(x=0.0, y=0.0, z=2.4)
        )
        
        self.lidar = self.world.spawn_actor(
            lidar_bp, lidar_transform, attach_to=self.ego_vehicle)
        self.lidar.listen(self._on_lidar_data)
        self.sensors.append(self.lidar)
        
        # GNSS
        gnss_bp = bp_lib.find('sensor.other.gnss')
        self.gnss = self.world.spawn_actor(
            gnss_bp, carla.Transform(), attach_to=self.ego_vehicle)
        self.gnss.listen(self._on_gnss_data)
        self.sensors.append(self.gnss)
        
        # IMU
        imu_bp = bp_lib.find('sensor.other.imu')
        self.imu = self.world.spawn_actor(
            imu_bp, carla.Transform(), attach_to=self.ego_vehicle)
        self.imu.listen(self._on_imu_data)
        self.sensors.append(self.imu)
        
        print(f"[StateEstimator] Spawned {len(self.sensors)} sensors")
    
    def _on_camera_data(self, image, cam_name):
        """Callback for camera sensor data."""
        with self.camera_lock:
            array = np.frombuffer(image.raw_data, dtype=np.uint8)
            array = array.reshape((CAMERA_H, CAMERA_W, 4))
            self.camera_data[cam_name] = array[:, :, :3].copy()  # RGB only
    
    def _on_lidar_data(self, data):
        """Callback for lidar sensor data."""
        with self.lidar_lock:
            points = np.frombuffer(data.raw_data, dtype=np.float32)
            points = points.reshape(-1, 4)  # x, y, z, intensity
            self.lidar_data = points.copy()
    
    def _on_gnss_data(self, data):
        """Callback for GNSS sensor data."""
        self.gnss_data = {
            'lat': data.latitude,
            'lon': data.longitude,
            'alt': data.altitude
        }
    
    def _on_imu_data(self, data):
        """Callback for IMU sensor data."""
        self.imu_data = {
            'accel': np.array([data.accelerometer.x, 
                              data.accelerometer.y, 
                              data.accelerometer.z]),
            'gyro': np.array([data.gyroscope.x, 
                             data.gyroscope.y, 
                             data.gyroscope.z]),
            'compass': data.compass
        }
    
    def tick(self):
        """
        Process one frame of sensor data from all cameras.
        Performs object detection, tracking, and mosaic visualization.
        """
        # Clear previous detections
        self.detected_positions = []
        self.tracked_vehicle_detections = set()
        self.frame_count += 1
        
        # Process LiDAR at reduced frequency (every 5 frames = 4Hz instead of 20Hz)
        if self.frame_count % self.LIDAR_PROCESS_EVERY_N == 0:
            with self.lidar_lock:
                self.lidar_clusters = self._lightweight_lidar_clustering()
        
        # Get all camera frames
        with self.camera_lock:
            if not self.camera_data:
                return False
            camera_frames = {name: img.copy() for name, img in self.camera_data.items()}
        
        # Process detections from all cameras
        if self.yolo_model is not None:
            for cam_name, frame in camera_frames.items():
                try:
                    # YOLOv5 torch.hub inference
                    results = self.yolo_model(frame)
                    
                    # Parse results (torch.hub YOLOv5 format)
                    # results.xyxy[0] contains [x1, y1, x2, y2, conf, class]
                    detections = results.xyxy[0].cpu().numpy()
                    
                    for det in detections:
                        x1, y1, x2, y2, conf, cls = det
                        cls = int(cls)
                        
                        # Only track high-confidence vehicle detections (class 2 = car)
                        if cls == 2 and conf > 0.5:
                            center_x = (x1 + x2) / 2
                            center_y = (y1 + y2) / 2
                            
                            detection_idx = len(self.detected_positions)
                            self.detected_positions.append({
                                'bbox': (x1, y1, x2, y2),
                                'center': (center_x, center_y),
                                'conf': float(conf),
                                'class': cls,
                                'camera': cam_name,
                                'index': detection_idx
                            })
                        
                        # Draw detection on frame
                        cv2.rectangle(frame, 
                                    (int(x1), int(y1)), 
                                    (int(x2), int(y2)), 
                                    (0, 255, 0), 2)
                        cv2.putText(frame, f'{conf:.2f}', 
                                  (int(x1), int(y1) - 5),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                                  (0, 255, 0), 1)
                except Exception as e:
                    pass  # Silent fail if detection fails
        
        # Redraw tracked vehicles with BLUE boxes to distinguish them
        if len(self.tracked_vehicles) > 0:
            ego_transform = self.ego_vehicle.get_transform()
            ego_pos = ego_transform.location
            
            # For each tracked vehicle, find matching detections and redraw in blue
            for veh_id in self.tracked_vehicles:
                try:
                    if veh_id not in traci.vehicle.getIDList():
                        continue
                    
                    # Get SUMO position
                    sumo_x, sumo_y = traci.vehicle.getPosition(veh_id)
                    
                    # Check all cameras to find where this vehicle appears
                    for cam_name in self.camera_names:
                        frame = camera_frames.get(cam_name)
                        if frame is None:
                            continue
                        
                        # Get camera transform
                        camera_mounts = self._build_camera_mounts()
                        cam_relative = camera_mounts[cam_name]
                        cam_world_transform = carla.Transform(
                            ego_transform.location + cam_relative.location,
                            carla.Rotation(
                                yaw=ego_transform.rotation.yaw + cam_relative.rotation.yaw,
                                pitch=ego_transform.rotation.pitch + cam_relative.rotation.pitch,
                                roll=ego_transform.rotation.roll + cam_relative.rotation.roll
                            )
                        )
                        
                        # Transform SUMO to CARLA coordinates
                        offset = self.sumo_carla_offset
                        carla_x = sumo_x - offset[0]
                        carla_y = -(sumo_y - offset[1])
                        world_pos = carla.Location(x=carla_x, y=carla_y, z=ego_pos.z)
                        
                        # Project to this camera
                        cam_coords = self._world_to_camera_coords(world_pos, cam_world_transform)
                        if cam_coords is None:
                            continue
                        
                        img_coords = self._project_to_image(cam_coords)
                        if img_coords is None:
                            continue
                        
                        # Find closest detection in this camera
                        for detection in self.detected_positions:
                            if detection['camera'] != cam_name:
                                continue
                            
                            det_center = detection['center']
                            pixel_dist = np.sqrt((img_coords[0] - det_center[0])**2 + 
                                               (img_coords[1] - det_center[1])**2)
                            
                            # If detection matches this tracked vehicle, redraw in blue
                            if pixel_dist < 150:
                                x1, y1, x2, y2 = detection['bbox']
                                conf = detection['conf']
                                
                                # Redraw with BLUE box (thicker)
                                cv2.rectangle(frame, 
                                            (int(x1), int(y1)), 
                                            (int(x2), int(y2)), 
                                            (255, 0, 0), 3)  # Blue, thicker
                                cv2.putText(frame, f'{conf:.2f} [T-{veh_id}]', 
                                          (int(x1), int(y1) - 5),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                                          (255, 0, 0), 2)
                                break  # Only redraw first match per camera
                except:
                    continue
        
        # Create 2x4 mosaic visualization
        mosaic = self._create_mosaic(camera_frames)
        
        # Add info overlay
        ego_transform = self.ego_vehicle.get_transform()
        
        # Get velocity from SUMO if available, otherwise fallback to CARLA
        if self.ego_id and self.ego_id in traci.vehicle.getIDList():
            speed_ms = traci.vehicle.getSpeed(self.ego_id)
            speed_kmh = speed_ms * 3.6
        else:
            velocity = self.ego_vehicle.get_velocity()
            speed_ms = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
            speed_kmh = speed_ms * 3.6
        
        info_text = [
            f'Speed: {speed_kmh:.1f} km/h',
            f'Detections: {len(self.detected_positions)}',
            f'Pos: ({ego_transform.location.x:.1f}, {ego_transform.location.y:.1f})'
        ]
        
        y_offset = 30
        for text in info_text:
            cv2.putText(mosaic, text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_offset += 30
        
        # Display mosaic
        if self.display is None:
            cv2.namedWindow('State Estimator - Surround View', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('State Estimator - Surround View', 1280, 640)
        
        cv2.imshow('State Estimator - Surround View', mosaic)
        key = cv2.waitKey(1)
        
        return key == 27  # Return True if ESC pressed
    
    def _create_mosaic(self, camera_frames):
        """Create 2x4 mosaic from 8 camera views."""
        h, w = CAMERA_H, CAMERA_W
        mosaic = np.zeros((h*2, w*4, 3), dtype=np.uint8)
        
        # Arrange cameras in 2x4 grid (matches phase9 layout)
        layout = [
            ['cam1', 'cam0', 'cam7', 'cam2'],  # Top row
            ['cam3', 'cam4', 'cam5', 'cam6']   # Bottom row
        ]
        
        for r, row in enumerate(layout):
            for c, cam_name in enumerate(row):
                if cam_name in camera_frames:
                    img = camera_frames[cam_name]
                    # Add camera label in top right corner
                    text_size = cv2.getTextSize(cam_name, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    text_x = w - text_size[0] - 10  # 10px padding from right edge
                    cv2.putText(img, cam_name, (text_x, 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    mosaic[r*h:(r+1)*h, c*w:(c+1)*w] = img
        
        return mosaic
    
    def notify_merged_on_mainline(self):
        """No longer gates tracking - kept for backwards compatibility."""
        pass
    
    def _lightweight_lidar_clustering(self):
        """Fast LiDAR clustering - groups nearby points without complex algorithms."""
        if self.lidar_data is None or len(self.lidar_data) == 0:
            return []
        
        points = self.lidar_data
        
        # Quick filtering (vectorized for speed)
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        dist = np.sqrt(x**2 + y**2)
        
        # Filter: reasonable height, not too close/far, exclude ego vehicle
        mask = (
            (z > -1.3) & (z < 2.8) &  # Vehicle height range
            (dist > 3.0) & (dist < 50.0) &  # 3-50m range
            ~((np.abs(x) < 2.7) & (np.abs(y) < 1.6))  # Exclude ego
        )
        
        filtered = points[mask]
        if len(filtered) < 10:
            return []
        
        # Simple grid-based clustering (much faster than iterative methods)
        grid_size = 3.0  # 3m grid cells
        grid_dict = {}
        
        for point in filtered[:, :2]:  # Only x, y
            grid_x = int(point[0] / grid_size)
            grid_y = int(point[1] / grid_size)
            key = (grid_x, grid_y)
            
            if key not in grid_dict:
                grid_dict[key] = []
            grid_dict[key].append(point)
        
        # Create clusters from grid cells with enough points
        clusters = []
        for cell_points in grid_dict.values():
            if len(cell_points) >= 8:  # Minimum points per cluster
                centroid = np.mean(cell_points, axis=0)
                clusters.append({
                    'centroid': centroid,
                    'count': len(cell_points)
                })
        
        return clusters
    
    def get_detected_count(self):
        """
        Get the number of detected vehicles in the last frame.
        
        Returns
        -------
        int
            Number of high-confidence vehicle detections
        """
        return len(self.detected_positions)
    
    def _world_to_camera_coords(self, world_pos, camera_transform):
        """
        Convert world position to camera coordinate system.
        
        Parameters
        ----------
        world_pos : carla.Location
            Position in world coordinates
        camera_transform : carla.Transform
            Camera transform (world space)
            
        Returns
        -------
        tuple
            (x, y, z) in camera space, or None if behind camera
        """
        # Get relative position
        dx = world_pos.x - camera_transform.location.x
        dy = world_pos.y - camera_transform.location.y
        dz = world_pos.z - camera_transform.location.z
        
        # Rotate to camera frame (CARLA uses left-hand Z-up coordinate system)
        yaw_rad = np.radians(camera_transform.rotation.yaw)
        pitch_rad = np.radians(camera_transform.rotation.pitch)
        
        # Simplified rotation (yaw only for now)
        cos_yaw = np.cos(yaw_rad)
        sin_yaw = np.sin(yaw_rad)
        
        # Camera-relative coordinates (forward=x, right=y, up=z in camera frame)
        cam_x = dx * cos_yaw + dy * sin_yaw
        cam_y = -dx * sin_yaw + dy * cos_yaw
        cam_z = dz
        
        # Check if behind camera
        if cam_x < 0:
            return None
            
        return (cam_x, cam_y, cam_z)
    
    def _project_to_image(self, cam_coords):
        """
        Project camera coordinates to image pixel coordinates.
        
        Parameters
        ----------
        cam_coords : tuple
            (x, y, z) in camera coordinate system
            
        Returns
        -------
        tuple
            (u, v) pixel coordinates, or None if out of view
        """
        cam_x, cam_y, cam_z = cam_coords
        
        # Pinhole camera projection
        fov_rad = np.radians(CAMERA_FOV)
        focal_length = CAMERA_W / (2.0 * np.tan(fov_rad / 2.0))
        
        # Project to normalized image plane
        u = (cam_y / cam_x) * focal_length + CAMERA_W / 2.0
        v = -(cam_z / cam_x) * focal_length + CAMERA_H / 2.0  # Negative because y-axis points down
        
        # Check if in image bounds
        if 0 <= u < CAMERA_W and 0 <= v < CAMERA_H:
            return (u, v)
        return None
    
    def match_detections_with_sumo(self, sumo_vehicles, ego_id, ego_sumo_pos=None):
        """
        Match YOLO detections with SUMO ground truth positions.
        Selects 2-3 nearby vehicles to track for position error based on proximity.
        
        Parameters
        ----------
        sumo_vehicles : dict
            Dictionary mapping vehicle_id -> (x, y) SUMO position
        ego_id : str
            Ego vehicle ID to exclude
        ego_sumo_pos : tuple, optional
            Ego position in SUMO coordinates (x, y). If None, fetches from traci.
            
        Returns
        -------
        list
            List of matched vehicles with position errors
        """
        matched = []
        
        # Get ego position in SUMO coordinates for accurate distance calculation
        if ego_sumo_pos is None:
            # Fallback: fetch from traci if not provided
            if ego_id and ego_id in traci.vehicle.getIDList():
                ego_sumo_pos = traci.vehicle.getPosition(ego_id)
            else:
                return matched  # Can't calculate without ego position
        
        ego_x, ego_y = ego_sumo_pos
        
        # Also get CARLA ego position for camera projections
        ego_transform = self.ego_vehicle.get_transform()
        ego_pos = ego_transform.location
        
        # If we don't have tracked vehicles yet, select 2-3 based on proximity
        if len(self.tracked_vehicles) < 3 and len(sumo_vehicles) > 0:
            # Find all vehicles with distances for debugging
            all_distances = []
            candidate_vehicles = []
            for veh_id, (sumo_x, sumo_y) in sumo_vehicles.items():
                if veh_id == ego_id or veh_id in self.tracked_vehicles:
                    continue
                
                # Calculate distance from ego using SUMO coordinates
                dist_from_ego = np.sqrt((sumo_x - ego_x)**2 + (sumo_y - ego_y)**2)
                all_distances.append(dist_from_ego)
                
                # Relaxed range: 5-80m (closer vehicles and full camera range)
                if 5.0 <= dist_from_ego <= 80.0:
                    candidate_vehicles.append((veh_id, dist_from_ego, sumo_x, sumo_y))
            
            # Sort by distance and take closest 3
            candidate_vehicles.sort(key=lambda x: x[1])
            
            # Track which SUMO vehicles we've already attempted to add (deduplication)
            attempted_vehicles = set()
            
            for veh_id, dist, sumo_x, sumo_y in candidate_vehicles[:10]:  # Check more candidates
                if len(self.tracked_vehicles) >= 3:
                    break
                
                # Skip if we already tried this vehicle
                if veh_id in attempted_vehicles:
                    continue
                attempted_vehicles.add(veh_id)
                
                # STRICT: Must match SUMO vehicle position with a detection
                best_detection = None
                best_pixel_dist = float('inf')
                best_cam = None
                
                for cam_name in self.camera_names:
                    camera_mounts = self._build_camera_mounts()
                    cam_relative = camera_mounts[cam_name]
                    cam_world_transform = carla.Transform(
                        ego_transform.location + cam_relative.location,
                        carla.Rotation(
                            yaw=ego_transform.rotation.yaw + cam_relative.rotation.yaw,
                            pitch=ego_transform.rotation.pitch + cam_relative.rotation.pitch,
                            roll=ego_transform.rotation.roll + cam_relative.rotation.roll
                        )
                    )
                    
                    # Transform SUMO coordinates to CARLA coordinates
                    offset = self.sumo_carla_offset
                    carla_x = sumo_x - offset[0]
                    carla_y = -(sumo_y - offset[1])  # Y axis is flipped
                    world_pos = carla.Location(x=carla_x, y=carla_y, z=ego_pos.z)
                    cam_coords = self._world_to_camera_coords(world_pos, cam_world_transform)
                    if cam_coords is not None:
                        img_coords = self._project_to_image(cam_coords)
                        if img_coords is not None:
                            # Find closest detection to SUMO vehicle projection in this camera
                            for detection in self.detected_positions:
                                if detection['camera'] != cam_name:
                                    continue
                                det_center = detection['center']
                                pixel_dist = np.sqrt((img_coords[0] - det_center[0])**2 + 
                                                   (img_coords[1] - det_center[1])**2)
                                if pixel_dist < best_pixel_dist and pixel_dist < 80:  # Very strict 80px threshold for initial tracking
                                    best_pixel_dist = pixel_dist
                                    best_detection = detection
                                    best_cam = cam_name
                
                # Only track if we found a strong match with SUMO vehicle position
                if best_detection and best_pixel_dist < 100:
                    self.tracked_vehicles.append(veh_id)
                    self.position_errors[veh_id] = []
                    self.position_error_times[veh_id] = []
                    self.drift_rates[veh_id] = []
                    self.tracking_frames[veh_id] = 0  # Initialize frame counter
                    self.lost_tracking_frames[veh_id] = 0  # Initialize lost tracking counter
                    print(f"[StateEstimator] ✓ Now tracking vehicle {veh_id} for position error (distance={dist:.1f}m, conf={best_detection['conf']:.2f}, cam={best_cam})")
        
        # Track which vehicles to remove (either left simulation or lost tracking)
        vehicles_to_remove = []
        for veh_id in self.tracked_vehicles:
            if veh_id not in sumo_vehicles:
                # Vehicle left simulation
                vehicles_to_remove.append(veh_id)
                print(f"[StateEstimator] Vehicle {veh_id} left simulation, removing from tracking")
        
        # Remove vehicles that left simulation and clean up state
        for veh_id in vehicles_to_remove:
            self.tracked_vehicles.remove(veh_id)
            # Clean up ALL tracking state dictionaries
            self.tracking_frames.pop(veh_id, None)
            self.lost_tracking_frames.pop(veh_id, None)
            self.smoothed_velocities.pop(veh_id, None)
            self.position_errors.pop(veh_id, None)
            self.position_error_times.pop(veh_id, None)
            self.drift_rates.pop(veh_id, None)
        
        # Calculate position errors for tracked vehicles
        for veh_id in self.tracked_vehicles:
            if veh_id not in sumo_vehicles:
                continue  # Vehicle left simulation (shouldn't happen after cleanup above)
            
            sumo_x, sumo_y = sumo_vehicles[veh_id]
            
            # Increment frame counter for this vehicle
            if veh_id in self.tracking_frames:
                self.tracking_frames[veh_id] += 1
            else:
                self.tracking_frames[veh_id] = 1
            
            # Find best matching detection across all cameras
            best_detection = None
            best_pixel_dist = float('inf')
            best_cam_name = None
            
            for cam_name in self.camera_names:
                # Get camera transform
                camera_mounts = self._build_camera_mounts()
                cam_relative = camera_mounts[cam_name]
                cam_world_transform = carla.Transform(
                    ego_transform.location + cam_relative.location,
                    carla.Rotation(
                        yaw=ego_transform.rotation.yaw + cam_relative.rotation.yaw,
                        pitch=ego_transform.rotation.pitch + cam_relative.rotation.pitch,
                        roll=ego_transform.rotation.roll + cam_relative.rotation.roll
                    )
                )
                
                # Project SUMO position to this camera (transform to CARLA coords first)
                offset = self.sumo_carla_offset
                carla_x = sumo_x - offset[0]
                carla_y = -(sumo_y - offset[1])
                world_pos = carla.Location(x=carla_x, y=carla_y, z=ego_pos.z)
                cam_coords = self._world_to_camera_coords(world_pos, cam_world_transform)
                if cam_coords is None:
                    continue
                
                img_coords = self._project_to_image(cam_coords)
                if img_coords is None:
                    continue
                
                # Find closest detection in this camera
                for detection in self.detected_positions:
                    if detection['camera'] != cam_name:
                        continue
                    
                    det_center = detection['center']
                    pixel_dist = np.sqrt((img_coords[0] - det_center[0])**2 + 
                                       (img_coords[1] - det_center[1])**2)
                    
                    if pixel_dist < best_pixel_dist:
                        best_pixel_dist = pixel_dist
                        best_detection = detection
                        best_cam_name = cam_name
            
            # Track consecutive lost frames for each vehicle
            # STRICT MATCHING: Only accept if pixel distance is below threshold AND detection not already claimed
            if best_detection and best_pixel_dist < 80:  # Tightened threshold: 80px to prevent false matches
                # Check if this detection is already claimed by another tracked vehicle
                detection_key = (best_cam_name, best_detection['index'])
                if detection_key not in self.tracked_vehicle_detections:
                    # Reset lost tracking counter when detection found
                    self.lost_tracking_frames[veh_id] = 0
                    
                    # Mark this detection as belonging to a tracked vehicle
                    self.tracked_vehicle_detections.add(detection_key)
                    
                    # Only compute position error if tracked for minimum duration (filter out transient detections)
                    if self.tracking_frames[veh_id] >= self.MIN_TRACKING_FRAMES:
                        lidar_confidence = 1.0
                        if len(self.lidar_clusters) > 0:
                            # Transform SUMO to CARLA ego frame for comparison with LiDAR
                            offset = self.sumo_carla_offset
                            carla_x_veh = sumo_x - offset[0]
                            carla_y_veh = -(sumo_y - offset[1])
                            veh_ego_x = carla_x_veh - ego_transform.location.x
                            veh_ego_y = carla_y_veh - ego_transform.location.y
                            
                            # Check if any LiDAR cluster is near this vehicle
                            for cluster in self.lidar_clusters:
                                cluster_x, cluster_y = cluster['centroid']
                                cluster_dist = np.sqrt((cluster_x - veh_ego_x)**2 + (cluster_y - veh_ego_y)**2)
                                if cluster_dist < 4.0:  # LiDAR cluster within 4m of vehicle
                                    lidar_confidence = 0.95  # Slightly reduce error (more confident)
                                    break
                        
                        # For position error, we use 2D world distance between SUMO and detected position
                        # Since we don't have depth from YOLO, we estimate based on bbox size
                        bbox_area = (best_detection['bbox'][2] - best_detection['bbox'][0]) * \
                                   (best_detection['bbox'][3] - best_detection['bbox'][1])
                        # Rough distance estimate: larger bbox = closer
                        estimated_distance = 60.0 / (np.sqrt(bbox_area) / 100.0 + 1.0)
                        
                        # Position error: compare SUMO ground truth distance with estimated distance
                        # Get vehicle length and heading to adjust the reference point
                        try:
                            veh_length = traci.vehicle.getLength(veh_id)
                            veh_angle = traci.vehicle.getAngle(veh_id)  # degrees, 0=North, 90=East
                            # Convert angle to radians and adjust for coordinate system (SUMO angle: 0=North, 90=East)
                            angle_rad = np.deg2rad(90 - veh_angle)  # Convert to standard math angle (0=East)
                            
                            # Vector from ego to vehicle (in SUMO coordinates)
                            dx = sumo_x - ego_x
                            dy = sumo_y - ego_y
                            angle_to_vehicle = np.arctan2(dy, dx)
                            
                            # If vehicle is moving away from ego, bbox shows rear; if approaching, shows front
                            angle_diff = angle_to_vehicle - angle_rad
                            # Normalize to [-pi, pi]
                            angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi
                            
                            # Use half vehicle length as offset along vehicle's heading
                            if abs(angle_diff) < np.pi / 2:
                                # Seeing front - SUMO center is behind the visible point
                                offset_dist = veh_length / 2.0
                            else:
                                # Seeing rear - SUMO center is ahead of the visible point
                                offset_dist = -veh_length / 2.0
                            
                            # Adjust SUMO position to front/rear reference point (in SUMO coordinates)
                            adjusted_sumo_x = sumo_x + offset_dist * np.cos(angle_rad)
                            adjusted_sumo_y = sumo_y + offset_dist * np.sin(angle_rad)
                            
                            # Transform adjusted SUMO position to CARLA coordinates
                            offset = self.sumo_carla_offset
                            adjusted_carla_x = adjusted_sumo_x - offset[0]
                            adjusted_carla_y = -(adjusted_sumo_y - offset[1])
                            
                            # Transform ego SUMO position to CARLA coordinates
                            ego_carla_x = ego_x - offset[0]
                            ego_carla_y = -(ego_y - offset[1])
                            
                            # Calculate distance in CARLA coordinate space
                            actual_distance = np.sqrt((adjusted_carla_x - ego_carla_x)**2 + (adjusted_carla_y - ego_carla_y)**2)
                        except Exception as e:
                            # Fallback: use center point if vehicle info unavailable
                            # Transform to CARLA coordinates
                            offset = self.sumo_carla_offset
                            veh_carla_x = sumo_x - offset[0]
                            veh_carla_y = -(sumo_y - offset[1])
                            ego_carla_x = ego_x - offset[0]
                            ego_carla_y = -(ego_y - offset[1])
                            actual_distance = np.sqrt((veh_carla_x - ego_carla_x)**2 + (veh_carla_y - ego_carla_y)**2)
                        
                        position_error = abs(actual_distance - estimated_distance)
                        
                        # Apply lidar confidence adjustment if available (reduces error slightly when LiDAR confirms)
                        if lidar_confidence < 1.0:
                            position_error = position_error * lidar_confidence
                        
                        matched.append({
                            'vehicle_id': veh_id,
                            'sumo_pos': (sumo_x, sumo_y),
                            'actual_distance': actual_distance,
                            'estimated_distance': estimated_distance,
                            'position_error': position_error,
                            'confidence': best_detection['conf'],
                            'camera': best_cam_name,
                            'pixel_distance': best_pixel_dist
                        })
                        
                        self.position_errors[veh_id].append(position_error)
                        
                        # Calculate drift rate with velocity smoothing for robustness
                        if hasattr(self, 'ego_vehicle') and self.ego_vehicle is not None:
                            current_time = self.ego_vehicle.get_world().get_snapshot().timestamp.elapsed_seconds
                            self.position_error_times[veh_id].append(current_time)
                            
                            # Calculate instantaneous drift rate
                            if len(self.position_errors[veh_id]) >= 2:
                                prev_error = self.position_errors[veh_id][-2]
                                curr_error = position_error
                                prev_time = self.position_error_times[veh_id][-2]
                                curr_time = current_time
                                time_delta = curr_time - prev_time
                                
                                if time_delta > 0:
                                    raw_drift_rate = (curr_error - prev_error) / time_delta
                                
                                # Apply exponential moving average for smoothing
                                if veh_id in self.smoothed_velocities:
                                    smoothed_drift = (self.velocity_alpha * raw_drift_rate + 
                                                    (1 - self.velocity_alpha) * self.smoothed_velocities[veh_id])
                                else:
                                    smoothed_drift = raw_drift_rate
                                
                                self.smoothed_velocities[veh_id] = smoothed_drift
                                self.drift_rates[veh_id].append(smoothed_drift)
                else:
                    # Detection already claimed by another tracked vehicle - treat as no match
                    if veh_id not in self.lost_tracking_frames:
                        self.lost_tracking_frames[veh_id] = 0
                    self.lost_tracking_frames[veh_id] += 1
            else:
                # No detection found for this tracked vehicle (pixel distance > 80px or no detection at all)
                if veh_id not in self.lost_tracking_frames:
                    self.lost_tracking_frames[veh_id] = 0
                self.lost_tracking_frames[veh_id] += 1
        
        # Remove vehicles that have lost tracking for too long
        vehicles_to_remove = []
        for veh_id in self.tracked_vehicles:
            if veh_id in self.lost_tracking_frames and self.lost_tracking_frames[veh_id] >= self.MAX_LOST_FRAMES:
                vehicles_to_remove.append(veh_id)
                print(f"[StateEstimator] Vehicle {veh_id} lost tracking for {self.lost_tracking_frames[veh_id]} frames, removing from tracking")
        
        for veh_id in vehicles_to_remove:
            self.tracked_vehicles.remove(veh_id)
            # Clean up ALL tracking state dictionaries
            self.tracking_frames.pop(veh_id, None)
            self.lost_tracking_frames.pop(veh_id, None)
            self.smoothed_velocities.pop(veh_id, None)
            self.position_errors.pop(veh_id, None)
            self.position_error_times.pop(veh_id, None)
            self.drift_rates.pop(veh_id, None)
            # Clean up tracking state dictionaries
            self.tracking_frames.pop(veh_id, None)
            self.lost_tracking_frames.pop(veh_id, None)
            self.smoothed_velocities.pop(veh_id, None)
        
        return matched
    
    def destroy(self):
        """Clean up sensors and visualization."""
        print("[StateEstimator] Destroying sensors...")
        
        for sensor in self.sensors:
            if sensor is not None and sensor.is_alive:
                sensor.stop()
                sensor.destroy()
        
        cv2.destroyAllWindows()
        for _ in range(4):
            cv2.waitKey(1)
        
        print("[StateEstimator] Cleanup complete")