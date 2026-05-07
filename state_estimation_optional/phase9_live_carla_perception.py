# ============================================
# PHASE 9 — WORKING BASE + CLEAN MOSAIC
# ============================================
import glob
import importlib.util
import os
import sys

def bootstrap_carla_api():
    carla_found = importlib.util.find_spec("carla") is not None
    version_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    search_roots = [
        os.environ.get("CARLA_ROOT"),
        os.path.expanduser("~/CARLA"),
        os.path.expanduser("~/carla"),
        "/opt/carla-simulator",
        "D:\\CARLA",
        "C:\\CARLA",
    ]
    search_roots.extend(sorted(glob.glob(os.path.expanduser("~/CARLA_*"))))
    search_roots.extend(sorted(glob.glob("/opt/CARLA*")))
    search_roots = [p for p in dict.fromkeys(search_roots) if p]
    for root in search_roots:
        if not root or not os.path.isdir(root):
            continue
        pattern = os.path.join(root, "PythonAPI", "carla", "dist", f"carla-*-{version_tag}-*.whl")
        matches = sorted(glob.glob(pattern))
        if matches and not carla_found:
            sys.path.insert(0, matches[-1])
            carla_found = True
        agents_dir = os.path.join(root, "PythonAPI", "carla")
        if os.path.isdir(agents_dir) and agents_dir not in sys.path:
            sys.path.insert(0, agents_dir)
        if matches and carla_found:
            return matches[-1]
    return None

bootstrap_carla_api()
import carla
import numpy as np
import cv2
import math
import random
import csv
from collections import defaultdict
from ultralytics import YOLO
import threading
import time
try:
    from agents.navigation.local_planner import LocalPlanner, RoadOption
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "CARLA Python agents package not found. Set CARLA_ROOT to your CARLA installation "
        "folder (for example: export CARLA_ROOT=~/CARLA_0.9.15) so PythonAPI/carla is added "
        "to sys.path."
    ) from exc

STEP = 0.05
NPC_COUNT = 30
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
LIDAR_SELF_HALF_X = 2.7
LIDAR_SELF_HALF_Y = 1.6
LIDAR_GRID_M = 0.6
MAX_TRACK_AGE = 20  # increased for longer persistence
DET_EVERY_N = 1
DET_SCALE = 0.75
# UI options
CAM_HITS_CONFIRM = 2
# 3D visualization
SHOW_3D_BOX = True
SHOW_HEADING_VECTOR = True
SHOW_PREDICTION_TRAIL = True
BOX_LENGTH = 4.0
BOX_WIDTH = 2.0
BOX_HEIGHT = 1.5
VECTOR_DISTANCE = 3.0
WGS84_R = 6378137.0

# Fusion / quality thresholds  (tuned for smoothness)
LIDAR_CLUSTER_DIST = 2.5
LIDAR_CLUSTER_MIN_POINTS = 6
LIDAR_MATCH_DIST = 5.0  # tighter gating
CAM_MATCH_PIX = 110.0
CAM_UPDATE_DIST = 7.0
CAM_SEED_RANGE_M = 55.0
MIN_DET_AREA_PX = 100
MAX_DET_ASPECT = 4.5
BOTTOM_CLIP_RATIO = 0.97
BOTTOM_SELF_AREA_RATIO = 0.08
YOLO_MIN_CONF = 0.18
MIN_VEHICLE_SHORT_SIDE_M = 0.6
MAX_VEHICLE_SHORT_SIDE_M = 3.2
MIN_VEHICLE_LONG_SIDE_M = 1.0
MAX_VEHICLE_LONG_SIDE_M = 8.5
MIN_VEHICLE_HEIGHT_M = 0.8
MAX_VEHICLE_HEIGHT_M = 4.0
ROAD_RELEVANCE_DIST_M = 3.5
LANE_KEEP_PROJECTION_MAX_ERR_M = 2.6
LANE_KEEP_MIN_FRACTION = 0.60
LANE_PATH_SAMPLE_STEP_M = 3.0
EGO_PATH_LOOKAHEAD_M = 45.0
TURN_LANE_CLEARANCE_M = 3.4
TURN_LANE_RISK_SCALE = 0.50
TURN_LANE_RISK_CAP = 0.35
BEV_CLUSTER_ASSOC_DIST_M = 5.5
CAMERA_SEED_CONFIRM_FRAMES = 2
CAMERA_PERSIST_CONFIRM_HITS = 6
CAMERA_PERSIST_STRONG_HITS = 10
CAMERA_PERSIST_RANGE_M = 45.0
CAMERA_PERSIST_LATERAL_M = 8.5
CAMERA_PERSIST_MIN_CONFIDENCE = 0.32
CAMERA_PERSIST_MAX_MISS = 3
CAMERA_SEED_GRID_X_M = 2.5
CAMERA_SEED_GRID_Y_M = 1.8
CAMERA_SEED_TTL_FRAMES = 6
MAX_CAMERA_ONLY_RANGE_M = 28.0
CAMERA_FRONT_OBS_MIN_X_M = -12.0
CAMERA_SIDE_REAR_OBS_MIN_X_M = -18.0
CAMERA_FRONT_SEED_MIN_X_M = -6.0
CAMERA_SIDE_REAR_SEED_MIN_X_M = -10.0
SURROUND_CAMS_PER_CYCLE = 8
TRAFFIC_LIGHT_LOOKAHEAD_M = 38.0
TRAFFIC_LIGHT_LATERAL_M = 5.0
TRAFFIC_LIGHT_STOP_BUFFER_M = 1.0
TRAFFIC_LIGHT_QUEUE_GAP_M = 4.5
STANLEY_GAIN = 1.4
STANLEY_SOFTENING = 1.5
OBSTACLE_SENSOR_DISTANCE_M = 8.0
OBSTACLE_SENSOR_RADIUS_M = 0.4
OBSTACLE_EVENT_TTL_S = 0.25
OBSTACLE_STOP_BUFFER_M = 1.5
OBSTACLE_BRAKE_DISTANCE_M = 5.0
OBSTACLE_LATERAL_GATE_M = 1.35
LIDAR_CORRIDOR_RANGE_M = 16.0
LIDAR_CORRIDOR_HALF_WIDTH_M = 1.15
LIDAR_CORRIDOR_MIN_POINTS = 8
CONFIRMATION_CONFIDENCE = 0.6
DECONFIRM_CONFIDENCE = 0.2
MIN_SPEED_MPS = 0.15  # lower threshold for consistency check
CRUISE_SPEED_MPS = 31.3  # 70 mph highway cap; city roads still follow CARLA limits.
SPEED_LIMIT_TRACK_FACTOR = 1.00
MIN_FOLLOW_DISTANCE_M = 5.0
FOLLOW_TIME_GAP_S = 1.45
IDM_MIN_GAP_M = 3.0
IDM_TIME_HEADWAY_S = 1.4
IDM_COMFORT_ACCEL_MPS2 = 1.8
IDM_COMFORT_BRAKE_MPS2 = 2.7
IDM_DELTA = 4.0
DRIVE_MAX_ACCEL_MPS2 = 2.1
DRIVE_MAX_DECEL_MPS2 = 2.4
DRIVE_CAUTION_DECEL_MPS2 = 3.0
DRIVE_STOP_DECEL_MPS2 = 4.0
DRIVE_EMERGENCY_DECEL_MPS2 = 6.0
BRAKE_APPLY_RATE = 2.2
BRAKE_RELEASE_RATE = 3.2
GREEN_RESUME_ACCEL_MPS2 = 1.4
GREEN_RESUME_HOLD_S = 0.8
TURN_PREVIEW_M = 28.0
TURN_PREVIEW_STEP_M = 4.0
TURN_SPEED_CAP_HARD_MPS = 9.0
TURN_SPEED_CAP_MED_MPS = 11.8
TURN_SPEED_CAP_SOFT_MPS = 14.2
JUNCTION_APPROACH_SPEED_MPS = 12.5
JUNCTION_APPROACH_DIST_M = 18.0
STRAIGHT_JUNCTION_MAX_YAW_DEG = 12.0
ROUTE_SOFT_TURN_YAW_DEG = 20.0
ROUTE_MED_TURN_YAW_DEG = 34.0
ROUTE_HARD_TURN_YAW_DEG = 52.0
HIGHWAY_SPEED_LIMIT_MIN_MPS = 24.0
HIGHWAY_TARGET_SPEED_MPS = 29.0
HIGHWAY_MIN_LANES = 2
LANE_CHANGE_YAW_TOL_DEG = 32.0
LANE_END_LOOKAHEAD_M = 85.0
LANE_CHANGE_LOOKAHEAD_M = 16.0
LANE_CHANGE_COMMIT_DIST_M = 42.0
LANE_CHANGE_MIN_FRONT_GAP_M = 20.0
LANE_CHANGE_MIN_REAR_GAP_M = 14.0
LANE_CHANGE_FRONT_TTC_S = 2.6
LANE_CHANGE_REAR_TTC_S = 3.0
MERGE_WAIT_SPEED_MPS = 8.0
CAUTION_TTC_S = 3.0
EMERGENCY_TTC_S = 1.5
CAUTION_COLLISION_S = 2.5
EMERGENCY_COLLISION_S = 1.25
PREDICTION_HORIZON_S = 3.5
PREDICTION_DT_S = 0.5
LANE_HALF_WIDTH_M = 1.8
EGO_HALF_LENGTH_M = 2.4
EGO_HALF_WIDTH_M = 1.1
GNSS_STD_M = 1.5
R_POS = GNSS_STD_M ** 2
SPEED_STD = 0.5
YAWRATE_STD = 4.0
R_SPEED = SPEED_STD ** 2
R_YAWRATE = YAWRATE_STD ** 2
IMU_ACCEL_CLAMP_MPS2 = 8.0
EGO_PROCESS_POS_Q = 0.4
EGO_PROCESS_VEL_Q = 1.2
EGO_PROCESS_YAW_Q = 0.6
EGO_PROCESS_YAWRATE_Q = 2.5
TL_CLASS_ID = 9
TL_MIN_CONF = 0.16
TL_MIN_AREA_PX = 18
TL_MEMORY_S = 0.7
TL_SCORE_MIN = 0.015
TL_CONTROL_CONF_MIN = 0.08
TL_DET_BOTTOM_RATIO = 0.82
TL_JUNCTION_STEP_M = 4.0
TL_YELLOW_COMMIT_DISTANCE_M = 10.0
TL_YELLOW_COMMIT_SPEED_MPS = 7.5
RADAR_RANGE_M = 90.0
RADAR_HFOV_DEG = 70.0
RADAR_VFOV_DEG = 12.0
RADAR_POINTS_PER_SECOND = 1800
RADAR_Z = 0.7
RADAR_X_FRONT = 2.0
RADAR_X_REAR = -2.0
RADAR_Y_LEFT = -0.9
RADAR_Y_RIGHT = 0.9
RADAR_MEMORY_S = 0.30
RADAR_MIN_Z = -0.8
RADAR_MAX_Z = 3.5
RADAR_ASSOC_DIST_M = 4.0
RADAR_GRID_M = 2.0
RADAR_CLUSTER_MIN_POINTS = 2
MERGE_RADAR_LATERAL_MIN_M = 2.0
MERGE_RADAR_LATERAL_MAX_M = 9.5
MERGE_RADAR_REAR_X_MIN_M = -35.0
MERGE_RADAR_FRONT_X_MAX_M = 35.0
MERGE_CAUTION_TTC_S = 2.1
MERGE_EMERGENCY_TTC_S = 1.3
MERGE_MIN_CLOSING_SPEED_MPS = 3.0
CAM_TRACK_HOLD_S = 1.3
SIDE_REAR_CAM_TRACK_HOLD_S = 1.9
RECENT_CAM_WINDOW_S = 0.95
RECENT_LIDAR_WINDOW_S = 0.45
RECENT_RADAR_WINDOW_S = 0.40
CAM_GLOBAL_CLUSTER_DIST_M = 3.5
CAM_GLOBAL_ASSOC_DIST_M = 4.5
CAMERA_ADJ_CLUSTER_BONUS_M = 1.4
CAMERA_ADJ_ASSOC_BONUS_M = 1.8
CAMERA_SIDE_REAR_ANCHOR_RATIO = 0.80
CAMERA_REAR_ANCHOR_RATIO = 0.74
TRACK_FORWARD_RELEVANCE_M = 90.0
TRACK_REAR_RELEVANCE_M = 35.0
TRACK_LATERAL_RELEVANCE_M = 10.5
MERGE_TRACK_LATERAL_M = 14.0
MAX_PUBLISHED_SIDE_TRACKS_PER_SIDE = 2
MAX_PUBLISHED_TRACKS = 8
CLOSE_SIDE_CAMERA_X_MIN_M = -6.0
CLOSE_SIDE_CAMERA_X_MAX_M = 18.0
CLOSE_SIDE_CAMERA_LATERAL_MIN_M = 4.5
CLOSE_SIDE_CAMERA_MIN_HITS = 3
LIDAR_ONLY_FRONT_MAX_X_M = 28.0
LIDAR_ONLY_LATERAL_M = 4.2
LIDAR_ONLY_WIDE_LATERAL_M = 8.5
MIN_STRONG_LIDAR_POINTS = 10
PARALLEL_LANE_YAW_ERR_DEG = 28.0
PARALLEL_PASSER_MAX_VY_MPS = 1.8
PARALLEL_PASSER_RISK_SCALE = 0.28
PARALLEL_PASSER_RISK_CAP = 0.18
MERGE_LATERAL_APPROACH_MIN_MPS = 0.55
MERGE_CAUTION_DISTANCE_M = 18.0
MERGE_EMERGENCY_DISTANCE_M = 11.0
MERGE_PATH_OVERLAP_GATE_M = 2.7
RADAR_SEED_CONFIRM_FRAMES = 2
RADAR_SEED_GRID_X_M = 4.0
RADAR_SEED_GRID_Y_M = 2.5
RADAR_SEED_TTL_FRAMES = 8
GT_EVAL_RANGE_M = 85.0
GT_MATCH_DIST_M = 8.0
GT_EVAL_WRITE_EVERY = 5
USE_TM_LANE_KEEP = True
ENABLE_OBSTACLE_BACKUP = False
USE_BUILTIN_ROUTE_DRIVER = False
ROUTE_REPLAN_MIN_DIST_M = 120.0
ROUTE_REACHED_DIST_M = 18.0
TM_STALL_TARGET_MPS = 5.0
TM_STALL_SPEED_RATIO = 0.45
TM_STALL_HOLD_S = 8.0
ENABLE_TRAFFIC_LIGHT_CONTROL = True
GT_EVAL_FORWARD_M = 75.0
GT_EVAL_REAR_M = 25.0
GT_EVAL_LATERAL_M = 14.0
FRONT_CAMERA_NAMES = {"front", "cam0", "cam1", "cam7"}
CAMERA_RING_ORDER = ("cam0", "cam1", "cam2", "cam3", "cam4", "cam5", "cam6", "cam7")
SAVE_TEST_VIDEO = True
TEST_VIDEO_BASENAME = "phase9 test video"
TEST_VIDEO_FPS = 20.0
TEST_VIDEO_CODEC_OPTIONS = (
    ("MJPG", ".avi"),
    ("XVID", ".avi"),
    ("mp4v", ".mp4"),
)
MAX_RUN_SECONDS = float(os.environ.get("PHASE9_MAX_RUN_SECONDS", "0") or 0.0)

img_rgb = None
lidar_data = None
cams = {}
cams_transforms = {}
cams_dets = {}
cams_frames = {}
cams_tl_dets = {}
camera_seed_memory = {}
radar_seed_memory = {}
obstacle_events = {}
gnss_data = None
imu_data = None
radar_returns = {}
traffic_light_memory = {
    "state_name": "None",
    "score": 0.0,
    "bbox": None,
    "ts": 0.0,
}
ego_state_filter = None
eval_state = {}
control_state = {}

# Inference thread control
inference_running = False
frame_idx = 0
frame_lock = threading.Lock()

model = YOLO("yolov8n.pt")

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def speed_xy(x, y):
    return float(math.hypot(x, y))

def blended_control_speed(actor_speed, estimated_speed):
    actor_speed = max(0.0, float(actor_speed))
    if estimated_speed is None or not np.isfinite(estimated_speed):
        return actor_speed
    estimated_speed = max(0.0, float(estimated_speed))
    disagreement = abs(estimated_speed - actor_speed)
    if disagreement >= 3.0:
        return actor_speed
    weight = clamp(1.0 - disagreement / 3.0, 0.0, 0.45)
    return (1.0 - weight) * actor_speed + weight * estimated_speed

def wrap_deg(a):
    while a > 180.0:
        a -= 360.0
    while a < -180.0:
        a += 360.0
    return a

def ll_to_enu_m(lat_deg, lon_deg, lat0_deg, lon0_deg):
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    lat0 = math.radians(lat0_deg)
    lon0 = math.radians(lon0_deg)
    east = (lon - lon0) * math.cos(lat0) * WGS84_R
    north = (lat - lat0) * WGS84_R
    return east, north

def imu_compass_to_yaw_deg(compass_rad):
    # CARLA compass is 0 at north and pi/2 at east, while actor yaw is 0 at +X.
    return wrap_deg(math.degrees(compass_rad) - 90.0)

def traffic_light_state_name(state):
    mapping = {
        carla.TrafficLightState.Red: "Red",
        carla.TrafficLightState.Yellow: "Yellow",
        carla.TrafficLightState.Green: "Green",
        carla.TrafficLightState.Off: "Off",
        carla.TrafficLightState.Unknown: "Unknown",
    }
    return mapping.get(state, "None")

def ego_location_from_state(ego_state):
    return carla.Location(
        x=float(ego_state["x"]),
        y=float(ego_state["y"]),
        z=float(ego_state.get("z", 0.0)),
    )

def ego_transform_from_state(ego_state):
    return carla.Transform(
        ego_location_from_state(ego_state),
        carla.Rotation(yaw=float(ego_state["yaw_deg"])),
    )

def world_to_ego_xy(x_world, y_world, ego_state):
    dx = float(x_world - ego_state["x"])
    dy = float(y_world - ego_state["y"])
    yaw = math.radians(float(ego_state["yaw_deg"]))
    cy = math.cos(yaw)
    sy = math.sin(yaw)
    return cy * dx + sy * dy, -sy * dx + cy * dy

def ego_state_from_actor(ego):
    tf = ego.get_transform()
    vel = ego.get_velocity()
    return {
        "x": float(tf.location.x),
        "y": float(tf.location.y),
        "z": float(tf.location.z),
        "vx": float(vel.x),
        "vy": float(vel.y),
        "speed": speed_xy(vel.x, vel.y),
        "yaw_deg": float(tf.rotation.yaw),
        "yaw_rate_dps": 0.0,
        "source": "CARLA_GT",
    }

def init_eval_state():
    out_dir = os.path.join(os.path.dirname(__file__), "eval")
    os.makedirs(out_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(out_dir, f"phase9_tracking_eval_{ts}.csv")
    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    writer = csv.writer(csv_file)
    writer.writerow([
        "frame",
        "matched",
        "gt_count",
        "track_count",
        "misses",
        "false_tracks",
        "mean_pos_err_m",
        "mean_vel_err_mps",
        "ego_pos_err_m",
        "ego_yaw_err_deg",
        "id_switches_total",
    ])
    return {
        "csv_file": csv_file,
        "writer": writer,
        "csv_path": csv_path,
        "gt_to_track": {},
        "id_switches": 0,
        "matched_events": 0,
        "sum_pos_err": 0.0,
        "sum_vel_err": 0.0,
        "sum_misses": 0,
        "sum_false_tracks": 0,
        "frames": 0,
    }

def surrounding_vehicle_truth(ego, world, max_range_m=GT_EVAL_RANGE_M):
    ego_tf = ego.get_transform()
    ego_loc = ego_tf.location
    ego_vel = ego.get_velocity()
    yaw = math.radians(ego_tf.rotation.yaw)
    cy = math.cos(yaw)
    sy = math.sin(yaw)
    truth = []
    for actor in world.get_actors().filter("vehicle.*"):
        if actor.id == ego.id:
            continue
        loc = actor.get_location()
        dx = float(loc.x - ego_loc.x)
        dy = float(loc.y - ego_loc.y)
        x_rel = cy * dx + sy * dy
        y_rel = -sy * dx + cy * dy
        dist = speed_xy(x_rel, y_rel)
        if dist > max_range_m:
            continue
        if x_rel < -GT_EVAL_REAR_M or x_rel > GT_EVAL_FORWARD_M or abs(y_rel) > GT_EVAL_LATERAL_M:
            continue
        vel = actor.get_velocity()
        dvx = float(vel.x - ego_vel.x)
        dvy = float(vel.y - ego_vel.y)
        vx_rel = cy * dvx + sy * dvy
        vy_rel = -sy * dvx + cy * dvy
        truth.append({
            "actor_id": actor.id,
            "x": x_rel,
            "y": y_rel,
            "vx": vx_rel,
            "vy": vy_rel,
            "distance": dist,
            "type_id": actor.type_id,
        })
    truth.sort(key=lambda item: item["distance"])
    return truth

def evaluate_tracking_against_gt(ego, world, world_model, frame_idx):
    global eval_state
    gt = surrounding_vehicle_truth(ego, world)
    preds = [
        entry for entry in world_model["tracks"]
        if entry["distance"] <= GT_EVAL_RANGE_M
        and -GT_EVAL_REAR_M <= entry["x"] <= GT_EVAL_FORWARD_M
        and abs(entry["y"]) <= GT_EVAL_LATERAL_M
    ]

    pairs = []
    for gi, g in enumerate(gt):
        for pi, p in enumerate(preds):
            pos_err = speed_xy(g["x"] - p["x"], g["y"] - p["y"])
            if pos_err <= GT_MATCH_DIST_M:
                vel_err = speed_xy(g["vx"] - p["vx"], g["vy"] - p["vy"])
                cost = pos_err + 0.35 * vel_err
                pairs.append((cost, gi, pi, pos_err, vel_err))

    pairs.sort(key=lambda item: item[0])
    matched_gt = set()
    matched_pred = set()
    matches = []

    for _, gi, pi, pos_err, vel_err in pairs:
        if gi in matched_gt or pi in matched_pred:
            continue
        matched_gt.add(gi)
        matched_pred.add(pi)
        matches.append((gt[gi], preds[pi], pos_err, vel_err))

    matched = len(matches)
    misses = len(gt) - matched
    false_tracks = len(preds) - matched
    mean_pos_err = float(np.mean([m[2] for m in matches])) if matches else float("nan")
    mean_vel_err = float(np.mean([m[3] for m in matches])) if matches else float("nan")

    for g, p, _, _ in matches:
        prev = eval_state["gt_to_track"].get(g["actor_id"])
        if prev is not None and prev != p["id"]:
            eval_state["id_switches"] += 1
        eval_state["gt_to_track"][g["actor_id"]] = p["id"]

    eval_state["frames"] += 1
    eval_state["matched_events"] += matched
    eval_state["sum_misses"] += misses
    eval_state["sum_false_tracks"] += false_tracks
    if matches:
        eval_state["sum_pos_err"] += float(np.sum([m[2] for m in matches]))
        eval_state["sum_vel_err"] += float(np.sum([m[3] for m in matches]))

    ego_est = world_model.get("ego_estimate")
    ego_gt = ego_state_from_actor(ego)
    ego_pos_err = float("nan")
    ego_yaw_err = float("nan")
    if ego_est is not None:
        ego_pos_err = speed_xy(ego_est["x"] - ego_gt["x"], ego_est["y"] - ego_gt["y"])
        ego_yaw_err = abs(wrap_deg(ego_est["yaw_deg"] - ego_gt["yaw_deg"]))

    if frame_idx % GT_EVAL_WRITE_EVERY == 0:
        eval_state["writer"].writerow([
            frame_idx,
            matched,
            len(gt),
            len(preds),
            misses,
            false_tracks,
            mean_pos_err,
            mean_vel_err,
            ego_pos_err,
            ego_yaw_err,
            eval_state["id_switches"],
        ])
        eval_state["csv_file"].flush()

    running_mean_pos = eval_state["sum_pos_err"] / max(1, eval_state["matched_events"])
    running_mean_vel = eval_state["sum_vel_err"] / max(1, eval_state["matched_events"])

    if frame_idx % 60 == 0 and len(gt) > 0:
        gt_dbg = [(item["actor_id"], round(item["x"], 1), round(item["y"], 1)) for item in gt[:6]]
        pred_dbg = [(item["id"], round(item["x"], 1), round(item["y"], 1)) for item in preds[:6]]
        print(f"[GTDBG] frame={frame_idx} gt={gt_dbg} preds={pred_dbg} matched={matched}")

    return {
        "matched": matched,
        "gt_count": len(gt),
        "track_count": len(preds),
        "misses": misses,
        "false_tracks": false_tracks,
        "mean_pos_err": mean_pos_err,
        "mean_vel_err": mean_vel_err,
        "running_pos_err": running_mean_pos,
        "running_vel_err": running_mean_vel,
        "id_switches": eval_state["id_switches"],
        "ego_pos_err": ego_pos_err,
        "ego_yaw_err": ego_yaw_err,
        "csv_path": eval_state["csv_path"],
    }

def det_footpoint(det):
    return det[4], det[3]

def det_conf(det):
    return det[6] if len(det) > 6 else 0.5

def det_contains(det, u, v, margin=0):
    x1, y1, x2, y2 = det[:4]
    return (x1 - margin) <= u <= (x2 + margin) and (y1 - margin) <= v <= (y2 + margin)

def cluster_vehicle_like(length, width, height):
    long_side = max(length, width)
    short_side = min(length, width)
    return (
        MIN_VEHICLE_LONG_SIDE_M <= long_side <= MAX_VEHICLE_LONG_SIDE_M and
        MIN_VEHICLE_SHORT_SIDE_M <= short_side <= MAX_VEHICLE_SHORT_SIDE_M and
        MIN_VEHICLE_HEIGHT_M <= height <= MAX_VEHICLE_HEIGHT_M
    )

def cluster_vehicle_candidate(length, width, height, range_m):
    long_side = max(length, width)
    short_side = min(length, width)

    if range_m > 35.0:
        min_long = 0.45
        min_short = 0.18
        min_height = 0.25
    elif range_m > 20.0:
        min_long = 0.70
        min_short = 0.25
        min_height = 0.40
    else:
        min_long = MIN_VEHICLE_LONG_SIDE_M
        min_short = MIN_VEHICLE_SHORT_SIDE_M
        min_height = MIN_VEHICLE_HEIGHT_M

    return (
        min_long <= long_side <= MAX_VEHICLE_LONG_SIDE_M and
        min_short <= short_side <= MAX_VEHICLE_SHORT_SIDE_M and
        min_height <= height <= MAX_VEHICLE_HEIGHT_M
    )

def track_vehicle_like(track):
    return cluster_vehicle_like(track.length, track.width, track.height)

def recent_track_support(track, now):
    recent_cam = now - track.last_cam_seen < RECENT_CAM_WINDOW_S
    recent_lidar = now - track.last_lidar_seen < RECENT_LIDAR_WINDOW_S
    recent_radar = now - track.last_radar_seen < RECENT_RADAR_WINDOW_S
    recent_multiview = now - track.last_multiview_seen < RECENT_CAM_WINDOW_S
    support_score = 0.0
    if recent_lidar:
        support_score += 1.15
    if recent_cam:
        support_score += 1.00
    if recent_radar:
        support_score += 0.70
    if recent_multiview:
        support_score += 0.35
    return recent_cam, recent_lidar, recent_radar, recent_multiview, support_score

def track_in_tracking_corridor(x, y):
    return -TRACK_REAR_RELEVANCE_M <= x <= TRACK_FORWARD_RELEVANCE_M and abs(y) <= TRACK_LATERAL_RELEVANCE_M

def camera_is_front_facing(cam_name):
    return cam_name in FRONT_CAMERA_NAMES

def camera_ring_name(cam_name):
    return "cam0" if cam_name == "front" else cam_name

def camera_ring_index(cam_name):
    name = camera_ring_name(cam_name)
    if name not in CAMERA_RING_ORDER:
        return None
    return CAMERA_RING_ORDER.index(name)

def camera_side(cam_name):
    name = camera_ring_name(cam_name)
    if name in {"cam1", "cam2", "cam3"}:
        return "right"
    if name in {"cam5", "cam6", "cam7"}:
        return "left"
    return "center"

def cameras_adjacent(cam_a, cam_b):
    idx_a = camera_ring_index(cam_a)
    idx_b = camera_ring_index(cam_b)
    if idx_a is None or idx_b is None:
        return False
    diff = abs(idx_a - idx_b)
    return diff == 1 or diff == len(CAMERA_RING_ORDER) - 1

def camera_obs_min_x(cam_name):
    return CAMERA_FRONT_OBS_MIN_X_M if camera_is_front_facing(cam_name) else CAMERA_SIDE_REAR_OBS_MIN_X_M

def camera_seed_min_x(cam_name):
    return CAMERA_FRONT_SEED_MIN_X_M if camera_is_front_facing(cam_name) else CAMERA_SIDE_REAR_SEED_MIN_X_M

def camera_projection_anchor(det, cam_name):
    x1, y1, x2, y2 = det[:4]
    u = int(round(0.5 * (x1 + x2)))
    h = max(1, y2 - y1)
    name = camera_ring_name(cam_name)
    if name == "cam4":
        v = int(round(y1 + CAMERA_REAR_ANCHOR_RATIO * h))
    elif name in {"cam2", "cam3", "cam5", "cam6"}:
        v = int(round(y1 + CAMERA_SIDE_REAR_ANCHOR_RATIO * h))
    else:
        v = int(y2)
    return int(clamp(u, x1, x2)), int(clamp(v, y1, y2))

def camera_obs_cluster_gate(cam_a, cam_b):
    gate = CAM_GLOBAL_CLUSTER_DIST_M
    if not camera_is_front_facing(cam_a) or not camera_is_front_facing(cam_b):
        gate += 0.5 * CAMERA_ADJ_CLUSTER_BONUS_M
    if cameras_adjacent(cam_a, cam_b):
        gate += CAMERA_ADJ_CLUSTER_BONUS_M
    elif camera_side(cam_a) != "center" and camera_side(cam_a) == camera_side(cam_b):
        gate += 0.7 * CAMERA_ADJ_CLUSTER_BONUS_M
    return gate

def camera_assoc_bonus(last_cam_name, obs_cams):
    if last_cam_name is None:
        return 0.0
    bonus = 0.0
    for cam_name in obs_cams:
        if camera_ring_name(cam_name) == camera_ring_name(last_cam_name):
            bonus = max(bonus, CAMERA_ADJ_ASSOC_BONUS_M)
        elif cameras_adjacent(last_cam_name, cam_name):
            bonus = max(bonus, 0.85 * CAMERA_ADJ_ASSOC_BONUS_M)
        elif camera_side(last_cam_name) != "center" and camera_side(last_cam_name) == camera_side(cam_name):
            bonus = max(bonus, 0.55 * CAMERA_ADJ_ASSOC_BONUS_M)
    return bonus

def track_camera_hold_window(track):
    x, y = track.pos()
    if (
        track.last_camera_name is not None
        and not camera_is_front_facing(track.last_camera_name)
        and track.cam_hits >= CAMERA_SEED_CONFIRM_FRAMES
    ):
        return SIDE_REAR_CAM_TRACK_HOLD_S
    if (
        track.cam_hits >= CAMERA_SEED_CONFIRM_FRAMES
        and MERGE_RADAR_REAR_X_MIN_M <= x <= CLOSE_SIDE_CAMERA_X_MAX_M
        and CLOSE_SIDE_CAMERA_LATERAL_MIN_M <= abs(y) <= MERGE_TRACK_LATERAL_M
    ):
        return SIDE_REAR_CAM_TRACK_HOLD_S
    return CAM_TRACK_HOLD_S

def build_surround_camera_mounts():
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

def cluster_camera_observations(cams_dets, cams_transforms, K):
    assoc_cams_dets = {cam: dets for cam, dets in cams_dets.items() if cam != "front"}
    if not assoc_cams_dets and "front" in cams_dets:
        assoc_cams_dets = {"front": cams_dets["front"]}

    raw_obs = []
    for cam, yolo_dets in assoc_cams_dets.items():
        cam_tf = cams_transforms.get(cam)
        if cam_tf is None:
            continue
        for det in yolo_dets:
            if det_conf(det) < max(0.22, YOLO_MIN_CONF):
                continue
            fu, fv = camera_projection_anchor(det, cam)
            cam_pos = camera_to_vehicle(fu, fv, K, cam_tf)
            if cam_pos is None:
                continue
            x, y = float(cam_pos[0]), float(cam_pos[1])
            if x < camera_obs_min_x(cam) or x > CAM_SEED_RANGE_M or abs(y) > (TRACK_LATERAL_RELEVANCE_M + 8.0):
                continue
            if -4.0 < x < 3.0 and abs(y) < 1.8:
                continue
            raw_obs.append({
                "x": x,
                "y": y,
                "cam": cam,
                "conf": float(det_conf(det)),
                "det": det,
            })

    raw_obs.sort(key=lambda item: item["conf"], reverse=True)
    clusters = []
    for obs in raw_obs:
        best = None
        best_dist = float("inf")
        for cluster in clusters:
            dist = speed_xy(obs["x"] - cluster["x"], obs["y"] - cluster["y"])
            cluster_gate = max(camera_obs_cluster_gate(obs["cam"], cam_name) for cam_name in cluster["cams"])
            if dist <= cluster_gate and dist < best_dist:
                best = cluster
                best_dist = dist
        if best is None:
            clusters.append({
                "x": obs["x"],
                "y": obs["y"],
                "weight": obs["conf"],
                "conf": obs["conf"],
                "cams": {obs["cam"]},
                "dets": [obs["det"]],
                "count": 1,
                "best_cam": obs["cam"],
            })
            continue
        total_w = best["weight"] + obs["conf"]
        if total_w > 1e-6:
            best["x"] = (best["x"] * best["weight"] + obs["x"] * obs["conf"]) / total_w
            best["y"] = (best["y"] * best["weight"] + obs["y"] * obs["conf"]) / total_w
        best["weight"] = total_w
        if obs["conf"] >= best["conf"]:
            best["best_cam"] = obs["cam"]
        best["conf"] = max(best["conf"], obs["conf"])
        best["cams"].add(obs["cam"])
        best["dets"].append(obs["det"])
        best["count"] += 1

    return clusters

def track_strength_score(track):
    score = 0.0
    score += 6.0 if track.confirmed else 0.0
    score += 0.04 * track.hits
    score += 0.05 * track.cam_hits
    score += 0.10 * track.lidar_hits
    score += 0.06 * track.radar_hits
    score += 2.0 * track.confidence
    return score

def merge_duplicate_tracks(now):
    global tracks
    ids = list(tracks.keys())
    to_drop = set()
    for i, tid_a in enumerate(ids):
        if tid_a in to_drop or tid_a not in tracks:
            continue
        ta = tracks[tid_a]
        ax, ay = ta.pos()
        avx, avy = ta.vel()
        for tid_b in ids[i + 1:]:
            if tid_b in to_drop or tid_b not in tracks:
                continue
            tb = tracks[tid_b]
            bx, by = tb.pos()
            bvx, bvy = tb.vel()
            pos_dist = speed_xy(ax - bx, ay - by)
            vel_dist = speed_xy(avx - bvx, avy - bvy)
            same_side_camera = (
                ta.last_camera_name is not None
                and tb.last_camera_name is not None
                and camera_side(ta.last_camera_name) == camera_side(tb.last_camera_name)
                and camera_side(ta.last_camera_name) != "center"
            )
            merge_gate = 5.5 if same_side_camera else 3.2
            if pos_dist > merge_gate or vel_dist > 6.0:
                continue
            if same_side_camera and abs(ay - by) > 2.4:
                continue

            recent_a = max(ta.last_cam_seen, ta.last_lidar_seen, ta.last_radar_seen)
            recent_b = max(tb.last_cam_seen, tb.last_lidar_seen, tb.last_radar_seen)
            if now - max(recent_a, recent_b) > 1.2:
                continue

            keep_id, drop_id = (tid_a, tid_b)
            if track_strength_score(tb) > track_strength_score(ta):
                keep_id, drop_id = tid_b, tid_a
            keep = tracks[keep_id]
            drop = tracks[drop_id]

            if drop.last_update > keep.last_update:
                keep.x = drop.x.copy()
                keep.P = drop.P.copy()

            keep.miss = min(keep.miss, drop.miss)
            keep.hits = max(keep.hits, drop.hits)
            keep.confirmed = keep.confirmed or drop.confirmed
            keep.cam_set.update(drop.cam_set)
            keep.cam_hits = max(keep.cam_hits, drop.cam_hits)
            keep.lidar_hits = max(keep.lidar_hits, drop.lidar_hits)
            keep.radar_hits = max(keep.radar_hits, drop.radar_hits)
            keep.last_cam_seen = max(keep.last_cam_seen, drop.last_cam_seen)
            keep.last_lidar_seen = max(keep.last_lidar_seen, drop.last_lidar_seen)
            keep.last_radar_seen = max(keep.last_radar_seen, drop.last_radar_seen)
            keep.last_multiview_seen = max(keep.last_multiview_seen, drop.last_multiview_seen)
            keep.last_lidar_points = max(keep.last_lidar_points, drop.last_lidar_points)
            keep.radar_support = max(keep.radar_support, drop.radar_support)
            keep.radar_closing_speed = max(keep.radar_closing_speed, drop.radar_closing_speed)
            keep.confidence = max(keep.confidence, drop.confidence)
            keep.last_camera_obs_count = max(keep.last_camera_obs_count, drop.last_camera_obs_count)
            if keep.last_camera_name is None:
                keep.last_camera_name = drop.last_camera_name
            to_drop.add(drop_id)
            if drop_id == tid_a:
                ta = keep
                ax, ay = ta.pos()
                avx, avy = ta.vel()

    for tid in to_drop:
        tracks.pop(tid, None)

def ekf_init(x0, y0, yaw0_deg):
    x = np.array([[x0], [y0], [0.0], [0.0], [yaw0_deg], [0.0]], dtype=float)
    P = np.eye(6) * 10.0
    return x, P

def ekf_predict(x, P, dt, q_pos=EGO_PROCESS_POS_Q, q_vel=EGO_PROCESS_VEL_Q, q_yaw=EGO_PROCESS_YAW_Q, q_yawrate=EGO_PROCESS_YAWRATE_Q):
    F = np.eye(6)
    F[0, 2] = dt
    F[1, 3] = dt
    F[4, 5] = dt
    Q = np.diag([q_pos, q_pos, q_vel, q_vel, q_yaw, q_yawrate])
    x = F @ x
    x[4, 0] = wrap_deg(float(x[4, 0]))
    P = F @ P @ F.T + Q
    return x, P

def ekf_update_pos(x, P, z_x, z_y, r_pos):
    H = np.zeros((2, 6))
    H[0, 0] = 1.0
    H[1, 1] = 1.0
    z = np.array([[z_x], [z_y]], dtype=float)
    R = np.eye(2) * r_pos
    y = z - (H @ x)
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)
    x = x + K @ y
    P = (np.eye(6) - K @ H) @ P
    return x, P

def ekf_update_speed_yawrate(x, P, z_speed, z_yawrate, r_speed, r_yawrate):
    vx = float(x[2, 0])
    vy = float(x[3, 0])
    v = math.sqrt(vx * vx + vy * vy)
    if v < 1e-3:
        dv_dvx = 0.0
        dv_dvy = 0.0
    else:
        dv_dvx = vx / v
        dv_dvy = vy / v

    h = np.array([[v], [float(x[5, 0])]], dtype=float)
    H = np.zeros((2, 6))
    H[0, 2] = dv_dvx
    H[0, 3] = dv_dvy
    H[1, 5] = 1.0
    z = np.array([[z_speed], [z_yawrate]], dtype=float)
    R = np.diag([r_speed, r_yawrate])
    y = z - h
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)
    x = x + K @ y
    x[4, 0] = wrap_deg(float(x[4, 0]))
    P = (np.eye(6) - K @ H) @ P
    return x, P

class EgoStateEstimator:
    def __init__(self, bootstrap_tf, world_map=None):
        self.bootstrap_tf = bootstrap_tf
        self.world_map = world_map
        self.initialized = False
        self.lat0 = None
        self.lon0 = None
        self.alt0 = None
        self.x = None
        self.P = None
        self.last_ts = None
        self.last_gnss_xy = None
        self.last_gnss_ts = None
        self.source = "BOOTSTRAP"
        self.compass_offset_deg = None

    def initialize(self, gnss):
        if gnss is None or self.initialized:
            return False
        self.lat0 = float(gnss.latitude)
        self.lon0 = float(gnss.longitude)
        self.alt0 = float(gnss.altitude)
        self.x, self.P = ekf_init(
            float(self.bootstrap_tf.location.x),
            float(self.bootstrap_tf.location.y),
            float(self.bootstrap_tf.rotation.yaw),
        )
        ts = float(getattr(gnss, "timestamp", 0.0))
        self.last_ts = ts
        self.last_gnss_ts = ts
        self.last_gnss_xy = (float(self.bootstrap_tf.location.x), float(self.bootstrap_tf.location.y))
        self.compass_offset_deg = None
        self.initialized = True
        self.source = "GNSS+IMU_EKF"
        return True

    def step(self, gnss, imu):
        if not self.initialized and not self.initialize(gnss):
            return None

        ts_candidates = []
        if gnss is not None and hasattr(gnss, "timestamp"):
            ts_candidates.append(float(gnss.timestamp))
        if imu is not None and hasattr(imu, "timestamp"):
            ts_candidates.append(float(imu.timestamp))
        ts = max(ts_candidates) if ts_candidates else (self.last_ts + STEP if self.last_ts is not None else STEP)
        dt = clamp(ts - self.last_ts if self.last_ts is not None else STEP, 0.01, 0.20)
        self.last_ts = ts

        self.x, self.P = ekf_predict(self.x, self.P, dt)

        if imu is not None:
            yaw_deg = float(self.x[4, 0])
            yaw_rad = math.radians(yaw_deg)
            ax_body = clamp(float(imu.accelerometer.x), -IMU_ACCEL_CLAMP_MPS2, IMU_ACCEL_CLAMP_MPS2)
            ay_body = clamp(float(imu.accelerometer.y), -IMU_ACCEL_CLAMP_MPS2, IMU_ACCEL_CLAMP_MPS2)
            yawrate_deg = float(imu.gyroscope.z) * (180.0 / math.pi)
            ax_world = math.cos(yaw_rad) * ax_body - math.sin(yaw_rad) * ay_body
            ay_world = math.sin(yaw_rad) * ax_body + math.cos(yaw_rad) * ay_body
            self.x[0, 0] += 0.5 * ax_world * dt * dt
            self.x[1, 0] += 0.5 * ay_world * dt * dt
            self.x[2, 0] += ax_world * dt
            self.x[3, 0] += ay_world * dt
            self.x[5, 0] = 0.70 * float(self.x[5, 0]) + 0.30 * yawrate_deg
            self.x[4, 0] = wrap_deg(float(self.x[4, 0]) + 0.35 * yawrate_deg * dt)
            if hasattr(imu, "compass"):
                raw_compass_yaw = imu_compass_to_yaw_deg(float(imu.compass))
                if self.compass_offset_deg is None:
                    self.compass_offset_deg = wrap_deg(float(self.bootstrap_tf.rotation.yaw) - raw_compass_yaw)
                compass_yaw = wrap_deg(raw_compass_yaw + float(self.compass_offset_deg))
                yaw_err = wrap_deg(compass_yaw - float(self.x[4, 0]))
                self.x[4, 0] = wrap_deg(float(self.x[4, 0]) + 0.18 * yaw_err)

        if gnss is not None:
            z_x = None
            z_y = None
            if self.world_map is not None and hasattr(self.world_map, "geolocation_to_transform"):
                try:
                    geo = carla.GeoLocation(float(gnss.latitude), float(gnss.longitude), float(gnss.altitude))
                    geo_loc = self.world_map.geolocation_to_transform(geo)
                    z_x = float(geo_loc.x)
                    z_y = float(geo_loc.y)
                except Exception:
                    z_x = None
                    z_y = None
            if z_x is None or z_y is None:
                east_m, north_m = ll_to_enu_m(float(gnss.latitude), float(gnss.longitude), self.lat0, self.lon0)
                z_x = float(self.bootstrap_tf.location.x + east_m)
                z_y = float(self.bootstrap_tf.location.y - north_m)
            self.x, self.P = ekf_update_pos(self.x, self.P, z_x, z_y, r_pos=R_POS)

            if self.last_gnss_xy is not None and self.last_gnss_ts is not None:
                gdt = max(0.05, ts - self.last_gnss_ts)
                gvx = (z_x - self.last_gnss_xy[0]) / gdt
                gvy = (z_y - self.last_gnss_xy[1]) / gdt
                gspeed = speed_xy(gvx, gvy)
                yawrate_deg = float(self.x[5, 0])
                if imu is not None:
                    yawrate_deg = float(imu.gyroscope.z) * (180.0 / math.pi)
                self.x, self.P = ekf_update_speed_yawrate(
                    self.x, self.P, gspeed, yawrate_deg, r_speed=R_SPEED, r_yawrate=R_YAWRATE
                )
                self.x[2, 0] = 0.65 * float(self.x[2, 0]) + 0.35 * gvx
                self.x[3, 0] = 0.65 * float(self.x[3, 0]) + 0.35 * gvy
                if gspeed > 1.0:
                    course_yaw = math.degrees(math.atan2(gvy, gvx))
                    yaw_err = wrap_deg(course_yaw - float(self.x[4, 0]))
                    self.x[4, 0] = wrap_deg(float(self.x[4, 0]) + 0.12 * yaw_err)

            self.last_gnss_xy = (z_x, z_y)
            self.last_gnss_ts = ts

        vx = float(self.x[2, 0])
        vy = float(self.x[3, 0])
        speed = speed_xy(vx, vy)
        yaw_deg = float(self.x[4, 0])
        if speed > 0.5:
            vel_yaw = math.degrees(math.atan2(vy, vx))
            yaw_err = wrap_deg(vel_yaw - yaw_deg)
            yaw_deg = wrap_deg(yaw_deg + 0.05 * yaw_err)
            self.x[4, 0] = yaw_deg

        return {
            "x": float(self.x[0, 0]),
            "y": float(self.x[1, 0]),
            "z": float(self.bootstrap_tf.location.z),
            "vx": vx,
            "vy": vy,
            "speed": speed,
            "yaw_deg": yaw_deg,
            "yaw_rate_dps": float(self.x[5, 0]),
            "source": self.source,
        }

def classify_traffic_light_state(img, box):
    x1, y1, x2, y2 = box
    crop = img[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
    if crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 4:
        return "Unknown", 0.0

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1].astype(np.float32) / 255.0
    v = hsv[:, :, 2].astype(np.float32) / 255.0
    bright = (s > 0.30) & (v > 0.45)

    red_mask = (((h <= 12) | (h >= 170)) & bright)
    yellow_mask = ((h >= 15) & (h <= 40) & bright)
    green_mask = ((h >= 42) & (h <= 95) & bright)

    height, width = crop.shape[:2]
    vertical = height >= int(1.15 * width)
    if vertical:
        thirds = [0, height // 3, (2 * height) // 3, height]
        red_score = float(red_mask[thirds[0]:thirds[1], :].mean()) + 0.25 * float(red_mask.mean())
        yellow_score = float(yellow_mask[thirds[1]:thirds[2], :].mean()) + 0.25 * float(yellow_mask.mean())
        green_score = float(green_mask[thirds[2]:thirds[3], :].mean()) + 0.25 * float(green_mask.mean())
    else:
        thirds = [0, width // 3, (2 * width) // 3, width]
        red_score = float(red_mask[:, thirds[0]:thirds[1]].mean()) + 0.25 * float(red_mask.mean())
        yellow_score = float(yellow_mask[:, thirds[1]:thirds[2]].mean()) + 0.25 * float(yellow_mask.mean())
        green_score = float(green_mask[:, thirds[2]:thirds[3]].mean()) + 0.25 * float(green_mask.mean())

    scores = {"Red": red_score, "Yellow": yellow_score, "Green": green_score}
    state_name = max(scores, key=scores.get)
    score = scores[state_name]
    if score < TL_SCORE_MIN:
        return "Unknown", score
    return state_name, score

def run_yolo_resized(img, scale=1.0):
    if scale != 1.0:
        h,w = img.shape[:2]
        small = cv2.resize(img, (int(w*scale), int(h*scale)))
    else:
        small = img
    results = model(small, verbose=False)[0]
    dets = []
    tl_dets = []
    for box in results.boxes:
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        x1,y1,x2,y2 = map(int, box.xyxy[0])
        if scale != 1.0:
            x1 = int(x1/scale); y1 = int(y1/scale); x2 = int(x2/scale); y2 = int(y2/scale)
        w_box = max(1, x2 - x1)
        h_box = max(1, y2 - y1)
        area = w_box * h_box
        cx = (x1+x2)//2
        cy = (y1+y2)//2
        if cls in [2,3,5,7]:
            if conf < YOLO_MIN_CONF:
                continue
            aspect = h_box / max(1, w_box)
            if area < MIN_DET_AREA_PX or aspect > MAX_DET_ASPECT:
                continue
            if y2 > int(BOTTOM_CLIP_RATIO * img.shape[0]) and area > int(BOTTOM_SELF_AREA_RATIO * img.shape[0] * img.shape[1]):
                continue
            dets.append((x1,y1,x2,y2,cx,cy,conf,cls))
        elif cls == TL_CLASS_ID and conf >= TL_MIN_CONF:
            if area < TL_MIN_AREA_PX or y2 > int(TL_DET_BOTTOM_RATIO * img.shape[0]):
                continue
            state_name, state_score = classify_traffic_light_state(img, (x1, y1, x2, y2))
            tl_dets.append({
                "bbox": (x1, y1, x2, y2),
                "cx": cx,
                "cy": cy,
                "conf": conf,
                "score": state_score,
                "state_name": state_name,
            })
    return dets, tl_dets

# ======================
# CALLBACKS
# ======================
def rgb_cb(data):
    global img_rgb
    try:
        arr = np.frombuffer(data.raw_data, dtype=np.uint8)
        img_rgb = arr.reshape((data.height, data.width, 4))[:,:,:3]
        # update front frame buffer
        with frame_lock:
            cams_frames['front'] = img_rgb
    except:
        pass

def make_cam_cb(name):
    def cb(data):
        try:
            arr = np.frombuffer(data.raw_data, dtype=np.uint8)
            frame = arr.reshape((data.height, data.width, 4))[:,:,:3]
            cams[name] = frame
            with frame_lock:
                cams_frames[name] = frame
        except:
            pass
    return cb

def lidar_cb(data):
    global lidar_data
    lidar_data = data

def gnss_cb(data):
    global gnss_data
    gnss_data = data

def imu_cb(data):
    global imu_data
    imu_data = data

def radar_meas_to_ego_points(radar_meas, radar_tf):
    rows = []
    theta = math.radians(radar_tf.rotation.yaw)
    ct = math.cos(theta)
    st = math.sin(theta)
    for det in radar_meas:
        depth = float(det.depth)
        az = float(det.azimuth)
        alt = float(det.altitude)
        xs = depth * math.cos(alt) * math.cos(az)
        ys = depth * math.cos(alt) * math.sin(az)
        zs = depth * math.sin(alt)
        xe = float(radar_tf.location.x + ct * xs - st * ys)
        ye = float(radar_tf.location.y + st * xs + ct * ys)
        ze = float(radar_tf.location.z + zs)
        if not (RADAR_MIN_Z <= ze <= RADAR_MAX_Z):
            continue
        rows.append({
            "x": xe,
            "y": ye,
            "z": ze,
            "depth": depth,
            "velocity": float(det.velocity),
        })
    return rows

def make_radar_cb(name, radar_tf):
    def cb(data):
        global radar_returns
        try:
            radar_returns[name] = {
                "ts": time.time(),
                "points": radar_meas_to_ego_points(data, radar_tf),
            }
        except Exception:
            pass
    return cb

def make_obstacle_cb(name):
    def cb(event):
        global obstacle_events
        try:
            other_actor = event.other_actor
            type_id = other_actor.type_id if other_actor is not None else "unknown"
            if other_actor is not None and other_actor.id == event.actor.id:
                return
            if float(event.distance) <= 0.75:
                return
            if isinstance(type_id, str) and type_id.startswith("traffic."):
                return
            obstacle_events[name] = {
                "ts": time.time(),
                "distance": float(event.distance),
                "actor_id": None if other_actor is None else other_actor.id,
                "type_id": type_id,
                "name": name,
                "forward": float(event.distance),
                "lateral": 0.0,
            }
        except Exception:
            pass
    return cb

# ======================
# TRACK (UNCHANGED)
# ======================
class Track:
    def __init__(self, tid, x, y):
        self.id = tid
        self.x = np.array([[x],[y],[0],[0]], float)
        self.P = np.eye(4)*5
        self.miss = 0
        self.hits = 0
        self.confirmed = False
        self.cam_set = set()
        self.cam_hits = 0
        self.lidar_hits = 0
        self.last_cam_seen = 0.0
        self.last_lidar_seen = 0.0
        self.history = []  # recent (ts, x, y)
        self.age = 0
        self.confidence = 0.15
        self.consistent_velocity_frames = 0
        self.first_seen = time.time()
        self.last_update = self.first_seen
        self.length = BOX_LENGTH
        self.width = BOX_WIDTH
        self.height = BOX_HEIGHT
        self.last_heading = 0.0
        self.future = []
        self.risk = 0.0
        self.ttc = float("inf")
        self.collision_time = float("inf")
        self.front_hazard = False
        self.cross_hazard = False
        self.radar_hits = 0
        self.last_radar_seen = 0.0
        self.radar_closing_speed = 0.0
        self.radar_support = 0.0
        self.last_camera_name = None
        self.world_history = []
        self.world_motion_speed = 0.0
        self.last_lidar_points = 0
        self.last_camera_obs_count = 0
        self.last_multiview_seen = 0.0

    def predict(self, dt):
        F = np.array([[1,0,dt,0],
                      [0,1,0,dt],
                      [0,0,1,0],
                      [0,0,0,1]])
        Q = np.eye(4)*0.15  # lower process noise for smoother predictions
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        self.miss += 1
        self.age += 1

    def update(self, z, from_lidar=False, meas_var=None):
        H = np.array([[1,0,0,0],[0,1,0,0]])
        if meas_var is None:
            meas_var = 1.5 if from_lidar else 4.5
        R = np.eye(2)*meas_var

        y = z - H@self.x
        S = H@self.P@H.T + R
        K = self.P@H.T@np.linalg.inv(S)

        self.x += K@y
        self.P = (np.eye(4)-K@H)@self.P

        self.miss = 0
        self.hits += 1
        self.last_update = time.time()
        self.history.append((self.last_update, self.x[0,0], self.x[1,0]))
        if len(self.history) > 15:
            self.history.pop(0)

        if from_lidar:
            self.lidar_hits += 1
            self.last_lidar_seen = self.last_update
            self.confidence = min(1.0, self.confidence + 0.22)
        else:
            self.confidence = min(1.0, self.confidence + 0.08)

    def pos(self):
        return self.x[0,0], self.x[1,0]

    def vel(self):
        return self.x[2,0], self.x[3,0]

    def speed(self):
        vx, vy = self.vel()
        return speed_xy(vx, vy)

    def heading(self):
        vx, vy = self.vel()
        if self.speed() > 0.25:
            self.last_heading = math.atan2(vy, vx)
        return self.last_heading

    def set_extent(self, length, width, height):
        self.length = clamp(0.75 * self.length + 0.25 * length, 2.0, 12.0)
        self.width = clamp(0.75 * self.width + 0.25 * width, 1.2, 4.0)
        self.height = clamp(0.75 * self.height + 0.25 * height, 1.2, 4.5)

    def predict_future(self, horizon=PREDICTION_HORIZON_S, dt=PREDICTION_DT_S):
        x0, y0 = self.pos()
        vx, vy = self.vel()
        future = []
        t = dt
        while t <= horizon + 1e-6:
            future.append((t, x0 + vx * t, y0 + vy * t))
            t += dt
        self.future = future
        return future

    def estimate_global(self, ego_tf):
        # Convert ego-relative track position into global map coordinates
        ex = ego_tf.location.x
        ey = ego_tf.location.y
        yaw = math.radians(ego_tf.rotation.yaw)
        xr, yr = self.pos()
        xg = ex + math.cos(yaw)*xr - math.sin(yaw)*yr
        yg = ey + math.sin(yaw)*xr + math.cos(yaw)*yr
        return xg, yg

def track_is_road_relevant(track, ego_state, world_map):
    x, y = track.pos()
    if -12.0 <= x <= 70.0 and abs(y) <= 9.0:
        return True
    ego_tf = ego_transform_from_state(ego_state)
    gx, gy = track.estimate_global(ego_tf)
    loc = carla.Location(x=float(gx), y=float(gy), z=float(ego_state.get("z", 0.0)))
    lane_mask = carla.LaneType.Driving | carla.LaneType.Bidirectional | carla.LaneType.Parking
    wp = world_map.get_waypoint(loc, project_to_road=True, lane_type=lane_mask)
    if wp is None:
        return False
    return speed_xy(loc.x - wp.transform.location.x, loc.y - wp.transform.location.y) <= ROAD_RELEVANCE_DIST_M

def persistent_camera_track_candidate(track, recent_multiview=False, road_relevant=True):
    x, y = track.pos()
    front_facing_cam = track.last_camera_name in FRONT_CAMERA_NAMES
    in_publish_band = -6.0 <= x <= CAMERA_PERSIST_RANGE_M and abs(y) <= CAMERA_PERSIST_LATERAL_M
    strong_camera_history = track.cam_hits >= CAMERA_PERSIST_CONFIRM_HITS and track.confidence >= CAMERA_PERSIST_MIN_CONFIDENCE
    stable_recent = track.miss <= CAMERA_PERSIST_MAX_MISS
    return (
        road_relevant
        and in_publish_band
        and strong_camera_history
        and stable_recent
        and (recent_multiview or front_facing_cam or track.cam_hits >= CAMERA_PERSIST_STRONG_HITS)
    )

def choose_continuing_waypoint(candidates, ref_yaw_deg):
    if not candidates:
        return None
    return min(candidates, key=lambda cand: abs(wrap_deg(cand.transform.rotation.yaw - ref_yaw_deg)))

def sample_waypoint_chain(start_wp, lookahead_m, step_m=LANE_PATH_SAMPLE_STEP_M):
    if start_wp is None:
        return []
    points = [start_wp]
    traveled = 0.0
    current = start_wp
    ref_yaw = start_wp.transform.rotation.yaw
    while traveled < lookahead_m:
        next_wps = current.next(step_m)
        if not next_wps:
            break
        current = choose_continuing_waypoint(next_wps, ref_yaw)
        if current is None:
            break
        points.append(current)
        ref_yaw = current.transform.rotation.yaw
        traveled += step_m
    return points

def point_to_path_distance(x, y, path_points):
    if not path_points:
        return float("inf")
    return min(speed_xy(x - px, y - py) for px, py in path_points)

def analyze_track_lane_behavior(track, future, ego_state, world_map):
    info = {
        "valid_points": 0,
        "lane_fraction": 0.0,
        "parallel_fraction": 0.0,
        "stays_in_lane": False,
        "same_direction": False,
        "adjacent_parallel": False,
        "ego_path_overlap": False,
        "ego_path_min_dist": float("inf"),
        "stays_outside_ego_path": False,
    }
    ego_loc = ego_location_from_state(ego_state)
    lane_mask = carla.LaneType.Driving | carla.LaneType.Bidirectional
    ego_wp = world_map.get_waypoint(ego_loc, project_to_road=True, lane_type=lane_mask)
    if ego_wp is None:
        return info

    ego_path_wps = sample_waypoint_chain(ego_wp, EGO_PATH_LOOKAHEAD_M)
    if not ego_path_wps:
        return info

    ego_path_ids = {(wp.road_id, wp.section_id, wp.lane_id) for wp in ego_path_wps}
    ego_path_points = [(wp.transform.location.x, wp.transform.location.y) for wp in ego_path_wps]
    ego_path_yaw = ego_wp.transform.rotation.yaw

    ego_tf = ego_transform_from_state(ego_state)
    gx0, gy0 = track.estimate_global(ego_tf)
    samples = [(gx0, gy0)]
    samples.extend(
        (
            ego_tf.location.x + math.cos(math.radians(ego_tf.rotation.yaw)) * px - math.sin(math.radians(ego_tf.rotation.yaw)) * py,
            ego_tf.location.y + math.sin(math.radians(ego_tf.rotation.yaw)) * px + math.cos(math.radians(ego_tf.rotation.yaw)) * py,
        )
        for tau, px, py in future[::2]
        if tau <= min(PREDICTION_HORIZON_S, 3.0)
    )

    on_lane = 0
    parallel_hits = 0
    overlap_hits = 0
    min_path_dist = float("inf")
    for gx, gy in samples:
        loc = carla.Location(x=float(gx), y=float(gy), z=float(ego_state.get("z", 0.0)))
        wp = world_map.get_waypoint(loc, project_to_road=True, lane_type=lane_mask)
        if wp is None:
            continue
        info["valid_points"] += 1
        lane_err = speed_xy(loc.x - wp.transform.location.x, loc.y - wp.transform.location.y)
        if lane_err <= LANE_KEEP_PROJECTION_MAX_ERR_M:
            on_lane += 1
        if abs(wrap_deg(wp.transform.rotation.yaw - ego_path_yaw)) <= PARALLEL_LANE_YAW_ERR_DEG:
            parallel_hits += 1
        if (wp.road_id, wp.section_id, wp.lane_id) in ego_path_ids:
            overlap_hits += 1
        min_path_dist = min(min_path_dist, point_to_path_distance(loc.x, loc.y, ego_path_points))

    if info["valid_points"] <= 0:
        return info

    info["lane_fraction"] = on_lane / float(info["valid_points"])
    info["parallel_fraction"] = parallel_hits / float(info["valid_points"])
    info["stays_in_lane"] = info["valid_points"] >= 2 and info["lane_fraction"] >= LANE_KEEP_MIN_FRACTION
    info["same_direction"] = info["valid_points"] >= 2 and info["parallel_fraction"] >= LANE_KEEP_MIN_FRACTION
    info["ego_path_overlap"] = overlap_hits > 0
    info["ego_path_min_dist"] = min_path_dist
    info["stays_outside_ego_path"] = (
        np.isfinite(min_path_dist)
        and min_path_dist > TURN_LANE_CLEARANCE_M
        and not info["ego_path_overlap"]
    )
    info["adjacent_parallel"] = (
        info["same_direction"]
        and info["stays_in_lane"]
        and info["stays_outside_ego_path"]
        and np.isfinite(min_path_dist)
        and min_path_dist < 8.0
    )
    return info

def lateral_error_to_waypoint(location, waypoint):
    yaw = math.radians(waypoint.transform.rotation.yaw)
    dx = location.x - waypoint.transform.location.x
    dy = location.y - waypoint.transform.location.y
    return -math.sin(yaw) * dx + math.cos(yaw) * dy

def heading_error_to_waypoint(ego_yaw_deg, waypoint):
    return wrap_deg(waypoint.transform.rotation.yaw - ego_yaw_deg)

def advance_target_waypoint(seed_wp, ego_yaw_deg, lookahead_m, step_m=3.0):
    if seed_wp is None:
        return None
    wp = seed_wp
    travelled = 0.0
    while travelled < lookahead_m:
        next_wps = [cand for cand in wp.next(step_m) if cand.lane_type == carla.LaneType.Driving]
        if not next_wps:
            break

        def waypoint_cost(cand):
            yaw_change = abs(wrap_deg(cand.transform.rotation.yaw - wp.transform.rotation.yaw))
            ego_err = abs(wrap_deg(cand.transform.rotation.yaw - ego_yaw_deg))
            lane_penalty = 0.0 if cand.lane_id == wp.lane_id else 28.0
            road_penalty = 0.0 if cand.road_id == wp.road_id else (6.0 if wp.is_junction or cand.is_junction else 20.0)
            reverse_penalty = 120.0 if yaw_change > 80.0 else 0.0
            return reverse_penalty + 1.8 * yaw_change + 0.35 * ego_err + lane_penalty + road_penalty

        wp = min(next_wps, key=waypoint_cost)
        travelled += step_m
    return wp

def lateral_speed_toward_ego(y, vy):
    if abs(y) < 1e-3:
        return abs(vy)
    return max(0.0, -math.copysign(1.0, y) * vy)

def get_radar_points():
    now = time.time()
    stale = [name for name, data in radar_returns.items() if now - data["ts"] > RADAR_MEMORY_S]
    for name in stale:
        radar_returns.pop(name, None)
    points = []
    for data in radar_returns.values():
        points.extend(data["points"])
    return points

def cluster_radar_points(points):
    if not points:
        return []
    cell_to_points = defaultdict(list)
    for idx, point in enumerate(points):
        key = (
            int(math.floor(point["x"] / RADAR_GRID_M)),
            int(math.floor(point["y"] / RADAR_GRID_M)),
        )
        cell_to_points[key].append(idx)

    clusters = []
    visited = set()
    for key in list(cell_to_points.keys()):
        if key in visited:
            continue
        queue = [key]
        visited.add(key)
        idxs = []
        while queue:
            cx, cy = queue.pop()
            idxs.extend(cell_to_points[(cx, cy)])
            for nx in range(cx - 1, cx + 2):
                for ny in range(cy - 1, cy + 2):
                    nkey = (nx, ny)
                    if nkey in cell_to_points and nkey not in visited:
                        visited.add(nkey)
                        queue.append(nkey)
        if len(idxs) < RADAR_CLUSTER_MIN_POINTS:
            continue
        arr = np.array([[points[i]["x"], points[i]["y"], points[i]["velocity"]] for i in idxs], dtype=float)
        centroid = arr[:, :2].mean(axis=0)
        positive_closing = np.maximum(arr[:, 2], 0.0)
        clusters.append({
            "x": float(centroid[0]),
            "y": float(centroid[1]),
            "count": int(len(idxs)),
            "distance": float(speed_xy(centroid[0], centroid[1])),
            "closing_speed": float(np.quantile(positive_closing, 0.75) if positive_closing.size else 0.0),
        })
    return clusters

def merge_relevant_radar_cluster(cluster):
    x = float(cluster["x"])
    y = float(cluster["y"])
    lateral = abs(y)
    closing_speed = float(cluster["closing_speed"])
    return (
        MERGE_RADAR_REAR_X_MIN_M <= x <= 35.0
        and MERGE_RADAR_LATERAL_MIN_M <= lateral <= TRACK_LATERAL_RELEVANCE_M
        and closing_speed >= max(2.0, 0.75 * MERGE_MIN_CLOSING_SPEED_MPS)
    )

def fuse_radar_tracks(radar_points):
    global tracks, next_id, radar_seed_memory
    now = time.time()
    for t in tracks.values():
        if now - t.last_radar_seen > RADAR_MEMORY_S:
            t.radar_support = 0.0
            t.radar_closing_speed *= 0.8

    radar_clusters = cluster_radar_points(radar_points)
    for t in tracks.values():
        tx, ty = t.pos()
        gate = RADAR_ASSOC_DIST_M + 0.25 * max(t.length, t.width)
        if abs(ty) >= MERGE_RADAR_LATERAL_MIN_M:
            gate += 1.25
        if tx < 12.0:
            gate += 0.50
        matches = [p for p in radar_points if speed_xy(tx - p["x"], ty - p["y"]) <= gate]
        if not matches:
            continue
        best = min(matches, key=lambda p: speed_xy(tx - p["x"], ty - p["y"]))
        t.radar_hits += len(matches)
        t.last_radar_seen = now
        t.radar_support = min(1.0, 0.30 + 0.12 * len(matches))
        t.radar_closing_speed = max(0.6 * t.radar_closing_speed, max(0.0, float(best["velocity"])))
        t.confidence = min(1.0, t.confidence + 0.04)
        if not t.confirmed and t.lidar_hits >= 2 and t.radar_hits >= 2 and (t.cam_hits >= 1 or t.speed() > 0.5):
            t.confirmed = True
            t.confidence = max(t.confidence, 0.45)

    stale_keys = [k for k, seed in radar_seed_memory.items() if frame_idx - seed["frame"] > RADAR_SEED_TTL_FRAMES]
    for key in stale_keys:
        radar_seed_memory.pop(key, None)

    for cluster in radar_clusters:
        if not merge_relevant_radar_cluster(cluster):
            continue
        if any(speed_xy(cluster["x"] - t.pos()[0], cluster["y"] - t.pos()[1]) < 4.0 for t in tracks.values()):
            continue
        key = (
            int(round(cluster["x"] / RADAR_SEED_GRID_X_M)),
            int(round(cluster["y"] / RADAR_SEED_GRID_Y_M)),
        )
        seed = radar_seed_memory.get(key)
        if seed is None:
            seed = {
                "count": 0,
                "x": float(cluster["x"]),
                "y": float(cluster["y"]),
                "frame": frame_idx,
                "closing_speed": float(cluster["closing_speed"]),
            }
        seed["count"] += 1
        seed["x"] = 0.55 * seed["x"] + 0.45 * float(cluster["x"])
        seed["y"] = 0.55 * seed["y"] + 0.45 * float(cluster["y"])
        seed["frame"] = frame_idx
        seed["closing_speed"] = max(seed["closing_speed"], float(cluster["closing_speed"]))
        radar_seed_memory[key] = seed

        if seed["count"] >= RADAR_SEED_CONFIRM_FRAMES:
            tr = Track(next_id, seed["x"], seed["y"])
            tr.radar_hits = seed["count"]
            tr.last_radar_seen = now
            tr.radar_support = 0.55
            tr.radar_closing_speed = seed["closing_speed"]
            tr.confidence = 0.26
            tracks[next_id] = tr
            next_id += 1
            radar_seed_memory.pop(key, None)

def get_lidar_corridor_info(lidar):
    if lidar is None:
        return {
            "distance": float("inf"),
            "type_id": None,
            "active": False,
            "name": None,
            "forward": float("inf"),
            "lateral": float("inf"),
            "source": None,
        }

    pts = np.frombuffer(lidar.raw_data, dtype=np.float32).reshape(-1, 4)
    xyz = pts[:, :3]
    min_forward = max(2.8, EGO_HALF_LENGTH_M + 0.8)
    corridor_half_width = min(LIDAR_CORRIDOR_HALF_WIDTH_M, 0.95)
    mask = (xyz[:, 0] > min_forward) & (xyz[:, 0] < LIDAR_CORRIDOR_RANGE_M)
    mask &= np.abs(xyz[:, 1]) < corridor_half_width
    mask &= xyz[:, 2] > max(LIDAR_MIN_Z, -0.35)
    mask &= xyz[:, 2] < min(LIDAR_MAX_Z, 2.5)
    corridor = xyz[mask]
    if len(corridor) < LIDAR_CORRIDOR_MIN_POINTS:
        return {
            "distance": float("inf"),
            "type_id": None,
            "active": False,
            "name": None,
            "forward": float("inf"),
            "lateral": float("inf"),
            "source": None,
        }

    keep = np.ones(len(corridor), dtype=bool)
    for t in tracks.values():
        if not t.confirmed:
            continue
        tx, ty = t.pos()
        keep &= ~(
            (np.abs(corridor[:, 0] - tx) < max(1.5, 0.7 * t.length)) &
            (np.abs(corridor[:, 1] - ty) < max(1.0, 0.8 * t.width))
        )

    corridor = corridor[keep]
    if len(corridor) < LIDAR_CORRIDOR_MIN_POINTS:
        return {
            "distance": float("inf"),
            "type_id": None,
            "active": False,
            "name": None,
            "forward": float("inf"),
            "lateral": float("inf"),
            "source": None,
        }

    distance = float(np.quantile(corridor[:, 0], 0.10))
    lateral = float(np.median(corridor[:, 1]))
    return {
        "distance": distance,
        "type_id": "lidar_corridor",
        "active": distance < min(OBSTACLE_BRAKE_DISTANCE_M, 4.0) and abs(lateral) < corridor_half_width,
        "name": "lidar_corridor",
        "forward": distance,
        "lateral": lateral,
        "source": "lidar",
    }

def fuse_obstacle_sources(obstacle_info, lidar_corridor_info):
    if lidar_corridor_info["active"] and lidar_corridor_info["distance"] < obstacle_info["distance"]:
        return lidar_corridor_info
    if obstacle_info["active"]:
        type_id = obstacle_info.get("type_id") or ""
        if obstacle_info["distance"] <= 0.75 or (isinstance(type_id, str) and type_id.startswith("traffic.")):
            obstacle_info["active"] = False
            obstacle_info["distance"] = float("inf")
            obstacle_info["forward"] = float("inf")
            obstacle_info["lateral"] = float("inf")
            obstacle_info["type_id"] = None
            obstacle_info["name"] = None
            obstacle_info["source"] = None
            return obstacle_info
        if obstacle_info["distance"] > 3.5:
            obstacle_info["active"] = False
            obstacle_info["distance"] = float("inf")
            obstacle_info["forward"] = float("inf")
            obstacle_info["lateral"] = float("inf")
            obstacle_info["type_id"] = None
            obstacle_info["name"] = None
            obstacle_info["source"] = None
            return obstacle_info
        obstacle_info["source"] = "obstacle_sensor"
    else:
        obstacle_info["source"] = None
    return obstacle_info

def distance_to_next_junction(current_wp, max_dist=TRAFFIC_LIGHT_LOOKAHEAD_M):
    step = TL_JUNCTION_STEP_M
    dist = 0.0
    wp = current_wp
    while wp is not None and dist < max_dist:
        if wp.is_junction and dist > 0.5:
            return dist
        next_wps = wp.next(step)
        if not next_wps:
            break
        wp = min(next_wps, key=lambda cand: abs(wrap_deg(cand.transform.rotation.yaw - wp.transform.rotation.yaw)))
        dist += step
    return float("inf")

def traffic_light_stop_distance(current_wp, ego_state, traffic_light):
    best = float("inf")
    stop_wps = traffic_light.get_stop_waypoints()
    for stop_wp in stop_wps:
        if stop_wp.lane_type != carla.LaneType.Driving:
            continue
        same_lane = stop_wp.road_id == current_wp.road_id and stop_wp.lane_id == current_wp.lane_id
        if not same_lane:
            continue
        rel_x, rel_y = world_to_ego_xy(stop_wp.transform.location.x, stop_wp.transform.location.y, ego_state)
        if rel_x < -2.0 or rel_x > TRAFFIC_LIGHT_LOOKAHEAD_M + 5.0:
            continue
        if abs(rel_y) > TRAFFIC_LIGHT_LATERAL_M:
            continue
        if rel_x < best:
            best = rel_x
    return best

def get_traffic_light_info(ego, ego_state, world_map, world):
    info = {
        "state": None,
        "state_name": "None",
        "distance": float("inf"),
        "id": None,
        "confidence": 0.0,
        "bbox": None,
    }
    if ego is None or ego_state is None:
        return info

    loc = ego_location_from_state(ego_state)
    current_wp = world_map.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)
    if current_wp is None:
        return info

    candidates = {}
    active_tl = None
    if ego.is_at_traffic_light():
        try:
            active_tl = ego.get_traffic_light()
        except Exception:
            active_tl = None
        if active_tl is not None:
            candidates[active_tl.id] = active_tl
    if world is not None and hasattr(world, "get_traffic_lights_from_waypoint"):
        try:
            for tl in world.get_traffic_lights_from_waypoint(current_wp, TRAFFIC_LIGHT_LOOKAHEAD_M):
                candidates[tl.id] = tl
        except Exception:
            pass

    for tl in candidates.values():
        dist = traffic_light_stop_distance(current_wp, ego_state, tl)
        if not np.isfinite(dist) and active_tl is not None and tl.id == active_tl.id:
            dist = 0.0
        if not np.isfinite(dist) or dist > TRAFFIC_LIGHT_LOOKAHEAD_M + 5.0:
            continue
        if dist < info["distance"]:
            info["state"] = tl.get_state()
            info["state_name"] = traffic_light_state_name(info["state"])
            info["distance"] = dist
            info["id"] = tl.id
            info["confidence"] = 1.0
    return info

def route_speed_cap_mps(ego_state, world_map):
    loc = ego_location_from_state(ego_state)
    wp = world_map.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)
    if wp is None:
        return float("inf")

    chain = sample_waypoint_chain(wp, TURN_PREVIEW_M, step_m=TURN_PREVIEW_STEP_M)
    cap = float("inf")
    max_delta = 0.0
    if len(chain) >= 2:
        yaw0 = chain[0].transform.rotation.yaw
        max_delta = max(abs(wrap_deg(node.transform.rotation.yaw - yaw0)) for node in chain[1:])
        if max_delta > ROUTE_HARD_TURN_YAW_DEG:
            cap = min(cap, TURN_SPEED_CAP_HARD_MPS)
        elif max_delta > ROUTE_MED_TURN_YAW_DEG:
            cap = min(cap, TURN_SPEED_CAP_MED_MPS)
        elif max_delta > ROUTE_SOFT_TURN_YAW_DEG:
            cap = min(cap, TURN_SPEED_CAP_SOFT_MPS)

    dist_to_junction = distance_to_next_junction(wp, max_dist=TURN_PREVIEW_M)
    if np.isfinite(dist_to_junction) and dist_to_junction < JUNCTION_APPROACH_DIST_M:
        turn_like_junction = max_delta > STRAIGHT_JUNCTION_MAX_YAW_DEG
        if turn_like_junction:
            ratio = clamp(dist_to_junction / max(1.0, JUNCTION_APPROACH_DIST_M), 0.0, 1.0)
            cap = min(cap, JUNCTION_APPROACH_SPEED_MPS + 3.5 * ratio)

    return cap

def waypoint_same_direction(wp_a, wp_b, yaw_tol_deg=LANE_CHANGE_YAW_TOL_DEG):
    if wp_a is None or wp_b is None:
        return False
    return abs(wrap_deg(wp_a.transform.rotation.yaw - wp_b.transform.rotation.yaw)) <= yaw_tol_deg

def adjacent_driving_lane(current_wp, side):
    if current_wp is None:
        return None
    candidate = current_wp.get_left_lane() if side == "left" else current_wp.get_right_lane()
    if candidate is None or candidate.lane_type != carla.LaneType.Driving:
        return None
    if not waypoint_same_direction(current_wp, candidate):
        return None
    return candidate

def count_same_direction_lanes(current_wp):
    if current_wp is None:
        return 0
    total = 1
    for side in ("left", "right"):
        walker = current_wp
        while True:
            walker = adjacent_driving_lane(walker, side)
            if walker is None:
                break
            total += 1
    return total

def distance_to_lane_end(current_wp, max_dist=LANE_END_LOOKAHEAD_M):
    if current_wp is None:
        return float("inf")
    step = max(3.0, TURN_PREVIEW_STEP_M)
    dist = 0.0
    wp = current_wp
    while wp is not None and dist < max_dist:
        next_wps = [cand for cand in wp.next(step) if cand.lane_type == carla.LaneType.Driving]
        if not next_wps:
            return dist
        next_wp = min(next_wps, key=lambda cand: abs(wrap_deg(cand.transform.rotation.yaw - wp.transform.rotation.yaw)))
        if not waypoint_same_direction(wp, next_wp, yaw_tol_deg=55.0) and dist > 2.0:
            return dist
        wp = next_wp
        dist += step
    return float("inf")

def road_target_speed_mps(current_wp, speed_limit_mps):
    speed_limit_mps = max(0.0, float(speed_limit_mps))
    if current_wp is None:
        return min(CRUISE_SPEED_MPS, speed_limit_mps) if speed_limit_mps > 0.0 else CRUISE_SPEED_MPS

    lane_count = count_same_direction_lanes(current_wp)
    highway_like = (
        speed_limit_mps >= HIGHWAY_SPEED_LIMIT_MIN_MPS
        and lane_count >= HIGHWAY_MIN_LANES
        and not current_wp.is_junction
    )
    if highway_like:
        base = max(speed_limit_mps, HIGHWAY_TARGET_SPEED_MPS)
    elif speed_limit_mps > 0.0:
        base = speed_limit_mps
    else:
        base = CRUISE_SPEED_MPS
    return min(CRUISE_SPEED_MPS, base)

def choose_route_destination(spawn_points, current_location, min_dist_m=ROUTE_REPLAN_MIN_DIST_M):
    candidates = []
    for tf in spawn_points:
        loc = tf.location
        dist = current_location.distance(loc)
        if dist >= min_dist_m:
            candidates.append((dist, loc))
    if not candidates:
        for tf in spawn_points:
            loc = tf.location
            dist = current_location.distance(loc)
            if dist >= 40.0:
                candidates.append((dist, loc))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    top_k = min(6, len(candidates))
    return random.choice(candidates[:top_k])[1]

def build_forward_route_plan(start_wp, distance_m=180.0, step_m=2.0):
    plan = []
    if start_wp is None:
        return plan
    wp = start_wp
    travelled = 0.0
    while wp is not None and travelled < distance_m:
        road_option = RoadOption.STRAIGHT if wp.is_junction else RoadOption.LANEFOLLOW
        plan.append((wp, road_option))
        next_wps = [cand for cand in wp.next(step_m) if cand.lane_type == carla.LaneType.Driving]
        if not next_wps:
            break
        wp = min(next_wps, key=lambda cand: abs(wrap_deg(cand.transform.rotation.yaw - wp.transform.rotation.yaw)))
        travelled += step_m
    return plan

def make_route_driver(ego, world_map, spawn_points):
    planner = LocalPlanner(
        ego,
        opt_dict={
            "target_speed": max(20.0, 3.6 * min(CRUISE_SPEED_MPS, 18.0)),
            "sampling_radius": 2.0,
            "follow_speed_limits": False,
            "max_throttle": 0.75,
            "max_brake": 0.35,
            "max_steering": 0.75,
        },
        map_inst=world_map,
    )
    start_wp = world_map.get_waypoint(ego.get_location(), project_to_road=True, lane_type=carla.LaneType.Driving)
    route = build_forward_route_plan(start_wp)
    goal = route[-1][0].transform.location if route else None
    if route:
        planner.set_global_plan(route, stop_waypoint_creation=True, clean_queue=True)
    return {"planner": planner}, goal

def maybe_refresh_route_driver(route_agent, ego, spawn_points, route_goal):
    if route_agent is None:
        return None
    planner = route_agent["planner"]
    current_loc = ego.get_location()
    goal_reached = route_goal is None or current_loc.distance(route_goal) < ROUTE_REACHED_DIST_M
    if goal_reached or planner.done():
        current_wp = ego.get_world().get_map().get_waypoint(current_loc, project_to_road=True, lane_type=carla.LaneType.Driving)
        route = build_forward_route_plan(current_wp)
        if route:
            planner.set_global_plan(route, stop_waypoint_creation=True, clean_queue=True)
            return route[-1][0].transform.location
        return None
    return route_goal

def assess_target_lane_gap(world_model, target_wp, ego_state):
    lane_center_x, lane_center_y = world_to_ego_xy(
        target_wp.transform.location.x,
        target_wp.transform.location.y,
        ego_state,
    )
    lane_width = max(3.2, float(target_wp.lane_width))
    lane_band = max(1.35, 0.45 * lane_width + 0.35)
    front_gap = float("inf")
    rear_gap = float("inf")
    front_ttc = float("inf")
    rear_ttc = float("inf")
    front_track_id = None
    rear_track_id = None

    for entry in world_model.get("tracks", []):
        if abs(entry["y"] - lane_center_y) > lane_band:
            continue
        rel_x = float(entry["x"] - lane_center_x)
        rel_vx = float(entry.get("vx", 0.0))
        if rel_x >= 0.0:
            if rel_x < front_gap:
                front_gap = rel_x
                front_track_id = entry["id"]
                closing = max(0.0, -rel_vx)
                front_ttc = rel_x / max(0.1, closing) if closing > 0.1 else float("inf")
        else:
            rear_dist = -rel_x
            if rear_dist < rear_gap:
                rear_gap = rear_dist
                rear_track_id = entry["id"]
                closing = max(0.0, rel_vx)
                rear_ttc = rear_dist / max(0.1, closing) if closing > 0.1 else float("inf")

    return {
        "lane_center_y": lane_center_y,
        "front_gap": front_gap,
        "rear_gap": rear_gap,
        "front_ttc": front_ttc,
        "rear_ttc": rear_ttc,
        "front_track_id": front_track_id,
        "rear_track_id": rear_track_id,
    }

def choose_lane_change_candidate(current_wp, ego_state, world_model):
    candidates = []
    for side in ("left", "right"):
        target_wp = adjacent_driving_lane(current_wp, side)
        if target_wp is None:
            continue
        gaps = assess_target_lane_gap(world_model, target_wp, ego_state)
        min_gap = min(gaps["front_gap"], gaps["rear_gap"])
        safe = (
            gaps["front_gap"] >= LANE_CHANGE_MIN_FRONT_GAP_M
            and gaps["rear_gap"] >= LANE_CHANGE_MIN_REAR_GAP_M
            and gaps["front_ttc"] >= LANE_CHANGE_FRONT_TTC_S
            and gaps["rear_ttc"] >= LANE_CHANGE_REAR_TTC_S
        )
        score = min_gap
        if safe:
            score += 100.0
        if side == "left":
            score += 2.0
        candidates.append({
            "side": side,
            "target_wp": target_wp,
            "gaps": gaps,
            "safe": safe,
            "score": score,
        })
    if not candidates:
        return None
    return max(candidates, key=lambda item: item["score"])

def build_route_context(ego_state, world_map, world_model):
    global control_state
    context = {
        "current_wp": None,
        "lane_count": 0,
        "highway_like": False,
        "lane_end_distance_m": float("inf"),
        "lane_change_plan": {
            "required": False,
            "active": False,
            "safe": False,
            "side": None,
            "target_wp": None,
            "front_gap": float("inf"),
            "rear_gap": float("inf"),
            "front_ttc": float("inf"),
            "rear_ttc": float("inf"),
            "lane_center_y": 0.0,
            "reason": "clear",
        },
    }

    loc = ego_location_from_state(ego_state)
    current_wp = world_map.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)
    if current_wp is None:
        return context

    speed_limit_mps = float(world_model.get("ego_speed_limit_mps", 0.0))
    lane_count = count_same_direction_lanes(current_wp)
    lane_end_distance = distance_to_lane_end(current_wp, max_dist=LANE_END_LOOKAHEAD_M)
    highway_like = (
        speed_limit_mps >= HIGHWAY_SPEED_LIMIT_MIN_MPS
        and lane_count >= HIGHWAY_MIN_LANES
        and not current_wp.is_junction
    )

    plan = context["lane_change_plan"]
    plan["required"] = (
        lane_count >= 2
        and not current_wp.is_junction
        and np.isfinite(lane_end_distance)
        and lane_end_distance < LANE_END_LOOKAHEAD_M
    )

    commit = control_state.get("lane_change_commit")
    if commit is not None:
        if time.time() > commit.get("expires_ts", 0.0):
            control_state["lane_change_commit"] = None
            commit = None
        elif current_wp.lane_id == commit.get("lane_id") and current_wp.road_id == commit.get("road_id"):
            control_state["lane_change_commit"] = None
            commit = None

    candidate = None
    if commit is not None:
        target_wp = adjacent_driving_lane(current_wp, commit["side"])
        if target_wp is not None:
            candidate = {
                "side": commit["side"],
                "target_wp": target_wp,
                "gaps": assess_target_lane_gap(world_model, target_wp, ego_state),
                "safe": True,
                "score": 999.0,
            }
        else:
            control_state["lane_change_commit"] = None
    if candidate is None and plan["required"]:
        candidate = choose_lane_change_candidate(current_wp, ego_state, world_model)
        if (
            candidate is not None
            and candidate["safe"]
            and np.isfinite(lane_end_distance)
            and lane_end_distance < LANE_CHANGE_COMMIT_DIST_M
        ):
            control_state["lane_change_commit"] = {
                "side": candidate["side"],
                "lane_id": candidate["target_wp"].lane_id,
                "road_id": candidate["target_wp"].road_id,
                "expires_ts": time.time() + 4.0,
            }

    if candidate is not None:
        plan.update(candidate["gaps"])
        plan["side"] = candidate["side"]
        plan["target_wp"] = candidate["target_wp"]
        plan["safe"] = bool(candidate["safe"])
        plan["active"] = commit is not None or (plan["required"] and candidate["safe"] and np.isfinite(lane_end_distance))
        if plan["active"]:
            plan["reason"] = f"merge_{candidate['side']}"
        elif plan["required"]:
            plan["reason"] = f"wait_for_{candidate['side']}_gap"
    elif plan["required"]:
        plan["reason"] = "lane_end_no_target_lane"

    context.update({
        "current_wp": current_wp,
        "lane_count": lane_count,
        "highway_like": highway_like,
        "lane_end_distance_m": lane_end_distance,
        "lane_change_plan": plan,
    })
    return context

def get_obstacle_info():
    now = time.time()
    best = {
        "distance": float("inf"),
        "type_id": None,
        "active": False,
        "name": None,
        "forward": float("inf"),
        "lateral": float("inf"),
    }

    stale = [name for name, event in obstacle_events.items() if now - event["ts"] > OBSTACLE_EVENT_TTL_S]
    for name in stale:
        obstacle_events.pop(name, None)

    for name, event in obstacle_events.items():
        if event.get("forward", event["distance"]) < 0.0:
            continue
        if abs(event.get("lateral", 0.0)) > OBSTACLE_LATERAL_GATE_M:
            continue
        if event["distance"] < best["distance"]:
            best = {
                "distance": event["distance"],
                "type_id": event["type_id"],
                "active": True,
                "name": name,
                "forward": event.get("forward", event["distance"]),
                "lateral": event.get("lateral", 0.0),
            }

    return best

tracks = {}
next_id = 0

# ======================
# LIDAR
# ======================
def lidar_clusters(lidar):
    pts = np.frombuffer(lidar.raw_data, dtype=np.float32).reshape(-1,4)
    xyz = pts[:,:3]
    xy = xyz[:,:2]
    rng = np.linalg.norm(xy, axis=1)

    mask = rng > LIDAR_MIN_RANGE_M
    mask &= rng < LIDAR_RANGE_M
    mask &= xyz[:,2] > LIDAR_MIN_Z
    mask &= xyz[:,2] < LIDAR_MAX_Z
    mask &= ~(
        (xyz[:,0] > -2.0) &
        (xyz[:,0] < 4.5) &
        (np.abs(xyz[:,1]) < 2.2) &
        (xyz[:,2] > -1.0)
    )

    xyz = xyz[mask]
    if len(xyz) == 0:
        return []

    cells = np.floor(xyz[:,:2] / LIDAR_GRID_M).astype(int)
    cell_to_points = defaultdict(list)
    for idx, cell in enumerate(cells):
        cell_to_points[(int(cell[0]), int(cell[1]))].append(idx)

    clusters = []
    visited = set()

    for key in list(cell_to_points.keys()):
        if key in visited:
            continue

        queue = [key]
        visited.add(key)
        member_idx = []

        while queue:
            cx, cy = queue.pop()
            member_idx.extend(cell_to_points[(cx, cy)])
            for nx in range(cx - 1, cx + 2):
                for ny in range(cy - 1, cy + 2):
                    nkey = (nx, ny)
                    if nkey in cell_to_points and nkey not in visited:
                        visited.add(nkey)
                        queue.append(nkey)

        group = xyz[member_idx]
        min_xy = group[:,:2].min(axis=0)
        max_xy = group[:,:2].max(axis=0)
        extent = max_xy - min_xy
        if extent[0] > 18.0 or extent[1] > 10.0:
            continue
        centroid_xy = np.median(group[:,:2], axis=0)
        range_m = speed_xy(float(centroid_xy[0]), float(centroid_xy[1]))
        min_points = 4 if range_m > 35.0 else LIDAR_CLUSTER_MIN_POINTS
        if len(group) < min_points:
            continue

        raw_height = float(max(0.15, group[:,2].max() - group[:,2].min()))
        raw_length = float(max(0.05, extent[0]))
        raw_width = float(max(0.05, extent[1]))
        if not cluster_vehicle_candidate(raw_length, raw_width, raw_height, range_m):
            continue

        height = float(max(1.2, raw_height))
        length = float(max(1.0, raw_length))
        width = float(max(0.8, raw_width))

        clusters.append({
            "centroid": centroid_xy,
            "min_xy": min_xy,
            "max_xy": max_xy,
            "length": length,
            "width": width,
            "height": height,
            "points": int(len(group)),
        })

    return clusters

# ======================
# PROJECTION (UNCHANGED)
# ======================
def get_K(w,h,fov=CAMERA_FOV):
    f = w/(2*math.tan(math.radians(fov)/2))
    return np.array([[f,0,w/2],
                     [0,f,h/2],
                     [0,0,1]])

def project_xyz(x, y, z, K, cam_x=CAMERA_POS_X, cam_y=CAMERA_POS_Y, cam_z=CAMERA_POS_Z, cam_yaw=0.0):
    # Transform ego-frame track coordinates into the local camera frame.
    dx = x - cam_x
    dy = y - cam_y

    theta = math.radians(cam_yaw)
    x_cam = math.cos(theta) * dx + math.sin(theta) * dy
    y_cam = -math.sin(theta) * dx + math.cos(theta) * dy

    if x_cam <= 0.5:
        return None

    f = K[0,0]
    cx = K[0,2]
    cy = K[1,2]

    u = f * (y_cam / x_cam) + cx
    v = f * ((cam_z - z) / x_cam) + cy

    if not np.isfinite(u) or not np.isfinite(v):
        return None
    return int(u), int(v)

def project(x, y, K, cam_x=CAMERA_POS_X, cam_y=CAMERA_POS_Y, cam_z=CAMERA_POS_Z, cam_yaw=0.0):
    return project_xyz(x, y, 0.0, K, cam_x=cam_x, cam_y=cam_y, cam_z=cam_z, cam_yaw=cam_yaw)


def camera_to_vehicle(u, v, K, cam_tf):
    # Convert a 2D ground-contact pixel into ego-frame ground coordinates.
    fx = K[0,0]
    fy = K[1,1]
    cx = K[0,2]
    cy = K[1,2]
    cam_x = cam_tf.location.x
    cam_y = cam_tf.location.y
    cam_z = cam_tf.location.z
    cam_yaw = cam_tf.rotation.yaw

    denom = v - cy
    if abs(denom) < 1e-3:
        return None
    x_cam = fy * cam_z / denom
    if x_cam <= 0.5:
        return None

    y_cam = (u - cx) * x_cam / fx

    theta = math.radians(cam_yaw)
    x = cam_x + math.cos(theta) * x_cam - math.sin(theta) * y_cam
    y = cam_y + math.sin(theta) * x_cam + math.cos(theta) * y_cam

    return x, y

def get_3d_box_corners(track):
    x, y = track.pos()
    heading = track.heading()
    hl = 0.5 * track.length
    hw = 0.5 * track.width
    h = track.height

    c = math.cos(heading)
    s = math.sin(heading)
    corners = []
    local = [
        (+hl, +hw, 0.0), (+hl, -hw, 0.0), (-hl, -hw, 0.0), (-hl, +hw, 0.0),
        (+hl, +hw, h), (+hl, -hw, h), (-hl, -hw, h), (-hl, +hw, h),
    ]
    for px, py, pz in local:
        gx = x + c * px - s * py
        gy = y + s * px + c * py
        corners.append((gx, gy, pz))
    return corners

# ======================
# ASSOC (LIDAR-MASTER + CAMERA CONFIRMATION)
# ======================
def associate(cams_dets, lidar_dets, cams_transforms, K):
    global tracks, next_id, camera_seed_memory
    now = time.time()

    # 1) Global LiDAR association
    lidar_pairs = []
    for tid, t in tracks.items():
        gate = LIDAR_MATCH_DIST + min(4.0, 0.6 * t.speed())
        tx, ty = t.pos()
        for i, cluster in enumerate(lidar_dets):
            dist = speed_xy(tx - cluster["centroid"][0], ty - cluster["centroid"][1])
            if dist <= gate:
                size_penalty = 0.08 * abs(t.length - cluster["length"]) + 0.10 * abs(t.width - cluster["width"])
                lidar_pairs.append((dist + size_penalty, tid, i))

    lidar_pairs.sort(key=lambda item: item[0])
    assigned_tracks = set()
    assigned_lidar = set()

    for _, tid, idx in lidar_pairs:
        if tid in assigned_tracks or idx in assigned_lidar:
            continue
        cluster = lidar_dets[idx]
        if not cluster_vehicle_like(cluster["length"], cluster["width"], cluster["height"]):
            continue
        z = np.array([[cluster["centroid"][0]], [cluster["centroid"][1]]], float)
        tracks[tid].update(z, from_lidar=True, meas_var=1.2)
        tracks[tid].set_extent(cluster["length"], cluster["width"], cluster["height"])
        tracks[tid].last_lidar_points = int(cluster["points"])
        assigned_tracks.add(tid)
        assigned_lidar.add(idx)

    for tid, t in tracks.items():
        if tid not in assigned_tracks:
            t.confidence = max(0.0, t.confidence - 0.05)

    # 2) Seed new LiDAR tracks
    for i, cluster in enumerate(lidar_dets):
        if i in assigned_lidar:
            continue
        if not cluster_vehicle_like(cluster["length"], cluster["width"], cluster["height"]):
            continue
        pos = cluster["centroid"]
        if any(speed_xy(t.pos()[0] - pos[0], t.pos()[1] - pos[1]) < 2.0 for t in tracks.values()):
            continue
        tnew = Track(next_id, float(pos[0]), float(pos[1]))
        tnew.set_extent(cluster["length"], cluster["width"], cluster["height"])
        tnew.update(np.array([[pos[0]], [pos[1]]], float), from_lidar=True, meas_var=1.2)
        tnew.last_lidar_points = int(cluster["points"])
        tnew.confidence = max(tnew.confidence, 0.2)
        tracks[next_id] = tnew
        next_id += 1

    # 3) Global camera confirmation and soft updates
    cam_observations = cluster_camera_observations(cams_dets, cams_transforms, K)
    cam_pairs = []
    for obs_idx, obs in enumerate(cam_observations):
        ox = obs["x"]
        oy = obs["y"]
        for tid, t in tracks.items():
            recent_cam, recent_lidar, recent_radar, recent_multiview, _ = recent_track_support(t, now)
            if not (t.confirmed or recent_cam or recent_lidar or recent_radar):
                continue
            handoff_bonus = camera_assoc_bonus(t.last_camera_name, obs["cams"])
            camera_history_bonus = 0.0
            if recent_cam:
                camera_history_bonus += 0.60
            if recent_multiview:
                camera_history_bonus += 0.35
            if t.cam_hits >= CAMERA_SEED_CONFIRM_FRAMES:
                camera_history_bonus += 0.25
            if (
                t.last_camera_name is not None
                and any(
                    camera_side(t.last_camera_name) == camera_side(cam_name)
                    and camera_side(cam_name) != "center"
                    for cam_name in obs["cams"]
                )
            ):
                camera_history_bonus += 0.35
            camera_penalty = 0.0
            if not recent_cam and t.cam_hits < 1 and t.lidar_hits > 0:
                camera_penalty += 0.60
            gate = CAM_GLOBAL_ASSOC_DIST_M + min(5.0, 0.35 * t.speed()) + 0.4 * max(0, len(obs["cams"]) - 1) + handoff_bonus + 0.35 * camera_history_bonus
            dist = speed_xy(ox - t.pos()[0], oy - t.pos()[1])
            cost = dist - 0.50 * obs["conf"] - 0.20 * max(0, len(obs["cams"]) - 1) - 0.35 * handoff_bonus - 0.45 * camera_history_bonus + camera_penalty
            if cost <= gate:
                cam_pairs.append((cost, tid, obs_idx))

    cam_pairs.sort(key=lambda item: item[0])
    matched_tracks = set()
    matched_obs = set()

    for _, tid, obs_idx in cam_pairs:
        if tid in matched_tracks or obs_idx in matched_obs:
            continue
        matched_tracks.add(tid)
        matched_obs.add(obs_idx)

        obs = cam_observations[obs_idx]
        tr = tracks[tid]
        handoff_bonus = camera_assoc_bonus(tr.last_camera_name, obs["cams"])
        tr.cam_set.update(obs["cams"])
        tr.cam_hits += max(1, obs["count"])
        tr.last_cam_seen = now
        tr.last_camera_name = obs["best_cam"]
        tr.last_camera_obs_count = len(obs["cams"])
        if len(obs["cams"]) >= 2 or handoff_bonus >= 0.85 * CAMERA_ADJ_ASSOC_BONUS_M:
            tr.last_multiview_seen = now
        tr.confidence = min(1.0, tr.confidence + 0.04 + 0.02 * min(3, len(obs["cams"])))

        cam_dist = speed_xy(obs["x"] - tr.pos()[0], obs["y"] - tr.pos()[1])
        if cam_dist < CAM_UPDATE_DIST + 0.75 * max(0, len(obs["cams"]) - 1) + 0.75 * handoff_bonus:
            cam_var = 2.8 if (len(obs["cams"]) >= 2 or handoff_bonus >= 0.85 * CAMERA_ADJ_ASSOC_BONUS_M) else 4.2
            tr.update(np.array([[obs["x"]], [obs["y"]]], float), from_lidar=False, meas_var=cam_var)

        if tr.lidar_hits >= 2 and tr.cam_hits >= CAM_HITS_CONFIRM:
            tr.confirmed = True

    # 4) Reacquire from unmatched global camera observations
    for obs_idx, obs in enumerate(cam_observations):
        if obs_idx in matched_obs or obs["conf"] < 0.25:
            continue
        if speed_xy(obs["x"], obs["y"]) > CAM_SEED_RANGE_M or obs["x"] < camera_seed_min_x(obs["best_cam"]):
            continue
        if any(speed_xy(obs["x"] - t.pos()[0], obs["y"] - t.pos()[1]) < 4.0 for t in tracks.values()):
            continue

        key = (
            int(round(obs["x"] / CAMERA_SEED_GRID_X_M)),
            int(round(obs["y"] / CAMERA_SEED_GRID_Y_M)),
        )
        seed = camera_seed_memory.get(key)
        if seed is None or frame_idx - seed["frame"] > CAMERA_SEED_TTL_FRAMES:
            seed = {"count": 0, "x": obs["x"], "y": obs["y"], "frame": frame_idx, "conf": obs["conf"], "cams": set()}

        seed["count"] += max(1, obs["count"])
        seed["x"] = 0.6 * seed["x"] + 0.4 * obs["x"]
        seed["y"] = 0.6 * seed["y"] + 0.4 * obs["y"]
        seed["frame"] = frame_idx
        seed["conf"] = max(seed["conf"], obs["conf"])
        seed["cams"].update(obs["cams"])
        camera_seed_memory[key] = seed

        if seed["count"] >= CAMERA_SEED_CONFIRM_FRAMES and (len(seed["cams"]) >= 2 or seed["x"] < 25.0):
            candidate = Track(next_id, seed["x"], seed["y"])
            candidate.confidence = 0.20
            candidate.cam_hits = seed["count"]
            candidate.last_cam_seen = now
            candidate.last_camera_obs_count = len(seed["cams"])
            candidate.cam_set.update(seed["cams"])
            candidate.last_camera_name = obs["best_cam"]
            if len(seed["cams"]) >= 2:
                candidate.last_multiview_seen = now
            tracks[next_id] = candidate
            next_id += 1
            camera_seed_memory.pop(key, None)

    stale_keys = [k for k, seed in camera_seed_memory.items() if frame_idx - seed["frame"] > CAMERA_SEED_TTL_FRAMES]
    for key in stale_keys:
        camera_seed_memory.pop(key, None)

    # 5) confirmation decision and persistence
    for t in tracks.values():
        speed = t.speed()
        if speed >= MIN_SPEED_MPS:
            t.consistent_velocity_frames += 1
        else:
            t.consistent_velocity_frames = max(0, t.consistent_velocity_frames - 1)

        recent_cam, recent_lidar, recent_radar, recent_multiview, support_score = recent_track_support(t, now)
        if not t.confirmed:
            close_forward = t.pos()[0] > -2.0 and t.pos()[0] < LIDAR_ONLY_FRONT_MAX_X_M and abs(t.pos()[1]) < LIDAR_ONLY_WIDE_LATERAL_M
            persistent_camera = persistent_camera_track_candidate(t, recent_multiview=recent_multiview, road_relevant=True)
            merge_side = -TRACK_REAR_RELEVANCE_M <= t.pos()[0] <= 40.0 and MERGE_RADAR_LATERAL_MIN_M <= abs(t.pos()[1]) <= MERGE_TRACK_LATERAL_M
            moving_track = t.consistent_velocity_frames >= 4 and t.speed() > 0.8
            strong_lidar = t.last_lidar_points >= MIN_STRONG_LIDAR_POINTS
            if recent_lidar and recent_cam and track_vehicle_like(t) and (recent_multiview or t.cam_hits >= CAM_HITS_CONFIRM or moving_track):
                t.confirmed = True
                t.confidence = max(t.confidence, CONFIRMATION_CONFIDENCE)
            elif recent_lidar and recent_radar and track_vehicle_like(t) and (merge_side or moving_track or t.cam_hits >= 1):
                t.confirmed = True
                t.confidence = max(t.confidence, 0.60)
            elif recent_cam and recent_radar and merge_side and track_vehicle_like(t) and (t.cam_hits >= 1 or t.radar_hits >= RADAR_SEED_CONFIRM_FRAMES):
                t.confirmed = True
                t.confidence = max(t.confidence, 0.48)
            elif recent_radar and merge_side and track_vehicle_like(t) and t.radar_hits >= 3 and t.radar_support >= 0.55 and t.radar_closing_speed >= max(2.5, 0.8 * MERGE_MIN_CLOSING_SPEED_MPS):
                t.confirmed = True
                t.confidence = max(t.confidence, 0.40)
            elif recent_lidar and close_forward and strong_lidar and track_vehicle_like(t) and (moving_track or t.cam_hits >= 1 or abs(t.pos()[1]) < LIDAR_ONLY_LATERAL_M):
                t.confirmed = True
                t.confidence = max(t.confidence, 0.52)
            elif recent_cam and track_vehicle_like(t) and persistent_camera:
                t.confirmed = True
                t.confidence = max(t.confidence, 0.44)
            elif recent_cam and recent_multiview and t.cam_hits >= CAMERA_SEED_CONFIRM_FRAMES and close_forward and track_vehicle_like(t):
                t.confirmed = True
                t.confidence = max(t.confidence, 0.42)

        # keep confirmed through brief occlusions
        if t.confirmed and support_score > 0.0:
            t.miss = max(0, t.miss - 1)

        if t.miss > 0:
            t.confidence = max(0.0, t.confidence - 0.025)

    merge_duplicate_tracks(now)

    # 6) prune stale or weak tracks
    tracks = {
        k: v for k, v in tracks.items()
        if v.miss < MAX_TRACK_AGE
        and track_vehicle_like(v)
        and (
            (
                v.confirmed and (
                    recent_track_support(v, now)[0]
                    or recent_track_support(v, now)[2]
                    or (
                        recent_track_support(v, now)[1]
                        and v.pos()[0] < LIDAR_ONLY_FRONT_MAX_X_M
                        and abs(v.pos()[1]) < LIDAR_ONLY_LATERAL_M
                        and v.last_lidar_points >= MIN_STRONG_LIDAR_POINTS
                    )
                )
            )
            or v.confidence > 0.1
            or v.lidar_hits > 1
            or now - v.last_cam_seen < track_camera_hold_window(v)
        )
    }

    # debug
    if frame_idx % 30 == 0:
        active = [(t.id, t.confirmed, round(t.confidence,2), len(t.cam_set), t.cam_hits, t.lidar_hits, (float(t.x[0,0]), float(t.x[1,0])), t.consistent_velocity_frames) for t in tracks.values()]
        print(f"[TRACK] active={len(active)} -> {active}")

def risk_color(score):
    if score >= 0.8:
        return (0, 0, 255)
    if score >= 0.45:
        return (0, 180, 255)
    return (0, 255, 0)

def idm_target_speed(current_speed, desired_speed, gap_m, closing_speed_mps):
    desired_speed = max(1.0, float(desired_speed))
    gap = max(0.5, float(gap_m) - 3.0)
    closing = max(0.0, float(closing_speed_mps))
    comfort_term = 2.0 * math.sqrt(max(0.1, IDM_COMFORT_ACCEL_MPS2 * IDM_COMFORT_BRAKE_MPS2))
    s_star = IDM_MIN_GAP_M + max(0.0, current_speed * IDM_TIME_HEADWAY_S + (current_speed * closing) / comfort_term)
    accel = IDM_COMFORT_ACCEL_MPS2 * (1.0 - (current_speed / desired_speed) ** IDM_DELTA - (s_star / max(gap, 0.5)) ** 2)
    next_speed = current_speed + accel * STEP
    return clamp(next_speed, 0.0, desired_speed)

def published_track_priority(entry):
    ttc = entry["ttc"] if np.isfinite(entry["ttc"]) else 999.0
    collision = entry["collision_time"] if np.isfinite(entry["collision_time"]) else 999.0
    return (
        0 if entry["front_hazard"] else 1,
        0 if entry.get("merge_actionable", False) else 1,
        0 if entry.get("same_direction_adjacent", False) else 1,
        min(ttc, collision),
        -entry["recent_support_score"],
        -entry["risk"],
        -entry["radar_support"],
        entry["distance"],
    )

def prune_published_tracks(entries):
    if not entries:
        return []

    kept = []
    seen = set()
    side_candidates = {"left": [], "right": []}
    top_candidate = max(entries, key=lambda item: (item["risk"], -item["distance"]))

    def add_entry(entry):
        if entry["id"] in seen:
            return
        kept.append(entry)
        seen.add(entry["id"])

    for entry in entries:
        x = entry["x"]
        y = entry["y"]
        merge_band = (
            MERGE_RADAR_REAR_X_MIN_M <= x <= 40.0
            and MERGE_RADAR_LATERAL_MIN_M <= abs(y) <= MERGE_TRACK_LATERAL_M
        )
        in_lane = x > -4.0 and abs(y) < (LANE_HALF_WIDTH_M + 0.8)
        strong_support = (
            entry["recent_support_score"] >= 1.45
            or entry["radar_support"] >= 0.60
            or entry["risk"] >= 0.30
            or entry.get("close_side_camera_ok", False)
            or entry.get("same_direction_adjacent", False)
        )
        strong_in_lane_support = (
            strong_support
            or (
                entry.get("ego_path_overlap", False)
                and entry.get("recent_support_score", 0.0) >= 1.15
            )
            or (
                entry.get("radar_support", 0.0) >= 0.35
                and entry["distance"] < 35.0
            )
        )

        if in_lane:
            if strong_in_lane_support:
                add_entry(entry)
            continue
        if merge_band:
            if x < -12.0 and not strong_support and not entry["camera_only_ok"]:
                continue
            side = "right" if y > 0.0 else "left"
            side_candidates[side].append(entry)
            continue
        if entry["road_relevant"] and strong_support and entry["distance"] < 45.0:
            add_entry(entry)

    for side in ("left", "right"):
        side_candidates[side].sort(key=published_track_priority)
        for entry in side_candidates[side][:MAX_PUBLISHED_SIDE_TRACKS_PER_SIDE]:
            add_entry(entry)

    if top_candidate["risk"] >= 0.35:
        add_entry(top_candidate)

    kept.sort(key=lambda item: item["distance"])
    return kept[:MAX_PUBLISHED_TRACKS]

def update_track_world_motion(track, ego_tf, now):
    gx, gy = track.estimate_global(ego_tf)
    track.world_history.append((now, gx, gy))
    if len(track.world_history) > 20:
        track.world_history.pop(0)

    motion_speed = 0.0
    if len(track.world_history) >= 2:
        newest = track.world_history[-1]
        oldest = None
        for item in reversed(track.world_history[:-1]):
            if newest[0] - item[0] >= 0.75:
                oldest = item
                break
        if oldest is None:
            oldest = track.world_history[0]
        dt = max(1e-3, newest[0] - oldest[0])
        motion_speed = speed_xy(newest[1] - oldest[1], newest[2] - oldest[2]) / dt
    track.world_motion_speed = motion_speed
    return gx, gy, motion_speed

def build_world_model(ego_state, world_map, radar_points):
    ego_tf = ego_transform_from_state(ego_state)
    ego_speed = float(ego_state["speed"])
    entries = []
    radar_clusters = cluster_radar_points(radar_points)
    merge_monitor = None

    for cluster in radar_clusters:
        x = cluster["x"]
        y = cluster["y"]
        lateral = abs(y)
        closing_speed = cluster["closing_speed"]
        if lateral < MERGE_RADAR_LATERAL_MIN_M or lateral > MERGE_RADAR_LATERAL_MAX_M:
            continue
        if x < MERGE_RADAR_REAR_X_MIN_M or x > MERGE_RADAR_FRONT_X_MAX_M:
            continue
        if closing_speed < MERGE_MIN_CLOSING_SPEED_MPS:
            continue
        ref_dist = abs(x) if abs(x) > 1.0 else cluster["distance"]
        ttc = ref_dist / max(0.1, closing_speed)
        if merge_monitor is None or ttc < merge_monitor["ttc"]:
            merge_monitor = {
                "x": x,
                "y": y,
                "distance": cluster["distance"],
                "closing_speed": closing_speed,
                "ttc": ttc,
                "side": "right" if y > 0.0 else "left",
                "actionable": False,
            }

    now = time.time()
    merge_track_hazard = None
    for t in tracks.values():
        _, _, world_motion_speed = update_track_world_motion(t, ego_tf, now)
        if not t.confirmed or not track_vehicle_like(t):
            t.future = []
            t.risk = 0.0
            t.ttc = float("inf")
            t.collision_time = float("inf")
            continue

        recent_cam, recent_lidar, recent_radar, recent_multiview, support_score = recent_track_support(t, now)
        x, y = t.pos()
        vx, vy = t.vel()
        dist = speed_xy(x, y)
        road_relevant = track_is_road_relevant(t, ego_state, world_map)
        close_forward = x > -2.0 and x < LIDAR_ONLY_FRONT_MAX_X_M and abs(y) < LIDAR_ONLY_WIDE_LATERAL_M
        corridor_relevant = track_in_tracking_corridor(x, y)
        merge_side = (
            -TRACK_REAR_RELEVANCE_M <= x <= 40.0
            and MERGE_RADAR_LATERAL_MIN_M <= abs(y) <= MERGE_TRACK_LATERAL_M
        )
        lidar_only_ok = recent_lidar and close_forward and t.last_lidar_points >= MIN_STRONG_LIDAR_POINTS and world_motion_speed > 0.5
        camera_only_ok = (
            recent_cam
            and t.lidar_hits < 1
            and persistent_camera_track_candidate(t, recent_multiview=recent_multiview, road_relevant=road_relevant)
        )
        close_side_camera_ok = (
            t.confirmed
            and recent_cam
            and t.lidar_hits < 1
            and CLOSE_SIDE_CAMERA_X_MIN_M <= x <= CLOSE_SIDE_CAMERA_X_MAX_M
            and CLOSE_SIDE_CAMERA_LATERAL_MIN_M <= abs(y) <= MERGE_TRACK_LATERAL_M
            and t.cam_hits >= CLOSE_SIDE_CAMERA_MIN_HITS
            and t.confidence >= 0.28
            and t.miss <= CAMERA_PERSIST_MAX_MISS
            and (
                road_relevant
                or recent_multiview
                or len(t.cam_set) >= 2
                or t.last_camera_obs_count >= 2
            )
        )
        radar_only_merge = (
            recent_radar
            and merge_side
            and t.radar_support >= 0.5
            and t.radar_closing_speed >= max(2.0, 0.75 * MERGE_MIN_CLOSING_SPEED_MPS)
        )
        merge_publish_ok = (
            t.confirmed
            and merge_side
            and recent_cam
            and recent_radar
            and (t.radar_hits >= RADAR_SEED_CONFIRM_FRAMES or t.radar_support >= 0.60)
            and t.confidence >= 0.35
        )
        if support_score < 1.45 and not lidar_only_ok and not radar_only_merge and not camera_only_ok and not close_side_camera_ok and not merge_publish_ok:
            t.future = []
            t.risk = 0.0
            t.ttc = float("inf")
            t.collision_time = float("inf")
            continue
        if not recent_cam and not recent_radar and not lidar_only_ok and not close_side_camera_ok and not merge_publish_ok:
            t.future = []
            t.risk = 0.0
            t.ttc = float("inf")
            t.collision_time = float("inf")
            continue
        if abs(y) > 6.5 and not recent_cam and not recent_radar and not close_side_camera_ok and not merge_publish_ok:
            t.future = []
            t.risk = 0.0
            t.ttc = float("inf")
            t.collision_time = float("inf")
            continue
        if x < -5.0 and not recent_cam and not recent_radar and not close_side_camera_ok and not merge_publish_ok:
            t.future = []
            t.risk = 0.0
            t.ttc = float("inf")
            t.collision_time = float("inf")
            continue
        if not corridor_relevant and not road_relevant and not recent_radar and not recent_multiview and not close_side_camera_ok and not merge_publish_ok:
            t.future = []
            t.risk = 0.0
            t.ttc = float("inf")
            t.collision_time = float("inf")
            continue
        if dist > 65.0 and not recent_cam and not recent_radar:
            t.future = []
            t.risk = 0.0
            t.ttc = float("inf")
            t.collision_time = float("inf")
            continue
        if dist > 45.0 and not recent_radar and not recent_multiview and not camera_only_ok and not close_side_camera_ok and not merge_publish_ok:
            t.future = []
            t.risk = 0.0
            t.ttc = float("inf")
            t.collision_time = float("inf")
            continue
        if t.lidar_hits < 1 and not camera_only_ok and not close_side_camera_ok and not merge_publish_ok and not radar_only_merge and (dist > max(MAX_CAMERA_ONLY_RANGE_M, 38.0) or t.cam_hits < CAMERA_SEED_CONFIRM_FRAMES):
            t.future = []
            t.risk = 0.0
            t.ttc = float("inf")
            t.collision_time = float("inf")
            continue
        if t.lidar_hits < 1 and not camera_only_ok and not close_side_camera_ok and not merge_publish_ok and not radar_only_merge and (not recent_multiview or (abs(y) > 6.5 and not close_side_camera_ok) or (-4.0 < x < 3.0 and abs(y) < 1.8)):
            t.future = []
            t.risk = 0.0
            t.ttc = float("inf")
            t.collision_time = float("inf")
            continue
        future = t.predict_future()
        lane_behavior = analyze_track_lane_behavior(t, future, ego_state, world_map)
        lane_clear_turn = (
            lane_behavior["stays_in_lane"]
            and lane_behavior["stays_outside_ego_path"]
            and not lane_behavior["ego_path_overlap"]
        )
        same_direction_adjacent = (
            lane_behavior["adjacent_parallel"]
            and abs(y) > (LANE_HALF_WIDTH_M + 0.45)
            and abs(vy) <= PARALLEL_PASSER_MAX_VY_MPS
            and x >= MERGE_RADAR_REAR_X_MIN_M
        )
        lateral_toward_ego = lateral_speed_toward_ego(y, vy)

        radial_closing = 0.0
        if dist > 1e-3:
            radial_closing = max(0.0, -(x * vx + y * vy) / dist)

        ttc = float("inf")
        strongest_closing = max(radial_closing, t.radar_closing_speed)
        if strongest_closing > 0.1:
            ttc = dist / strongest_closing

        front_ttc = float("inf")
        if x > 0.5 and abs(y) < (LANE_HALF_WIDTH_M + 0.5 * t.width) and vx < -0.1:
            front_ttc = x / max(0.1, -vx)

        collision_time = float("inf")
        lane_intrusion = False
        cross_hazard = False
        min_future_dist = dist

        for tau, px, py in future:
            min_future_dist = min(min_future_dist, speed_xy(px, py))
            if px > -2.0 and abs(py) < (LANE_HALF_WIDTH_M + 0.5 * t.width):
                lane_intrusion = True
            if abs(py) < (LANE_HALF_WIDTH_M + 0.6) and lateral_toward_ego > 0.4 and px > -5.0:
                cross_hazard = True
            if abs(px) < (EGO_HALF_LENGTH_M + 0.5 * t.length) and abs(py) < (EGO_HALF_WIDTH_M + 0.5 * t.width):
                collision_time = min(collision_time, tau)

        if lane_clear_turn and not np.isfinite(collision_time):
            lane_intrusion = False
            cross_hazard = False
        if same_direction_adjacent and not np.isfinite(collision_time):
            lane_intrusion = False
            cross_hazard = False

        risk = 0.0
        if lane_intrusion:
            risk += 0.20
        if cross_hazard:
            risk += 0.20
        if dist < 25.0:
            risk += 0.20 * (1.0 - clamp(dist / 25.0, 0.0, 1.0))
        base_ttc = min(ttc, front_ttc)
        if same_direction_adjacent and not lane_intrusion and not cross_hazard and not np.isfinite(collision_time):
            base_ttc = front_ttc
        if np.isfinite(base_ttc):
            risk += 0.35 * (1.0 - clamp(base_ttc / 4.0, 0.0, 1.0))
        if np.isfinite(collision_time):
            risk += 0.45 * (1.0 - clamp(collision_time / PREDICTION_HORIZON_S, 0.0, 1.0))
        if t.radar_support > 0.0:
            radar_risk_scale = 0.15 if same_direction_adjacent and not np.isfinite(collision_time) else 1.0
            risk += 0.12 * clamp(t.radar_closing_speed / 12.0, 0.0, 1.0) * t.radar_support * radar_risk_scale
        if lane_clear_turn:
            risk *= TURN_LANE_RISK_SCALE
            if abs(y) > (LANE_HALF_WIDTH_M + 0.4) and not np.isfinite(collision_time):
                risk = min(risk, TURN_LANE_RISK_CAP)
        if same_direction_adjacent:
            risk *= PARALLEL_PASSER_RISK_SCALE
            if not np.isfinite(collision_time):
                risk = min(risk, PARALLEL_PASSER_RISK_CAP)
        if t.confidence < 0.35:
            risk *= 0.7
        if t.lidar_hits < 1:
            risk *= 0.70 if close_side_camera_ok else (0.65 if camera_only_ok else 0.45)

        merge_actionable = (
            merge_side
            and not same_direction_adjacent
            and (
                np.isfinite(collision_time)
                or (
                    lane_intrusion
                    and (
                        lateral_toward_ego >= MERGE_LATERAL_APPROACH_MIN_MPS
                        or lane_behavior["ego_path_min_dist"] < MERGE_PATH_OVERLAP_GATE_M
                        or abs(y) < (LANE_HALF_WIDTH_M + 1.4)
                    )
                )
                or (
                    cross_hazard
                    and lateral_toward_ego >= MERGE_LATERAL_APPROACH_MIN_MPS
                )
                or (
                    lane_behavior["ego_path_overlap"]
                    and (
                        lateral_toward_ego >= MERGE_LATERAL_APPROACH_MIN_MPS
                        or lane_behavior["ego_path_min_dist"] < MERGE_PATH_OVERLAP_GATE_M
                    )
                )
                or lateral_toward_ego > max(PARALLEL_PASSER_MAX_VY_MPS, 1.2)
            )
        )
        if merge_actionable:
            merge_ttc = collision_time if np.isfinite(collision_time) else base_ttc
            if (
                merge_track_hazard is None
                or (np.isfinite(merge_ttc) and merge_ttc < merge_track_hazard["ttc"])
                or (not np.isfinite(merge_track_hazard["ttc"]) and risk > merge_track_hazard.get("risk", 0.0))
            ):
                merge_track_hazard = {
                    "x": x,
                    "y": y,
                    "distance": dist,
                    "closing_speed": strongest_closing,
                    "ttc": merge_ttc,
                    "side": "right" if y > 0.0 else "left",
                    "actionable": True,
                    "track_id": t.id,
                    "risk": risk,
                    "lateral_toward_ego": lateral_toward_ego,
                }

        t.risk = clamp(risk, 0.0, 1.0)
        t.ttc = base_ttc
        t.collision_time = collision_time
        front_support_ok = (
            support_score >= 1.15
            or t.radar_support >= 0.40
            or (recent_cam and recent_lidar)
        )
        front_path_overlap = (
            lane_behavior["ego_path_overlap"]
            or lane_behavior["ego_path_min_dist"] < (0.6 * LANE_HALF_WIDTH_M + 0.25 * t.width)
        )
        t.front_hazard = (
            x > 0.0
            and abs(y) < (0.95 * LANE_HALF_WIDTH_M + 0.10 * t.width)
            and front_support_ok
            and front_path_overlap
            and not same_direction_adjacent
        )
        t.cross_hazard = cross_hazard

        entry = {
            "id": t.id,
            "x": x,
            "y": y,
            "vx": vx,
            "vy": vy,
            "speed": t.speed(),
            "distance": dist,
            "risk": t.risk,
            "ttc": t.ttc,
            "collision_time": collision_time,
            "front_hazard": t.front_hazard,
            "cross_hazard": cross_hazard,
            "road_relevant": road_relevant,
            "radar_closing_speed": t.radar_closing_speed,
            "radar_support": t.radar_support,
            "world_motion_speed": world_motion_speed,
            "recent_support_score": support_score,
            "lane_keep_confidence": lane_behavior["lane_fraction"],
            "lane_keep_clear": lane_clear_turn,
            "parallel_fraction": lane_behavior["parallel_fraction"],
            "same_direction_adjacent": same_direction_adjacent,
            "front_support_ok": front_support_ok,
            "ego_path_overlap": lane_behavior["ego_path_overlap"],
            "ego_path_min_dist": lane_behavior["ego_path_min_dist"],
            "lateral_toward_ego": lateral_toward_ego,
            "camera_only_ok": camera_only_ok,
            "close_side_camera_ok": close_side_camera_ok,
            "merge_actionable": merge_actionable,
            "future": future,
        }
        entries.append(entry)

    entries = prune_published_tracks(entries)
    merge_hazard = merge_track_hazard if merge_track_hazard is not None else merge_monitor
    lead = None
    top = None
    for entry in entries:
        if entry["front_hazard"] and entry["x"] > 0.0:
            if lead is None or entry["x"] < lead["x"]:
                lead = entry
        if top is None or (entry["risk"], -entry["distance"]) > (top["risk"], -top["distance"]):
            top = entry

    entries.sort(key=lambda item: item["distance"])
    return {
        "ego_speed": ego_speed,
        "ego_source": ego_state.get("source", "UNKNOWN"),
        "tracks": entries,
        "lead": lead,
        "top": top,
        "merge_hazard": merge_hazard,
    }

def decide_action(world_model, traffic_light_info, obstacle_info):
    cruise_speed = CRUISE_SPEED_MPS
    road_speed_limit = float(world_model.get("ego_speed_limit_mps", 0.0))
    if road_speed_limit > 1.0:
        cruise_speed = min(cruise_speed, SPEED_LIMIT_TRACK_FACTOR * road_speed_limit)
    route_speed_cap = float(world_model.get("route_speed_cap_mps", float("inf")))
    if np.isfinite(route_speed_cap):
        cruise_speed = min(cruise_speed, route_speed_cap)
    decision = {
        "mode": "CRUISE",
        "target_speed": cruise_speed,
        "brake": 0.0,
        "reason": "clear",
        "lead_track_id": None,
        "top_track_id": None,
        "top_risk": 0.0,
        "ttc": float("inf"),
        "collision_time": float("inf"),
        "traffic_light_state": traffic_light_info["state_name"],
        "traffic_light_distance": traffic_light_info["distance"],
        "traffic_light_confidence": traffic_light_info.get("confidence", 0.0),
        "lane_error_m": 0.0,
        "heading_error_deg": 0.0,
        "speed_limit": cruise_speed,
        "obstacle_distance": obstacle_info["distance"],
        "obstacle_type": obstacle_info["type_id"],
        "obstacle_lateral": obstacle_info.get("lateral", float("inf")),
        "obstacle_source": obstacle_info.get("source"),
        "ego_source": world_model.get("ego_source", "UNKNOWN"),
        "ego_control_speed_mps": world_model.get("ego_speed_for_control", world_model["ego_speed"]),
        "merge_side": None,
        "merge_ttc": float("inf"),
        "merge_closing_speed": 0.0,
    }

    ego_speed = world_model.get("ego_speed_for_control", world_model["ego_speed"])
    lead = world_model["lead"]
    top = world_model["top"]

    if top is not None:
        decision["top_track_id"] = top["id"]
        decision["top_risk"] = top["risk"]
        decision["ttc"] = top["ttc"]
        decision["collision_time"] = top["collision_time"]

    safe_gap = max(MIN_FOLLOW_DISTANCE_M, 3.0 + FOLLOW_TIME_GAP_S * max(ego_speed, 2.0))

    if lead is not None:
        gap = lead["x"]
        decision["lead_track_id"] = lead["id"]
        lead_closing_speed = clamp(max(0.0, -lead.get("vx", 0.0)), 0.0, 15.0)
        if lead.get("front_support_ok", False) or lead.get("radar_support", 0.0) >= 0.35 or gap < 18.0:
            idm_speed = idm_target_speed(ego_speed, decision["target_speed"], gap, lead_closing_speed)
            if gap > safe_gap + 2.0 and lead["risk"] < 0.45 and not np.isfinite(lead["collision_time"]):
                idm_speed = max(
                    idm_speed,
                    min(decision["target_speed"], max(1.5, 0.22 * gap))
                )
            decision["target_speed"] = min(decision["target_speed"], idm_speed)
        if gap < safe_gap:
            decision["mode"] = "FOLLOW"
            decision["reason"] = f"lead T{lead['id']} gap={gap:.1f}m"
            decision["brake"] = max(decision["brake"], 0.12 + 0.33 * clamp((safe_gap - gap) / max(1.0, safe_gap), 0.0, 1.0))
        if lead["collision_time"] < EMERGENCY_COLLISION_S or lead["ttc"] < EMERGENCY_TTC_S:
            decision["mode"] = "EMERGENCY_BRAKE"
            decision["reason"] = f"imminent front collision T{lead['id']}"
            decision["target_speed"] = 0.0
            decision["brake"] = 1.0
        elif lead["collision_time"] < CAUTION_COLLISION_S or lead["ttc"] < CAUTION_TTC_S:
            decision["mode"] = "CAUTION"
            decision["reason"] = f"front ttc={lead['ttc']:.1f}s"
            decision["target_speed"] = min(decision["target_speed"], max(0.0, gap / 1.6))
            decision["brake"] = max(decision["brake"], 0.24)

    if top is not None and top is not lead:
        if not top.get("same_direction_adjacent", False) or top.get("merge_actionable", False):
            if top["cross_hazard"] and top["collision_time"] < 1.6 and top["risk"] > 0.45:
                decision["mode"] = "YIELD"
                decision["reason"] = f"cross traffic T{top['id']}"
                decision["target_speed"] = min(decision["target_speed"], 4.5)
                decision["brake"] = max(decision["brake"], 0.28)
            if top["risk"] > 0.92 or top["collision_time"] < EMERGENCY_COLLISION_S:
                decision["mode"] = "EMERGENCY_BRAKE"
                decision["reason"] = f"predicted collision T{top['id']}"
                decision["target_speed"] = 0.0
                decision["brake"] = 1.0

    merge_hazard = world_model.get("merge_hazard")
    if merge_hazard is not None:
        decision["merge_side"] = merge_hazard["side"]
        decision["merge_ttc"] = merge_hazard["ttc"]
        decision["merge_closing_speed"] = merge_hazard["closing_speed"]
        merge_distance = float(merge_hazard.get("distance", float("inf")))
        merge_lateral = abs(float(merge_hazard.get("y", 99.0)))
        merge_lateral_rate = float(merge_hazard.get("lateral_toward_ego", 0.0))
        merge_near_lane = merge_lateral < (LANE_HALF_WIDTH_M + 2.2)
        immediate_merge_intrusion = (
            merge_distance < 8.0
            and merge_lateral < (LANE_HALF_WIDTH_M + 0.9)
            and merge_lateral_rate >= max(0.8, MERGE_LATERAL_APPROACH_MIN_MPS)
        )
        if (
            merge_hazard.get("actionable", False)
            and merge_hazard["ttc"] < MERGE_EMERGENCY_TTC_S
            and merge_distance < MERGE_EMERGENCY_DISTANCE_M
            and immediate_merge_intrusion
            and (merge_near_lane or merge_lateral_rate >= MERGE_LATERAL_APPROACH_MIN_MPS)
        ):
            decision["mode"] = "YIELD" if decision["mode"] != "EMERGENCY_BRAKE" else decision["mode"]
            decision["reason"] = f"{merge_hazard['side']} merge hazard {merge_hazard['ttc']:.1f}s"
            decision["target_speed"] = min(decision["target_speed"], 3.5)
            decision["brake"] = max(decision["brake"], 0.45)
        elif (
            merge_hazard.get("actionable", False)
            and merge_hazard["ttc"] < MERGE_CAUTION_TTC_S
            and merge_distance < MERGE_CAUTION_DISTANCE_M
            and (immediate_merge_intrusion and merge_hazard["ttc"] < 1.6)
            and (merge_near_lane or merge_lateral_rate >= MERGE_LATERAL_APPROACH_MIN_MPS)
        ):
            if decision["mode"] == "CRUISE":
                decision["mode"] = "CAUTION"
                decision["reason"] = f"{merge_hazard['side']} closing traffic"
            decision["target_speed"] = min(decision["target_speed"], 10.0)
            decision["brake"] = max(decision["brake"], 0.10)

    light_conf = traffic_light_info.get("confidence", 0.0)
    light_state = traffic_light_info["state"]
    if ENABLE_TRAFFIC_LIGHT_CONTROL and light_conf >= TL_CONTROL_CONF_MIN and light_state in (carla.TrafficLightState.Red, carla.TrafficLightState.Yellow):
        dist = traffic_light_info["distance"]
        queue_stop = False
        if np.isfinite(dist) and dist < TRAFFIC_LIGHT_LOOKAHEAD_M:
            if light_state == carla.TrafficLightState.Yellow and dist < TL_YELLOW_COMMIT_DISTANCE_M and ego_speed > TL_YELLOW_COMMIT_SPEED_MPS:
                dist = float("inf")
            if lead is not None and lead["front_hazard"] and 0.0 < lead["x"] < dist + 8.0:
                queued_gap = max(0.0, lead["x"] - TRAFFIC_LIGHT_QUEUE_GAP_M)
                if queued_gap < dist:
                    dist = queued_gap
                    queue_stop = True
            stop_profile_divisor = 1.6 if queue_stop else 1.3
            stop_speed = max(0.0, (dist - TRAFFIC_LIGHT_STOP_BUFFER_M) / stop_profile_divisor) if np.isfinite(dist) else decision["target_speed"]
        else:
            dist = float("inf")
        if np.isfinite(dist) and dist < TRAFFIC_LIGHT_LOOKAHEAD_M:
            decision["target_speed"] = min(decision["target_speed"], stop_speed)
            decision["mode"] = "LIGHT_STOP" if decision["mode"] != "EMERGENCY_BRAKE" else decision["mode"]
            reason_suffix = " queue" if queue_stop else ""
            decision["reason"] = f"{traffic_light_info['state_name']} light{reason_suffix} {dist:.1f}m"
            decision["brake"] = max(
                decision["brake"],
                0.10 + 0.68 * clamp((TRAFFIC_LIGHT_LOOKAHEAD_M - dist) / TRAFFIC_LIGHT_LOOKAHEAD_M, 0.0, 1.0)
            )
            hard_stop_dist = 3.5 if queue_stop else 6.0
            if dist <= hard_stop_dist:
                decision["target_speed"] = 0.0
                decision["brake"] = max(decision["brake"], 0.95)

    if obstacle_info["active"]:
        dist = obstacle_info["distance"]
        if dist < OBSTACLE_BRAKE_DISTANCE_M:
            stop_speed = max(0.0, (dist - OBSTACLE_STOP_BUFFER_M) / 1.0)
            decision["target_speed"] = min(decision["target_speed"], stop_speed)
            decision["mode"] = "OBSTACLE_STOP" if decision["mode"] != "EMERGENCY_BRAKE" else decision["mode"]
            reason_type = obstacle_info["type_id"] if obstacle_info["type_id"] is not None else "unknown"
            decision["reason"] = f"obstacle {reason_type} {dist:.1f}m"
            decision["brake"] = max(
                decision["brake"],
                0.30 + 0.70 * clamp((OBSTACLE_BRAKE_DISTANCE_M - dist) / OBSTACLE_BRAKE_DISTANCE_M, 0.0, 1.0)
            )
        if dist <= 3.0:
            decision["target_speed"] = 0.0
            decision["brake"] = max(decision["brake"], 1.0)
            decision["mode"] = "EMERGENCY_BRAKE"
            decision["reason"] = f"hard obstacle stop {dist:.1f}m"

    return decision

def smooth_control_command(decision, ego_speed):
    global control_state
    now = time.time()
    if not control_state:
        control_state = {
            "target_speed": max(0.0, float(ego_speed)),
            "brake": 0.0,
            "last_mode": decision["mode"],
            "last_light_stop_ts": None,
        }

    mode = decision["mode"]
    prev_target = float(control_state.get("target_speed", max(0.0, float(ego_speed))))
    prev_brake = float(control_state.get("brake", 0.0))

    desired_speed = max(0.0, float(decision["target_speed"]))
    desired_brake = float(decision.get("brake", 0.0))

    accel_limit = DRIVE_MAX_ACCEL_MPS2
    decel_limit = DRIVE_MAX_DECEL_MPS2
    if mode in {"CAUTION", "FOLLOW", "YIELD"}:
        accel_limit = 1.4
        decel_limit = DRIVE_CAUTION_DECEL_MPS2
    if mode in {"LIGHT_STOP", "OBSTACLE_STOP"}:
        accel_limit = 0.8
        decel_limit = DRIVE_STOP_DECEL_MPS2
    if mode == "EMERGENCY_BRAKE":
        accel_limit = 0.0
        decel_limit = DRIVE_EMERGENCY_DECEL_MPS2

    if mode == "LIGHT_STOP":
        control_state["last_light_stop_ts"] = now
    elif (
        control_state.get("last_mode") == "LIGHT_STOP"
        and control_state.get("last_light_stop_ts") is not None
        and now - control_state["last_light_stop_ts"] < GREEN_RESUME_HOLD_S
        and desired_speed > prev_target
    ):
        accel_limit = min(accel_limit, GREEN_RESUME_ACCEL_MPS2)

    up_step = accel_limit * STEP
    down_step = decel_limit * STEP
    if desired_speed >= prev_target:
        target_speed = min(desired_speed, prev_target + up_step)
    else:
        target_speed = max(desired_speed, prev_target - down_step)

    apply_rate = BRAKE_APPLY_RATE
    release_rate = BRAKE_RELEASE_RATE
    if mode in {"LIGHT_STOP", "OBSTACLE_STOP"}:
        apply_rate = max(apply_rate, 2.8)
    if mode == "EMERGENCY_BRAKE":
        apply_rate = 8.0
        desired_brake = max(desired_brake, 0.95)

    if desired_brake >= prev_brake:
        brake_cmd = min(desired_brake, prev_brake + apply_rate * STEP)
    else:
        brake_cmd = max(desired_brake, prev_brake - release_rate * STEP)

    if target_speed > ego_speed + 0.8 and brake_cmd < 0.08:
        brake_cmd = 0.0
    if mode == "CRUISE" and desired_speed > ego_speed and brake_cmd < 0.12:
        brake_cmd = 0.0

    control_state["target_speed"] = target_speed
    control_state["brake"] = brake_cmd
    control_state["last_mode"] = mode

    smoothed = dict(decision)
    smoothed["raw_target_speed"] = desired_speed
    smoothed["raw_brake"] = desired_brake
    smoothed["target_speed"] = target_speed
    smoothed["brake"] = brake_cmd
    return smoothed

def compute_vehicle_control(world_map, decision, ego_state, world_model=None, route_agent=None):
    global control_state
    ego_speed = float(ego_state["speed"])
    control_speed = float(decision.get("ego_control_speed_mps", ego_speed))
    ego_loc = ego_location_from_state(ego_state)
    wp = world_map.get_waypoint(ego_loc, project_to_road=True, lane_type=carla.LaneType.Driving)
    if wp is None:
        return carla.VehicleControl(throttle=0.0, brake=0.5, steer=0.0)

    lookahead = clamp(10.0 + 1.6 * ego_speed, 10.0, 30.0)
    next_wps = wp.next(lookahead)
    target_wp = wp if not next_wps else min(next_wps, key=lambda cand: abs(wrap_deg(cand.transform.rotation.yaw - ego_state["yaw_deg"])))
    lane_error = lateral_error_to_waypoint(ego_loc, wp)
    heading_error_deg = heading_error_to_waypoint(ego_state["yaw_deg"], target_wp)

    speed_limit_mps = float(decision.get("speed_limit", CRUISE_SPEED_MPS))
    decision["speed_limit"] = speed_limit_mps
    decision["lane_error_m"] = lane_error
    decision["heading_error_deg"] = heading_error_deg

    target_speed_cmd = min(max(0.0, decision["target_speed"]), max(2.5, speed_limit_mps))
    speed_error = target_speed_cmd - control_speed
    speed_error_int = float(control_state.get("speed_error_int", 0.0))
    if decision["mode"] in {"LIGHT_STOP", "OBSTACLE_STOP", "EMERGENCY_BRAKE"}:
        speed_error_int = 0.0
    else:
        speed_error_int = clamp(speed_error_int + speed_error * STEP, -6.0, 6.0)
    control_state["speed_error_int"] = speed_error_int

    heading_error_rad = math.radians(heading_error_deg)
    stanley_term = math.atan2(0.85 * STANLEY_GAIN * lane_error, ego_speed + STANLEY_SOFTENING + 1.0)
    raw_steer = clamp(0.75 * heading_error_rad + 0.60 * stanley_term, -0.75, 0.75)
    if abs(lane_error) < 0.25 and abs(heading_error_deg) < 4.0:
        raw_steer *= 0.6

    prev_steer = float(control_state.get("manual_steer", 0.0))
    steer_step = clamp(0.08 + 0.015 * ego_speed, 0.08, 0.18)
    steer = prev_steer + clamp(raw_steer - prev_steer, -steer_step, steer_step)
    steer = clamp(steer, -0.75, 0.75)
    control_state["manual_steer"] = steer

    throttle = clamp(0.28 * speed_error + 0.035 * speed_error_int, 0.0, 0.88)
    brake = clamp(-0.24 * speed_error, 0.0, 0.82)
    if target_speed_cmd > max(12.0, control_speed + 5.0) and abs(steer) < 0.22:
        throttle = min(0.92, max(throttle, 0.42))

    if abs(steer) > 0.4:
        throttle = min(throttle, 0.35)

    if decision["mode"] in {"FOLLOW", "CAUTION", "YIELD", "EMERGENCY_BRAKE", "LIGHT_STOP"}:
        brake = max(brake, decision["brake"])
        if brake > 0.05:
            throttle = 0.0

    if decision["mode"] == "EMERGENCY_BRAKE":
        steer = clamp(0.5 * steer, -0.5, 0.5)
        brake = max(brake, 0.95)

    if decision["mode"] in {"LIGHT_STOP", "OBSTACLE_STOP"} and ego_speed < 1.5:
        steer *= 0.5
    return carla.VehicleControl(throttle=float(throttle), brake=float(brake), steer=float(steer))

def draw_hud(rgb, world_model, decision, control):
    vis = rgb.copy()
    tl_state = decision.get("traffic_light_state", "None")
    tl_dist = decision.get("traffic_light_distance", float("inf"))
    tl_conf = decision.get("traffic_light_confidence", 0.0)
    tl_text = f"TL: {tl_state}"
    if np.isfinite(tl_dist):
        tl_text += f" {tl_dist:.1f}m"
    if tl_conf > 0.0:
        tl_text += f" c={tl_conf:.2f}"
    merge_text = "merge: clear"
    if np.isfinite(decision.get("merge_ttc", float("inf"))):
        merge_text = f"merge {decision.get('merge_side', '?')} ttc={decision['merge_ttc']:.1f}s vc={decision.get('merge_closing_speed', 0.0):.1f}"
    eval_info = world_model.get("eval", {})
    lines = [
        f"Ego speed: {world_model['ego_speed']:.1f} m/s | src={decision.get('ego_source', 'UNKNOWN')}",
        f"Decision: {decision['mode']} | target {decision['target_speed']:.1f} m/s | reason {decision['reason']}",
        f"Control: throttle {control.throttle:.2f} brake {control.brake:.2f} steer {control.steer:.2f}",
        f"Top risk: T{decision['top_track_id'] if decision['top_track_id'] is not None else '-'} risk={decision['top_risk']:.2f} ttc={decision['ttc']:.1f}s",
        f"{tl_text} | obs={decision.get('obstacle_distance', float('inf')):.1f}m {decision.get('obstacle_source')} | {merge_text}",
        f"lane_err={decision.get('lane_error_m', 0.0):.2f}m | head_err={decision.get('heading_error_deg', 0.0):.1f}deg | speed_limit={decision.get('speed_limit', 0.0):.1f}m/s",
        f"Eval: matched {eval_info.get('matched', 0)}/{eval_info.get('gt_count', 0)} miss={eval_info.get('misses', 0)} false={eval_info.get('false_tracks', 0)} idsw={eval_info.get('id_switches', 0)}",
        f"Err: pos={eval_info.get('running_pos_err', float('nan')):.2f}m vel={eval_info.get('running_vel_err', float('nan')):.2f}m/s ego={eval_info.get('ego_pos_err', float('nan')):.2f}m yaw={eval_info.get('ego_yaw_err', float('nan')):.1f}deg",
    ]
    for idx, line in enumerate(lines):
        y = 24 + idx * 22
        cv2.putText(vis, line, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)
        cv2.putText(vis, line, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20,20,20), 1)

    state_color = (160,160,160)
    if tl_state == "Red":
        state_color = (0,0,255)
    elif tl_state == "Yellow":
        state_color = (0,255,255)
    elif tl_state == "Green":
        state_color = (0,255,0)
    cv2.circle(vis, (vis.shape[1] - 28, 28), 10, state_color, -1)
    return vis

def resize_with_letterbox(img, size):
    width, height = size
    if img is None or not hasattr(img, "shape") or img.shape[0] == 0 or img.shape[1] == 0:
        return np.zeros((height, width, 3), dtype=np.uint8)
    src_h, src_w = img.shape[:2]
    scale = min(width / max(1, src_w), height / max(1, src_h))
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    resized = cv2.resize(img, (new_w, new_h))
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    off_x = (width - new_w) // 2
    off_y = (height - new_h) // 2
    canvas[off_y:off_y + new_h, off_x:off_x + new_w] = resized
    return canvas

def compose_recording_frame(rgb, mosaic, bev):
    rgb_panel = resize_with_letterbox(rgb, (960, 540))
    bev_panel = resize_with_letterbox(bev, (480, 540))
    top_row = np.hstack([rgb_panel, bev_panel])
    mosaic_panel = resize_with_letterbox(mosaic, (1440, 720))
    return np.vstack([top_row, mosaic_panel])

def open_test_video_writer(frame):
    frame_h, frame_w = frame.shape[:2]
    base_dir = os.path.dirname(__file__)
    for codec_name, ext in TEST_VIDEO_CODEC_OPTIONS:
        video_path = os.path.join(base_dir, f"{TEST_VIDEO_BASENAME}{ext}")
        try:
            if os.path.exists(video_path):
                os.remove(video_path)
        except OSError:
            pass
        fourcc = cv2.VideoWriter_fourcc(*codec_name)
        writer = cv2.VideoWriter(video_path, fourcc, TEST_VIDEO_FPS, (frame_w, frame_h))
        if writer is not None and writer.isOpened():
            print(f"[VIDEO] recording to {video_path} codec={codec_name}")
            return writer, video_path, codec_name
        try:
            writer.release()
        except Exception:
            pass
    print("[VIDEO] unable to open any configured video writer")
    return None, None, None

# ======================
# DRAW (REUSED LOGIC)
# ======================
def draw_on(img, yolo, K, cam_x=CAMERA_POS_X, cam_y=CAMERA_POS_Y, cam_yaw=0.0, cam_z=CAMERA_POS_Z, traffic_lights=None, world_model=None):
    if img is None or not hasattr(img, 'shape') or img.shape[0] == 0 or img.shape[1] == 0:
        return np.zeros((2,2,3), dtype=np.uint8)

    vis = img.copy()
    visible_entries = {}
    if world_model is not None:
        visible_entries = {entry["id"]: entry for entry in world_model.get("tracks", [])}

    for det in traffic_lights or []:
        if det.get("state_name") not in {"Red", "Yellow", "Green"}:
            continue
        x1, y1, x2, y2 = det["bbox"]
        color = (0, 0, 255) if det["state_name"] == "Red" else (0, 255, 255) if det["state_name"] == "Yellow" else (0, 255, 0)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv2.putText(vis, f"TL {det['state_name']}", (x1, max(14, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

    for t in tracks.values():
        entry = visible_entries.get(t.id)
        if world_model is not None and entry is None:
            continue
        close_forward = t.pos()[0] > 0.0 and t.pos()[0] < 20.0 and abs(t.pos()[1]) < 3.5
        if not t.confirmed or not track_vehicle_like(t):
            continue
        if entry is None and t.cam_hits < 1 and t.lidar_hits < 2 and t.speed() < 0.1 and speed_xy(*t.pos()) > 65.0 and not close_forward:
            continue

        x, y = t.pos()
        vx, vy = t.vel()
        d = speed_xy(x, y)
        v = t.speed()

        p = project(x, y, K, cam_x=cam_x, cam_y=cam_y, cam_z=cam_z, cam_yaw=cam_yaw)
        if p is None:
            continue

        u, v_img = p
        if u < 0 or u >= vis.shape[1] or v_img < 0 or v_img >= vis.shape[0]:
            continue

        matched = None
        for det in yolo:
            fu, fv = det_footpoint(det)
            if det_contains(det, u, v_img, margin=20) or speed_xy(fu - u, fv - v_img) < 90.0:
                matched = det
                break

        risk_value = entry["risk"] if entry is not None else t.risk
        color = risk_color(risk_value)
        label = f"T{t.id}|v={v:.1f}|d={d:.1f}|r={risk_value:.2f}"

        if matched:
            x1,y1,x2,y2 = matched[:4]
            cv2.rectangle(vis,(x1,y1),(x2,y2),color,2)
            cv2.circle(vis, det_footpoint(matched), 4, color, -1)
            cv2.putText(vis, label, (x1,max(14,y1-8)), cv2.FONT_HERSHEY_SIMPLEX,0.4,color,2)
        else:
            size = int(clamp(180.0 / max(2.0, d), 10, 40))
            cv2.circle(vis,(u,v_img),size,color,2)
            cv2.putText(vis, label, (u+10,max(14,v_img-10)), cv2.FONT_HERSHEY_SIMPLEX,0.4,color,1)

        if SHOW_HEADING_VECTOR and v > 0.2:
            tip = project(x + 0.8 * vx, y + 0.8 * vy, K, cam_x=cam_x, cam_y=cam_y, cam_z=cam_z, cam_yaw=cam_yaw)
            if tip:
                cv2.arrowedLine(vis, (u, v_img), tip, color, 2, tipLength=0.25)

        if SHOW_PREDICTION_TRAIL and t.future:
            prev = (u, v_img)
            for _, px, py in t.future[:4]:
                proj = project(px, py, K, cam_x=cam_x, cam_y=cam_y, cam_z=cam_z, cam_yaw=cam_yaw)
                if proj:
                    cv2.circle(vis, proj, 3, color, -1)
                    cv2.line(vis, prev, proj, color, 1)
                    prev = proj

        if SHOW_3D_BOX:
            projected = []
            for px, py, pz in get_3d_box_corners(t):
                proj = project_xyz(px, py, pz, K, cam_x=cam_x, cam_y=cam_y, cam_z=cam_z, cam_yaw=cam_yaw)
                if not proj:
                    projected = []
                    break
                projected.append(proj)
            if len(projected) == 8:
                for i in range(4):
                    cv2.line(vis, projected[i], projected[(i+1)%4], (255,128,0), 1)
                    cv2.line(vis, projected[i+4], projected[4 + (i+1)%4], (255,128,0), 1)
                    cv2.line(vis, projected[i], projected[i+4], (255,128,0), 1)

    return vis


def draw_bev(tracks, clusters, world_model, decision):
    scale = 8.0
    size = 700
    bev = np.zeros((size, size, 3), dtype=np.uint8)
    cx = size//2
    cy = size - 80

    for meters in range(-30, 31, 5):
        px = int(cx + meters * scale)
        if 0 <= px < size:
            cv2.line(bev, (px,0), (px,size), (35,35,35), 1)
    for meters in range(-10, 75, 5):
        py = int(cy - meters * scale)
        if 0 <= py < size:
            cv2.line(bev, (0,py), (size,py), (35,35,35), 1)

    lane_left = int(cx - LANE_HALF_WIDTH_M * scale)
    lane_right = int(cx + LANE_HALF_WIDTH_M * scale)
    cv2.line(bev, (lane_left,0), (lane_left,size), (60,60,100), 1)
    cv2.line(bev, (lane_right,0), (lane_right,size), (60,60,100), 1)

    valid_ids = {entry["id"] for entry in world_model["tracks"]}
    visible_positions = [
        (entry["x"], entry["y"])
        for entry in world_model["tracks"]
    ]

    for c in clusters:
        x,y = c["centroid"][0], c["centroid"][1]
        if visible_positions and not any(speed_xy(x - tx, y - ty) <= BEV_CLUSTER_ASSOC_DIST_M for tx, ty in visible_positions):
            continue
        px = int(cx + y * scale)
        py = int(cy - x * scale)
        if 0 <= px < size and 0 <= py < size:
            cv2.circle(bev, (px,py), 3, (0,0,255), -1)

    for t in tracks.values():
        if not t.confirmed or t.id not in valid_ids:
            continue
        x,y = t.pos()
        px = int(cx + y * scale)
        py = int(cy - x * scale)
        color = risk_color(t.risk)
        if 0 <= px < size and 0 <= py < size:
            cv2.circle(bev, (px,py), 7, color, -1)
            cv2.putText(bev, f"T{t.id}", (px+8, py-6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)
            if SHOW_PREDICTION_TRAIL and t.future:
                prev = (px, py)
                for _, fx, fy in t.future:
                    fpx = int(cx + fy * scale)
                    fpy = int(cy - fx * scale)
                    if 0 <= fpx < size and 0 <= fpy < size:
                        cv2.circle(bev, (fpx, fpy), 3, color, -1)
                        cv2.line(bev, prev, (fpx, fpy), color, 1)
                        prev = (fpx, fpy)

    ex1 = int(cx - EGO_HALF_WIDTH_M * scale)
    ex2 = int(cx + EGO_HALF_WIDTH_M * scale)
    ey1 = int(cy - EGO_HALF_LENGTH_M * scale)
    ey2 = int(cy + EGO_HALF_LENGTH_M * scale)
    cv2.rectangle(bev, (ex1, ey1), (ex2, ey2), (255,255,255), 2)
    cv2.putText(bev, "EGO", (ex2+8, ey2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    header = [
        f"mode={decision['mode']}",
        f"target={decision['target_speed']:.1f}m/s",
        f"brake={decision['brake']:.2f}",
        f"risk={decision['top_risk']:.2f}",
    ]
    cv2.putText(bev, " | ".join(header), (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)
    cv2.putText(bev, f"reason={decision['reason']}", (20, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (210,210,210), 1)

    merge_hazard = world_model.get("merge_hazard")
    if merge_hazard is not None:
        mx = int(cx + merge_hazard["y"] * scale)
        my = int(cy - merge_hazard["x"] * scale)
        if 0 <= mx < size and 0 <= my < size:
            cv2.circle(bev, (mx, my), 10, (0, 255, 255), 2)
            cv2.putText(
                bev,
                f"{merge_hazard['side']} merge {merge_hazard['ttc']:.1f}s",
                (mx + 10, max(20, my - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 255),
                1,
            )

    for idx, entry in enumerate(world_model["tracks"][:5]):
        text = f"T{entry['id']} x={entry['x']:.1f} y={entry['y']:.1f} v={entry['speed']:.1f} ttc={entry['ttc']:.1f} risk={entry['risk']:.2f}"
        cv2.putText(bev, text, (20, 84 + idx * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (190,190,190), 1)
    return bev


def inference_worker(selected=None):
    global inference_running, frame_idx
    last_idx = -1
    surround_keys = [f"cam{i}" for i in range(8)]
    while inference_running:
        with frame_lock:
            idx = frame_idx
            frames = dict(cams_frames)
        if idx != last_idx and idx % DET_EVERY_N == 0:
            if selected is not None:
                keys = selected
            else:
                keys = (["front"] if "front" in frames else []) + surround_keys[:min(SURROUND_CAMS_PER_CYCLE, len(surround_keys))]
            for k in keys:
                img = frames.get(k)
                if img is None:
                    continue
                try:
                    dets, tl_dets = run_yolo_resized(img, scale=DET_SCALE)
                except Exception:
                    dets, tl_dets = [], []
                cams_dets[k] = dets
                if k == "front":
                    cams_tl_dets[k] = tl_dets
            last_idx = idx
        time.sleep(0.01)

def configure_camera_bp(cam_bp):
    cam_bp.set_attribute("image_size_x", str(CAMERA_W))
    cam_bp.set_attribute("image_size_y", str(CAMERA_H))
    cam_bp.set_attribute("fov", str(CAMERA_FOV))
    cam_bp.set_attribute("sensor_tick", str(STEP))
    if cam_bp.has_attribute("motion_blur_intensity"):
        cam_bp.set_attribute("motion_blur_intensity", "0.0")

def configure_lidar_bp(lidar_bp):
    lidar_bp.set_attribute("channels", "64")
    lidar_bp.set_attribute("range", str(LIDAR_RANGE_M))
    lidar_bp.set_attribute("points_per_second", "90000")
    lidar_bp.set_attribute("rotation_frequency", str(int(round(1.0 / STEP))))
    lidar_bp.set_attribute("upper_fov", "10.0")
    lidar_bp.set_attribute("lower_fov", "-25.0")
    lidar_bp.set_attribute("sensor_tick", str(STEP))

def configure_obstacle_bp(obstacle_bp):
    obstacle_bp.set_attribute("distance", str(OBSTACLE_SENSOR_DISTANCE_M))
    obstacle_bp.set_attribute("hit_radius", str(OBSTACLE_SENSOR_RADIUS_M))
    obstacle_bp.set_attribute("only_dynamics", "false")
    obstacle_bp.set_attribute("sensor_tick", str(STEP))

def configure_gnss_bp(gnss_bp):
    if gnss_bp.has_attribute("sensor_tick"):
        gnss_bp.set_attribute("sensor_tick", str(STEP))
    deg_noise = GNSS_STD_M / 111139.0
    for key in ["noise_lat_stddev", "noise_lon_stddev"]:
        if gnss_bp.has_attribute(key):
            gnss_bp.set_attribute(key, f"{deg_noise:.8f}")
    if gnss_bp.has_attribute("noise_alt_stddev"):
        gnss_bp.set_attribute("noise_alt_stddev", "1.0")

def configure_imu_bp(imu_bp):
    if imu_bp.has_attribute("sensor_tick"):
        imu_bp.set_attribute("sensor_tick", str(STEP))
    for key in ["noise_accel_stddev_x", "noise_accel_stddev_y", "noise_accel_stddev_z"]:
        if imu_bp.has_attribute(key):
            imu_bp.set_attribute(key, "0.12")
    for key in ["noise_gyro_stddev_x", "noise_gyro_stddev_y", "noise_gyro_stddev_z"]:
        if imu_bp.has_attribute(key):
            imu_bp.set_attribute(key, "0.01")

def configure_radar_bp(radar_bp):
    radar_bp.set_attribute("horizontal_fov", str(RADAR_HFOV_DEG))
    radar_bp.set_attribute("vertical_fov", str(RADAR_VFOV_DEG))
    radar_bp.set_attribute("range", str(RADAR_RANGE_M))
    radar_bp.set_attribute("points_per_second", str(RADAR_POINTS_PER_SECOND))
    radar_bp.set_attribute("sensor_tick", str(STEP))

def build_radar_mounts():
    return {
        "radar_front_left": carla.Transform(
            carla.Location(x=RADAR_X_FRONT, y=RADAR_Y_LEFT, z=RADAR_Z),
            carla.Rotation(yaw=-35.0),
        ),
        "radar_front_right": carla.Transform(
            carla.Location(x=RADAR_X_FRONT, y=RADAR_Y_RIGHT, z=RADAR_Z),
            carla.Rotation(yaw=35.0),
        ),
        "radar_rear_left": carla.Transform(
            carla.Location(x=RADAR_X_REAR, y=RADAR_Y_LEFT, z=RADAR_Z),
            carla.Rotation(yaw=-145.0),
        ),
        "radar_rear_right": carla.Transform(
            carla.Location(x=RADAR_X_REAR, y=RADAR_Y_RIGHT, z=RADAR_Z),
            carla.Rotation(yaw=145.0),
        ),
    }

def cleanup_phase9_actors(world):
    stale = []
    for actor in world.get_actors():
        try:
            role_name = actor.attributes.get("role_name", "")
        except Exception:
            role_name = ""
        if role_name.startswith("phase9_"):
            stale.append(actor)
    for actor in stale:
        try:
            actor.destroy()
        except Exception:
            pass

def spawn_actor_from_points(world, blueprint, spawn_points):
    for transform in spawn_points:
        actor = world.try_spawn_actor(blueprint, transform)
        if actor is not None:
            return actor, transform
    return None, None

def connect_to_carla(host="localhost", port=2000, timeout_s=20.0, attempts=3):
    last_exc = None
    for _ in range(max(1, attempts)):
        try:
            client = carla.Client(host, port)
            client.set_timeout(timeout_s)
            world = client.get_world()
            return client, world
        except RuntimeError as exc:
            last_exc = exc
            time.sleep(2.0)
    raise last_exc

# ======================
# MAIN
# ======================
def main():
    global obstacle_events, traffic_light_memory, radar_returns, ego_state_filter, gnss_data, imu_data, cams_tl_dets, tracks, next_id, camera_seed_memory, radar_seed_memory, eval_state, control_state
    random.seed(7)
    np.random.seed(7)
    obstacle_events = {}
    radar_returns = {}
    traffic_light_memory = {"state_name": "None", "score": 0.0, "bbox": None, "ts": 0.0}
    gnss_data = None
    imu_data = None
    cams_tl_dets = {}
    tracks = {}
    next_id = 0
    camera_seed_memory = {}
    radar_seed_memory = {}
    eval_state = init_eval_state()
    control_state = {}
    lane_keep_state = {"use_tm": USE_TM_LANE_KEEP, "stall_since": None}
    video_writer = None
    active_video_path = None
    active_video_codec = None
    run_started_ts = time.time()
    client, world = connect_to_carla("localhost", 2000, timeout_s=20.0, attempts=3)
    world_map = world.get_map()
    traffic_manager = client.get_trafficmanager(8000)
    traffic_manager.set_synchronous_mode(True)
    if hasattr(traffic_manager, "set_random_device_seed"):
        traffic_manager.set_random_device_seed(7)
    if hasattr(traffic_manager, "set_global_distance_to_leading_vehicle"):
        traffic_manager.set_global_distance_to_leading_vehicle(2.5)
    else:
        traffic_manager.global_distance_to_leading_vehicle(2.5)

    original_settings = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = STEP
    world.apply_settings(settings)
    cleanup_phase9_actors(world)
    world.tick()

    bp = world.get_blueprint_library()
    sp = world_map.get_spawn_points()

    ego_bp = bp.filter("model3")[0]
    if ego_bp.has_attribute("role_name"):
        ego_bp.set_attribute("role_name", "phase9_ego")
    ego, ego_spawn_tf = spawn_actor_from_points(world, ego_bp, sp)
    if ego is None:
        raise RuntimeError("Unable to spawn phase9 ego vehicle at any spawn point")
    ego.set_autopilot(False)
    ego.apply_control(carla.VehicleControl(brake=1.0))
    route_agent = None
    route_goal = None
    if USE_BUILTIN_ROUTE_DRIVER:
        try:
            route_agent, route_goal = make_route_driver(ego, world_map, sp)
        except Exception as exc:
            route_agent = None
            route_goal = None
            print(f"[ROUTE] route_agent_init_failed: {exc}")
    if USE_TM_LANE_KEEP:
        ego.set_autopilot(True, traffic_manager.get_port())
        traffic_manager.auto_lane_change(ego, False)
        traffic_manager.ignore_vehicles_percentage(ego, 100.0)
        traffic_manager.ignore_walkers_percentage(ego, 100.0)
        traffic_manager.ignore_lights_percentage(ego, 100.0)
        traffic_manager.ignore_signs_percentage(ego, 100.0)
        traffic_manager.distance_to_leading_vehicle(ego, 0.0)
        traffic_manager.set_desired_speed(ego, 0.0)

    actors = [ego]

    npc_blueprints = bp.filter("vehicle.*")
    for s in sp[1:NPC_COUNT]:
        npc_bp = random.choice(npc_blueprints)
        if npc_bp.has_attribute("role_name"):
            npc_bp.set_attribute("role_name", "phase9_npc")
        npc = world.try_spawn_actor(npc_bp, s)
        if npc:
            npc.set_autopilot(True, traffic_manager.get_port())
            actors.append(npc)

    # FRONT CAM
    front_mount = carla.Transform(carla.Location(x=CAMERA_POS_X, y=CAMERA_POS_Y, z=CAMERA_POS_Z), carla.Rotation(yaw=0.0))
    cam_bp = bp.find("sensor.camera.rgb")
    configure_camera_bp(cam_bp)
    cam = world.spawn_actor(cam_bp, front_mount, attach_to=ego)
    cam.listen(rgb_cb)
    actors.append(cam)
    cams_transforms['front'] = front_mount

    # 8 CAMS
    surround_mounts = build_surround_camera_mounts()
    for cam_name, mount in surround_mounts.items():
        c = world.spawn_actor(cam_bp, mount, attach_to=ego)
        c.listen(make_cam_cb(cam_name))
        cams_transforms[cam_name] = mount
        actors.append(c)

    lidar_bp = bp.find("sensor.lidar.ray_cast")
    configure_lidar_bp(lidar_bp)
    lidar = world.spawn_actor(lidar_bp,
        carla.Transform(carla.Location(z=2.5)), attach_to=ego)
    lidar.listen(lidar_cb)
    actors.append(lidar)

    gnss_bp = bp.find("sensor.other.gnss")
    configure_gnss_bp(gnss_bp)
    gnss = world.spawn_actor(gnss_bp, carla.Transform(carla.Location(z=2.0)), attach_to=ego)
    gnss.listen(gnss_cb)
    actors.append(gnss)

    imu_bp = bp.find("sensor.other.imu")
    configure_imu_bp(imu_bp)
    imu = world.spawn_actor(imu_bp, carla.Transform(carla.Location(z=2.0)), attach_to=ego)
    imu.listen(imu_cb)
    actors.append(imu)

    radar_bp = bp.find("sensor.other.radar")
    configure_radar_bp(radar_bp)
    for name, mount in build_radar_mounts().items():
        radar = world.spawn_actor(radar_bp, mount, attach_to=ego)
        radar.listen(make_radar_cb(name, mount))
        actors.append(radar)

    obstacle_bp = bp.find("sensor.other.obstacle")
    configure_obstacle_bp(obstacle_bp)
    obstacle_sensor = world.spawn_actor(
        obstacle_bp,
        carla.Transform(carla.Location(x=2.2, z=1.1)),
        attach_to=ego
    )
    obstacle_sensor.listen(make_obstacle_cb("front"))
    actors.append(obstacle_sensor)

    while img_rgb is None or lidar_data is None or len(cams)<8 or gnss_data is None or imu_data is None or len(radar_returns) < 4:
        world.tick()

    # compute intrinsic K from camera resolution
    first_cam = sorted(cams.keys())[0]
    h,w,_ = cams[first_cam].shape
    K = get_K(w,h)

    # start background inference thread
    global inference_running, frame_idx
    inference_running = True
    worker_thread = threading.Thread(target=inference_worker, args=(None,), daemon=True)
    worker_thread.start()
    ego_state_filter = EgoStateEstimator(ego_spawn_tf, world_map)

    try:
        while True:
            world.tick()

            if img_rgb is None:
                continue
            try:
                rgb = img_rgb.copy()
            except:
                continue

            # increment frame index for worker and use latest detections
            with frame_lock:
                frame_idx += 1
            yolo = cams_dets.get('front', [])

            clusters = lidar_clusters(lidar_data)
            radar_points = get_radar_points()

            for t in tracks.values():
                t.predict(STEP)

            # Aggregate all detections for association (lidar master, camera confirmation)
            associate(cams_dets, clusters, cams_transforms, K)
            fuse_radar_tracks(radar_points)
            estimated_ego_state = ego_state_filter.step(gnss_data, imu_data)
            if estimated_ego_state is None:
                continue
            ego_state = ego_state_from_actor(ego)
            if route_agent is not None:
                route_goal = maybe_refresh_route_driver(route_agent, ego, sp, route_goal)
            world_model = build_world_model(ego_state, world_map, radar_points)
            world_model["ego_estimate"] = estimated_ego_state
            world_model["ego_speed_for_control"] = blended_control_speed(
                world_model["ego_speed"],
                estimated_ego_state.get("speed", world_model["ego_speed"]),
            )
            try:
                world_model["ego_speed_limit_mps"] = max(0.0, float(ego.get_speed_limit()) / 3.6)
            except Exception:
                world_model["ego_speed_limit_mps"] = CRUISE_SPEED_MPS
            world_model["route_speed_cap_mps"] = route_speed_cap_mps(ego_state, world_map)
            world_model["eval"] = evaluate_tracking_against_gt(ego, world, world_model, frame_idx)
            traffic_light_info = get_traffic_light_info(ego, ego_state, world_map, world)
            if ENABLE_OBSTACLE_BACKUP:
                obstacle_info = fuse_obstacle_sources(get_obstacle_info(), get_lidar_corridor_info(lidar_data))
            else:
                obstacle_info = {
                    "distance": float("inf"),
                    "type_id": None,
                    "active": False,
                    "name": None,
                    "forward": float("inf"),
                    "lateral": float("inf"),
                    "source": None,
                }
            decision = decide_action(world_model, traffic_light_info, obstacle_info)
            decision = smooth_control_command(decision, ego_state["speed"])
            use_tm_lane_keep = lane_keep_state["use_tm"]
            if use_tm_lane_keep:
                tm_should_progress = (
                    decision["mode"] not in {"LIGHT_STOP", "OBSTACLE_STOP", "EMERGENCY_BRAKE"}
                    and decision["target_speed"] >= TM_STALL_TARGET_MPS
                    and decision["brake"] < 0.08
                    and traffic_light_info["state"] not in (carla.TrafficLightState.Red, carla.TrafficLightState.Yellow)
                    and not obstacle_info["active"]
                )
                if tm_should_progress and ego_state["speed"] < 0.6:
                    if lane_keep_state["stall_since"] is None:
                        lane_keep_state["stall_since"] = time.time()
                    elif time.time() - lane_keep_state["stall_since"] > TM_STALL_HOLD_S:
                        lane_keep_state["use_tm"] = False
                        use_tm_lane_keep = False
                        lane_keep_state["stall_since"] = None
                        try:
                            ego.set_autopilot(False)
                        except Exception:
                            pass
                        print(f"[CTRL] frame={frame_idx} fallback=manual_lane_keep reason=tm_stall ego={ego_state['speed']:.1f} target={decision['target_speed']:.1f}")
                else:
                    lane_keep_state["stall_since"] = None
            if use_tm_lane_keep:
                # CARLA TrafficManager.set_desired_speed expects km/h, while our policy uses m/s.
                tm_target_speed = 3.6 * max(0.0, float(decision["target_speed"]))
                tl_dist = float(decision.get("traffic_light_distance", float("inf")))
                if decision["mode"] == "LIGHT_STOP" and (tm_target_speed < 10.8 or (np.isfinite(tl_dist) and tl_dist < 18.0)):
                    tm_target_speed = 0.0
                traffic_manager.set_desired_speed(ego, tm_target_speed)
                hud_brake = 0.0
                if decision["mode"] == "EMERGENCY_BRAKE":
                    hud_brake = max(0.8, float(decision["brake"]))
                elif decision["mode"] == "OBSTACLE_STOP" and (decision["brake"] > 0.18 or obstacle_info["distance"] < 4.5):
                    hud_brake = float(decision["brake"])
                elif decision["mode"] == "LIGHT_STOP" and (
                    decision["brake"] > 0.16
                    or (np.isfinite(tl_dist) and tl_dist < 16.0)
                    or tm_target_speed <= 0.5
                ):
                    hud_brake = max(hud_brake, float(decision["brake"]))
                elif decision["mode"] in {"FOLLOW", "CAUTION", "YIELD"} and decision["brake"] > 0.28:
                    hud_brake = max(hud_brake, float(decision["brake"]))
                if hud_brake > 0.05:
                    ego.apply_control(carla.VehicleControl(throttle=0.0, brake=hud_brake, steer=0.0))
                speed_ref = max(1.0, float(world_model.get("ego_speed_limit_mps", CRUISE_SPEED_MPS)))
                hud_throttle = 0.0 if hud_brake > 0.05 else clamp(tm_target_speed / speed_ref, 0.0, 0.75)
                control = carla.VehicleControl(throttle=hud_throttle, brake=hud_brake, steer=0.0)
                decision["lane_error_m"] = 0.0
                decision["heading_error_deg"] = 0.0
                decision["speed_limit"] = speed_ref
            else:
                control = compute_vehicle_control(
                    world_map,
                    decision,
                    ego_state,
                    world_model=world_model,
                    route_agent=route_agent,
                )
                ego.apply_control(control)


            # DRAW FRONT (use front camera extrinsics)
            front_tf = cams_transforms.get('front')
            if front_tf is not None:
                rgb = draw_on(
                    rgb, yolo, K,
                    cam_x=front_tf.location.x,
                    cam_y=front_tf.location.y,
                    cam_yaw=front_tf.rotation.yaw,
                    cam_z=front_tf.location.z,
                    world_model=world_model
                )
            else:
                rgb = draw_on(rgb, yolo, K, world_model=world_model)
            rgb = draw_hud(rgb, world_model, decision, control)

            # DRAW MOSAIC (per-camera detections)
            if frame_idx % 30 == 0:
                cam_status = {k: (None if cams.get(k) is None else cams[k].shape) for k in sorted(cams.keys())}
                print(f"[DBG] frame {frame_idx} cam_status: {cam_status}")
            imgs = []
            for k in sorted(cams.keys()):
                if cams[k] is not None:
                    dets_k = cams_dets.get(k, [])
                    imgs.append(draw_on(
                        cams[k], dets_k, K,
                        cam_x=cams_transforms[k].location.x,
                        cam_y=cams_transforms[k].location.y,
                        cam_yaw=cams_transforms[k].rotation.yaw,
                        cam_z=cams_transforms[k].location.z,
                        world_model=world_model
                    ))
                else:
                    imgs.append(None)

            valid_imgs = [img for img in imgs[:8] if img is not None]
            if valid_imgs:
                h,w,_ = valid_imgs[0].shape
                mosaic = np.zeros((h*2,w*4,3),dtype=np.uint8)

                for i,img in enumerate(imgs[:8]):
                    if img is not None:
                        r=i//4; c=i%4
                        mosaic[r*h:(r+1)*h, c*w:(c+1)*w] = img
            else:
                mosaic = np.zeros((600,800,3),dtype=np.uint8)  # fallback

            if rgb is not None and hasattr(rgb, 'shape') and rgb.shape[0] > 0 and rgb.shape[1] > 0:
                cv2.imshow("RGB Fusion", rgb)
            if mosaic is not None and hasattr(mosaic, 'shape') and mosaic.shape[0] > 0 and mosaic.shape[1] > 0:
                cv2.imshow("MOSAIC", mosaic)
            # BEV debug overlay
            bev = draw_bev(tracks, clusters, world_model, decision)
            if bev is not None and hasattr(bev, 'shape') and bev.shape[0] > 0 and bev.shape[1] > 0:
                cv2.imshow("BEV / Prediction", bev)

            if SAVE_TEST_VIDEO:
                record_frame = compose_recording_frame(rgb, mosaic, bev)
                if video_writer is None:
                    video_writer, active_video_path, active_video_codec = open_test_video_writer(record_frame)
                if video_writer is not None and video_writer.isOpened():
                    video_writer.write(record_frame)

            if frame_idx % 20 == 0:
                lead = world_model["lead"]
                lead_txt = "none" if lead is None else f"T{lead['id']} x={lead['x']:.1f} risk={lead['risk']:.2f}"
                merge_txt = "clear" if not np.isfinite(decision.get("merge_ttc", float("inf"))) else f"{decision.get('merge_side')}:{decision['merge_ttc']:.1f}s"
                ev = world_model.get("eval", {})
                lane_mode = "TM" if lane_keep_state["use_tm"] else "MAN"
                print(f"[AV] frame={frame_idx} mode={decision['mode']} lane={lane_mode} ego={ego_state['speed']:.1f} target={decision['target_speed']:.1f}/{decision.get('raw_target_speed', decision['target_speed']):.1f} tl={decision['traffic_light_state']} obs={decision['obstacle_distance']:.1f} merge={merge_txt} throttle={control.throttle:.2f} brake={control.brake:.2f} steer={control.steer:.2f} lead={lead_txt} eval={ev.get('matched',0)}/{ev.get('gt_count',0)} miss={ev.get('misses',0)} false={ev.get('false_tracks',0)} pos={ev.get('running_pos_err',float('nan')):.2f} idsw={ev.get('id_switches',0)}")
            if frame_idx % 60 == 0:
                published = [
                    (
                        entry["id"],
                        round(entry["x"], 1),
                        round(entry["y"], 1),
                        round(entry.get("recent_support_score", 0.0), 2),
                    )
                    for entry in world_model["tracks"][:12]
                ]
                print(f"[PUB] frame={frame_idx} tracks={published}")

            if MAX_RUN_SECONDS > 0.0 and (time.time() - run_started_ts) >= MAX_RUN_SECONDS:
                print(f"[RUN] reached timed stop at {MAX_RUN_SECONDS:.1f}s, shutting down cleanly")
                break

            if cv2.waitKey(1)==ord('q'):
                break

    finally:
        # stop inference thread
        inference_running = False
        try:
            worker_thread.join(timeout=1.0)
        except Exception:
            pass
        try:
            if eval_state.get("csv_file") is not None:
                eval_state["csv_file"].close()
        except Exception:
            pass
        try:
            if video_writer is not None:
                video_writer.release()
        except Exception:
            pass
        if active_video_path:
            print(f"[VIDEO] saved {active_video_path} codec={active_video_codec}")
        for a in actors:
            try:
                a.destroy()
            except Exception:
                pass
        world.apply_settings(original_settings)
        traffic_manager.set_synchronous_mode(False)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
