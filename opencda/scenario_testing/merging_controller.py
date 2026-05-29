# -*- coding: utf-8 -*-
"""
DAROM merging controller adapter for CARLA-SUMO co-simulation.

Loads the pretrained DAROM-GRU SAC model and provides an interface to:
  1. Build observations from the live traci state (same format as the
     onRampMerging Gymnasium env).
  2. Run inference to get [acceleration, lane-change] actions.
  3. Apply those actions to the ego vehicle via traci.

The controller is designed to be called once per simulation tick from the
co-simulation main loop.  It does NOT create its own traci connection — it
reuses whichever connection the CoScenarioManager already started.
"""
# License: MIT

import sys
import os
import numpy as np
from collections import deque

# ---------------------------------------------------------------------------
# numpy 2.x -> 1.x compatibility: models saved with numpy 2.x reference
# numpy._core.* submodules that don't exist in numpy <2.  Patch them in.
# ---------------------------------------------------------------------------
if not hasattr(np, '_core'):
    np._core = np.core  # type: ignore[attr-defined]
import importlib
for _sub in ('numeric', 'multiarray', '_multiarray_umath',
             'fromnumeric', '_methods', 'function_base'):
    _full = f'numpy._core.{_sub}'
    if _full not in sys.modules:
        try:
            sys.modules[_full] = importlib.import_module(f'numpy.core.{_sub}')
        except ModuleNotFoundError:
            pass

try:
    import numpy.random._pickle as _rp
    _orig_bg_ctor = _rp.__bit_generator_ctor
    def _patched_bg_ctor(bit_gen_name, *args, **kwargs):
        if isinstance(bit_gen_name, type):
            bit_gen_name = bit_gen_name.__name__
        return _orig_bg_ctor(bit_gen_name, *args, **kwargs)
    _rp.__bit_generator_ctor = _patched_bg_ctor
except Exception:
    pass

# ---------------------------------------------------------------------------
# Ensure the onRampMerging source is importable
# ---------------------------------------------------------------------------
_ONRAMP_ROOT = os.environ.get(
    'ONRAMP_MERGING_ROOT',
    os.path.join(os.path.dirname(__file__), '..', '..', 'onRampMerging'))
if _ONRAMP_ROOT not in sys.path:
    sys.path.insert(0, _ONRAMP_ROOT)

from stable_baselines3 import SAC
from src.safety_controller import safetyCheck
from src.merging import parseAction
import traci


# ── Helpers (adapted from onRampMerging/src/utils.py) ─────────────────────

def _get_distance(pos1, pos2):
    return np.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)


def _flatten(xss):
    """Recursive flatten (mirrors onRampMerging/src/utils.py)."""
    flat = []
    for xs in xss:
        try:
            iter(xs)
            if isinstance(xs, str):
                raise TypeError
            for x in xs:
                try:
                    iter(x)
                    if isinstance(x, str):
                        raise TypeError
                    for y in x:
                        flat.append(round(float(y), 2))
                except TypeError:
                    flat.append(round(float(x), 2))
        except TypeError:
            flat.append(round(float(xs), 2))
    return flat


def _extract_lane_name(lane_str):
    if lane_str.startswith(':'):
        parts = lane_str.split('_')
        return parts[0][1:] if parts else lane_str
    parts = lane_str.split('_')
    return parts[0] if parts else lane_str


# ── Lane-index helpers for the PHMD5 network ─────────────────────────────

def _get_absolute_lane_idx(veh_id):
    """Return the lane index for *veh_id* (0 = rightmost)."""
    try:
        return traci.vehicle.getLaneIndex(veh_id)
    except Exception:
        return -1


# ── Observation builder ───────────────────────────────────────────────────

def _build_observation(ego_id: str, radius: float, num_entities: int = 31):
    """
    Build a (num_entities, 3) observation array identical in structure to
    ``Merging._get_obs()`` in the onRampMerging project.

    Row 0 = ego [x, y, speed].
    Rows 1..N = neighbours sorted by (y, x), expressed relative to ego.
    Remaining rows padded with zeros.
    """
    ego_pos = traci.vehicle.getPosition(ego_id)
    ego_vel = traci.vehicle.getSpeed(ego_id)
    ego = [[ego_pos[0], ego_pos[1], ego_vel]]

    intruders = []
    for veh_id in traci.vehicle.getIDList():
        if veh_id == ego_id:
            continue
        veh_pos = traci.vehicle.getPosition(veh_id)
        dist = _get_distance(veh_pos, ego_pos)
        if dist <= radius:
            veh_vel = traci.vehicle.getSpeed(veh_id)
            intruders.append([
                veh_pos[0] - ego_pos[0],
                veh_pos[1] - ego_pos[1],
                veh_vel - ego_vel,
            ])

    intruders = intruders[:num_entities - 1]
    intruders.sort(key=lambda x: (x[1], x[0]))
    while len(intruders) < num_entities - 1:
        intruders.append([0.0, 0.0, 0.0])

    return np.array(ego + intruders, dtype=np.float32)


# ── Delay wrapper logic (stand-alone, no Gymnasium dependency) ────────────

class DelayState:
    """
    Mirrors the state-augmentation logic of ``DelayWrapper`` from
    ``onRampMerging/src/wrapper.py`` without requiring a Gymnasium env.
    """

    def __init__(self, obs_shape, act_dim, max_delay=20,
                 mode='all', delay_mode='uniform'):
        self.obs_shape = obs_shape          
        self.act_dim = act_dim              
        self.max_delay = max_delay
        self.mode = mode
        self.delay_mode = delay_mode
        self.congested = False              

        self.observation_history = None
        self.action_history = None
        self.last_observation = None
        self.delay_of_last_observation = 0
        self.reset()

    # ── public API ────────────────────────────────────────────────────

    def reset(self, initial_obs=None):
        n_ent, feat = self.obs_shape
        self.observation_history = deque([
            {'observation': np.zeros((n_ent, feat), dtype=np.float32),
             'delay': 0}
            for _ in range(self.max_delay + 1)
        ], maxlen=self.max_delay + 1)
        self.action_history = deque([
            np.zeros(self.act_dim, dtype=np.float32)
            for _ in range(self.max_delay)
        ], maxlen=self.max_delay)
        self.congested = False

        if initial_obs is not None:
            self.last_observation = initial_obs.copy()
        else:
            self.last_observation = np.zeros((n_ent, feat), dtype=np.float32)
        self.delay_of_last_observation = 0
        return self._pack(self.last_observation, 0)

    def augment(self, raw_obs, last_action):
        """
        Given the *true* current observation and the action that was just
        applied, return the delay-augmented observation vector the SAC model
        expects.
        """
        delay = self._sample_delay()

        # shift action history
        self.action_history.pop()
        self.action_history.appendleft(
            np.array(last_action, dtype=np.float32))

        # store current obs at the correct delay slot
        self.observation_history[-delay - 1] = {
            'observation': raw_obs.copy(), 'delay': delay}

        # pop the oldest entry (what the agent "receives")
        received = self.observation_history.pop()
        received_obs = received['observation']
        delay_of_received = received['delay']
        self.observation_history.appendleft({
            'observation': np.zeros(self.obs_shape, dtype=np.float32),
            'delay': 0})

        # fall-back: use last observation if nothing useful arrived
        no_data = not received_obs.any()
        stale = (self.delay_of_last_observation + 1 < delay_of_received)
        if no_data or stale:
            received_obs = self.last_observation
            delay_of_received = self.delay_of_last_observation + 1

        self.last_observation = received_obs
        self.delay_of_last_observation = delay_of_received

        # ego row is never delayed
        received_obs[0, :] = raw_obs[0, :]

        return self._pack(received_obs, delay_of_received)

    # ── private helpers ───────────────────────────────────────────────

    def _pack(self, obs, delay):
        masked = np.array(list(self.action_history)).copy()
        for i in range(delay, len(masked)):
            masked[i] = np.zeros_like(masked[i])

        if self.mode == 'only_delayed_state':
            return obs
        elif self.mode == 'delayed_state_and_action':
            return np.array(_flatten([obs, masked]), dtype=np.float32)
        elif self.mode == 'delayed_state_and_delay':
            return np.array(_flatten([obs, [delay]]), dtype=np.float32)
        elif self.mode == 'all':
            return np.array(
                _flatten([obs, masked, [delay]]), dtype=np.float32)
        raise ValueError(f'Unknown mode: {self.mode}')

    def _sample_delay(self):
        md = self.max_delay
        if self.delay_mode == 'uniform':
            return np.random.randint(0, md + 1)
        elif self.delay_mode == 'exponential':
            return int(np.clip(round(np.random.exponential(md / 3.0)), 0, md))
        elif self.delay_mode == 'triangular':
            return int(round(np.random.triangular(0, md / 2.0, md)))
        elif self.delay_mode == 'bursty':
            if self.congested:
                self.congested = np.random.random() < 0.9
                return np.random.randint(md // 2, md + 1)
            else:
                self.congested = np.random.random() < 0.05
                return np.random.randint(0, md // 4 + 1)
        elif self.delay_mode == 'bimodal':
            if np.random.random() < 0.6:
                return np.random.randint(0, max(md // 5, 1) + 1)
            else:
                return np.random.randint(md * 3 // 5, md + 1)
        raise ValueError(f'Unknown delay_mode: {self.delay_mode}')


# ── Main controller class ─────────────────────────────────────────────────

class MergingController:
    """
    High-level controller that wraps DAROM model loading, observation
    building, delay augmentation, safety checking, and action application.

    Parameters
    ----------
    model_path : str
        Path to the pretrained SAC .zip checkpoint.
    ego_id : str
        The SUMO vehicle ID of the ego (default ``'ego'``).
    radius : float
        Observation radius in metres (default 50).
    max_delay : int
        Maximum communication delay in time-steps (default 20).
    delay_mode : str
        One of ``uniform | exponential | triangular | bursty | bimodal``.
    use_safety : bool
        Whether to run the physics-based safety controller.
    merge_edge : str
        The SUMO edge ID on which a successful merge is detected.
        For the PHMD5 network this is ``'-1'`` (the mainline after the
        ramp junction).  Adjust if your network differs.
    merge_min_lane : int
        Minimum lane index that counts as "merged" on *merge_edge*.
    """

    def __init__(self, model_path, ego_id='ego', radius=50, max_delay=20,
                 delay_mode='uniform', use_safety=True,
                 merge_edge='-1', merge_min_lane=1, log_file='merging_log.txt'):
        self.ego_id = ego_id
        self.radius = radius
        self.use_safety = use_safety
        self.merge_edge = merge_edge
        self.merge_min_lane = merge_min_lane
        self.log_file_path = log_file
        
        # Open log file in write mode (overrides existing file)
        self.log_file = open(self.log_file_path, 'w')
        self.log_file.write(f"Merging Controller Log\n")
        self.log_file.write(f"Model: {model_path}\n")
        self.log_file.write(f"Ego ID: {ego_id}, Radius: {radius}m, Max Delay: {max_delay}, Mode: {delay_mode}\n")
        self.log_file.write("="*80 + "\n\n")

        # Observation / delay state
        self.num_entities = 31
        self.feature_dim = 3
        obs_shape = (self.num_entities, self.feature_dim)
        act_dim = 2
        self.delay_state = DelayState(obs_shape, act_dim,
                                      max_delay=max_delay,
                                      mode='all',
                                      delay_mode=delay_mode)

        # Load pretrained SAC model 
        from gymnasium import spaces as gym_spaces
        obs_flat_dim = (self.num_entities * self.feature_dim
                        + max_delay * act_dim + 1)  # 'all' mode
        custom_objects = {
            'observation_space': gym_spaces.Box(
                low=-1e3, high=1e3,
                shape=(obs_flat_dim,), dtype=np.float32),
            'action_space': gym_spaces.Box(
                low=np.array([-5.0, -5.0], dtype=np.float32),
                high=np.array([5.0, 5.0], dtype=np.float32),
                dtype=np.float32),
        }
        self.model = SAC.load(model_path, custom_objects=custom_objects)
        print(f'[MergingController] loaded model from {model_path}')
        print(f'  ego_id={ego_id}  radius={radius}  '
              f'max_delay={max_delay}  delay_mode={delay_mode}  '
              f'safety={use_safety}')

        # Runtime state
        self._merged = False
        self._last_action = np.zeros(act_dim, dtype=np.float32)
        self._obs_augmented = None
        self._active = False     # becomes True once ego has departed

    # ── public API ────────────────────────────────────────────────────

    @property
    def is_active(self):
        return self._active

    @property
    def merged(self):
        return self._merged

    def reset(self):
        """Call at the start of each episode / scenario."""
        self._merged = False
        self._active = False
        self._last_action = np.zeros(2, dtype=np.float32)
        self._obs_augmented = None
        
        # Logging state
        self._log_enabled = True
        self._tick_count = 0
        self._last_lane = 0
        self._lane_change_cooldown = 0  # Cooldown in ticks before next lane change allowed
        self._merged_to_lane1 = False  # Once merged to lane 1, stay there
        
        # Reset log file
        if hasattr(self, 'log_file') and self.log_file:
            self.log_file.write("\n" + "="*80 + "\n")
            self.log_file.write("RESET - New Episode\n")
            self.log_file.write("="*80 + "\n\n")
            self.log_file.flush()
        
        self.delay_state.reset()

    def tick(self):
        """
        Run one control step.  Call this once per simulation tick *after*
        ``scenario_manager.tick()`` so that SUMO state is fresh.

        Returns
        -------
        info : dict
            ``{'active': bool, 'merged': bool, 'action': list|None,
               'parsed_action': list|None}``
        """
        info = {'active': False, 'merged': self._merged,
                'action': None, 'parsed_action': None}

        # Check whether ego exists in the simulation
        if self.ego_id not in traci.vehicle.getIDList():
            if self._active:
                self._active = False
            return info

        if not self._active:
            # First tick after ego appears — initialise
            self._active = True
            traci.vehicle.setSpeedMode(self.ego_id, 96)
            # Disable SUMO's lane-change model - we control it via TraCI changeLane() commands
            traci.vehicle.setLaneChangeMode(self.ego_id, 0)
            # Initialize lane tracking
            try:
                self._last_lane = traci.vehicle.getLaneIndex(self.ego_id)
            except Exception:
                self._last_lane = 0
            raw_obs = _build_observation(
                self.ego_id, self.radius, self.num_entities)
            self._obs_augmented = self.delay_state.reset(raw_obs)
            info['active'] = True
            return info

        # ── Build observation ──
        raw_obs = _build_observation(
            self.ego_id, self.radius, self.num_entities)
        self._obs_augmented = self.delay_state.augment(
            raw_obs, self._last_action)

        # ── Model inference ──
        action, _ = self.model.predict(self._obs_augmented,
                                       deterministic=True)

        # ── Safety controller ──
        if self.use_safety:
            action = self._safety_check(action)

        parsed = parseAction(action)
        
        # ── Log before applying ──
        if self._log_enabled:
            self._tick_count += 1
            try:
                cur_lane = traci.vehicle.getLaneIndex(self.ego_id)
                edge_id = traci.vehicle.getRoadID(self.ego_id)
                lane_id = traci.vehicle.getLaneID(self.ego_id)
                pos = traci.vehicle.getPosition(self.ego_id)
                speed = traci.vehicle.getSpeed(self.ego_id)
                
                lane_changed = (self._last_lane is not None and 
                               self._last_lane != cur_lane)
                
                if self._tick_count % 10 == 0 or parsed[1] != 0 or lane_changed:
                    log_msg = (f"[Tick {self._tick_count:04d}] ego: "
                              f"edge={edge_id} lane={cur_lane} "
                              f"pos=({pos[0]:.1f},{pos[1]:.1f}) "
                              f"speed={speed:.1f}m/s | "
                              f"action=[acc={parsed[0]:.1f}, lc={parsed[1]:.0f}] | "
                              f"lane_changed={lane_changed}\n")
                    if hasattr(self, 'log_file') and self.log_file:
                        self.log_file.write(log_msg)
                        self.log_file.flush()
                
                self._last_lane = cur_lane
            except Exception as e:
                pass

        # ── Apply action to ego ──
        self._apply_action(parsed)

        self._last_action = np.array(action, dtype=np.float32)

        # ── Merge detection ──
        lane_id = traci.vehicle.getLaneID(self.ego_id)
        edge_name = _extract_lane_name(lane_id)
        lane_idx = _get_absolute_lane_idx(self.ego_id)
        if edge_name == self.merge_edge and lane_idx >= self.merge_min_lane:
            if not self._merged:
                self._merged = True
                merge_msg = '[MergingController] ego has merged!\n'
                print('[MergingController] ego has merged!')
                if hasattr(self, 'log_file') and self.log_file:
                    self.log_file.write(merge_msg)
                    self.log_file.flush()

        info.update({
            'active': True,
            'merged': self._merged,
            'action': action.tolist() if hasattr(action, 'tolist') else action,
            'parsed_action': parsed,
        })
        return info

    def destroy(self):
        """Close the log file when done."""
        if hasattr(self, 'log_file') and self.log_file:
            self.log_file.close()
            print(f"[MergingController] Log file closed: {self.log_file_path}")

    # ── traci-based safety check (network-agnostic) ───────────────────

    def _safety_check(self, action):
        """
        Physics-based safety controller using **traci.vehicle.getLeader**
        and **getFollower** so it works across edge boundaries.
        Inspired by onRampMerging/src/safety_controller.py
        """
        MIN_GAP = 5         # meters - minimum safe gap for high-speed following
        MAX_DECEL = 10.0         # m/s²
        VEH_LENGTH = 5.0        # meters

        parsed = parseAction(action)
        acc, lc = parsed

        ego_speed = traci.vehicle.getSpeed(self.ego_id)
        cur_lane = traci.vehicle.getLaneIndex(self.ego_id)
        edge_id = traci.vehicle.getRoadID(self.ego_id)

        # Once merged to lane 1, block ALL further lane changes (limit to one merge only)
        if self._merged_to_lane1:
            return np.array([action[0], 0.0], dtype=np.float32)

        # On junction internals OR mainline edges -2, -3, -16 → no lane change, keep acceleration
        # EXCEPT allow merging during clusterJ1_cluster111_J1 and clusterJ2_cluster122_J2 junctions (critical merge points)
        # Disable RL lane changes on these edges to prevent side collisions
        is_merge_junction = ':clusterJ1_cluster111_J1' in edge_id or ':clusterJ2_cluster122_J2' in edge_id
        on_mainline = edge_id in ['-2', '-3', '-16']
        if (edge_id.startswith(':') and not is_merge_junction) or on_mainline:
            # Still check leader in current lane, but be less aggressive on mainline after merge
            leader = traci.vehicle.getLeader(self.ego_id, 50.0)
            if leader is not None and leader[0]:
                gap = leader[1]
                l_speed = traci.vehicle.getSpeed(leader[0])
                approach = max(ego_speed - l_speed, 0.0)
                sdist = (approach ** 2) / (2 * MAX_DECEL) if approach > 0 else 0
                # On mainline after merge, use relaxed following to avoid emergency braking
                # (vehicle can't change lanes anyway, so must follow traffic flow)
                safety_gap = MIN_GAP if not (on_mainline and self._merged_to_lane1) else MIN_GAP * 0.3
                if gap < sdist + safety_gap:
                    # Use gentler deceleration on mainline after merge
                    decel = -5.0 if not (on_mainline and self._merged_to_lane1) else -3.0
                    action = np.array([decel, 0.0], dtype=np.float32)
                    return action
            return np.array([action[0], 0.0], dtype=np.float32)

        num_lanes = traci.edge.getLaneNumber(edge_id)
        target_lane = cur_lane + int(lc) if lc != 0 else cur_lane
        target_lane = max(0, min(target_lane, num_lanes - 1))

        def _stopping_dist(approach_speed):
            if approach_speed <= 0:
                return 0.0
            return (approach_speed ** 2) / (2 * MAX_DECEL)

        # ── Check CURRENT lane leader (always, for forward collision) ──
        leader = traci.vehicle.getLeader(self.ego_id, 50.0)
        cur_front_safe = True
        if leader is not None and leader[0]:
            gap = leader[1]
            l_speed = traci.vehicle.getSpeed(leader[0])
            approach = max(ego_speed - l_speed, 0.0)
            
            # Handle negative gaps (already overlapping or past leader) - EMERGENCY!
            if gap < 0 or gap < 5.0:  # Very critical - less than 5m
                cur_front_safe = False
                if self._log_enabled and self._tick_count % 10 == 0:
                    msg = f"  [EMERGENCY] Critical gap: {gap:.1f}m, leader_speed={l_speed:.1f}m/s\n"
                    if hasattr(self, 'log_file') and self.log_file:
                        self.log_file.write(msg)
                        self.log_file.flush()
            elif gap < _stopping_dist(approach) + MIN_GAP:
                cur_front_safe = False

        # ── Check TARGET lane (only if lane-changing AND not already merged) ──
        lc_safe = True
        target_lane_leader_speed = None  # Track target lane leader for pre-merge speed matching
        # Skip target lane checks if already merged to lane 1 - prevents unnecessary braking
        if lc != 0 and target_lane != cur_lane and not self._merged_to_lane1:
            target_lane_id = f'{edge_id}_{target_lane}'
            # Scan all vehicles for target lane neighbours
            ego_lanepos = traci.vehicle.getLanePosition(self.ego_id)
            front_gap, rear_gap = float('inf'), float('inf')
            front_speed, rear_speed = 0.0, 0.0

            for vid in traci.vehicle.getIDList():
                if vid == self.ego_id:
                    continue
                try:
                    v_lane_id = traci.vehicle.getLaneID(vid)
                except Exception:
                    continue
                if v_lane_id != target_lane_id:
                    continue
                v_lanepos = traci.vehicle.getLanePosition(vid)
                v_speed = traci.vehicle.getSpeed(vid)
                longitudinal = v_lanepos - ego_lanepos

                if longitudinal > 0:
                    gap = longitudinal - VEH_LENGTH
                    if gap < front_gap:
                        front_gap = gap
                        front_speed = v_speed
                        target_lane_leader_speed = v_speed  # Store for pre-merge speed matching
                else:
                    gap = abs(longitudinal) - VEH_LENGTH
                    if gap < rear_gap:
                        rear_gap = gap
                        rear_speed = v_speed

            # Calculate approach speeds
            front_approach = max(ego_speed - front_speed, 0.0)
            rear_approach = max(rear_speed - ego_speed, 0.0)
            
            # Check if gaps are safe using stopping distance formula
            front_safe_gap = _stopping_dist(front_approach) + MIN_GAP
            rear_safe_gap = _stopping_dist(rear_approach) + MIN_GAP
            
            if front_gap < front_safe_gap or rear_gap < rear_safe_gap:
                lc_safe = False
                if self._log_enabled and self._tick_count % 20 == 0:
                    msg = f"  [SAFETY] Lane change blocked: front_gap={front_gap:.1f}m (need {front_safe_gap:.1f}m) rear_gap={rear_gap:.1f}m (need {rear_safe_gap:.1f}m)\n"
                    if hasattr(self, 'log_file') and self.log_file:
                        self.log_file.write(msg)
                        self.log_file.flush()

        # ── Urgent merge: force lane change if on merge edge after junction ─────
        # As soon as vehicle exits clusterJ1_cluster111_J1 junction onto edge -1, 
        # it needs to merge IMMEDIATELY with urgency, not wait until running out of space.
        urgent_merge = False
        if cur_lane == 0 and not edge_id.startswith(':'):
            # Trigger urgent merge immediately on edge -1 (main merge edge after junction)
            # This gives the vehicle a sense of urgency to merge right away
            if edge_id == '-1':
                urgent_merge = True
                if self._log_enabled and self._tick_count % 10 == 0:
                    msg = f"  [URGENT] On merge edge -1, must merge immediately!\n"
                    if hasattr(self, 'log_file') and self.log_file:
                        self.log_file.write(msg)
                        self.log_file.flush()
            else:
                # For other edges, check remaining distance as before
                try:
                    lane_length = traci.lane.getLength(f'{edge_id}_0')
                    lane_pos = traci.vehicle.getLanePosition(self.ego_id)
                    remaining = lane_length - lane_pos
                    # Urgent if less than 100 m remaining on the dead-end lane
                    if remaining < 100.0:
                        urgent_merge = True
                        if self._log_enabled and self._tick_count % 10 == 0:
                            msg = f"  [URGENT] Dead-end lane detected: {remaining:.1f}m remaining\n"
                            if hasattr(self, 'log_file') and self.log_file:
                                self.log_file.write(msg)
                                self.log_file.flush()
                except Exception:
                    pass

        # ── Decide ──
        if urgent_merge:
            # Must merge — force lane change right (+1), reduce gap
            # requirement but still avoid imminent crash
            URGENT_GAP = 5.0
            target_lane_id = f'{edge_id}_1'
            ego_lanepos = traci.vehicle.getLanePosition(self.ego_id)
            u_front_gap, u_rear_gap = float('inf'), float('inf')
            for vid in traci.vehicle.getIDList():
                if vid == self.ego_id:
                    continue
                try:
                    if traci.vehicle.getLaneID(vid) != target_lane_id:
                        continue
                    v_lp = traci.vehicle.getLanePosition(vid)
                    diff = v_lp - ego_lanepos
                    if diff > 0:
                        u_front_gap = min(u_front_gap, diff - VEH_LENGTH)
                    else:
                        u_rear_gap = min(u_rear_gap, abs(diff) - VEH_LENGTH)
                except Exception:
                    continue
            if u_front_gap > URGENT_GAP and u_rear_gap > URGENT_GAP:
                # Force merge right
                action = np.array([action[0], 5.0], dtype=np.float32)
            else:
                # Slow down to create a gap, keep trying
                action = np.array([-3.0, 5.0], dtype=np.float32)
        elif lc != 0 and not lc_safe:
            # Cancel lane change; use moderate braking if current lane is also unsafe
            # PRE-MERGE SPEED MATCHING: Brake to match target lane leader speed
            action = np.array([action[0], 0.0], dtype=np.float32)
            if not cur_front_safe:
                # Check if gap is critical (< 5m) or just unsafe
                leader = traci.vehicle.getLeader(self.ego_id, 50.0)
                if leader is not None and leader[0] and leader[1] < 5.0:
                    action[0] = -5.0  # Hard braking for critical gap
                else:
                    action[0] = -3.0  # Moderate braking for unsafe gap
            elif target_lane_leader_speed is not None and ego_speed > target_lane_leader_speed + 2.0:
                # Slow down to match target lane traffic before merging
                action[0] = -3.0
                if self._log_enabled and self._tick_count % 10 == 0:
                    msg = f"  [PRE-MERGE] Slowing to match target lane: ego={ego_speed:.1f} target_leader={target_lane_leader_speed:.1f}\n"
                    if hasattr(self, 'log_file') and self.log_file:
                        self.log_file.write(msg)
                        self.log_file.flush()
        elif not cur_front_safe:
            # Use hard braking only if gap is critical, otherwise moderate
            leader = traci.vehicle.getLeader(self.ego_id, 50.0)
            if leader is not None and leader[0] and leader[1] < 5.0:
                action = np.array([-5.0, action[1]], dtype=np.float32)
            else:
                action = np.array([-3.0, action[1]], dtype=np.float32)

        return action

    def _apply_action(self, parsed_action):
        """Send acceleration and lane-change commands to SUMO."""
        acc, lc = parsed_action

        # Decrement cooldown timer
        if self._lane_change_cooldown > 0:
            self._lane_change_cooldown -= 1

        # Track lane changes and detect merge completion
        try:
            cur_lane = traci.vehicle.getLaneIndex(self.ego_id)
            if cur_lane == 1 and not self._merged_to_lane1:
                self._merged_to_lane1 = True
                if self._log_enabled:
                    msg = "  [MERGE COMPLETE] Vehicle reached lane 1, blocking further lane changes\n"
                    if hasattr(self, 'log_file') and self.log_file:
                        self.log_file.write(msg)
                        self.log_file.flush()
            
            # Detect lane change completion and reset cooldown
            if cur_lane != self._last_lane:
                self._lane_change_cooldown = 30  # 3 seconds at 0.1s time step
                self._last_lane = cur_lane
        except Exception:
            pass

        if lc != 0:
            # Check cooldown first
            if self._lane_change_cooldown > 0:
                if self._log_enabled:
                    msg = f"  [COOLDOWN] Lane change blocked, {self._lane_change_cooldown} ticks remaining\n"
                    if hasattr(self, 'log_file') and self.log_file:
                        self.log_file.write(msg)
                        self.log_file.flush()
            else:
                # Use changeLane() instead of changeLaneRelative()
                # changeLane(vehID, laneIndex, duration) performs GRADUAL lane change over duration
                try:
                    cur_lane = traci.vehicle.getLaneIndex(self.ego_id)
                    edge_id = traci.vehicle.getRoadID(self.ego_id)
                    
                    # Once merged to lane 1, block changes back to lane 0
                    if self._merged_to_lane1 and lc < 0:
                        if self._log_enabled:
                            msg = f"  [MERGE LOCK] Lane change to right blocked - already merged to lane 1\n"
                            if hasattr(self, 'log_file') and self.log_file:
                                self.log_file.write(msg)
                                self.log_file.flush()
                    else:
                        num_lanes = traci.edge.getLaneNumber(edge_id)
                        target_lane = cur_lane + int(lc)
                        target_lane = max(0, min(target_lane, num_lanes - 1))
                        
                        if target_lane != cur_lane:
                            # changeLane() with duration creates smooth lane change over time
                            # Reduced duration for more responsive merging (was 5.0s)
                            duration = 3.0
                            traci.vehicle.changeLane(self.ego_id, target_lane, duration)
                            if self._log_enabled:
                                msg = f"  [LC START] changeLane: {cur_lane} -> {target_lane} over {duration}s, edge={edge_id}\n"
                                if hasattr(self, 'log_file') and self.log_file:
                                    self.log_file.write(msg)
                                    self.log_file.flush()
                except Exception as e:
                    if self._log_enabled:
                        msg = f"  [ERROR] Lane change failed: {e}\n"
                        if hasattr(self, 'log_file') and self.log_file:
                            self.log_file.write(msg)
                            self.log_file.flush()

        # Acceleration - apply AFTER lane change to ensure proper sequencing
        try:
            traci.vehicle.setAcceleration(self.ego_id, acc, 1)
            # Cap speed at 28 m/s (100 km/h) for safer merging - reduced from 32 m/s
            current_speed = traci.vehicle.getSpeed(self.ego_id)
            if current_speed > 28:
                traci.vehicle.setSpeed(self.ego_id, 28)
            # Additional safety: if accelerating but leader is close, cap speed to leader's speed
            if acc > 0:
                leader = traci.vehicle.getLeader(self.ego_id, 50.0)
                if leader is not None and leader[0]:
                    gap = leader[1]
                    if gap < 30.0:  # Close following - increased from 20m
                        leader_speed = traci.vehicle.getSpeed(leader[0])
                        # Cap at leader speed minus safety margin
                        if current_speed > leader_speed:
                            traci.vehicle.setSpeed(self.ego_id, max(leader_speed - 2.0, 0))
        except Exception:
            pass
