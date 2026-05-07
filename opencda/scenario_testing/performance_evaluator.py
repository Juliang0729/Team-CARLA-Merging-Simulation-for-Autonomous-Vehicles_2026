# -*- coding: utf-8 -*-
"""
Performance Metrics Evaluation Module for OpenCDA Co-Simulation
Tracks and computes various performance metrics during CARLA-SUMO co-simulation
"""

import time
import csv
import os
import numpy as np
from collections import defaultdict


class PerformanceEvaluator:
    """
    Evaluates co-simulation performance metrics
   
    """
    
    def __init__(self, log_file='performance_metrics.csv'):
        """
        Initialize performance evaluator.
        
        Parameters
        ----------
        log_file : str
            Path to CSV file for logging metrics
        """
        self.log_file = log_file
        self.metrics = defaultdict(list)
        
        # Time tracking
        self.carla_times = []
        self.sumo_times = []
        self.simulator_time_offsets = []
        self.control_latencies = []
        
        # Wall clock tracking for control latency
        self.last_wall_time = None
        
        # Lateral error tracking
        self.lateral_errors = []
        
        # Position error tracking (detection vs ground truth)
        self.detection_counts = []
        self.sumo_vehicle_counts = []
        self.position_errors = []  # List of position errors for tracked vehicles
        self.drift_rates = []  # List of position drift rates (m/s)
        
        # Collision tracking
        self.collision_occurred = []  # Binary: 1 if collision detected this timestep, 0 otherwise
        self.total_collisions = 0  # Cumulative collision count
        self.colliding_vehicles = set()  # Set of vehicle IDs currently in collision
        
        # Velocity deviation tracking (CARLA vs SUMO)
        self.velocity_deviations = []  # List of velocity differences (m/s)
        
        # Previous ego state for velocity calculation (SUMO-controlled vehicles don't have physics velocity)
        self.prev_ego_position = None
        self.prev_ego_time = None
        
        # Initialized flag
        self.initialized = False
        
        print(f"[PerformanceEvaluator] Initialized. Metrics will be logged to: {log_file}")
    
    def record_timestep(self, carla_time, sumo_time, ego_id=None, state_estimator=None):
        """
        Record timing and control information for the current simulation step.
        
        Parameters
        ----------
        carla_time : float
            Current CARLA simulation time (seconds)
        sumo_time : float
            Current SUMO simulation time (seconds)
        ego_id : str, optional
            SUMO ID of ego vehicle for lateral error tracking
        state_estimator : StateEstimator, optional
            State estimator instance for position error tracking and ego vehicle reference
        scenario_manager : CoScenarioManager, optional
            Scenario manager for accessing CARLA ego vehicle
        """
        # Measure wall clock time for control latency
        current_wall_time = time.time()
        if self.last_wall_time is not None:
            control_latency = current_wall_time - self.last_wall_time
            self.control_latencies.append(control_latency)
        else:
            # First iteration - no latency to compute
            self.control_latencies.append(0.0)
        self.last_wall_time = current_wall_time
        
        self.carla_times.append(carla_time)
        self.sumo_times.append(sumo_time)
        
        # Calculate Simulator Time Offset: Δt_sim(t_k) = |t_c(t_k) - t_s(t_k)|
        time_offset = abs(carla_time - sumo_time)
        self.simulator_time_offsets.append(time_offset)
        
        # Lateral error (offset from lane centerline)
        lateral_error = 0.0
        if ego_id is not None:
            try:
                import traci
                veh_list = traci.vehicle.getIDList()
                if ego_id in veh_list:
                    # getLateralLanePosition returns offset from lane center (meters)
                    # Positive = right of center, Negative = left of center
                    lateral_error = abs(traci.vehicle.getLateralLanePosition(ego_id))
                    if not self.initialized:
                        print(f"[PerformanceEvaluator] Ego '{ego_id}' found at timestep {len(self.carla_times)}")
                        print(f"[PerformanceEvaluator] Tracking lateral error from lane center")
                        self.initialized = True
            except Exception as e:
                # If ego not found or traci error, use 0
                if not self.initialized:
                    print(f"[PerformanceEvaluator] Warning: Could not get lateral position: {e}")
                    self.initialized = True
                lateral_error = 0.0
        elif not self.initialized:
            print(f"[PerformanceEvaluator] Warning: ego_id is None, lateral error disabled")
            self.initialized = True
        self.lateral_errors.append(lateral_error)
        
        # Velocity deviation (CARLA vs SUMO)
        velocity_deviation = 0.0
        if ego_id is not None and state_estimator is not None:
            try:
                import traci
                veh_list = traci.vehicle.getIDList()
                if ego_id in veh_list:
                    # Get SUMO velocity (m/s)
                    sumo_velocity = traci.vehicle.getSpeed(ego_id)
                    
                    # Get CARLA ego vehicle from state estimator (correct reference)
                    carla_ego = state_estimator.ego_vehicle
                    if carla_ego is not None:
                        # IMPORTANT: SUMO-controlled vehicles use set_transform() which doesn't update
                        # physics velocity, so get_velocity() returns 0. Calculate from position delta.
                        ego_pos = carla_ego.get_transform().location
                        
                        if self.prev_ego_position is not None and self.prev_ego_time is not None:
                            dt = carla_time - self.prev_ego_time
                            if dt > 0:
                                dx = ego_pos.x - self.prev_ego_position.x
                                dy = ego_pos.y - self.prev_ego_position.y
                                carla_velocity = ((dx**2 + dy**2)**0.5) / dt
                                
                                # Calculate deviation (absolute difference)
                                velocity_deviation = abs(carla_velocity - sumo_velocity)
                        
                        # Update previous state for next frame
                        self.prev_ego_position = ego_pos
                        self.prev_ego_time = carla_time
            except Exception as e:
                if not hasattr(self, '_vel_error_logged'):
                    print(f"[PerformanceEvaluator] Velocity deviation error: {e}")
                    self._vel_error_logged = True
        self.velocity_deviations.append(velocity_deviation)
        
        # Position error (detection accuracy): track detected vs actual vehicle count
        # Only count SUMO vehicles within camera range (realistic comparison)
        detection_count = 0
        sumo_count = 0
        avg_position_error = 0.0
        if state_estimator is not None:
            try:
                detection_count = state_estimator.get_detected_count()
                import traci
                
                # Build dict of SUMO vehicle positions within range
                sumo_vehicles = {}
                if ego_id and ego_id in traci.vehicle.getIDList():
                    ego_pos = traci.vehicle.getPosition(ego_id)
                    
                    # Count only vehicles within 80m radius (realistic camera detection range)
                    DETECTION_RADIUS_M = 80.0
                    for veh_id in traci.vehicle.getIDList():
                        if veh_id == ego_id:
                            continue  # Don't count ego itself
                        veh_pos = traci.vehicle.getPosition(veh_id)
                        dist = ((veh_pos[0] - ego_pos[0])**2 + (veh_pos[1] - ego_pos[1])**2)**0.5
                        if dist <= DETECTION_RADIUS_M:
                            sumo_count += 1
                            sumo_vehicles[veh_id] = veh_pos
                    
                    # Match detections with SUMO ground truth for position error
                    # Pass ego_pos in SUMO coordinates for accurate distance calculation
                    matched = state_estimator.match_detections_with_sumo(sumo_vehicles, ego_id, ego_pos)
                    if matched:
                        errors = [m['position_error'] for m in matched]
                        avg_position_error = np.mean(errors)
                else:
                    # If ego not found, count all vehicles (fallback)
                    sumo_count = len(traci.vehicle.getIDList())
            except Exception as e:
                pass  # Silent fail if error occurs
        
        self.detection_counts.append(detection_count)
        self.sumo_vehicle_counts.append(sumo_count)
        self.position_errors.append(avg_position_error)
        
        # Calculate average drift rate across all tracked vehicles
        avg_drift_rate = 0.0
        if state_estimator is not None and hasattr(state_estimator, 'drift_rates'):
            all_drift_rates = []
            for veh_id, rates in state_estimator.drift_rates.items():
                if len(rates) > 0:
                    all_drift_rates.append(rates[-1])  # Take most recent drift rate
            if all_drift_rates:
                avg_drift_rate = np.mean(all_drift_rates)
        self.drift_rates.append(avg_drift_rate)
        
        # Collision detection using SUMO
        collision_this_step = 0
        try:
            import traci
            colliding_ids = traci.simulation.getCollidingVehiclesIDList()
            
            if colliding_ids:
                # New collisions (not previously colliding)
                new_collisions = set(colliding_ids) - self.colliding_vehicles
                if new_collisions:
                    self.total_collisions += len(new_collisions)
                    collision_this_step = 1
                    for veh_id in new_collisions:
                        print(f"[PerformanceEvaluator] Collision detected: vehicle '{veh_id}' at t={carla_time:.2f}s")
                
                # Update currently colliding vehicles set
                self.colliding_vehicles = set(colliding_ids)
            else:
                # No collisions this step
                self.colliding_vehicles.clear()
        except Exception as e:
            pass  # Silent fail if traci not available
        
        self.collision_occurred.append(collision_this_step)
        
        self.metrics['carla_time'].append(carla_time)
        self.metrics['sumo_time'].append(sumo_time)
        self.metrics['time_offset'].append(time_offset)
        self.metrics['control_latency'].append(self.control_latencies[-1])
        self.metrics['lateral_error'].append(lateral_error)
        self.metrics['detected_vehicles'].append(detection_count)
        self.metrics['sumo_vehicles'].append(sumo_count)
        self.metrics['position_error'].append(avg_position_error)
        self.metrics['drift_rate'].append(avg_drift_rate)
        self.metrics['velocity_deviation'].append(velocity_deviation)
    
    def get_summary_statistics(self):
        """
        Compute summary statistics for all metrics.
        
        Returns
        -------
        dict
            Dictionary containing statistics for all metrics
        """
        if not self.simulator_time_offsets:
            return {
                'time_offset': {
                    'mean': 0.0, 'max': 0.0, 'min': 0.0, 'std': 0.0, 'samples': 0
                },
                'control_latency': {
                    'mean': 0.0, 'max': 0.0, 'min': 0.0, 'std': 0.0, 'samples': 0
                },
                'lateral_error': {
                    'mean': 0.0, 'max': 0.0, 'min': 0.0, 'std': 0.0, 'samples': 0
                },
                'detection_accuracy': {
                    'detected_vehicles_mean': 0.0, 'sumo_vehicles_mean': 0.0,
                    'detection_rate': 0.0, 'samples': 0
                },
                'position_error': {
                    'mean': 0.0, 'max': 0.0, 'min': 0.0, 'std': 0.0, 'samples': 0
                },
                'drift_rate': {
                    'mean': 0.0, 'max': 0.0, 'min': 0.0, 'std': 0.0, 'samples': 0
                },
                'velocity_deviation': {
                    'mean': 0.0, 'max': 0.0, 'min': 0.0, 'std': 0.0, 'samples': 0
                },
                'collisions': {
                    'total_count': 0, 'frequency': 0.0, 'samples': 0
                }
            }
        
        offsets = np.array(self.simulator_time_offsets)
        latencies = np.array(self.control_latencies[1:])
        lateral_errs = np.array(self.lateral_errors)
        detected = np.array(self.detection_counts)
        sumo_counts = np.array(self.sumo_vehicle_counts)
        pos_errors = np.array(self.position_errors)
        drift_rates = np.array(self.drift_rates)
        vel_deviations = np.array(self.velocity_deviations)
        
        # Calculate detection rate (avoid division by zero)
        valid_mask = sumo_counts > 0
        if valid_mask.sum() > 0:
            detection_rates = detected[valid_mask] / sumo_counts[valid_mask]
            avg_detection_rate = float(np.mean(detection_rates))
        else:
            avg_detection_rate = 0.0
        
        # Calculate average position error (only for non-zero values)
        pos_error_mask = pos_errors > 0
        if pos_error_mask.sum() > 0:
            pos_error_mean = float(np.mean(pos_errors[pos_error_mask]))
            pos_error_max = float(np.max(pos_errors[pos_error_mask]))
            pos_error_min = float(np.min(pos_errors[pos_error_mask]))
            pos_error_std = float(np.std(pos_errors[pos_error_mask]))
            pos_error_samples = int(pos_error_mask.sum())
        else:
            pos_error_mean = 0.0
            pos_error_max = 0.0
            pos_error_min = 0.0
            pos_error_std = 0.0
            pos_error_samples = 0
        
        return {
            'time_offset': {
                'mean': float(np.mean(offsets)),
                'max': float(np.max(offsets)),
                'min': float(np.min(offsets)),
                'std': float(np.std(offsets)),
                'samples': len(offsets)
            },
            'control_latency': {
                'mean': float(np.mean(latencies)) if len(latencies) > 0 else 0.0,
                'max': float(np.max(latencies)) if len(latencies) > 0 else 0.0,
                'min': float(np.min(latencies)) if len(latencies) > 0 else 0.0,
                'std': float(np.std(latencies)) if len(latencies) > 0 else 0.0,
                'samples': len(latencies)
            },
            'lateral_error': {
                'mean': float(np.mean(lateral_errs)),
                'max': float(np.max(lateral_errs)),
                'min': float(np.min(lateral_errs)),
                'std': float(np.std(lateral_errs)),
                'samples': len(lateral_errs)
            },
            'detection_accuracy': {
                'detected_vehicles_mean': float(np.mean(detected)),
                'sumo_vehicles_mean': float(np.mean(sumo_counts)),
                'detection_rate': avg_detection_rate,
                'samples': len(detected)
            },
            'position_error': {
                'mean': pos_error_mean,
                'max': pos_error_max,
                'min': pos_error_min,
                'std': pos_error_std,
                'samples': pos_error_samples
            },
            'drift_rate': {
                'mean': float(np.mean(np.abs(drift_rates[drift_rates != 0]))) if np.any(drift_rates != 0) else 0.0,
                'max': float(np.max(np.abs(drift_rates))) if len(drift_rates) > 0 else 0.0,
                'min': float(np.min(np.abs(drift_rates[drift_rates != 0]))) if np.any(drift_rates != 0) else 0.0,
                'std': float(np.std(drift_rates[drift_rates != 0])) if np.any(drift_rates != 0) else 0.0,
                'samples': int(np.sum(drift_rates != 0))
            },
            'velocity_deviation': {
                'mean': float(np.mean(vel_deviations[vel_deviations != 0])) if np.any(vel_deviations != 0) else 0.0,
                'max': float(np.max(vel_deviations)) if len(vel_deviations) > 0 else 0.0,
                'min': float(np.min(vel_deviations[vel_deviations != 0])) if np.any(vel_deviations != 0) else 0.0,
                'std': float(np.std(vel_deviations[vel_deviations != 0])) if np.any(vel_deviations != 0) else 0.0,
                'samples': int(np.sum(vel_deviations != 0))
            },
            'collisions': {
                'total_count': self.total_collisions,
                'frequency': float(self.total_collisions / len(self.collision_occurred)) if len(self.collision_occurred) > 0 else 0.0,
                'samples': len(self.collision_occurred)
            }
        }
    
    def save_to_csv(self):
        """Save all metrics to CSV file with summary statistics at the top."""
        if not self.metrics['carla_time']:
            print("[PerformanceEvaluator] No data to save.")
            return
        
        # Expand user path
        log_path = os.path.expanduser(self.log_file)
        
        # Get summary statistics
        stats = self.get_summary_statistics()
        
        # Write metrics to CSV
        with open(log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write summary statistics as header comments
            writer.writerow(['# PERFORMANCE EVALUATION SUMMARY'])
            writer.writerow(['#'])
            writer.writerow(['# Simulator Time Offset (seconds):'])
            writer.writerow(['#   Mean:', f"{stats['time_offset']['mean']:.6f}"])
            writer.writerow(['#   Max:', f"{stats['time_offset']['max']:.6f}"])
            writer.writerow(['#   Min:', f"{stats['time_offset']['min']:.6f}"])
            writer.writerow(['#   Std Dev:', f"{stats['time_offset']['std']:.6f}"])
            writer.writerow(['#   Samples:', stats['time_offset']['samples']])
            writer.writerow(['#'])
            writer.writerow(['# Control Synchronization Latency (seconds):'])
            writer.writerow(['#   Mean:', f"{stats['control_latency']['mean']:.6f}"])
            writer.writerow(['#   Max:', f"{stats['control_latency']['max']:.6f}"])
            writer.writerow(['#   Min:', f"{stats['control_latency']['min']:.6f}"])
            writer.writerow(['#   Std Dev:', f"{stats['control_latency']['std']:.6f}"])
            writer.writerow(['#   Samples:', stats['control_latency']['samples']])
            writer.writerow(['#'])
            writer.writerow(['# Lateral Error (meters):'])
            writer.writerow(['#   Mean:', f"{stats['lateral_error']['mean']:.6f}"])
            writer.writerow(['#   Max:', f"{stats['lateral_error']['max']:.6f}"])
            writer.writerow(['#   Min:', f"{stats['lateral_error']['min']:.6f}"])
            writer.writerow(['#   Std Dev:', f"{stats['lateral_error']['std']:.6f}"])
            writer.writerow(['#   Samples:', stats['lateral_error']['samples']])
            writer.writerow(['#'])
            writer.writerow(['# Detection Accuracy (Position Error Proxy):'])
            writer.writerow(['#   Avg Detected Vehicles:', f"{stats['detection_accuracy']['detected_vehicles_mean']:.2f}"])
            writer.writerow(['#   Avg SUMO Vehicles:', f"{stats['detection_accuracy']['sumo_vehicles_mean']:.2f}"])
            writer.writerow(['#   Detection Rate:', f"{stats['detection_accuracy']['detection_rate']:.2%}"])
            writer.writerow(['#   Samples:', stats['detection_accuracy']['samples']])
            writer.writerow(['#'])
            writer.writerow(['# Position Error (meters):'])
            writer.writerow(['#   Mean:', f"{stats['position_error']['mean']:.6f}"])
            writer.writerow(['#   Max:', f"{stats['position_error']['max']:.6f}"])
            writer.writerow(['#   Min:', f"{stats['position_error']['min']:.6f}"])
            writer.writerow(['#   Std Dev:', f"{stats['position_error']['std']:.6f}"])
            writer.writerow(['#   Samples:', stats['position_error']['samples']])
            writer.writerow(['#'])
            writer.writerow(['# Position Drift Rate (m/s):'])
            writer.writerow(['#   Mean (absolute):', f"{stats['drift_rate']['mean']:.6f}"])
            writer.writerow(['#   Max (absolute):', f"{stats['drift_rate']['max']:.6f}"])
            writer.writerow(['#   Min (absolute):', f"{stats['drift_rate']['min']:.6f}"])
            writer.writerow(['#   Std Dev:', f"{stats['drift_rate']['std']:.6f}"])
            writer.writerow(['#   Samples:', stats['drift_rate']['samples']])
            writer.writerow(['#'])
            writer.writerow(['# Velocity Deviation (m/s):'])
            writer.writerow(['#   Mean:', f"{stats['velocity_deviation']['mean']:.6f}"])
            writer.writerow(['#   Max:', f"{stats['velocity_deviation']['max']:.6f}"])
            writer.writerow(['#   Min:', f"{stats['velocity_deviation']['min']:.6f}"])
            writer.writerow(['#   Std Dev:', f"{stats['velocity_deviation']['std']:.6f}"])
            writer.writerow(['#   Samples:', stats['velocity_deviation']['samples']])
            writer.writerow(['#'])
            writer.writerow(['# Collision Frequency:'])
            writer.writerow(['#   Total Collisions:', stats['collisions']['total_count']])
            writer.writerow(['#   Collision Frequency:', f"{stats['collisions']['frequency']:.6f}"])
            writer.writerow(['#   Samples:', stats['collisions']['samples']])
            writer.writerow(['#'])
            
            # Data header
            writer.writerow(['timestep', 'carla_time', 'sumo_time', 'time_offset_seconds', 
                           'control_latency_seconds', 'lateral_error_meters',
                           'detected_vehicles', 'sumo_vehicles', 'position_error_meters', 
                           'drift_rate_mps', 'velocity_deviation_mps', 'collision'])
            
            # Data rows
            num_steps = len(self.metrics['carla_time'])
            for i in range(num_steps):
                writer.writerow([
                    i,
                    self.metrics['carla_time'][i],
                    self.metrics['sumo_time'][i],
                    self.metrics['time_offset'][i],
                    self.metrics['control_latency'][i],
                    self.metrics['lateral_error'][i],
                    self.metrics['detected_vehicles'][i],
                    self.metrics['sumo_vehicles'][i],
                    self.metrics['position_error'][i],
                    self.metrics['drift_rate'][i],
                    self.metrics['velocity_deviation'][i],
                    self.collision_occurred[i]
                ])
        
        print(f"[PerformanceEvaluator] Metrics saved to: {log_path}")
    
    def print_summary(self):
        """Print summary statistics to console (disabled to reduce clutter)."""
        pass
    
    def destroy(self):
        """Clean up and save final metrics."""
        self.save_to_csv()
