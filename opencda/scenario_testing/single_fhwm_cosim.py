# -*- coding: utf-8 -*-
"""
FHWM highway co-simulation scenario **with DAROM merging control**.

SUMO handles traffic spawning and background vehicle motion.
CARLA acts as a 3D visualizer for the SUMO traffic.
The **ego vehicle** on the on-ramp is controlled by a pretrained
DAROM-GRU SAC agent (from the onRampMerging project) that sends
acceleration and lane-change commands via traci every tick.

The spectator camera is fixed in a bird's-eye view above the highway.
"""
# License: MIT

import os
import sys
import logging
import argparse

import carla

import opencda.scenario_testing.utils.cosim_api as sim_api
from opencda.core.common.cav_world import CavWorld
from opencda.scenario_testing.utils.yaml_utils import add_current_time
from opencda.scenario_testing.merging_controller import MergingController
from opencda.scenario_testing.birdseye_camera import BirdseyeCamera
from opencda.scenario_testing.state_estimator import StateEstimator
from opencda.scenario_testing.performance_evaluator import PerformanceEvaluator


# Suppress "vtype customType not found" warnings from co-simulation bridge
class VTypeWarningFilter(logging.Filter):
    def filter(self, record):
        return 'vtype customType not found' not in record.getMessage()

logging.getLogger().addFilter(VTypeWarningFilter())


def run_scenario(opt, scenario_params):
    scenario_manager = None
    merging_ctrl = None
    birdseye_cam = None
    state_estimator = None
    evaluator = None

    try:
        scenario_params = add_current_time(scenario_params)

        cav_world = CavWorld(opt.apply_ml)

        current_path = os.path.dirname(os.path.realpath(__file__))
        sumo_cfg = os.path.join(current_path,
                                '../assets/FHWM_SUMO')

        scenario_manager = \
            sim_api.CoScenarioManager(scenario_params,
                                      opt.apply_ml,
                                      opt.version,
                                      town='FHWM',
                                      cav_world=cav_world,
                                      sumo_file_parent_path=sumo_cfg)

        spectator = scenario_manager.world.get_spectator()
        spectator.set_transform(
            carla.Transform(
                carla.Location(x=0.0, y=10.0, z=350.0),
                carla.Rotation(pitch=-90.0, yaw=0.0, roll=0.0)))

        # ── Merging controller setup ──────────────────────────────────
        merging_params = scenario_params.get('merging_controller', {})
        _default_model = os.path.join(
            os.path.dirname(__file__), '..', '..', 'onRampMerging',
            'models', 'GRU-uniform-delay', 'GRU-uniform-delay_best')
        model_path = os.path.expanduser(str(merging_params.get('model_path', _default_model)))
        merging_ctrl = MergingController(
            model_path=model_path,
            ego_id=merging_params.get('ego_id', 'ego'),
            radius=merging_params.get('radius', 50),
            max_delay=merging_params.get('max_delay', 20),
            delay_mode=merging_params.get('delay_mode', 'uniform'),
            use_safety=merging_params.get('use_safety', True),
            merge_edge=merging_params.get('merge_edge', '-1'),
            merge_min_lane=merging_params.get('merge_min_lane', 1),
            log_file=merging_params.get('log_file', 'merging_log.txt'),
        )
        merging_ctrl.reset()

        # ── Performance evaluation setup ──────────────────────────────
        if opt.evaluation:
            eval_log = merging_params.get('evaluation_log', 
                                         'performance_metrics.csv')
            evaluator = PerformanceEvaluator(log_file=eval_log)
            print('[cosim] Performance evaluation enabled.')

        ego_id_str = merging_params.get('ego_id', 'ego')

        print('Co-simulation running with DAROM merging control. '
              'Press Ctrl+C to stop.')
        while True:
            scenario_manager.tick()

            # Record timing metrics if evaluation enabled
            if evaluator is not None:
                carla_time = scenario_manager.world.get_snapshot().timestamp.elapsed_seconds
                import traci
                sumo_time = traci.simulation.getTime()
                evaluator.record_timestep(carla_time, sumo_time, 
                                        ego_id=ego_id_str,
                                        state_estimator=state_estimator)

            # Let the RL agent control the ego vehicle
            ctrl_info = merging_ctrl.tick()
            if ctrl_info.get('merged') and not getattr(
                    run_scenario, '_merge_logged', False):
                run_scenario._merge_logged = True
                print('[cosim] Ego vehicle has merged onto the mainline.')
                if state_estimator is not None:
                    state_estimator.notify_merged_on_mainline()

            # Attach bird's-eye camera once ego appears in CARLA unless state estimator is enabled
            if birdseye_cam is None and state_estimator is None:
                carla_id = scenario_manager.sumo2carla_ids.get(ego_id_str)
                if carla_id is not None:
                    ego_actor = scenario_manager.world.get_actor(carla_id)
                    if ego_actor is not None:
                        if opt.state_estimator:
                            # Use state estimator instead of birdseye camera
                            yolo_path = merging_params.get('yolo_model_path', None)
                            state_estimator = StateEstimator(
                                scenario_manager.world, ego_actor, 
                                yolo_model_path=yolo_path)
                            if getattr(run_scenario, '_merge_logged', False):
                                state_estimator.notify_merged_on_mainline()
                            print('[cosim] State estimator attached '
                                  'to ego vehicle.')
                        else:
                            # Use birdseye camera
                            birdseye_cam = BirdseyeCamera(
                                scenario_manager.world, ego_actor,
                                width=800, height=600, z=50, fov=90)
                            print('[cosim] Bird\'s-eye camera attached '
                                  'to ego vehicle.')

            # Check if ego vehicle despawned and cleanup visualizer
            carla_id = scenario_manager.sumo2carla_ids.get(ego_id_str)
            if carla_id is None or scenario_manager.world.get_actor(carla_id) is None:
                if state_estimator is not None:
                    print('\n[cosim] Ego vehicle despawned, closing state estimator window.')
                    state_estimator.destroy()
                    state_estimator = None
                    break
                elif birdseye_cam is not None:
                    print('\n[cosim] Ego vehicle despawned, closing birdseye camera.')
                    birdseye_cam.destroy()
                    birdseye_cam = None
                    break

            # Tick the active visualizer
            if state_estimator is not None:
                should_quit = state_estimator.tick()
                if should_quit:
                    print('\nState estimator window closed.')
                    break
            elif birdseye_cam is not None:
                birdseye_cam.tick()
                if birdseye_cam.should_quit():
                    print('\nBirdseye window closed.')
                    break

    except KeyboardInterrupt:
        print('\nSimulation stopped by user.')

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f'\n[ERROR] Scenario failed: {e}')

    finally:
        if evaluator is not None:
            evaluator.destroy()
        if merging_ctrl is not None:
            merging_ctrl.destroy()
        if state_estimator is not None:
            state_estimator.destroy()
        if birdseye_cam is not None:
            birdseye_cam.destroy()
        if scenario_manager is not None:
            scenario_manager.close()
