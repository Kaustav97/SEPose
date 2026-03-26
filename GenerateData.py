#!/usr/bin/env python

# Copyright (c) 2021 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""CARLA simulation script for generating pose estimation training data."""

import argparse
import copy
import glob
import logging
import os
import queue
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from numpy import random

try:
    sys.path.append("carla-0.9.15-py3.7-linux-x86_64.egg")
except Exception as e:    
    print(e)

from draw_skeleton import get_screen_points, draw_skeleton
from config import load_config, apply_cli_overrides, get_nested
import carla

# Global config (loaded in main)
CONFIG: Dict[str, Any] = {}

# Constants (not configurable)
OUT_DIR = ""
KEYPOINTS = [
    "crl_Head__C", "crl_eye__R", "crl_eye__L", "crl_shoulder__R",
    "crl_shoulder__L", "crl_arm__R", "crl_arm__L", "crl_foreArm__R",
    "crl_foreArm__L", "crl_hips__C", "crl_thigh__R", "crl_thigh__L",
    "crl_leg__R", "crl_leg__L", "crl_foot__R", "crl_foot__L"
]
NUM_KEYPOINTS = len(KEYPOINTS)


def check_keypoint_visibility(
    point_2d: Tuple[float, float],
    image_w: int,
    image_h: int
) -> int:
    """Check keypoint visibility based on image bounds.
    
    Args:
        point_2d: 2D point coordinates (x, y)
        image_w: Image width in pixels
        image_h: Image height in pixels
    
    Returns:
        Visibility flag: 0 (not visible), 2 (visible)
        Note: Occlusion detection (flag 1) requires depth/semantic data
    """
    x, y = point_2d[:2]  # points_2d includes depth as 3rd element
    
    # Check if point is within image boundaries
    if 0 <= x < image_w and 0 <= y < image_h:
        return 2  # Visible
    return 0  # Not visible (out of bounds)


def compute_bbox_from_keypoints(
    keypoints: np.ndarray,
    visibility: np.ndarray,
    image_w: int,
    image_h: int,
    padding: float = None
) -> Optional[Tuple[float, float, float, float]]:
    """Compute normalized bounding box from visible keypoints.
    
    Args:
        keypoints: Array of shape (N, 2) with keypoint coordinates
        visibility: Array of shape (N,) with visibility flags
        image_w: Image width in pixels
        image_h: Image height in pixels
        padding: Padding ratio to add around bounding box (uses config if None)
    
    Returns:
        Tuple of (x_center, y_center, width, height) normalized to [0, 1]
        Returns None if no visible keypoints
    """
    if padding is None:
        padding = get_nested(CONFIG, 'pose', 'bbox_padding', default=0.1)
    # Filter visible keypoints only
    visible_mask = visibility > 0
    if not np.any(visible_mask):
        return None
    
    visible_kps = keypoints[visible_mask]
    
    # Calculate min/max bounds
    x_min = np.min(visible_kps[:, 0])
    x_max = np.max(visible_kps[:, 0])
    y_min = np.min(visible_kps[:, 1])
    y_max = np.max(visible_kps[:, 1])
    
    # Add padding
    width = x_max - x_min
    height = y_max - y_min
    x_min = max(0, x_min - width * padding)
    x_max = min(image_w, x_max + width * padding)
    y_min = max(0, y_min - height * padding)
    y_max = min(image_h, y_max + height * padding)
    
    # Calculate center and dimensions
    bbox_width = x_max - x_min
    bbox_height = y_max - y_min
    x_center = x_min + bbox_width / 2.0
    y_center = y_min + bbox_height / 2.0
    
    # Normalize to [0, 1]
    x_center_norm = x_center / image_w
    y_center_norm = y_center / image_h
    width_norm = bbox_width / image_w
    height_norm = bbox_height / image_h
    
    return (x_center_norm, y_center_norm, width_norm, height_norm)


def getCamXforms(map_name: str) -> Tuple[carla.Location, carla.Rotation]:
    """Get camera transform (location and rotation) for a given map.
    
    Args:
        map_name: Name of the CARLA map
    
    Returns:
        Tuple of (Location, Rotation) for camera placement
    
    Raises:
        ValueError: If map_name is not supported in config
    """
    transforms = get_nested(CONFIG, 'camera', 'transforms', default={})
    
    if map_name not in transforms:
        supported_maps = ', '.join(transforms.keys())
        raise ValueError(
            f"Unsupported map: '{map_name}'. Supported maps: {supported_maps}"
        )
    
    t = transforms[map_name]
    loc = t['location']
    rot = t['rotation']
    
    return (
        carla.Location(x=loc['x'], y=loc['y'], z=loc['z']),
        carla.Rotation(pitch=rot['pitch'], yaw=rot['yaw'], roll=rot['roll'])
    )


def GenerateGTPose(
    image,
    image_h: int,
    image_w: int,
    K: np.ndarray,
    camera,
    peds: List
) -> None:
    """Generate ground truth pose annotations in YOLO-pose format.
    
    Processes pedestrians and outputs:
    - Visualization image with skeleton overlay (GT folder)
    - YOLO-pose format TXT annotations (Annot folder)
    
    YOLO-pose format per line:
    <class> <x_center> <y_center> <width> <height> 
    <kp1_x> <kp1_y> <kp1_vis> ... <kp16_x> <kp16_y> <kp16_vis>
    
    All coordinates normalized to [0, 1].
    """
    buf = np.zeros((image_h, image_w, 3), dtype=np.uint8)
    yolo_annotations = []
    max_ped_distance = get_nested(
        CONFIG, 'simulation', 'max_pedestrian_distance', default=50.0
    )
    
    for ped in peds:
        try:
            dist = ped.get_transform().location.distance(
                camera.get_transform().location
            )
            if dist >= max_ped_distance:
                continue
            
            forward_vec = camera.get_transform().get_forward_vector()
            ray = ped.get_transform().location - \
                  camera.get_transform().location
            
            if forward_vec.dot(ray) <= 0:
                continue
            
            # Get bone transforms from walker
            # API: Walker.get_bones() returns WalkerBoneControlOut
            # WalkerBoneControlOut.bone_transforms is list of bone_transform_out objects
            # Each bone_transform_out has attributes: name, world, component, relative
            # world/component/relative are Transform objects with .location attribute
            bones = ped.get_bones()
            bone_index = {
                x.name: i for i, x in enumerate(bones.bone_transforms)
            }
            points = [x.world.location for x in bones.bone_transforms]
            points2d = get_screen_points(
                camera, K, image_w, image_h, points
            )
            
            # Extract keypoint locations for the 16 defined keypoints
            keypoint_coords = np.array([
                points2d[bone_index[kp_name]] for kp_name in KEYPOINTS
            ])
            
            # Compute visibility for each keypoint
            visibility = np.array([
                check_keypoint_visibility(kp, image_w, image_h)
                for kp in keypoint_coords
            ])
            
            # Skip if no visible keypoints
            if not np.any(visibility > 0):
                continue
            
            # Compute bounding box from visible keypoints
            bbox = compute_bbox_from_keypoints(
                keypoint_coords, visibility, image_w, image_h
            )
            
            if bbox is None:
                continue
            
            # Normalize keypoint coordinates to [0, 1]
            kp_normalized = keypoint_coords.copy()
            kp_normalized[:, 0] /= image_w
            kp_normalized[:, 1] /= image_h
            
            # Prepare YOLO-pose annotation
            # Format: class x_c y_c w h kp1_x kp1_y kp1_v ... kp16_x kp16_y kp16_v
            annotation = [0] + list(bbox)  # class=0 for person
            for i in range(NUM_KEYPOINTS):
                annotation.extend([
                    kp_normalized[i, 0],
                    kp_normalized[i, 1],
                    visibility[i]
                ])
            
            yolo_annotations.append(annotation)
            
            # Draw skeleton on visualization buffer
            draw_skeleton(
                buf, image_w, image_h, bone_index, points2d,
                (0, 255, 0), 3
            )
            
        except Exception as e:
            logging.warning(f"Error processing pedestrian: {e}", exc_info=True)
    
    # Save visualization image
    cv2.imwrite(f"{OUT_DIR}/GT/{image.frame}.png", buf)
    
    # Write YOLO-pose format annotations
    with open(f"{OUT_DIR}/Annot/{image.frame}.txt", 'w') as f:
        for annotation in yolo_annotations:
            # Write class, bbox, and keypoints
            class_id = int(annotation[0])
            bbox_vals = annotation[1:5]
            kp_vals = annotation[5:]
            
            f.write(f"{class_id} ")
            f.write(" ".join(f"{v:.6f}" for v in bbox_vals))
            f.write(" ")
            f.write(" ".join(
                f"{kp_vals[i*3]:.6f} {kp_vals[i*3+1]:.6f} {int(kp_vals[i*3+2])}"
                for i in range(NUM_KEYPOINTS)
            ))
            f.write("\n") 



def ProcessDVSImage(image) -> None:
    """Process and save DVS (Dynamic Vision Sensor) image data.
    
    Args:
        image: CARLA DVS sensor image with raw event data
    """
    dvs_events = np.frombuffer(image.raw_data, dtype=np.dtype([
            ('x', np.uint16), ('y', np.uint16), ('t', np.int64),
            ('pol', np.bool_)]))

    dvs_img = np.zeros((image.height, image.width, 3), dtype=np.uint8)
    dvs_img[
        dvs_events[:]['y'], dvs_events[:]['x'], dvs_events[:]['pol'] * 2
    ] = 255
    array2 = copy.deepcopy(dvs_img)

    cv2.imwrite(f"{OUT_DIR}/events/{image.frame}.png", array2)
    print(f"{OUT_DIR}/events/{image.frame}.png")


def ProcessRGBImage(image) -> None:
    """Process and save RGB camera image.
    
    Args:
        image: CARLA RGB sensor image with raw pixel data
    """
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    # make the array writeable doing a deep copy
    array2 = copy.deepcopy(array)
    cv2.imwrite(f"{OUT_DIR}/RGB/{image.frame}_RGB.png", array2)

def build_projection_matrix(w: int, h: int, fov: float) -> np.ndarray:
    """Build camera projection matrix from intrinsic parameters.
    
    Args:
        w: Image width in pixels
        h: Image height in pixels
        fov: Field of view in degrees
    
    Returns:
        3x3 camera intrinsic matrix K
    """
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K


def generate_random_weather() -> carla.WeatherParameters:
    """Generate random weather from predefined templates with variations.
    
    Randomly selects from weather templates and adds parameter variations
    to create diverse weather conditions for training data.
    
    Returns:
        carla.WeatherParameters with randomized weather settings
    """
    weather_templates = get_nested(CONFIG, 'weather_presets', default={})
    
    if not weather_templates:
        logging.warning("No weather presets in config, using clear weather")
        return carla.WeatherParameters.ClearNoon
    
    # Randomly select a template
    template_name = random.choice(list(weather_templates.keys()))
    template = weather_templates[template_name].copy()
    
    # Add random variations (±20% for most parameters)
    variations = {
        'cloudiness': 0.2,
        'precipitation': 0.15,
        'precipitation_deposits': 0.15,
        'sun_altitude_angle': 15.0,  # ±15 degrees
        'sun_azimuth_angle': 30.0,   # ±30 degrees
        'wind_intensity': 0.2,
        'fog_density': 0.15,
        'fog_distance': 0.2,
        'wetness': 0.15,
    }
    
    for param, base_value in template.items():
        if param in variations:
            if param in ['sun_altitude_angle', 'sun_azimuth_angle']:
                # Additive variation for angles
                variation = random.uniform(
                    -variations[param], variations[param]
                )
            else:
                # Multiplicative variation for other params
                variation = random.uniform(
                    1.0 - variations[param], 1.0 + variations[param]
                )
                variation = base_value * variation - base_value
            
            template[param] = base_value + variation
    
    # Clamp values to valid CARLA ranges
    template['cloudiness'] = np.clip(template['cloudiness'], 0.0, 100.0)
    template['precipitation'] = np.clip(template['precipitation'], 0.0, 100.0)
    template['precipitation_deposits'] = np.clip(
        template['precipitation_deposits'], 0.0, 100.0
    )
    template['sun_altitude_angle'] = np.clip(
        template['sun_altitude_angle'], -90.0, 90.0
    )
    template['sun_azimuth_angle'] = np.clip(
        template['sun_azimuth_angle'], 0.0, 360.0
    )
    template['wind_intensity'] = np.clip(template['wind_intensity'], 0.0, 100.0)
    template['fog_density'] = np.clip(template['fog_density'], 0.0, 100.0)
    template['fog_distance'] = max(0.0, template['fog_distance'])  # 0 to infinity
    template['wetness'] = np.clip(template['wetness'], 0.0, 100.0)
    
    logging.info(f"Generated {template_name} weather with variations")
    
    return carla.WeatherParameters(**template)


def spawn_vehicles(
    world,
    client,
    config: Dict[str, Any],
    blueprints: List,
    spawn_points: List,
    traffic_manager,
    synchronous_master: bool
) -> List[int]:
    """Spawn vehicles in the simulation world.
    
    Args:
        world: CARLA world object
        client: CARLA client
        config: Configuration dictionary
        blueprints: List of vehicle blueprints
        spawn_points: List of spawn point transforms
        traffic_manager: Traffic manager instance
        synchronous_master: Whether running in synchronous mode
    
    Returns:
        List of spawned vehicle actor IDs
    """
    SpawnActor = carla.command.SpawnActor
    SetAutopilot = carla.command.SetAutopilot
    FutureActor = carla.command.FutureActor
    
    batch = []
    hero = get_nested(config, 'actors', 'vehicles', 'hero', default=False)
    num_vehicles = get_nested(config, 'actors', 'vehicles', 'count', default=30)
    
    for n, transform in enumerate(spawn_points):
        if n >= num_vehicles:
            break
        
        blueprint = random.choice(blueprints)
        
        if blueprint.has_attribute('color'):
            color = random.choice(
                blueprint.get_attribute('color').recommended_values
            )
            blueprint.set_attribute('color', color)
        
        if blueprint.has_attribute('driver_id'):
            driver_id = random.choice(
                blueprint.get_attribute('driver_id').recommended_values
            )
            blueprint.set_attribute('driver_id', driver_id)
        
        if hero:
            blueprint.set_attribute('role_name', 'hero')
            hero = False
        else:
            blueprint.set_attribute('role_name', 'autopilot')
        
        batch.append(
            SpawnActor(blueprint, transform).then(
                SetAutopilot(FutureActor, True, traffic_manager.get_port())
            )
        )
    
    vehicles_list = []
    for response in client.apply_batch_sync(batch, synchronous_master):
        if response.error:
            logging.error(f"Vehicle spawn error: {response.error}")
        else:
            vehicles_list.append(response.actor_id)
    
    # Set automatic vehicle lights if specified
    if get_nested(config, 'actors', 'vehicles', 'car_lights_on', default=False):
        all_vehicle_actors = world.get_actors(vehicles_list)
        for actor in all_vehicle_actors:
            traffic_manager.update_vehicle_lights(actor, True)
    
    logging.info(f"Spawned {len(vehicles_list)} vehicles")
    return vehicles_list


def spawn_walkers(
    world,
    client,
    config: Dict[str, Any],
    blueprints: List
) -> Tuple[List[Dict], List[int]]:
    """Spawn pedestrians in the simulation world.
    
    Args:
        world: CARLA world object
        client: CARLA client
        config: Configuration dictionary
        blueprints: List of walker blueprints
    
    Returns:
        Tuple of (walkers_list, all_id) where walkers_list contains
        walker dictionaries with 'id' and 'con' keys, and all_id
        is a flat list of all controller and walker IDs
    """
    SpawnActor = carla.command.SpawnActor
    
    percentagePedestriansRunning = get_nested(
        config, 'actors', 'walkers', 'percentage_running', default=0.0
    )
    percentagePedestriansCrossing = get_nested(
        config, 'actors', 'walkers', 'percentage_crossing', default=0.0
    )
    
    walker_seed = get_nested(config, 'actors', 'walkers', 'seed', default=0)
    if walker_seed:
        world.set_pedestrians_seed(walker_seed)
        random.seed(walker_seed)
    
    walker_count = get_nested(config, 'actors', 'walkers', 'count', default=10)
    
    # Generate spawn points
    spawn_points = []
    for i in range(walker_count):
        spawn_point = carla.Transform()
        loc = world.get_random_location_from_navigation()
        if loc is not None:
            spawn_point.location = loc
            spawn_points.append(spawn_point)
    
    # Spawn walker actors
    batch = []
    walker_speed = []
    for spawn_point in spawn_points:
        walker_bp = random.choice(blueprints)
        
        if walker_bp.has_attribute('is_invincible'):
            walker_bp.set_attribute('is_invincible', 'false')
        
        if walker_bp.has_attribute('speed'):
            if random.random() > percentagePedestriansRunning:
                walker_speed.append(
                    walker_bp.get_attribute('speed').recommended_values[1]
                )
            else:
                walker_speed.append(
                    walker_bp.get_attribute('speed').recommended_values[2]
                )
        else:
            walker_speed.append(0.0)
        
        batch.append(SpawnActor(walker_bp, spawn_point))
    
    results = client.apply_batch_sync(batch, True)
    walkers_list = []
    walker_speed2 = []
    
    for i in range(len(results)):
        if results[i].error:
            logging.error(f"Walker spawn error: {results[i].error}")
        else:
            walkers_list.append({"id": results[i].actor_id})
            walker_speed2.append(walker_speed[i])
    
    walker_speed = walker_speed2
    
    # Spawn walker controllers
    batch = []
    walker_controller_bp = world.get_blueprint_library().find(
        'controller.ai.walker'
    )
    for i in range(len(walkers_list)):
        batch.append(
            SpawnActor(
                walker_controller_bp, carla.Transform(), walkers_list[i]["id"]
            )
        )
    
    results = client.apply_batch_sync(batch, True)
    for i in range(len(results)):
        if results[i].error:
            logging.error(f"Controller spawn error: {results[i].error}")
        else:
            walkers_list[i]["con"] = results[i].actor_id
    
    # Build all_id list
    all_id = []
    for walker in walkers_list:
        all_id.append(walker["con"])
        all_id.append(walker["id"])
    
    # Initialize walker controllers
    all_actors = world.get_actors(all_id)
    world.set_pedestrians_cross_factor(percentagePedestriansCrossing)
    
    for i in range(0, len(all_id), 2):
        all_actors[i].start()
        all_actors[i].go_to_location(
            world.get_random_location_from_navigation()
        )
        all_actors[i].set_max_speed(float(walker_speed[int(i/2)]))
    
    logging.info(f"Spawned {len(walkers_list)} walkers")
    return walkers_list, all_id


def spawn_cameras(
    world,
    map_name: str
) -> Tuple:
    """Spawn DVS and RGB cameras at predefined locations.
    
    Args:
        world: CARLA world object
        map_name: Name of the map for camera positioning
    
    Returns:
        Tuple of (camera_dvs, camera_rgb, image_queue, rgb_image_queue,
                  image_w, image_h, fov, K)
    """
    # DVS camera
    camera_bp = world.get_blueprint_library().find('sensor.camera.dvs')
    
    # Apply DVS settings from config
    dvs_config = get_nested(CONFIG, 'camera', 'dvs', default={})
    camera_bp.set_attribute(
        'positive_threshold',
        str(dvs_config.get('positive_threshold', '0.7'))
    )
    camera_bp.set_attribute(
        'negative_threshold',
        str(dvs_config.get('negative_threshold', '0.7'))
    )
    camera_bp.set_attribute(
        'sigma_positive_threshold',
        str(dvs_config.get('sigma_positive_threshold', '0.7'))
    )
    camera_bp.set_attribute(
        'sigma_negative_threshold',
        str(dvs_config.get('sigma_negative_threshold', '0.7'))
    )
    camera_bp.set_attribute(
        'refractory_period_ns',
        str(dvs_config.get('refractory_period_ns', '330000'))
    )
    
    camera_dvs = world.spawn_actor(camera_bp, carla.Transform())
    cam_loc, cam_rot = getCamXforms(map_name)
    camera_dvs.set_transform(
        carla.Transform(location=cam_loc, rotation=cam_rot)
    )
    
    image_queue = queue.Queue()
    camera_dvs.listen(image_queue.put)
    
    image_w = camera_bp.get_attribute("image_size_x").as_int()
    image_h = camera_bp.get_attribute("image_size_y").as_int()
    fov = camera_bp.get_attribute("fov").as_float()
    K = build_projection_matrix(image_w, image_h, fov)
    
    # RGB camera
    camera_rgb_bp = world.get_blueprint_library().find('sensor.camera.rgb')
    camera_rgb = world.spawn_actor(camera_rgb_bp, carla.Transform())
    camera_rgb.set_transform(
        carla.Transform(location=cam_loc, rotation=cam_rot)
    )
    
    rgb_image_queue = queue.Queue()
    camera_rgb.listen(rgb_image_queue.put)
    
    # Move spectator for debugging
    spectator = world.get_spectator()
    spectator.set_transform(
        carla.Transform(location=cam_loc, rotation=cam_rot)
    )
    
    logging.info(f"Spawned cameras at {map_name} location")
    return (camera_dvs, camera_rgb, image_queue, rgb_image_queue,
            image_w, image_h, fov, K)


def destroy_actors(
    client,
    world,
    vehicles_list: List[int],
    all_id: List[int],
    cameras: List
) -> None:
    """Destroy all actors in the simulation.
    
    Args:
        client: CARLA client
        world: CARLA world
        vehicles_list: List of vehicle actor IDs
        all_id: List of all walker and controller IDs
        cameras: List of camera actors to destroy
    """
    # Stop walker controllers
    if all_id:
        all_actors = world.get_actors(all_id)
        for i in range(0, len(all_id), 2):
            try:
                all_actors[i].stop()
            except Exception as e:
                logging.warning(f"Error stopping walker controller: {e}")
    
    # Destroy vehicles
    if vehicles_list:
        logging.info(f'Destroying {len(vehicles_list)} vehicles')
        client.apply_batch(
            [carla.command.DestroyActor(x) for x in vehicles_list]
        )
    
    # Destroy walkers
    if all_id:
        logging.info(f'Destroying {len(all_id)//2} walkers')
        client.apply_batch(
            [carla.command.DestroyActor(x) for x in all_id]
        )
    
    # Destroy cameras
    for camera in cameras:
        if camera is not None:
            try:
                camera.destroy()
            except Exception as e:
                logging.warning(f"Error destroying camera: {e}")
    
    logging.info("All actors destroyed")


def get_actor_blueprints(world, filter: str, generation: str) -> List:
    """Get actor blueprints filtered by type and generation.
    
    Args:
        world: CARLA world object
        filter: Blueprint filter string (e.g., 'vehicle.*')
        generation: Generation filter ('1', '2', or 'all')
    
    Returns:
        List of filtered actor blueprints
    """
    bps = world.get_blueprint_library().filter(filter)

    if generation.lower() == "all":
        return bps

    # If the filter returns only one bp, we assume that this one needed
    # and therefore, we ignore the generation
    if len(bps) == 1:
        return bps

    try:
        int_generation = int(generation)
        # Check if generation is in available generations
        if int_generation in [1, 2]:
            bps = [x for x in bps if int(x.get_attribute('generation')) == int_generation]
            return bps
        else:
            logging.warning(
                f"Actor generation '{generation}' is not valid (must be 1 or 2). "
                "No actors will be spawned."
            )
            return []
    except ValueError:
        logging.warning(
            f"Actor generation '{generation}' is not a valid integer. "
            "No actors will be spawned."
        )
        return []

def main() -> None:
    """Main function to run CARLA simulation with pose estimation data generation.
    
    Spawns vehicles and pedestrians in CARLA, captures DVS and RGB images,
    generates ground truth pose annotations in YOLO format, and resets
    the simulation periodically with new random weather.
    """
    global CONFIG
    
    argparser = argparse.ArgumentParser(
        description=__doc__)

    # Config file argument (processed first)
    argparser.add_argument(
        '--config',
        metavar='PATH',
        default='configs/default.yaml',
        help='Path to YAML config file (default: configs/default.yaml)')
    argparser.add_argument(
        '-n', '--number-of-vehicles',
        metavar='N',
        default=None,
        type=int,
        help='Number of vehicles (overrides config)')
    argparser.add_argument(
        '-w', '--number-of-walkers',
        metavar='W',
        default=None,
        type=int,
        help='Number of walkers (overrides config)')
    argparser.add_argument(
        '--safe',
        action='store_true',
        help='Avoid spawning vehicles prone to accidents')
    argparser.add_argument(
        '--tm-port',
        metavar='P',
        default=None,
        type=int,
        help='Port to communicate with TM (overrides config)')
    argparser.add_argument(
        '--asynch',
        action='store_true',
        help='Activate asynchronous mode execution')
    argparser.add_argument(
        '--hybrid',
        action='store_true',
        help='Activate hybrid mode for Traffic Manager')
    argparser.add_argument(
        '-s', '--seed',
        metavar='S',
        type=int,
        help='Set random device seed and deterministic mode for Traffic Manager')
    argparser.add_argument(
        '--seedw',
        metavar='S',
        default=None,
        type=int,
        help='Set the seed for pedestrians module (overrides config)')
    argparser.add_argument(
        '--out-dir',        
        help='Output directory')

    args = argparser.parse_args()
    
    # Load config and apply CLI overrides
    logging.basicConfig(
        format='%(levelname)s: %(message)s', level=logging.INFO
    )
    
    CONFIG = load_config(args.config)
    CONFIG = apply_cli_overrides(CONFIG, args)
    logging.info(f"Loaded config from: {args.config}")

    global OUT_DIR
    # MAP_NAME = 'Town10HD'
    MAP_NAME = get_nested(CONFIG, 'carla', 'map', default='Town10HD')
    logging.info(f"Using map: {MAP_NAME}")
    
    # Get output directory from config or CLI
    out_dir = get_nested(CONFIG, 'output', 'directory', default='')
    if args.out_dir is not None:
        out_dir = args.out_dir
    assert out_dir, "Please specify an output directory via --out-dir or config"
    
    OUT_DIR = f"{out_dir}/{MAP_NAME}"
    os.makedirs(OUT_DIR, exist_ok=True)

    for dirname in ['events', 'RGB', 'GT', 'Annot']:
        os.makedirs(f"{OUT_DIR}/{dirname}", exist_ok=True)

    # Clean up old data
    for dirname in ['events', 'RGB', 'GT']:
        for file in glob.glob(f"{OUT_DIR}/{dirname}/*"):
            try:
                os.remove(file)
            except Exception as e:
                logging.warning(f"Could not remove {file}: {e}")

    vehicles_list = []
    walkers_list = []
    all_id = []
    
    # Get CARLA connection from config
    host = get_nested(CONFIG, 'carla', 'host', default='localhost')
    port = get_nested(CONFIG, 'carla', 'port', default=2000)
    client = carla.Client(host, port)
    client.set_timeout(40.0)
    synchronous_master = False
    
    seed = get_nested(CONFIG, 'carla', 'seed', default=None)
    random.seed(seed if seed is not None else int(time.time()))

    try:        
        client.load_world(MAP_NAME)
        world = client.get_world()
        world.set_weather(generate_random_weather())

        tm_port = get_nested(CONFIG, 'carla', 'tm_port', default=8000)
        traffic_manager = client.get_trafficmanager(tm_port)
        traffic_manager.set_global_distance_to_leading_vehicle(2.5)
        
        if get_nested(CONFIG, 'actors', 'vehicles', 'respawn', default=False):
            traffic_manager.set_respawn_dormant_vehicles(True)
        if get_nested(CONFIG, 'carla', 'hybrid', default=False):
            traffic_manager.set_hybrid_physics_mode(True)
            traffic_manager.set_hybrid_physics_radius(70.0)
        if seed is not None:
            traffic_manager.set_random_device_seed(seed)

        settings = world.get_settings()
        asynch_mode = get_nested(CONFIG, 'carla', 'asynch', default=False)
        if not asynch_mode:
            traffic_manager.set_synchronous_mode(True)
            if not settings.synchronous_mode:
                synchronous_master = True
                settings.synchronous_mode = True
                settings.fixed_delta_seconds = 0.05
            else:
                synchronous_master = False
        else:
            print("You are currently in asynchronous mode. If this is a traffic simulation, \
            you could experience some issues. If it's not working correctly, switch to synchronous \
            mode by using traffic_manager.set_synchronous_mode(True)")

        if get_nested(CONFIG, 'carla', 'no_rendering', default=False):
            settings.no_rendering_mode = True
        world.apply_settings(settings)

        # Get actor blueprints from config
        vehicle_filter = get_nested(CONFIG, 'actors', 'vehicles', 'filter', default='vehicle.*')
        vehicle_gen = get_nested(CONFIG, 'actors', 'vehicles', 'generation', default='All')
        walker_filter = get_nested(CONFIG, 'actors', 'walkers', 'filter', default='walker.pedestrian.*')
        walker_gen = get_nested(CONFIG, 'actors', 'walkers', 'generation', default='2')
        
        blueprints = get_actor_blueprints(world, vehicle_filter, vehicle_gen)
        blueprintsWalkers = get_actor_blueprints(world, walker_filter, walker_gen)

        if get_nested(CONFIG, 'actors', 'vehicles', 'safe', default=False):
            blueprints = [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == 4]
            blueprints = [x for x in blueprints if not x.id.endswith('microlino')]
            blueprints = [x for x in blueprints if not x.id.endswith('carlacola')]
            blueprints = [x for x in blueprints if not x.id.endswith('cybertruck')]
            blueprints = [x for x in blueprints if not x.id.endswith('t2')]
            blueprints = [x for x in blueprints if not x.id.endswith('sprinter')]
            blueprints = [x for x in blueprints if not x.id.endswith('firetruck')]
            blueprints = [x for x in blueprints if not x.id.endswith('ambulance')]

        blueprints = sorted(blueprints, key=lambda bp: bp.id)

        spawn_points = world.get_map().get_spawn_points()
        number_of_spawn_points = len(spawn_points)
        
        num_vehicles = get_nested(CONFIG, 'actors', 'vehicles', 'count', default=30)
        if num_vehicles < number_of_spawn_points:
            random.shuffle(spawn_points)
        elif num_vehicles > number_of_spawn_points:
            msg = 'requested %d vehicles, but could only find %d spawn points'
            logging.warning(msg, num_vehicles, number_of_spawn_points)
            num_vehicles = number_of_spawn_points

        # Initial weather setup
        initial_weather = generate_random_weather()
        world.set_weather(initial_weather)
        
        # Spawn actors using refactored functions
        vehicles_list = spawn_vehicles(
            world, client, CONFIG, blueprints, spawn_points,
            traffic_manager, synchronous_master
        )
        
        walkers_list, all_id = spawn_walkers(
            world, client, CONFIG, blueprintsWalkers
        )
        
        (camera, camera_rgb, image_queue, rgb_image_queue,
         image_w, image_h, _, K) = spawn_cameras(world, MAP_NAME)
        
        # Get pedestrian actors for pose generation
        peds = [x for x in world.get_actors() if 'pedestrian' in x.type_id]

        # Wait for a tick to ensure client receives last transforms
        if asynch_mode or not synchronous_master:
            world.wait_for_tick()
        else:
            world.tick()

        print(f'Spawned {len(vehicles_list)} vehicles and '
              f'{len(walkers_list)} walkers. Press Ctrl+C to exit.')

        # Traffic Manager parameters
        tm_speed_diff = get_nested(
            CONFIG, 'traffic_manager', 'global_speed_difference', default=30.0
        )
        traffic_manager.global_percentage_speed_difference(tm_speed_diff)

        # Initialize simulation timer for weather resets
        simulation_start_time = time.time()
        num_frames = get_nested(CONFIG, 'simulation', 'num_frames', default=12000)
        
        # Data collection cooldown after setup/reset
        data_collection_cooldown = get_nested(
            CONFIG, 'simulation', 'data_collection_cooldown', default=5.0
        )
        weather_reset_interval = get_nested(
            CONFIG, 'simulation', 'weather_reset_interval', default=20
        )
        last_reset_time = time.time()
        
        while num_frames >= 0:
            # Check if reset interval has elapsed
            elapsed_time = time.time() - simulation_start_time
            if elapsed_time >= weather_reset_interval:
                logging.info(
                    f"Resetting simulation after {elapsed_time:.1f} seconds"
                )
                
                # Destroy all current actors
                destroy_actors(
                    client, world, vehicles_list, all_id,
                    [camera, camera_rgb]
                )
                
                # Generate new random weather
                new_weather = generate_random_weather()
                world.set_weather(new_weather)
                
                # Respawn all actors
                vehicle_spawn_points = world.get_map().get_spawn_points()
                random.shuffle(vehicle_spawn_points)
                
                vehicles_list = spawn_vehicles(
                    world, client, CONFIG, blueprints,
                    vehicle_spawn_points, traffic_manager,
                    synchronous_master
                )
                
                walkers_list, all_id = spawn_walkers(
                    world, client, CONFIG, blueprintsWalkers
                )
                
                (camera, camera_rgb, image_queue, rgb_image_queue,
                 image_w, image_h, _, K) = spawn_cameras(
                    world, MAP_NAME
                )
                
                # Update pedestrian list
                peds = [
                    x for x in world.get_actors()
                    if 'pedestrian' in x.type_id
                ]
                
                # Reset timer
                simulation_start_time = time.time()
                
                # Wait for tick after reset
                if asynch_mode or not synchronous_master:
                    world.wait_for_tick()
                else:
                    world.tick()
                
                logging.info("Simulation reset complete")
                
                # Reset data collection cooldown
                last_reset_time = time.time()
            
            # Normal simulation tick
            if not asynch_mode and synchronous_master:
                world.tick()
            else:
                world.wait_for_tick()
            
            # Process images from queues (always consume to prevent backlog)
            image = image_queue.get()
            rgb_image = rgb_image_queue.get()
            
            # Skip data collection during cooldown period after reset
            if time.time() - last_reset_time < data_collection_cooldown:
                continue
            
            # Process and save data
            ProcessDVSImage(image)
            ProcessRGBImage(rgb_image)
            GenerateGTPose(image, image_h, image_w, K, camera, peds)
            num_frames -= 1

    finally:
        # Clean up simulation settings
        if not asynch_mode and synchronous_master:
            settings = world.get_settings()
            settings.synchronous_mode = False
            settings.no_rendering_mode = False
            settings.fixed_delta_seconds = None
            world.apply_settings(settings)

        # Destroy all actors
        destroy_actors(
            client, world, vehicles_list, all_id, [camera, camera_rgb]
        )


        time.sleep(0.5)

if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('\ndone.')
