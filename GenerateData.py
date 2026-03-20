#!/usr/bin/env python

# Copyright (c) 2021 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""Example script to generate traffic in the simulation"""

import glob
import os
import sys
import time

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

OUT_DIR= ""
# sys.path.append("/opt/carla-simulator/PythonAPI/examples")
sys.path.append("../")
from draw_skeleton import get_screen_points, draw_skeleton, draw_points_on_buffer

import carla

from carla import VehicleLightState as vls

import argparse
import logging
from numpy import random
import cv2

import numpy as np
import queue
import copy

def getCamXforms(map_name):
    if map_name == 'Town03':
        return carla.Location(x=71.634972, y=-213.649551, z=0.151049) , \
            carla.Rotation(pitch=0.000000, yaw=-87.941040, roll=0.000000)

    elif map_name== 'Town05':
        return carla.Location(x=-175.693878, y=76.624390, z=5.408956), \
                carla.Rotation(pitch=-16.103148, yaw=137.193390, roll=0.000127)

    elif map_name==  'Town04':
        return carla.Location(x=193.145935, y=-257.109344, z=6.300411), \
                carla.Rotation(pitch=-30.599081, yaw=49.880680, roll=0.000091)

    elif map_name== 'Town07':
        return carla.Location(x=8.499461, y=4.675543, z=1.850266) , \
                carla.Rotation(pitch=3.275424, yaw=-145.442154, roll=0.000480)
    
    elif map_name== 'Town07_Opt':
        # return carla.Location(x=5.589095, y=6.237842, z=9.058050) , \
        #         carla.Rotation(pitch=-39.062405, yaw=-132.940796, roll=0.000443)
        return carla.Location(x=3.249976, y=5.709602, z=2.096144),\
            carla.Rotation(pitch=-19.308350, yaw=-141.595810, roll=0.000442)
    
    elif map_name == 'Town10HD':
        return  carla.Location(x=-35.280499, y=-0.017508, z=1.532597), \
                carla.Rotation(pitch=4.182043, yaw=130.544815, roll=0.000170)

KEYPOINTS= ["crl_Head__C", "crl_eye__R", "crl_eye__L", "crl_shoulder__R", "crl_shoulder__L", \
            "crl_arm__R", "crl_arm__L", "crl_foreArm__R", "crl_foreArm__L", \
             "crl_hips__C", "crl_thigh__R", "crl_thigh__L", \
             "crl_leg__R", "crl_leg__L",  "crl_foot__R", "crl_foot__L"   ]

def GenerateGTPose(image,image_h,image_w, K,camera, peds):
    buf= np.zeros((image_h, image_w, 3), dtype=np.uint8)

    num_peds= 0
    kp_structs= []
    for ped in peds:
        dist = ped.get_transform().location.distance(camera.get_transform().location)
        if dist < 50.0:            
            try:
                forward_vec = camera.get_transform().get_forward_vector()
                ray = ped.get_transform().location - camera.get_transform().location

                if forward_vec.dot(ray) > 0:
                    bones= ped.get_bones()
                    boneIndex = { x.name:i for i,x in enumerate(bones.bone_transforms)  }          
                    points= [x.world.location for x in bones.bone_transforms]   

                    points2d = get_screen_points(camera, K, image_w, image_h, points)                    

                    locs= np.array( [ points2d[boneIndex[x]] for x in KEYPOINTS ])
                    if np.any(locs<0):
                        # print("NEG")
                        continue
                    
                    draw_skeleton(buf, image_w, image_h, boneIndex, points2d, (0, 255, 0), 3)
                    kp_structs.append(locs)
                    num_peds += 1


            except Exception as E:
                print(E)
    cv2.imwrite(f"{OUT_DIR}/GT/{image.frame}.png", buf)

    # print(f"WROTE {image.frame}_GT.png ")

    # TODO
    # one txt file per image: <class_idx> <px1> <py1> <px2> <py2> .... <pxn> <pyn> per line
    
    with open(f"{OUT_DIR}/Annot/{image.frame}.txt",'w') as F:        
        for person in kp_structs:
            F.write(f"0 ")
            for pnt in person: 
                F.write(f"{int(pnt[0])} {int(pnt[1])} ")
            F.write("\n")
    
    # change keypoint format to COCO-pose and 



def ProcessDVSImage(image):
    dvs_events = np.frombuffer(image.raw_data, dtype=np.dtype([
            ('x', np.uint16), ('y', np.uint16), ('t', np.int64), ('pol', np.bool)]))     

    dvs_img = np.zeros((image.height, image.width, 3), dtype=np.uint8)
    dvs_img[dvs_events[:]['y'], dvs_events[:]['x'], dvs_events[:]['pol'] * 2] = 255   
    array2 = copy.deepcopy(dvs_img)

    cv2.imwrite(f"{OUT_DIR}/events/{image.frame}.png", array2)
    print(f"{OUT_DIR}/events/{image.frame}.png")

def ProcessRGBImage(image):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    # make the array writeable doing a deep copy
    array2 = copy.deepcopy(array)
    cv2.imwrite(f"{OUT_DIR}/RGB/{image.frame}_RGB.png", array2)

def build_projection_matrix(w, h, fov):
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K

def get_actor_blueprints(world, filter, generation):
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
            print("   Warning! Actor Generation is not valid. No actor will be spawned.")
            return []
    except:
        print("   Warning! Actor Generation is not valid. No actor will be spawned.")
        return []

def main():
    argparser = argparse.ArgumentParser(
        description=__doc__)

    argparser.add_argument(
        '--map_name',        
        default='town05' )

    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-n', '--number-of-vehicles',
        metavar='N',
        default=30,
        type=int,
        help='Number of vehicles (default: 30)')
    argparser.add_argument(
        '-w', '--number-of-walkers',
        metavar='W',
        default=10,
        type=int,
        help='Number of walkers (default: 10)')
    argparser.add_argument(
        '--safe',
        action='store_true',
        help='Avoid spawning vehicles prone to accidents')
    argparser.add_argument(
        '--filterv',
        metavar='PATTERN',
        default='vehicle.*',
        help='Filter vehicle model (default: "vehicle.*")')
    argparser.add_argument(
        '--generationv',
        metavar='G',
        default='All',
        help='restrict to certain vehicle generation (values: "1","2","All" - default: "All")')
    argparser.add_argument(
        '--filterw',
        metavar='PATTERN',
        default='walker.pedestrian.*',
        help='Filter pedestrian type (default: "walker.pedestrian.*")')
    argparser.add_argument(
        '--generationw',
        metavar='G',
        default='2',
        help='restrict to certain pedestrian generation (values: "1","2","All" - default: "2")')
    argparser.add_argument(
        '--tm-port',
        metavar='P',
        default=8000,
        type=int,
        help='Port to communicate with TM (default: 8000)')
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
        default=0,
        type=int,
        help='Set the seed for pedestrians module')
    argparser.add_argument(
        '--car-lights-on',
        action='store_true',
        default=False,
        help='Enable automatic car light management')
    argparser.add_argument(
        '--hero',
        action='store_true',
        default=False,
        help='Set one of the vehicles as hero')
    argparser.add_argument(
        '--respawn',
        action='store_true',
        default=False,
        help='Automatically respawn dormant vehicles (only in large maps)')
    argparser.add_argument(
        '--no-rendering',
        action='store_true',
        default=False,
        help='Activate no rendering mode')

    args = argparser.parse_args()

    global OUT_DIR
    MAP_NAME= 'Town10HD'
    print(MAP_NAME)
    OUT_DIR= f"/home/local/ASUAD/kchanda3/carlaScripts/{MAP_NAME}"
    os.system(f"mkdir -p {OUT_DIR}")

    for dirname in ['events','RGB','GT', 'Annot']:
        os.system(f"mkdir -p {OUT_DIR}/{dirname}")

    os.system(f"rm {OUT_DIR}/events/*")
    os.system(f"rm {OUT_DIR}/RGB/*")
    os.system(f"rm {OUT_DIR}/GT/*")

    # RAINY
    weather_rainy = carla.WeatherParameters(
        cloudiness=1.0,
        precipitation=70.0,
        precipitation_deposits=70.0,
        sun_altitude_angle=10.0,
        sun_azimuth_angle = 90.0,        
        wind_intensity = 0.0,
        fog_density = 0.5,
        wetness = 70.0,
    )            

    # FOGGY
    weather_foggy = carla.WeatherParameters(
        cloudiness= 30.0,
        precipitation=00.0,
        precipitation_deposits=0.0,
        sun_altitude_angle=10.0,
        sun_azimuth_angle = 90.0,        
        wind_intensity = 0.0,
        fog_density = 40.0,
        wetness = 0.0,
    )

    # TWILIGHT
    weather_twilight = carla.WeatherParameters(
        cloudiness= 0.0,
        precipitation=0.0,
        precipitation_deposits=0.0,
        sun_altitude_angle=10.0,
        sun_azimuth_angle = 30.0,        
        wind_intensity = 0.0,
        fog_density = 0.0,
        wetness = 0.0,
    )

    # NIGHT
    weather_twilight = carla.WeatherParameters(
        cloudiness= 0.0,
        precipitation=0.0,
        precipitation_deposits=0.0,
        sun_altitude_angle= -10.0,
        sun_azimuth_angle = 90.0,        
        wind_intensity = 0.0,
        fog_density = 0.0,
        wetness = 0.0,
    )



    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    vehicles_list = []
    walkers_list = []
    all_id = []
    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    synchronous_master = False
    random.seed(args.seed if args.seed is not None else int(time.time()))

    try:
        
        client.load_world(MAP_NAME)
        world = client.get_world()
        world.set_weather(weather_twilight)

        traffic_manager = client.get_trafficmanager(args.tm_port)
        traffic_manager.set_global_distance_to_leading_vehicle(2.5)
        if args.respawn:
            traffic_manager.set_respawn_dormant_vehicles(True)
        if args.hybrid:
            traffic_manager.set_hybrid_physics_mode(True)
            traffic_manager.set_hybrid_physics_radius(70.0)
        if args.seed is not None:
            traffic_manager.set_random_device_seed(args.seed)

        settings = world.get_settings()
        if not args.asynch:
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

        if args.no_rendering:
            settings.no_rendering_mode = True
        world.apply_settings(settings)

        blueprints = get_actor_blueprints(world, args.filterv, args.generationv)
        blueprintsWalkers = get_actor_blueprints(world, args.filterw, args.generationw)

        if args.safe:
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

        if args.number_of_vehicles < number_of_spawn_points:
            random.shuffle(spawn_points)
        elif args.number_of_vehicles > number_of_spawn_points:
            msg = 'requested %d vehicles, but could only find %d spawn points'
            logging.warning(msg, args.number_of_vehicles, number_of_spawn_points)
            args.number_of_vehicles = number_of_spawn_points

        # @todo cannot import these directly.
        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        FutureActor = carla.command.FutureActor

        # --------------
        # Spawn vehicles
        # --------------
        batch = []
        hero = args.hero
        for n, transform in enumerate(spawn_points):
            if n >= args.number_of_vehicles:
                break
            blueprint = random.choice(blueprints)
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            if blueprint.has_attribute('driver_id'):
                driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
                blueprint.set_attribute('driver_id', driver_id)
            if hero:
                blueprint.set_attribute('role_name', 'hero')
                hero = False
            else:
                blueprint.set_attribute('role_name', 'autopilot')

            # spawn the cars and set their autopilot and light state all together
            batch.append(SpawnActor(blueprint, transform)
                .then(SetAutopilot(FutureActor, True, traffic_manager.get_port())))

        for response in client.apply_batch_sync(batch, synchronous_master):
            if response.error:
                logging.error(response.error)
            else:
                vehicles_list.append(response.actor_id)

        # Set automatic vehicle lights update if specified
        if args.car_lights_on:
            all_vehicle_actors = world.get_actors(vehicles_list)
            for actor in all_vehicle_actors:
                traffic_manager.update_vehicle_lights(actor, True)

        # -------------
        # Spawn Walkers
        # -------------
        # some settings
        percentagePedestriansRunning = 0.0      # how many pedestrians will run
        percentagePedestriansCrossing = 0.0     # how many pedestrians will walk through the road
        if args.seedw:
            world.set_pedestrians_seed(args.seedw)
            random.seed(args.seedw)
        # 1. take all the random locations to spawn
        spawn_points = []
        for i in range(args.number_of_walkers):
            spawn_point = carla.Transform()
            loc = world.get_random_location_from_navigation()
            if (loc != None):
                spawn_point.location = loc
                spawn_points.append(spawn_point)
        # 2. we spawn the walker object
        batch = []
        walker_speed = []
        for spawn_point in spawn_points:
            walker_bp = random.choice(blueprintsWalkers)
            # set as not invincible
            if walker_bp.has_attribute('is_invincible'):
                walker_bp.set_attribute('is_invincible', 'false')
            # set the max speed
            if walker_bp.has_attribute('speed'):
                if (random.random() > percentagePedestriansRunning):
                    # walking
                    walker_speed.append(walker_bp.get_attribute('speed').recommended_values[1])
                else:
                    # running
                    walker_speed.append(walker_bp.get_attribute('speed').recommended_values[2])
            else:
                print("Walker has no speed")
                walker_speed.append(0.0)
            batch.append(SpawnActor(walker_bp, spawn_point))
        results = client.apply_batch_sync(batch, True)
        walker_speed2 = []
        for i in range(len(results)):
            if results[i].error:
                logging.error(results[i].error)
            else:
                walkers_list.append({"id": results[i].actor_id})
                walker_speed2.append(walker_speed[i])
        walker_speed = walker_speed2
        # 3. we spawn the walker controller
        batch = []
        walker_controller_bp = world.get_blueprint_library().find('controller.ai.walker')
        for i in range(len(walkers_list)):
            batch.append(SpawnActor(walker_controller_bp, carla.Transform(), walkers_list[i]["id"]))
        results = client.apply_batch_sync(batch, True)
        for i in range(len(results)):
            if results[i].error:
                logging.error(results[i].error)
            else:
                walkers_list[i]["con"] = results[i].actor_id
        # 4. we put together the walkers and controllers id to get the objects from their id
        for i in range(len(walkers_list)):
            all_id.append(walkers_list[i]["con"])
            all_id.append(walkers_list[i]["id"])
        all_actors = world.get_actors(all_id)

        # SPAWN DVS CAMERA
        camera_bp = world.get_blueprint_library().find('sensor.camera.dvs')
        camera_bp.set_attribute('positive_threshold', '0.7')
        camera_bp.set_attribute('negative_threshold', '0.7')
        camera_bp.set_attribute('sigma_positive_threshold', '0.7')
        camera_bp.set_attribute('sigma_negative_threshold', '0.7')
        camera_bp.set_attribute('refractory_period_ns', '330000')

        camera = world.spawn_actor(camera_bp, carla.Transform())
        
        # cam_loc= carla.Location(x=-35.280499, y=-0.017508, z=1.532597) 
        # cam_rot= carla.Rotation(pitch=4.182043, yaw=130.544815, roll=0.000170)
        cam_loc, cam_rot = getCamXforms(MAP_NAME)
        camera.set_transform(carla.Transform(location=cam_loc, rotation=cam_rot ) )
        image_queue = queue.Queue()
        camera.listen(image_queue.put)

        image_w = camera_bp.get_attribute("image_size_x").as_int()
        image_h = camera_bp.get_attribute("image_size_y").as_int()
        fov = camera_bp.get_attribute("fov").as_float()
        K = build_projection_matrix(image_w, image_h, fov)
        
        # SPAWN RPG CAMERA
        camera_rgb_bp = world.get_blueprint_library().find('sensor.camera.rgb')
        camera_rgb = world.spawn_actor(camera_rgb_bp, carla.Transform())
        camera_rgb.set_transform(carla.Transform(location=cam_loc, rotation=cam_rot ) )
        rgb_image_queue = queue.Queue()
        camera_rgb.listen(rgb_image_queue.put)


        # MOVE SPECTATOR (FOR DEBUGGING)
        spectator= world.get_spectator()
        spectator.set_transform(carla.Transform(location=cam_loc, rotation=cam_rot ) )

        peds= [x for x in world.get_actors() if 'pedestrian' in x.type_id ]

        # wait for a tick to ensure client receives the last transform of the walkers we have just created
        if args.asynch or not synchronous_master:
            world.wait_for_tick()
        else:
            world.tick()

        # 5. initialize each controller and set target to walk to (list is [controler, actor, controller, actor ...])
        # set how many pedestrians can cross the road
        world.set_pedestrians_cross_factor(percentagePedestriansCrossing)
        for i in range(0, len(all_id), 2):
            # start walker
            all_actors[i].start()
            # set walk to random point
            all_actors[i].go_to_location(world.get_random_location_from_navigation())
            # max speed
            all_actors[i].set_max_speed(float(walker_speed[int(i/2)]))

        print('spawned %d vehicles and %d walkers, press Ctrl+C to exit.' % (len(vehicles_list), len(walkers_list)))

        # Example of how to use Traffic Manager parameters
        traffic_manager.global_percentage_speed_difference(30.0)

        num_frames= 12000
        while num_frames>=0 :                    
            if not args.asynch and synchronous_master:
                world.tick()
            else:
                # print("Waiting for tick...")
                world.wait_for_tick()
            image = image_queue.get()
            rgb_image= rgb_image_queue.get()
            ProcessDVSImage(image)
            ProcessRGBImage(rgb_image)         
            GenerateGTPose(image,image_h, image_w, K,camera, peds)
            num_frames-= 1
            # print(f"WROTE {image.frame}")

    finally:

        if not args.asynch and synchronous_master:
            settings = world.get_settings()
            settings.synchronous_mode = False
            settings.no_rendering_mode = False
            settings.fixed_delta_seconds = None
            world.apply_settings(settings)

        print('\ndestroying %d vehicles' % len(vehicles_list))
        client.apply_batch([carla.command.DestroyActor(x) for x in vehicles_list])

        # stop walker controllers (list is [controller, actor, controller, actor ...])
        for i in range(0, len(all_id), 2):
            all_actors[i].stop()

        print('\ndestroying %d walkers' % len(walkers_list))
        client.apply_batch([carla.command.DestroyActor(x) for x in all_id])

        camera.destroy()


        time.sleep(0.5)

if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('\ndone.')
