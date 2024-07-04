import glob
import os
import sys
import random
import time
import numpy as np
import cv2
import carla

try:
    sys.path.append(glob.glob('./dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

SHOW_PREVIEW = False

class CarEnv:
    STEER_AMT = 1.0
    SHOW_CAM = SHOW_PREVIEW
    SECONDS_PER_EPISODE = 240
    IMG_WIDTH = 200
    IMG_HEIGHT = 88
    front_camera = None

    def __init__(self):
        self.client = carla.Client("127.0.0.1", 2000)
        self.client.set_timeout(2.0)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self.car = self.blueprint_library.filter("model3")[0]
        self.reset()

    def reset(self):
        self.actor_list = []

        # Randomized spawn point
        self.transform = random.choice(self.world.get_map().get_spawn_points())

        # Spawn vehicle agent
        self.vehicle = self.world.spawn_actor(self.car, self.transform)
        self.actor_list.append(self.vehicle)

        # Add debug label at spawn point
        self.world.debug.draw_string(self.transform.location, "Start Point", life_time=60)

        # Enable autopilot without following traffic laws
        self.vehicle.set_autopilot(True, self.client.get_trafficmanager(8000).get_port())
        self.client.get_trafficmanager(8000).set_global_distance_to_leading_vehicle(1.0)
        self.client.get_trafficmanager(8000).global_percentage_speed_difference(10.0)  # Reduce speed further

        # Turn on all traffic lights
        for actor in self.world.get_actors():
            if actor.type_id.startswith('traffic.traffic_light'):
                actor.set_state(carla.TrafficLightState.Green)
                actor.freeze(True)

        # Spawn camera
        self.rgb_cam = self.blueprint_library.find("sensor.camera.rgb")
        self.rgb_cam.set_attribute("image_size_x", f"{self.IMG_WIDTH}")
        self.rgb_cam.set_attribute("image_size_y", f"{self.IMG_HEIGHT}")
        self.rgb_cam.set_attribute("fov", f"110")
        cam_transform = carla.Transform(carla.Location(x=2.5, z=1))
        self.rgb_sensor = self.world.spawn_actor(self.rgb_cam, cam_transform, attach_to=self.vehicle)
        self.actor_list.append(self.rgb_sensor)
        self.rgb_sensor.listen(lambda data: self.process_img(data))

        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        time.sleep(4)

        while self.front_camera is None:
            time.sleep(0.01)

        self.episode_start = time.time()
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.1, brake=0.0))  # Reduce throttle
        return self.front_camera

    def cleanup(self):
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.actor_list])

    def process_img(self, image):
        i = np.array(image.raw_data)
        i2 = i.reshape((self.IMG_HEIGHT, self.IMG_WIDTH, 4))
        i3 = i2[:, :, :3]
        if self.SHOW_CAM:
            cv2.imshow("", i3)
            cv2.waitKey(1)
        self.front_camera = i3

    def step(self):
        done = False

        if self.episode_start + self.SECONDS_PER_EPISODE < time.time():
            self.episode_end = time.time()
            done = True

        return self.front_camera, done, None

if __name__ == "__main__":
    FPS = 60
    random.seed(10)
    np.random.seed(10)

    env = CarEnv()
    throttle = 0.1  # Reduce throttle
    brake = 0.0

    while True:
        time.sleep(1 / FPS)
        next_image, done, _ = env.step()
        current_state = next_image

        if done:
            break

    env.cleanup()
    print(f'Car drove for {env.episode_end - env.episode_start} sec')