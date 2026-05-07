# -*- coding: utf-8 -*-
"""
Bird's-eye RGB camera viewer for a CARLA vehicle, rendered with pygame.

"""

import numpy as np

try:
    import pygame
except ImportError:
    raise RuntimeError("pygame is required: pip install pygame")

import carla


class BirdseyeCamera:
    """Attaches a downward-facing RGB camera to a vehicle and shows it
    in a pygame window."""

    def __init__(self, world, vehicle, *,
                 width=800, height=600, z=50, fov=90,
                 window_title="Bird's-Eye View"):
        self.world = world
        self.vehicle = vehicle
        self.width = width
        self.height = height
        self._surface = None
        self._quit = False

        pygame.init()
        pygame.display.set_caption(window_title)
        self.display = pygame.display.set_mode((width, height))
        self.clock = pygame.time.Clock()

        bp_lib = world.get_blueprint_library()
        cam_bp = bp_lib.find('sensor.camera.rgb')
        cam_bp.set_attribute('image_size_x', str(width))
        cam_bp.set_attribute('image_size_y', str(height))
        cam_bp.set_attribute('fov', str(fov))

        cam_transform = carla.Transform(
            carla.Location(x=0, y=0, z=z),
            carla.Rotation(pitch=-90, yaw=0, roll=0))

        self.sensor = world.spawn_actor(
            cam_bp, cam_transform, attach_to=vehicle)
        self.sensor.listen(self._on_image)

    def _on_image(self, image):
        """Convert CARLA image to a pygame surface."""
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((self.height, self.width, 4))[:, :, :3]
        array = array[:, :, ::-1]
        self._surface = pygame.surfarray.make_surface(
            array.swapaxes(0, 1))

    def tick(self):
        """Process pygame events and blit the latest frame."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._quit = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self._quit = True

        if self._surface is not None:
            self.display.blit(self._surface, (0, 0))
        pygame.display.flip()
        self.clock.tick(30)

    def should_quit(self):
        return self._quit

    def destroy(self):
        """Clean up sensor and pygame."""
        if self.sensor is not None:
            self.sensor.stop()
            self.sensor.destroy()
            self.sensor = None
        pygame.quit()
