import math
import time

import numpy as np
import wgpu
from rendercanvas.glfw import RenderCanvas, loop

from camera2d import Camera2D
from colormaterial import ColorMaterial
from engine import Engine
from node import Node
from pipeline import Pipeline
from renderer import Renderer
from scene import Scene
from shader import Shader
from shape import Shape
from transform import Transform
from texture import Texture
from sampler import Sampler
from textureset import TextureSet
from square import Square


class Disk(Shape):

    def __init__(self, device, segments = 96):
        vertices = [[0.0, 0.0]]
        uv = [[0.5, 0.5]]
        for i in range(segments):
            angle = 2 * math.pi * i / segments
            x = math.cos(angle)
            y = math.sin(angle)
            vertices.append([x, y])
            uv.append([(x + 1) / 2, (1 - y) / 2])

        indices = []
        for i in range(segments):
            indices.extend([0, i + 1, (i + 1) % segments + 1])

        self.index_count = len(indices)
        self.vertex_buffer = device.create_buffer_with_data(
            data=np.array(vertices, dtype="float32"),
            usage=wgpu.BufferUsage.VERTEX,
        )
        self.uv_buffer = device.create_buffer_with_data(
            data=np.array(uv, dtype="float32"), usage=wgpu.BufferUsage.VERTEX,
        )
        self.index_buffer = device.create_buffer_with_data(
            data=np.array(indices, dtype="uint32"),
            usage=wgpu.BufferUsage.INDEX,
        )

    def draw(self, st):
        instance = st.get_shader().commit_matrix(st)
        st.render_pass.set_vertex_buffer(0, self.vertex_buffer)
        st.render_pass.set_vertex_buffer(1, self.uv_buffer)
        st.render_pass.set_index_buffer(self.index_buffer, wgpu.IndexFormat.uint32)
        st.render_pass.draw_indexed(self.index_count, 1, 0, 0, instance)


class Rotate(Engine):

    def __init__(self, transform, degrees_per_second):
        self.transform = transform
        self.speed = degrees_per_second
        self.angle = 0.0

    def update(self, dt):
        self.angle = (self.angle + self.speed * dt) % 360
        self.transform.load_identity()
        self.transform.rotate(self.angle, 0, 0, 1)


def initialize(device, target_format):
    global scene, camera, background_transform
    camera = Camera2D(-10, 10, -10, 10)

    disk = Disk(device)

    sun_transform = Transform()
    sun_transform.scale(1.7, 1.7, 1)
    sun_material = ColorMaterial(2.5, 1.2, 0.06)
    sun = Node(trf=sun_transform, apps=[sun_material], shps=[disk])

    earth_orbit_transform = Transform()
    earth_orbit = Node(trf=earth_orbit_transform)

    earth_position_transform = Transform()
    earth_position_transform.translate(6, 0, 0)
    earth_position = Node(trf=earth_position_transform)
    earth_orbit.add_node(earth_position)

    earth_spin_transform = Transform()
    earth_spin = Node(trf=earth_spin_transform)
    earth_position.add_node(earth_spin)

    earth_size = Transform()
    earth_size.scale(0.6, 0.6, 1)
    earth_material = ColorMaterial(1.0, 1.0, 1.0)
    earth = Node(trf=earth_size, apps=[earth_material], shps=[disk])
    earth_spin.add_node(earth)

    moon_orbit_transform = Transform()
    moon_orbit = Node(trf=moon_orbit_transform)
    earth_position.add_node(moon_orbit)

    moon_position_transform = Transform()
    moon_position_transform.translate(1.35, 0, 0)
    moon_position = Node(trf=moon_position_transform)
    moon_orbit.add_node(moon_position)

    moon_size = Transform()
    moon_size.scale(0.22, 0.22, 1)
    moon_material = ColorMaterial(0.75, 0.75, 0.78)
    moon = Node(trf=moon_size, apps=[moon_material], shps=[disk])
    moon_position.add_node(moon)

    mercury_orbit_transform = Transform()
    mercury_orbit = Node(trf=mercury_orbit_transform)
    mercury_position_transform = Transform()
    mercury_position_transform.translate(3.5, 0, 0)
    mercury_position = Node(trf=mercury_position_transform)
    mercury_orbit.add_node(mercury_position)

    mercury_size = Transform()
    mercury_size.scale(0.36, 0.36, 1)
    mercury_material = ColorMaterial(0.85, 0.60, 0.35)
    mercury = Node(trf=mercury_size, apps=[mercury_material], shps=[disk])
    mercury_position.add_node(mercury)

    background_transform = Transform()
    background_transform.scale(10, 10, 1)
    background_material = ColorMaterial(1, 1, 1)
    background = Node(trf=background_transform, apps=[background_material],
                      shps=[Square(device)])

    shader = Shader(device, "solar_assets/solar.wgsl")
    shader.set_vertex_buffers([
        {"array_stride": 2 * 4, "step_mode": "vertex",
         "attributes": [{"format": "float32x2", "offset": 0, "var_name": "pos"}]},
        {"array_stride": 2 * 4, "step_mode": "vertex",
         "attributes": [{"format": "float32x2", "offset": 0, "var_name": "texcoord"}]},
    ])
    shader.add_material(sun_material)
    shader.add_material(earth_material)
    shader.add_material(moon_material)
    shader.add_material(mercury_material)
    shader.add_material(background_material)

    sampler = Sampler(device, "surface_sampler", address_mode_u="clamp-to-edge",
                      address_mode_v="clamp-to-edge")

    earth_image = Texture(device, "surface_texture", "solar_assets/terra.png")
    surface_image = Texture(device, "surface_texture", "solar_assets/superficie.png")
    space_image = Texture(device, "surface_texture", "solar_assets/espaco.png")

    earth_texture = TextureSet([earth_image, sampler])
    surface_texture = TextureSet([surface_image, sampler])
    space_texture = TextureSet([space_image, sampler])
    shader.add_texture_set(earth_texture)
    shader.add_texture_set(surface_texture)
    shader.add_texture_set(space_texture)
    earth.add_appearance(earth_texture)
    sun.add_appearance(surface_texture)
    moon.add_appearance(surface_texture)
    mercury.add_appearance(surface_texture)
    background.add_appearance(space_texture)

    pipeline = Pipeline(shader, target_format, depth_stencil=None)

    root = Node(pipeline, nodes=[background, sun, mercury_orbit, earth_orbit])
    scene = Scene(root)
    scene.add_engine(Rotate(earth_orbit_transform, 24))
    scene.add_engine(Rotate(earth_spin_transform, 90))
    scene.add_engine(Rotate(moon_orbit_transform, 100))
    scene.add_engine(Rotate(mercury_orbit_transform, 42))


def fit_background(transform, width, height):
    transform.load_identity()
    if width > height:
        transform.scale(10 * width / height, 10, 1)
    else:
        transform.scale(10, 10 * height / width, 1)


def draw():
    global last_time
    now = time.perf_counter()
    dt = now - last_time
    last_time = now
    scene.update(dt)
    target = context.get_current_texture()
    fit_background(background_transform, target.width, target.height)
    renderer.render(target, scene, camera)


def on_key(event):
    if event["key"] == "q":
        canvas.close()


def main():
    global canvas, context, renderer, last_time

    canvas = RenderCanvas(
        size=(800, 800), title="Projeto 1 - mini-sistema solar",
        update_mode="continuous", max_fps=60,
    )
    adapter = wgpu.gpu.request_adapter_sync()
    device = adapter.request_device_sync()
    context = canvas.get_context("wgpu")
    target_format = context.get_preferred_format(device.adapter)
    context.configure(device=device, format=target_format)
    renderer = Renderer(device, clear_value=(0.015, 0.025, 0.08, 1.0))
    initialize(device, target_format)

    canvas.add_event_handler(on_key, "key_down")
    last_time = time.perf_counter()
    canvas.request_draw(draw)
    loop.run()


if __name__ == "__main__":
    main()
