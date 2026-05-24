import os

if "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl"

import argparse
import pickle
from PIL import Image
from evojax import util
from mujoco_playground import registry


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--render-every", type=int, default=3)
    config, _ = parser.parse_known_args()
    return config


def render(mujoco_env, height, width, chunk_size, camera=None):
    config = parse_args()
    log_dir = f"log/mujoco/{mujoco_env}"
    logger = util.create_logger(name=f"Render-{mujoco_env}", log_dir=log_dir)
    trajectory_file = os.path.join(log_dir, "trajectory.pkl")
    logger.info(f"Loading episode: {trajectory_file}...")
    with open(trajectory_file, "rb") as f:
        data = pickle.load(f)
    trajectory_states = data["trajectory"]
    dt = data["dt"]
    frame_duration_ms = 1000.0 * dt * config.render_every

    base_env = registry.load(mujoco_env)
    base_env.mj_model.vis.global_.offwidth = max(640, width)
    base_env.mj_model.vis.global_.offheight = max(480, height)
    subsample = trajectory_states[:: config.render_every]
    gif_file = os.path.join(log_dir, "mujoco.gif")
    render_kwargs = {"height": height, "width": width}
    if camera is not None:
        render_kwargs["camera"] = camera
    frames = []
    for i in range(0, len(subsample), chunk_size):
        chunk_frames = base_env.render(subsample[i : i + chunk_size], **render_kwargs)
        for frame in chunk_frames:
            frames.append(Image.fromarray(frame))
    frames[0].save(
        gif_file,
        save_all=True,
        append_images=frames[1:],
        duration=frame_duration_ms,
        loop=0,
        optimize=False,
    )
    logger.info(f"GIF saved to: {gif_file}")
