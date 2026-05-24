from .mujoco_rendering import render


if __name__ == "__main__":
    render("CheetahRun", height=480, width=2200, chunk_size=200, camera="side")
