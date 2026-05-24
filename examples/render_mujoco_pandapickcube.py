from .mujoco_rendering import render


if __name__ == "__main__":
    render("PandaPickCube", height=480, width=2200, chunk_size=200)
