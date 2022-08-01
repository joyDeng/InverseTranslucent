import os

# replace this as home directory
HOME_DIR = "/home/dx/Research/"

ROOT_DIR = os.path.join(HOME_DIR, "psdr-bssrdf")
DATA_DIR = os.path.join(HOME_DIR, "data")
SCENES_DIR = os.path.join(ROOT_DIR, "examples/scenes/")

RAW_TEXTURE_DIR = os.path.join(ROOT_DIR, "examples/data/textures_raw/")
TEXTURE_DIR = os.path.join(DATA_DIR, "textures")
REMESH_DIR = os.path.join(HOME_DIR, "botsch-kobbelt-remesher-libigl/build")
RESULT_DIR = os.path.join(DATA_DIR, "results")
BLEND_SCENE = os.path.join(SCENES_DIR, "render.blend")
BLENDER_EXEC = "blender2.8" # Change this if you have a different blender installation
REAL_DIR = os.path.join(DATA_DIR, "realdata/")
# data folders
ESSEN_DIR = os.path.join(DATA_DIR, "essen/")
LIGHT_DIR = os.path.join(SCENES_DIR, "light/")
SHAPE_DIR = os.path.join(DATA_DIR, "smoothshape/")
