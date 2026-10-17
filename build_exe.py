
import PyInstaller.__main__
import os
import sys
import shutil
import config

# Ensure we are in the script directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Define input script
input_script = "main_gui.py"
app_name = "MeteorDetector"

# Paths to resources
icon_path = "icon.ico"
settings_file = "app_settings.json"
masks_file = "app_masks.npz"

# Bundle the first model supported by config.py from the project or parent dir.
model_search_dirs = [os.path.dirname(os.path.dirname(os.path.abspath(__file__))), os.getcwd()]
model_source_path = next(
    (
        os.path.join(directory, filename)
        for filename in config.MODEL_FILENAMES
        for directory in model_search_dirs
        if os.path.isfile(os.path.join(directory, filename))
    ),
    None,
)
if model_source_path is None:
    raise SystemExit(
        "Error: No supported classifier model found. Expected one of: "
        + ", ".join(config.MODEL_FILENAMES)
    )

# Data files to include: (source, dest_in_bundle)
datas = [
    # (settings_file, "."), # Don't bundle settings, copy it manually later
    (masks_file, "."),
    (model_source_path, "."),
    (icon_path, ".") if os.path.exists(icon_path) else None,
]
# Filter out None
datas = [d for d in datas if d is not None]

# Hidden imports
hidden_imports = [
    "tkinterdnd2",
    "PIL",
    "cv2",
    "numpy",
    "astropy",
    "torch",
    "torchvision",
    "status_panel",
    "ui_state",
    "network_copy",
    "download_pipeline",
    "meteor_sky_viewer",
    "coordinate_manager",
    "config",
    "file_utils",
    "video_processing",
    "astrometry",
    "image_processing",
    "model",
    "model_catalog",
    "utils",
    "location_utils",
    "sun_times",
    "auto_time_updater",
    "long_exposure_map",
    "distortion_correction",
    "meteor_angle_analysis",
    "lighten_blend_video",
    "lighten_blend_image",
    "timelapse_creator",
    "scipy.special.cython_special", # often needed for astropy/scipy
]

# Additional hooks/collects
from PyInstaller.utils.hooks import collect_all

# Collect tkinterdnd2
tmp_ret = collect_all('tkinterdnd2')
datas += tmp_ret[0]
binaries = tmp_ret[1]
hidden_imports += tmp_ret[2]

# Construct arguments
args = [
    input_script,
    "--name=%s" % app_name,
    "--noconsole",
    "--onedir", 
    "--clean",
]

# Add datas
for source, dest in datas:
    # PyInstaller expects 'source;dest' on Windows
    args.append(f"--add-data={source}{os.pathsep}{dest}")

# Add hidden imports
for imp in hidden_imports:
    args.append(f"--hidden-import={imp}")

# Add binaries if any (from collect_all, usually empty for pure python libs but tkinterdnd2 has 'tk' libs?)
# collect_all binaries format: (source, dest)
for source, dest in binaries:
     args.append(f"--add-binary={source}{os.pathsep}{dest}")

if os.path.exists(icon_path):
    args.append(f"--icon={icon_path}")

print("Running PyInstaller with arguments:", args)

PyInstaller.__main__.run(args)

# Post-build: Copy settings file to dist folder
dist_dir = os.path.join("dist", app_name)
if os.path.exists(dist_dir):
    print(f"Copying {settings_file} to {dist_dir}...")
    try:
        shutil.copy2(settings_file, os.path.join(dist_dir, settings_file))
        print("Settings file copied successfully.")
    except Exception as e:
        print(f"Error copying settings file: {e}")
else:
    print(f"Warning: Dist directory {dist_dir} not found. Settings file not copied.")
