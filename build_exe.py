
import PyInstaller.__main__
import os
import sys
import shutil

# Ensure we are in the script directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Define input script
input_script = "main_gui.py"
app_name = "MeteorDetector"

# Paths to resources
icon_path = "icon.ico"
settings_file = "app_settings.json"
masks_file = "app_masks.npz"

# Model file - assuming it's in the parent directory as per original config
# We want to bundle it into the exe
model_filename = "model_epoch_46.pth"
model_source_path = os.path.join("..", model_filename)

if not os.path.exists(model_source_path):
    print(f"Warning: Model file not found at {model_source_path}. Checking local...")
    if os.path.exists(model_filename):
        model_source_path = model_filename
    else:
        print("Error: Model file not found!")
        # sys.exit(1) # Don't exit, maybe user wants to build without it? 
        # But config.py expects it.

# Data files to include: (source, dest_in_bundle)
datas = [
    # (settings_file, "."), # Don't bundle settings, copy it manually later
    (masks_file, "."),
    (model_source_path, ".") if os.path.exists(model_source_path) else None,
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
    "noise_twin",
    "noise_twin_pipeline",
    "noise_twin_training",
    "noise_twin_worker",
    "temporal_mean",
    "astrometry",
    "image_processing",
    "model",
    "utils",
    "camera_plate_model",
    "camera_model_builder",
    "camera_model_monitor",
    "cloud_coverage",
    "local_wideangle_astrometry",
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
