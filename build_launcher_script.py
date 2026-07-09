import PyInstaller.__main__
import sys
import os

# Change to the directory of the script
os.chdir(os.path.dirname(os.path.abspath(__file__)))

print("Building bootstrap launcher...")
try:
    PyInstaller.__main__.run([
        'Launcher.spec',
        '--noconfirm',
        '--clean'
    ])
    print("Build successful.")
except Exception as e:
    print(f"Build failed: {e}")
    sys.exit(1)
