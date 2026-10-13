"""Legacy GUI entry point; spawned CPU workers never import Tk or model weights."""
import multiprocessing
import sys

if __name__ == "__main__":
    # PyInstaller intercepts worker/resource-tracker invocations here before
    # any GUI imports, model initialization or application lock acquisition.
    sys._meteor_cpu_spawn_ready = True
    multiprocessing.freeze_support()

if __name__ != "__mp_main__":
    from gui_app import *

if __name__ == "__main__":
    main()
