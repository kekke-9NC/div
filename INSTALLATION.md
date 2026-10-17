# Installation notes

Create the cross-platform CPU environment with:

```sh
conda env create -f environment.yml
```

On Windows or Linux with an NVIDIA GPU, use the CUDA 11.8 environment:

```sh
conda env create -f environment-cuda.yml
```

Both Conda environments include the FFmpeg command line tools. The PyInstaller build does not bundle FFmpeg or FFprobe. Frozen app users need both commands installed and available on the app's `PATH` for video concatenation; FFmpeg is also used for summary, timelapse, and video-composition exports.
