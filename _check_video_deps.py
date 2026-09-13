import importlib
import sys

pkgs = [
    "cv2",
    "av",
    "imageio",
    "imageio_ffmpeg",
    "torchvision",
    "torchcodec",
    "skvideo",
    "decord",
    "moviepy",
    "PIL",
    "numpy",
    "torch",
    "sklearn",
]
for p in pkgs:
    try:
        m = importlib.import_module(p)
        ver = getattr(m, "__version__", "?")
        print(f"OK {p} {ver}")
    except Exception as e:
        print(f"MISS {p}: {type(e).__name__}: {e}")

print("python", sys.version)

try:
    import cv2
    info = cv2.getBuildInformation()
    if "Video I/O" in info:
        chunk = info.split("Video I/O")[1][:1200]
        print("--- cv2 Video I/O ---")
        print(chunk)
except Exception as e:
    print("cv2 detail", e)

try:
    import torchvision
    print("torchvision.io.read_video", hasattr(torchvision.io, "read_video"))
except Exception as e:
    print("tv", e)
