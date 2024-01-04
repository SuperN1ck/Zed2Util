import copy
import dataclasses
import pathlib
import threading
import time

import numpy as np
import pyzed.sl as sl
import tyro
import yaml
from PIL import Image
from pynput import keyboard

from Zed2Utils.camera import ZED2Camera, ZED2Config


def default_field(obj):
    return dataclasses.field(default_factory=lambda: copy.copy(obj))


KEEP_GOING = True
START_STOP_KEY = "s"


def key_capture_thread():
    global KEEP_GOING
    KEEP_GOING = True

    with keyboard.Events() as events:
        # Block for as much as possible
        while True:
            event = events.get(1e6)
            if event.key != keyboard.KeyCode.from_char(START_STOP_KEY):
                continue
            break

    KEEP_GOING = False


def main(
    save_dir: pathlib.Path = pathlib.Path("recordings"),
    zed2config: ZED2Config = ZED2Config(fps=15),
):
    """
    Records images from the (a?) connected ZED2 camera. Files are saved in `save_dir`.
    save_dir: abc
    """
    camera = ZED2Camera(zed2config)
    save_dir.mkdir(exist_ok=True, parents=True)
    recording_count = 0

    while True:
        print("---+++=========+++---")
        print(f"Start new sequence recording by pressing '{START_STOP_KEY}'")
        threading.Thread(
            target=key_capture_thread, args=(), name="key_capture_thread", daemon=True
        ).start()
        # Waiting that something happens
        while KEEP_GOING:
            time.sleep(0.1)

        # Make sure the camera is open
        camera.open_camera()

        start = time.time()
        rgbs_left = []
        rgbs_right = []
        depths = []
        confidences = []

        print(
            f"{recording_count}-th recording started\nPress '{START_STOP_KEY}' to stop"
        )
        # We just assume they are using our naming convention =)
        recording_folder_count = len(list(save_dir.glob("*")))
        recording_save_dir = save_dir / f"{recording_folder_count:03}"
        # This shouldn't exist
        # recording_save_dir.mkdir(parents=True, exist_ok=False)
        recording_save_dir.mkdir(parents=True, exist_ok=True)

        # This automatically saves the zed2 to this path
        recording_parameters = sl.RecordingParameters(
            compression_mode=sl.SVO_COMPRESSION_MODE.H264,
            video_filename=str(recording_save_dir / "zed2.svo"),
        )
        err = camera.zed.enable_recording(recording_parameters)

        threading.Thread(
            target=key_capture_thread, args=(), name="key_capture_thread", daemon=True
        ).start()
        while KEEP_GOING:
            rgb_left_np, rgb_right_np, depth_np, confidence_np = camera.get_image()

            rgbs_left.append(rgb_left_np)
            rgbs_right.append(rgb_right_np)
            depths.append(depth_np)
            confidences.append(confidence_np)

        camera.zed.disable_recording()
        end = time.time()

        # Close camera when done
        camera.close_camera()

        n_images = len(rgbs_left)
        print(
            f"Recording stopped.\nImages:\t{n_images}\nTime:\t{end - start}\nFPS:\t{n_images / (end - start)}"
        )

        recording_count += 1

        print(f"Saving recording to {recording_save_dir}")
        # Save images for visualization
        rgb_dir = recording_save_dir / "rgb"
        rgb_dir.mkdir(exist_ok=True, parents=False)
        for i, rgb_img in enumerate(rgbs_left):
            im = Image.fromarray(rgb_img)
            im.save(rgb_dir / f"{i:05}.png")

        np.savez_compressed(
            str(recording_save_dir / "images.np"),
            rgbs_left=rgbs_left,
            rgbs_right=rgbs_right,
            depths=depths,
            confidences=confidences,
        )

        with open(str(recording_save_dir / "config.yaml"), "w") as file:
            file.write(tyro.to_yaml(zed2config))

        with open(str(recording_save_dir / "intrinsics.yaml"), "w") as file:
            yaml.dump(camera.get_intrinsics(as_dict=True), file)

        print(f"Recording saved, ready to start the next recording.")


if __name__ == "__main__":
    tyro.cli(main)
