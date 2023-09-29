import cv2
import numpy as np
import tyro

from Zed2Utils.camera import ZED2Camera, ZED2Config


def main(zed2config: ZED2Config):
    print("Press 'q' to close the window.")

    camera = ZED2Camera(zed2config)
    while True:
        rgb_np, _, depth_np, confidence_map_np = camera.get_image()
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_np, alpha=255 / 1.5), cv2.COLORMAP_JET
        )
        bgr_color = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)
        images = np.hstack((bgr_color, depth_colormap))
        cv2.namedWindow("Align Example", cv2.WINDOW_NORMAL)
        cv2.imshow("Align Example", images)
        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord("q") or key == 27:
            cv2.destroyAllWindows()
            break


if __name__ == "__main__":
    tyro.cli(main)
