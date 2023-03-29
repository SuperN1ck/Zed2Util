import cv2
import numpy as np
from camera import ZED2Camera, ZED2Config

if __name__ == "__main__":
    config = ZED2Config()
    camera = ZED2Camera(config)
    while True:
        rgb_np, depth_np, confidence_map_np = camera.get_image()
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
