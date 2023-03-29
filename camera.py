import numpy as np
import pyzed.sl as sl
import dataclasses



@dataclasses.dataclass
class ZED2Config:
    crop: bool = True
    downscale: float = 2.
    fps: int = 30
    min_distance: float = 0.3
    max_distance: float = 3.0
\

class ZED2Camera:
    def __init__(self, cfg: ZED2Config):
        self.zed = sl.Camera()

        # Set configuration parameters
        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.HD1080
        init_params.camera_fps = cfg.fps
        init_params.depth_mode = sl.DEPTH_MODE.NEURAL
        init_params.coordinate_units = sl.UNIT.METER
        init_params.depth_minimum_distance = cfg.min_distance
        init_params.depth_maximum_distance = cfg.max_distance

        self.runtime_parameters = sl.RuntimeParameters(
            sensing_mode=sl.SENSING_MODE.STANDARD,
            enable_depth=True,
            confidence_threshold=50,
            texture_confidence_threshold=100,
        )

        self.resolution = sl.get_resolution(init_params.camera_resolution)
        self.resolution.width //= cfg.downscale
        self.resolution.height //= cfg.downscale
        self.cfg = cfg

        # Open the camera
        err = self.zed.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            print("Camera open failed: {0}".format(err))
            exit(-1)
        return

    def get_image(self, ):
        err = self.zed.grab(self.runtime_parameters)
        if err != sl.ERROR_CODE.SUCCESS:
            print("Camera grab failed: {0}".format(err))
            exit(-1)

        rgb_left = sl.Mat()
        rgb_right = sl.Mat()
        depth_map = sl.Mat()
        confidence_map = sl.Mat()
        
        self.zed.retrieve_image(rgb_left, sl.VIEW.LEFT, resolution=self.resolution)
        self.zed.retrieve_image(rgb_right, sl.VIEW.RIGHT, resolution=self.resolution)
        self.zed.retrieve_measure(depth_map, sl.MEASURE.DEPTH, resolution=self.resolution)
        self.zed.retrieve_measure(confidence_map, sl.MEASURE.CONFIDENCE, resolution=self.resolution)
        
        rgb_left_np = rgb_left.get_data(deep_copy=True)
        rgb_right_np = rgb_right.get_data(deep_copy=True)
        depth_np = depth_map.get_data(deep_copy=True)
        confidence_map_np = confidence_map.get_data(deep_copy=True)

        # Postprocess rgb and depth
        rgb_left_np = rgb_left_np[:, :, [2, 1, 0]] # ZED2 returns BGR
        rgb_right_np = rgb_right_np[:, :, [2, 1, 0]] # ZED2 returns BGR
        depth_np = np.nan_to_num(depth_np, nan=0.0, posinf=0.0, neginf=0.0)
        depth_np = np.expand_dims(depth_np, axis=-1)
        confidence_map_np = np.expand_dims(confidence_map_np, axis=-1)

        # Crop
        if self.cfg.crop:
            rgb_left_np = rgb_left_np[14:526, ...]
            rgb_right_np = rgb_right_np[14:526, ...]
            depth_np = depth_np[14:526, ...]
            confidence_map_np = confidence_map_np[14:526, ...]

        # Order in memory (change to C order)
        rgb_left_np = np.ascontiguousarray(rgb_left_np)
        rgb_right_np = np.ascontiguousarray(rgb_right_np)
        depth_np = np.ascontiguousarray(depth_np)
        confidence_map_np = np.ascontiguousarray(confidence_map_np)

        return rgb_left_np, rgb_right_np, depth_np, confidence_map_np
