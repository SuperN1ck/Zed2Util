import numpy as np
import dataclasses
import pathlib
from typing import Union, Tuple, Dict

try:
    import pyzed.sl as sl
except:
    import logging

    logging.error("Could not load pyzed-SDK. ZED2Camera-class will not be avalable")
    # Fake everything that is exposed on a high level
    # TODO is there a better way to do this?
    import enum

    class RESOLUTION(enum.Enum):
        HD720 = enum.auto()

    import dataclasses

    @dataclasses.dataclass
    class sl:
        RESOLUTION = RESOLUTION.HD720


@dataclasses.dataclass
class ZED2Config:
    crop: bool = False
    downscale: float = 1.0
    fps: int = 30
    min_distance: float = 0.3
    max_distance: float = 3.0
    use_neural: bool = True
    depth_confidence_threshold: int = 50
    depth_texture_confidence_threshold: int = 100

    auto_exposure: bool = False
    gain: int = 100  # Will only be used if auto exposure is false
    exposure: int = 10  # Will only be used if auto exposure is false


class ZED2Camera:
    def __init__(self, cfg: ZED2Config, open_camera: bool = True):
        self.zed = sl.Camera()

        # Set configuration parameters
        self.init_params = sl.InitParameters()
        self.init_params.camera_resolution = sl.RESOLUTION.HD720
        self.init_params.camera_fps = cfg.fps
        self.init_params.depth_mode = (
            sl.DEPTH_MODE.NEURAL if cfg.use_neural else sl.DEPTH_MODE.ULTRA
        )
        self.init_params.coordinate_units = sl.UNIT.METER
        self.init_params.depth_stabilization = True
        self.init_params.depth_minimum_distance = cfg.min_distance
        self.init_params.depth_maximum_distance = cfg.max_distance

        self.runtime_parameters = sl.RuntimeParameters()
        self.runtime_parameters.enable_depth = True
        self.runtime_parameters.confidence_threshold = cfg.depth_confidence_threshold
        self.runtime_parameters.texture_confidence_threshold = (
            cfg.depth_texture_confidence_threshold
        )

        self.resolution = sl.get_resolution(self.init_params.camera_resolution)
        self.resolution.width //= cfg.downscale
        self.resolution.height //= cfg.downscale
        self.cfg = cfg

        # Open the camera
        if open_camera:
            self._open_camera()

    def _open_camera(self):
        err = self.zed.open(self.init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"Camera open failed: {err}")

        # Update parameters
        if self.cfg.auto_exposure:
            self.zed.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE, -1)
        else:
            print(
                "[Zed2Utils] Careful! Starting camera with manual exposure and gain control."
            )
            self.zed.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE, self.cfg.exposure)
            self.zed.set_camera_settings(sl.VIDEO_SETTINGS.GAIN, self.cfg.gain)

    def _grab(self):
        err = self.zed.grab(self.runtime_parameters)
        if err != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"Camera grab failed: {err}", err=err)

    def get_image(self, trigger_grab: bool = True):
        if trigger_grab:
            self._grab()

        rgb_left = sl.Mat()
        rgb_right = sl.Mat()
        depth_map = sl.Mat()
        confidence_map = sl.Mat()

        self.zed.retrieve_image(rgb_left, sl.VIEW.LEFT, resolution=self.resolution)
        self.zed.retrieve_image(rgb_right, sl.VIEW.RIGHT, resolution=self.resolution)
        self.zed.retrieve_measure(
            depth_map, sl.MEASURE.DEPTH, resolution=self.resolution
        )
        self.zed.retrieve_measure(
            confidence_map, sl.MEASURE.CONFIDENCE, resolution=self.resolution
        )

        rgb_left_np = rgb_left.get_data(deep_copy=True)
        rgb_right_np = rgb_right.get_data(deep_copy=True)
        depth_np = depth_map.get_data(deep_copy=True)
        confidence_map_np = confidence_map.get_data(deep_copy=True)

        # Postprocess rgb and depth
        rgb_left_np = rgb_left_np[:, :, [2, 1, 0]]  # ZED2 returns BGR
        rgb_right_np = rgb_right_np[:, :, [2, 1, 0]]  # ZED2 returns BGR
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

    def get_pc(self, trigger_grab: bool = True):
        if trigger_grab:
            self._grab()
        point_cloud = sl.Mat()
        self.zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)
        point_cloud_np = point_cloud.get_data(deep_copy=True)[..., :3]
        nan_points = np.any(np.isnan(point_cloud_np), axis=-1)
        return point_cloud_np[~nan_points]

    def get_intrinsics(
        self, as_dict: bool = False
    ) -> Union[Tuple[float, float, float, float], Dict[str, float]]:
        calibration_params = (
            self.zed.get_camera_information().camera_configuration.calibration_parameters
        )
        f_x = calibration_params.left_cam.fx
        f_y = calibration_params.left_cam.fy
        c_x = calibration_params.left_cam.cx
        c_y = calibration_params.left_cam.cy
        if as_dict:
            return {"f_x": f_x, "f_y": f_y, "c_x": c_x, "c_y": c_y}
        return f_x, f_y, c_x, c_y

    @staticmethod
    def get_default_intrinsics(cam_res: sl.RESOLUTION = sl.RESOLUTION.HD720):
        """
        Use this function with caution when devaiting the default height
        returns
            f_x, f_y, c_x, c_y
        """
        resolution = sl.get_resolution(cam_res)

        f = 1050 / 2.0
        c_x = resolution.width / 2
        c_y = resolution.height / 2
        return f, f, c_x, c_y


class ZED2CameraSVO(ZED2Camera):
    def __init__(
        self, cfg: ZED2Config, svo_file: pathlib.Path, real_time_svo: bool = True
    ):
        super().__init__(cfg, open_camera=False)

        self.init_params.set_from_svo_file(svo_file)
        self.init_params.svo_real_time_mode = real_time_svo

        self._open_camera()

    def set_svo_position(self, pos: int):
        if self.init_params.svo_real_time_mode:
            return

        # TODO Do something if pos > len?
        self.zed.cam.set_svo_position(pos)

    def step_svo(self):
        svo_position = self.zed.cam.get_svo_position()
        self.set_svo_position(svo_position + 1)

    def get_image(self):
        try:
            return super().get_image()
        except RuntimeError as e:
            if (
                e.err != sl.ERROR_CODE.END_OF_SVOFILE_REACHED
            ):  # Check if the .svo has ended
                raise  # if not we will raise the same error
            self.set_svo_position(0)
            return super().get_image()
