import tyro
from Zed2Utils.camera import ZED2Camera, ZED2Config
import matplotlib.pyplot as plt

import numpy as np
import open3d as o3d


def main(zed2config: ZED2Config):
    camera = ZED2Camera(zed2config)
    rgb_left, rgb_right, depth, confidence = camera.get_image()

    fig, axs = plt.subplots(2, 2)

    axs[0][0].imshow(rgb_left)
    axs[0][1].imshow(rgb_right)
    axs[1][0].imshow(depth)
    axs[1][1].imshow(confidence / 100.0)

    for ax in axs.flatten():
        ax.axis("off")
    fig.tight_layout()
    plt.show()

    # Get pc directly from Zed2
    pc_direct = camera.get_pc(trigger_grab=False)
    pcd_direct = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pc_direct))

    # Get depth with open3d
    rgb = o3d.geometry.Image(rgb_left)
    depth = o3d.geometry.Image(depth)
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        rgb,
        depth,
        depth_scale=1.0,
        depth_trunc=zed2config.max_distance,
        convert_rgb_to_intensity=False,
    )

    width = np.asarray(depth).shape[1]
    height = np.asarray(depth).shape[0]

    f_x, f_y, c_x, c_y = camera.get_intrinsics()

    K_o3d = o3d.camera.PinholeCameraIntrinsic(width, height, f_x, f_y, c_x, c_y)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, K_o3d)

    # They align.
    o3d.visualization.draw_geometries([pcd, pcd_direct])


if __name__ == "__main__":
    tyro.cli(main)
