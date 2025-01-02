import argparse
import open3d as o3d
import cv2
import mmcv
from pathlib import Path
import numpy as np
import torch
import os
import sys

CUR_DIR = Path(os.getcwd()).resolve()
sys.path.append(str(CUR_DIR))
print(CUR_DIR)
import hyper_dl
import hyper_data
import tools
# from pa.bev3d import Cyl_BEV3D_Dataset
from common.datasets import build_dataset, build_dataloader
from common.core.visualizer.image_vis import draw_cylcamera_bbox3d_on_img
from PIL import Image
from tools.dataset_converters import VirCylinderCamera
import math

bev_h = 1080
bev_w = 540
scale_h = bev_h / 100.0
scale_w = bev_w / 50
center = (bev_w // 2, bev_h - 10)
bottom_idx = [0, 1, 3, 6]
link_idx = [0, 1, 7, 2]


def parse_args():
    parser = argparse.ArgumentParser(description='show gt')
    parser.add_argument('--config', type=str,
                        default='configs/det_3d/bev3d/bevdet-cylin.py',
                        help='inference config file path, using test_config')
    # parser.add_argument('--out_dir', help='the dir to save inference results')
    # parser.add_argument(
    #     '--checkpoint', help='the checkpoint file to load')
    # parser.add_argument('--draw_boxes2d', action='store_true',
    #                     help='whether to draw 2d boxes')
    # parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    return args


def get_corner(bbox3d):
    center = bbox3d[:3]
    l, w, h = bbox3d[3:6]
    # print(size,center)
    size = [l, w, h]
    rotation = bbox3d[6]
    rotation = max(0.000001, rotation)
    obb = o3d.geometry.OrientedBoundingBox(center=center,
                                           extent=size,
                                           R=o3d.geometry.get_rotation_matrix_from_axis_angle(
                                               [0, 0, rotation]))
    corners = np.asarray(obb.get_box_points())

    return corners


def convert_lidar3d2cyl3d(cyl_cam, anno):
    """
    Coordinates in LiDAR same as local cylinder:

                    up z
                       ^   x front
                       |  /
                       | /
        left y <------ 0

    """
    center_lidar = np.array([anno['center3d']['x'], anno['center3d']['y'],
                             anno['center3d']['z']])
    center_cyl = cyl_cam.convert_lidar_to_vir_camera(center_lidar)
    loc = np.array(center_cyl).reshape(3, ).tolist()
    rot = cyl_cam.convert_lidar_rot_to_vir_rot(anno['yaw'])
    dim = [anno['dim']['length'], anno['dim']['width'], anno['dim']['height']]
    return loc, dim, [rot]


def project_cyl2cylimg(cyl_cam, anno, center3d):
    if anno['is_static_obs']:
        return [0, 0, 0]
    center_pt = np.matrix(center3d).reshape(3, 1)
    cyl_img_pt = cyl_cam.convert_vir_camera_to_vir_img(center_pt)
    # depth: sqrt(x^2 + y^2)
    depth = math.sqrt(center3d[0] ** 2 + center3d[1] ** 2)
    return [cyl_img_pt[0], cyl_img_pt[1]]


def limit_yaw_tensor(yaw):
    while ((yaw < -math.pi) | (yaw > math.pi)).any():
        yaw[yaw > math.pi] -= 2 * math.pi
        yaw[yaw < -math.pi] += 2 * math.pi
    return yaw


def limit_yaw(yaw):
    while yaw > math.pi:
        yaw -= 2 * math.pi
    while yaw <= -math.pi:
        yaw += 2 * math.pi
    return yaw


def convert_cyl3d2cam3d(cyl_box3d):
    """
    Coordinates cylinder:

                    up z
                       ^   x front (0)
                       |  /
                       | /
    (90)left y <------ 0

    Coordinates in camera:

            z front (-90)
           /
          /
         0 ------> x right (0)
         |
         |
         v
    down y
    """
    loc = [-cyl_box3d[1], -cyl_box3d[2], cyl_box3d[0]]
    dim = [cyl_box3d[3], cyl_box3d[5], cyl_box3d[4]]  # L, W, H =>  L, H, W
    cyl_rot = cyl_box3d[6]
    cam_rot = -math.pi / 2 - cyl_rot
    cam_rot = limit_yaw(cam_rot)
    return loc + dim + [cam_rot]


def convert_cam3d2cyl3d(cam3d):
    '''
    Coordinates in camera:
            z front (-90)
           /
          /
         0 ------> x right (0)
         |
         |
         v
    down y

    Coordinates cylinder:
                    up z
                       ^   x front (0)
                       |  /
                       | /
    (90)left y <------ 0
    '''
    box3d = cam3d
    rot = limit_yaw_tensor(box3d[:, 6])
    rot = -torch.pi / 2 - rot
    rot = limit_yaw_tensor(rot)
    box3d = torch.stack([box3d[:, 2], -box3d[:, 0], -box3d[:, 1],
                         box3d[:, 3], box3d[:, 5], box3d[:, 4], rot],
                        dim=1)
    return box3d.numpy()


def get_bbox3d_vertices(bbox3d):
    """
     x, y, z 与对应轴的尺寸一一对应则可求解
     corners: (8,3) array of vertices for the 3d box in following order:
         5 -------- 4
        /|         /|
       3 -------- 6 .
       | |        | |   --------> heading
       . 2 -------- 7
       |/         |/
       0 -------- 1
     """
    center = bbox3d[:3]
    size = bbox3d[3:6]
    rotation = bbox3d[6]
    obb = o3d.geometry.OrientedBoundingBox(center=center,
                                           extent=size,
                                           R=o3d.geometry.get_rotation_matrix_from_axis_angle(
                                               [0, 0, rotation]))
    corners = np.asarray(obb.get_box_points())
    line_set = o3d.geometry.LineSet.create_from_oriented_bounding_box(
        obb)
    return corners, line_set


def convert_camboxes3d_to_img(bboxes3d, img_pred, cyl_cam):
    height, width = img_pred.shape[:2]
    cube_edges = np.array([[1, 7], [7, 4], [4, 6], [6, 1], [0, 2], [2, 5], [5, 3],
                           [3, 0], [1, 0], [6, 3], [4, 5], [7, 2]])
    image_height, image_width = 576, 704
    # convert preds from CamInstance3D to cyl coord bbox3d
    cyl_bboxes3d = convert_cam3d2cyl3d(bboxes3d)
    # 2D: 基于cyl直接转换到图像上
    for cyl_3d in cyl_bboxes3d:
        corners, _ = get_bbox3d_vertices(cyl_3d.tolist())
        # get center2d
        center_pt = np.matrix(cyl_3d[:3]).reshape(3, 1)
        center2d = cyl_cam.convert_vir_camera_to_vir_img(center_pt)
        cv2.circle(img_pred, (int(center2d[0]), int(center2d[1])), 2, (0, 255, 0), 2)
        corners_cyl_img = [cyl_cam.convert_vir_camera_to_vir_img(
            np.matrix(pt).reshape(3, 1)
        ) for pt in corners]
        for edge in cube_edges:
            pt1 = corners_cyl_img[edge[0]]
            pt2 = corners_cyl_img[edge[1]]

            if 0 <= pt1[0] < image_width and 0 <= pt1[1] < image_height and 0 <= pt2[0] < image_width and pt2[
                1] >= 0 and pt2[1] < image_height:
                cv2.line(img_pred, tuple(pt1), tuple(pt2), (255, 0, 255), 2)
    return img_pred


def main():
    args = parse_args()
    config = mmcv.Config.fromfile(args.config)
    config_train = config.data.train
    dataset = build_dataset(config_train)
    print(len(dataset))
    data = dataset[0]


    # for test
    import pickle
    # 将对象序列化为pkl文件
    # with open('sample_data.pkl', 'wb') as file:
    #     pickle.dump(sample_data, file)
    #     exit(0)
    # test end

    cfg = {}
    cfg["focal_length"] = 220
    cfg["cx"] = 352
    cfg["cy"] = 248
    cfg["hfov"] = 183
    cfg["vfov"] = 104
    cfg["gdc_inputs_szie"] = [704, 576]
    cyl_cam = {}
    cam_str = ["FISHEYE_CAMERA_FRONT", "FISHEYE_CAMERA_BACK", "FISHEYE_CAMERA_LEFT", "FISHEYE_CAMERA_RIGHT"]
    order = [1, 0, 2, 3]
    cam_type = ["FISHEYE_CAMERA_FRONT", "FISHEYE_CAMERA_BACK", "FISHEYE_CAMERA_LEFT", "FISHEYE_CAMERA_RIGHT"]
    cam_gt_order = ['FISHEYE_CAMERA_BACK', 'FISHEYE_CAMERA_FRONT', 'FISHEYE_CAMERA_LEFT', 'FISHEYE_CAMERA_RIGHT']
    for i in range(4):
        sensor2egos = data['img_inputs'][1][order[i]].numpy()
        r = sensor2egos[:3, :3].T
        t = -sensor2egos[:3, 3]
        cyl_cam[cam_str[i]] = VirCylinderCamera(cfg)
        cyl_cam[cam_str[i]].init_local_camera(cam_type[i], r, t)


    grid_config = {
        'x': [-51.2, 51.2, 0.8],
        'y': [-51.2, 51.2, 0.8],
        'z': [-5, 3, 8],
        'depth': [1.0, 60.0, 1.0],
    }
    num_grid_x = int((grid_config['x'][1]-grid_config['x'][0])/grid_config['x'][-1])
    num_grid_y = int((grid_config['y'][1]-grid_config['y'][0])/grid_config['y'][-1])

    debug_id = 0
    for i in range(len(dataset)):
        if i !=debug_id:
            continue
        if i>debug_id:
            break

        data = dataset[i]
        name = f'bev_{i}.png'
        print(f"Index: {i}")
        img = data['img_inputs']
        img = img[0].cpu().numpy()
        center_3ds = []
        for nx in range(num_grid_x):
            for ny in range(num_grid_y):
                coor_x = grid_config['x'][0]+(nx+0.5)*grid_config['x'][-1]
                coor_y = grid_config['y'][0]+(ny+0.5)*grid_config['y'][-1]
                coor_z = -0.8
                center_3ds.append(
                    [[coor_x, coor_y, coor_z], [coor_x, coor_y, coor_z + 0.9], [coor_x, coor_y, coor_z - 0.9]])

        anchor = dict()
        for id in range(img.shape[0]):
            im = img[order[id]].copy()
            im = im.transpose(2, 1, 0)
            im = im.astype(np.uint8)
            im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
            cv2.putText(im, cam_str[id], (300, 100), 1, 2, (0, 255, 255), 2)
            cam_id = cam_gt_order.index(cam_str[id])
            one_anchor = np.zeros((num_grid_x,num_grid_x,3))
            for nx in range(num_grid_x):
                for ny in range(num_grid_y):
                    one_3ds = center_3ds[nx*num_grid_y+ny]
                    cam_3ds = [cyl_cam[cam_str[id]].convert_lidar_to_vir_img(pt) for pt in one_3ds]
                    u, v = cam_3ds[0]
                    cam_wid = abs(cam_3ds[1][1] - cam_3ds[2][1])
                    ih, iw = 576, 704
                    # print(n,u, v, cam_wid)

                    if u + cam_wid / 2 < 0:
                        u = cam_wid / 2
                    if u - cam_wid / 2 > iw - 1:
                        u = iw - 1 - cam_wid / 2

                    tx = int(u - cam_wid / 2)
                    tx = max(0, min(tx, iw - 1))
                    ty = int(v - cam_wid / 2)
                    ty = max(0, min(ty, ih - 1))

                    bx = int(u + cam_wid / 2)
                    bx = max(0, min(bx, iw - 1))
                    by = int(v + cam_wid / 2)
                    by = max(0, min(by, ih - 1))
                    if bx - tx < cam_wid - 1:
                        if tx < 2:
                            bx = tx + cam_wid
                            by = ty + cam_wid
                        elif bx > iw - 2:
                            tx = bx - cam_wid
                            ty = by - cam_wid

                    ncx = int((tx + bx) / 2)
                    ncy = int((ty + by) / 2)
                    nwid = abs(bx-tx)
                    assert(nwid !=0)
                    one_anchor[nx,ny]=[ncx,ncy,nwid]

            anchor[cam_str[id]] = one_anchor
        with open('/media/gpu/HDD/fk/cyl_mono_train_dataset/anchor.pkl', 'wb') as file:
            pickle.dump(anchor, file)
            print("----------- ok -------------")
            exit(0)

        print("------------------------")
    print("hello world")


if __name__ == "__main__":
    main()