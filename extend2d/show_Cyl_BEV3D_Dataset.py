import argparse
import open3d as o3d
import cv2
import mmcv
from pathlib import  Path
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
from common.datasets import build_dataset,build_dataloader
from common.core.visualizer.image_vis import draw_cylcamera_bbox3d_on_img
from PIL import Image
from tools.dataset_converters import VirCylinderCamera
import math
import pickle

bev_h = 1080
bev_w= 540
scale_h = bev_h/100.0
scale_w = bev_w/50
center = (bev_w//2,bev_h-10)
bottom_idx = [0,1,3,6]
link_idx = [0,1,7,2]
mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
def parse_args():
    parser = argparse.ArgumentParser(description='show gt')
    parser.add_argument('--config',type = str,default='/home/fengkai/Documents/code/other_dl/hyper_dl/configs/det_3d/bev3d/bevdet-cylin.py',
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
    l,w,h  = bbox3d[3:6]
    # print(size,center)
    size = [l,w,h]
    rotation = bbox3d[6]
    rotation = max(0.000001,rotation)
    obb = o3d.geometry.OrientedBoundingBox(center=center,
                                           extent=size,
                                           R=o3d.geometry.get_rotation_matrix_from_axis_angle(
                                               [0, 0,rotation]))
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
    ## dataloader
    runner_type = 'EpochBasedRunner' if 'runner' not in config else config.runner[
        'type']

    train_dataloader_default_args = dict(
            samples_per_gpu=2,
            workers_per_gpu=2,
            # `num_gpus` will be ignored if distributed
            num_gpus=1,
            dist=False,
            seed=1,
            runner_type=runner_type,
            persistent_workers=False)

    train_loader_cfg = {
        **train_dataloader_default_args,
        **config.data.get('train_dataloader', {})
    }

    # 1.初始化 data_loaders ，内部会初始化 GroupSampler
    # data_loaders = build_dataloader(dataset, **train_loader_cfg) 
    # iter_loader = iter(data_loaders)
    # for i in range(20):
    #     data = next(iter_loader)
    #     print(type(data))


    ##--end dataloader

    print(len(dataset))
    data = dataset[0]

    h = 576
    w = 704
    bev_h = h*2
    bev_w = h*2
    scale_h = bev_h/120
    scale_w= bev_w/120
    cx = bev_w//2
    cy = bev_h//2
    sample_data = dataset[0]
    #for test
    import pickle
    # 将对象序列化为pkl文件
    # with open('sample_data.pkl', 'wb') as file:
    #     pickle.dump(sample_data, file)
    #     exit(0)
    #test end
    cyl_k = np.array([220, 0, 352,
                      0, 220, 248,
                      0, 0, 1], dtype=np.float32).reshape(3, 3)
    cfg = {}
    cfg["focal_length"] = 220
    cfg["cx"] = 352
    cfg["cy"] = 248
    cfg["hfov"] = 183
    cfg["vfov"] =104
    cfg["gdc_inputs_szie"] = [704,576]
    cyl_cam = {}
    cam_gt_order = ['FISHEYE_CAMERA_BACK', 'FISHEYE_CAMERA_FRONT', 'FISHEYE_CAMERA_LEFT', 'FISHEYE_CAMERA_RIGHT']
    # cam_gt_order = ['FISHEYE_CAMERA_BACK', 'FISHEYE_CAMERA_LEFT', 'FISHEYE_CAMERA_RIGHT', 'FISHEYE_CAMERA_FRONT']
    cube_edges = np.array([[1, 7], [7, 4], [4, 6], [6, 1], [0, 2], [2, 5], [5, 3],
                           [3, 0], [1, 0], [6, 3], [4, 5], [7, 2]])
    image_height, image_width = 576, 704
    colors = []
    scale = 0.5

    grid_config = {
        'x': [-51.2, 51.2, 0.8],
        'y': [-51.2, 51.2, 0.8],
        'z': [-5, 3, 8],
        'depth': [1.0, 60.0, 1.0],
    }
    for i in range(100):
        b = np.random.randint(10, 255)
        g = np.random.randint(10, 255)
        r = np.random.randint(10, 255)
        colors.append([b, g, r])
    debug_id = 0
    with open('/media/data/train_data/bev_fisheyey_part0/anchor.pkl','rb') as fr:
        anchor = pickle.load(fr)
    # for i in range(20980,21020):
    for i in range(0,len(dataset),10):
        # i = 5
        # if i !=debug_id:
        #     continue
        # if i>debug_id:
        #     break
        bev_map = np.zeros((bev_h, bev_w, 3), dtype=np.uint8)

        cv2.putText(bev_map, str(i), (150, 100), 1, 3, (255, 0, 0), 2)
        cv2.circle(bev_map, (cx,cy), 4, (0, 255, 0), 5)
        for c in range(7):
            cr = int(c * 10 * scale_h)
            cv2.circle(bev_map, (cx, cy), cr, colors[c], 1)
            if c >0:
                cv2.putText(bev_map, f'{c*10}m', (cx+cr, cy), 1, 2, (255, 255, 255), 2)
        cv2.line(bev_map, (cx,cy), (cx+350,cy), (255, 0, 0), 2)
        cv2.putText(bev_map, 'x', (cx+350,cy), 1, 3, (255, 0, 0), 2)
        cv2.line(bev_map, (cx,cy), (cx,cy+300), (0, 255, 0), 2)
        cv2.putText(bev_map, 'y', (cx, cy+300), 1, 3, (0, 255, 0), 2)
        data = dataset[i]
        for ii in range(4):
            sensor2egos = data['img_inputs'][1][ii].numpy()#标注数据的理解有可能是反的；
            r = sensor2egos[:3, :3].T
            t = -sensor2egos[:3, 3]
            cyl_cam[cam_gt_order[ii]] = VirCylinderCamera(cfg)
            cyl_cam[cam_gt_order[ii]].init_local_camera(cam_gt_order[ii], r, t)#lidar2cam
        name = f'bev_{i}.png'
        '''test'''
        img_pt = torch.rand(2,4, 30, 8 ,4 ,3 ,1)
        vir_raw_pt = cyl_cam[cam_gt_order[0]].batch_convert_vir_img_to_vir_camera_by_vir_r(img_pt)
        '''end'''
        img_metas = data['img_metas'].data
        sample_idx = img_metas['sample_idx']
        print(f"Index: {i}", sample_idx)
        # for dw in range(-30,30,15):
        #     if abs(dw)<15:
        #         continue
        #     for dd in range(-30,30,15):
        #         if abs(dd) < 15:
        #             continue
        #         for dh in range(-2,2,1):
        #             center = np.array([dw,dd,dh])
        #             for id in range(4):
        #                 cam_raw_pt = cyl_cam[cam_gt_order[id]].convert_lidar_to_raw_camera(center)
        #                 vir_raw_pt = cyl_cam[cam_gt_order[id]].convert_raw_camera_to_vir_camera(cam_raw_pt)
        #
        #                 uv_origin = cyl_cam[cam_gt_order[id]].convert_vir_camera_to_vir_img(vir_raw_pt)
        #
        #                 if 0 <= uv_origin[0] < image_width - 1 and 0 <= uv_origin[1] < image_height - 1:
        #                     # print('-' * 10,dd,dw,dh,id)
        #                     r = np.sqrt(vir_raw_pt[0] * vir_raw_pt[0] + vir_raw_pt[1] * vir_raw_pt[1])
        #                     vir_pt = cyl_cam[cam_gt_order[id]].convert_vir_img_to_vir_camera_by_virdepth(uv_origin, r.item(0,0))
        #                     # print('vir_pt')
        #                     # print(vir_raw_pt[0, 0], vir_raw_pt[1, 0], vir_raw_pt[2, 0])
        #                     # print(vir_pt[0],vir_pt[1],vir_pt[2])
        #                     vir_pt = np.array(vir_pt)
        #                     vir_pt = vir_pt.reshape(3, 1)
        #                     tm_pt = cyl_cam[cam_gt_order[id]].convert_vir_camera_to_raw_camera(vir_pt)
        #                     # print(uv_origin)
        #                     d3_d2 =np.array([cam_raw_pt[0, 0], cam_raw_pt[1, 0], cam_raw_pt[2, 0]])
        #                     d2_d3=np.array([tm_pt[0, 0], tm_pt[1, 0], tm_pt[2, 0]])
        #                     d_diff = d3_d2-d2_d3
        #                     d_diff = abs(d_diff)
        #                     d_sum = d_diff.mean()
        #                     # print(d_sum)
        #                     assert d_sum <0.2
        #                     # print(cam_raw_pt[0, 0], cam_raw_pt[1, 0], cam_raw_pt[2, 0])
        #                     # print(tm_pt[0, 0], tm_pt[1, 0], tm_pt[2, 0])
        # continue

        img = data['img_inputs']
        img = img[0].cpu().numpy()

        print(i,img.shape)
        canvas = np.zeros((h*2,w*2,3),dtype =np.uint8)
        bboxes3d = data['gt_bboxes_3d'].data
        bboxes2d = data['gt_bboxes_2d'].data
        bboxes2d = bboxes2d.cpu().numpy()
        bboxes_label = data['gt_labels_3d'].data


        corners_3d = []
        show_3d = []

        center_3ds = []
        for bi,box in enumerate(bboxes3d.tensor):
            center3d = box[:3]
            x, y, z = center3d.numpy()
            center = center3d.numpy()
            for id in range(4):
                cam_raw_pt = cyl_cam[cam_gt_order[id]].convert_lidar_to_raw_camera(center)
                vir_raw_pt = cyl_cam[cam_gt_order[id]].convert_raw_camera_to_vir_camera(cam_raw_pt)

                uv_origin = cyl_cam[cam_gt_order[id]].convert_vir_camera_to_vir_img(vir_raw_pt)

                if 0 <= uv_origin[0] < image_width - 1 and 0 <= uv_origin[1] < image_height - 1:
                    # print('-' * 10,dd,dw,dh,id)
                    r = np.sqrt(vir_raw_pt[0] * vir_raw_pt[0] + vir_raw_pt[1] * vir_raw_pt[1])
                    vir_pt = cyl_cam[cam_gt_order[id]].CvtCylImgPt2CylCam(uv_origin, r.item(0,0))
                    img_pt = torch.zeros(1, 1, 1, 1, 1, 3, 1)
                    img_pt[...,0,0]=uv_origin[0]
                    img_pt[...,1,0]=uv_origin[1]
                    img_pt[...,2,0]=r.item(0,0)
                    # vir_pt_tmp = cyl_cam[cam_gt_order[id]].batch_convert_vir_img_to_vir_camera_by_vir_r(img_pt)

                    # print('vir_pt')
                    # print(vir_raw_pt[0, 0], vir_raw_pt[1, 0], vir_raw_pt[2, 0])
                    # print(vir_pt[0],vir_pt[1],vir_pt[2])
                    vir_pt = np.array(vir_pt)
                    vir_pt = vir_pt.reshape(3, 1)
                    tm_pt = cyl_cam[cam_gt_order[id]].convert_vir_camera_to_raw_camera(vir_pt)
                    # print(uv_origin)
                    d3_d2 =np.array([cam_raw_pt[0, 0], cam_raw_pt[1, 0], cam_raw_pt[2, 0]])
                    d2_d3=np.array([tm_pt[0, 0], tm_pt[1, 0], tm_pt[2, 0]])
                    d_diff = d3_d2-d2_d3
                    d_diff = abs(d_diff)
                    d_sum = d_diff.mean()
                    # print(d_sum)
                    try:
                        assert d_sum <0.2
                    except:
                        print(cam_raw_pt[0, 0], cam_raw_pt[1, 0], cam_raw_pt[2, 0])
                        print(tm_pt[0, 0], tm_pt[1, 0], tm_pt[2, 0])
            coor_x = (int(x/grid_config['x'][-1])+0.5)*grid_config['x'][-1]
            coor_y = (int(y/grid_config['y'][-1])+0.5)*grid_config['y'][-1]
            coor_z = -0.8
            center_3ds.append([[coor_x,coor_y,coor_z],[coor_x,coor_y,coor_z+0.9],[coor_x,coor_y,coor_z-0.9]])
            

            v = int(cy + y * scale_h)
            u = int(x * scale_w + cx)
            cv2.circle(bev_map, (u, v), 2, (0, 255, 255), 2)
            corner = get_corner(box)
            show_3d.append(corner)
            corners_3d.append(corner)

        for id in range(img.shape[0]):
            im = img[id].copy()
            im = im.transpose(2,1,0)
            im = im * std + mean
            im = im.astype(np.uint8)
            im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
            cv2.putText(im, cam_gt_order[id], (300, 100), 1, 2, (0, 255, 255), 2)
            cam_id = id
            one_anchor = anchor[cam_gt_order[id]]
            for nb in range(bboxes2d.shape[0]):
                if bboxes2d[nb][cam_id][0] !=0:
                    _, bcx, bcy, bw, bh,lft,rgt = bboxes2d[nb][cam_id]
                    assert(bw >0 and bh>0)
                    assert rgt>lft
                    bcx = bcx * scale
                    bcy = bcy * scale
                    bw = bw * scale
                    bh = bh * scale
                    x1 = bcx - bw / 2
                    x2 = bcx + bw / 2
                    y1 = bcy - bh / 2
                    y2 = bcy + bh / 2
                    cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)), colors[nb], 2)
                    cv2.putText(im, str(nb), (int(x1), int(y1)), 1, 3, colors[nb], 2)
                    head = [(0,0,0),(0,255,0),(255,0,0)]
                    lft = lft*scale
                    rgt = rgt*scale
                    cv2.rectangle(im, (int(lft), int(y1)), (int(rgt), int(y2)), (255,0,0), 2)



            for n,box in enumerate(show_3d):
                box = box.reshape(-1,3)
                uv_origin = [cyl_cam[cam_gt_order[id]].convert_lidar_to_vir_img(pt) for pt in box]
                if bboxes2d[n][cam_id][0] != 0:
                    #test pickle
                    box_tensor  = bboxes3d.tensor[n]
                    center3d = box_tensor[:3]
                    x, y, z = center3d.numpy()
                    coor_x = (int((x -grid_config['x'][0])/ grid_config['x'][-1]))
                    coor_y = (int((y -grid_config['y'][0])/ grid_config['y'][-1]))
                    u,v,cam_wid = one_anchor[coor_x,coor_y]
                    assert(cam_wid>0)
                    ncx = int(u)
                    ncy = int(v)
                    #--test end
                    ih,iw = 576,704

                    # if u+cam_wid / 2<0:
                    #     u = cam_wid/2
                    # if u - cam_wid / 2 >iw-1:
                    #     u = iw-1-cam_wid / 2

                    tx = int(u - cam_wid / 2)
                    tx = max(0, min(tx, iw - 1))
                    ty = int(v - cam_wid / 2)
                    ty = max(0, min(ty, ih - 1))

                    bx = int(u + cam_wid / 2)
                    bx = max(0, min(bx, iw - 1))
                    by = int(v + cam_wid / 2)
                    by = max(0, min(by, ih - 1))

                    cv2.circle(im, (ncx,ncy), 2, (255, 0, 0), 2)
                    cv2.rectangle(im, (tx, ty), (bx, by),(255, 0, 0), 1)
                    cv2.putText(im, str(n), uv_origin[cube_edges[0][0]], 1, 3, colors[n], 2)
                    for edge in cube_edges:
                        pt1 = uv_origin[edge[0]]
                        pt2 = uv_origin[edge[1]]

                        if 0 <= pt1[0] < image_width and 0 <= pt1[1] < image_height and 0 <= pt2[0] < image_width and pt2[
                            1] >= 0 and pt2[1] < image_height:
                            cv2.line(im, tuple(pt1), tuple(pt2), colors[n], 1)
            wi = id%2
            hi = int(id/2)
            canvas[h*hi:h*hi+h,w*wi:w*wi+w]=im



        corners_3d = np.array(corners_3d)
        for ii in range(corners_3d.shape[0]):
            data = corners_3d[ii]
            label = bboxes_label[ii].item()
            # if ii == 0:
                # print(data)
            bev_pt = []
            for idx in link_idx:
                x, y, z = data[idx]

                v = int(cy +y * scale_h )
                u = int(x*scale_w + cx)
                bev_pt.append((u, v))
                cv2.circle(bev_map, (u, v), 2, (255, 255, 0), 2)
                if idx ==0:
                    cv2.putText(bev_map, str(ii), (u, v), 1, 2, (0, 255, 255), 2)
            for i in range(4):
                if i == 3:
                    cv2.line(bev_map, bev_pt[i], bev_pt[0], colors[ii], 2)
                else:
                    cv2.line(bev_map, bev_pt[i], bev_pt[i + 1], colors[ii], 2)
        save_im = np.hstack([canvas, bev_map])
        # cv2.imwrite(f"{i}.png",save_im)
        save_im = Image.fromarray(save_im)
        save_im.save(name)
        # save_im.show()
        # exit(0)
        print("------------------------")
    print("hello world")
if __name__=="__main__":
    main()