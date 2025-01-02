import pickle
import json
import sys
import os
import glob
from tqdm import tqdm
import numpy as np
from pyquaternion import Quaternion

vehicle_classes = ['sedan', 'suv', 'bus', 'tiny_car', 'van', 'lorry', 'truck',
                   'engineering_vehicle', 'tanker', 'cementing_unit', 'semi-trailer',
                   'fire_trucks', 'ambulances', 'police_vehicle',
                   'bicycle', 'motor', 'fork_truck', 'travel_trailer']
person_classes = ['pedestrian']
rider_classes = ['rider']
rear_classes = ['rear', 'head', 'wheels']
static_obs_classes = ['animal', 'stacked_trolleys',
                      'non-motor_vehicle_group', 'crowd', 'cement_column', 'bollard',
                      'folding_warning_sign', 'traffic_cone', 'parking_lock_open',
                      'parking_lock_close', 'fire_hydrant_air', 'fire_hydrant_gnd',
                      'cement_sphere', 'water_barrier', 'parking_lot_gate_open', 'parking_lot_gate_close', 'trolley',
                      'charging_pile', 'reflector', 'distribution_box', 'single_fence', 'multi_fence', 'trash_can',
                      'parking_sign', 'triangle_warning',
                      'crash_barrel',
                      ]
ignore_classes = ['nonmotor_vehicle_group', 'fire_hydrant', 'undefined_vehicle', 'undefine_vehicle']
class_names = vehicle_classes + person_classes + rider_classes

valid_cameras = ['FISHEYE_CAMERA_BACK', 'FISHEYE_CAMERA_FRONT',
                 'FISHEYE_CAMERA_LEFT', 'FISHEYE_CAMERA_RIGHT',
                 'FISHEYE_CAMERA_REAR']
camera_ranks = {'FISHEYE_CAMERA_FRONT': 0,
                'FISHEYE_CAMERA_RIGHT': 1,
                'FISHEYE_CAMERA_BACK': 2,
                'FISHEYE_CAMERA_REAR': 2,
                'FISHEYE_CAMERA_LEFT': 3}


def filter_and_reorganize_camera_annos(camera_json_data):
    new_annotations = {}
    for anno in camera_json_data['annotations']:
        if anno['category_name'] in class_names:
            token = anno['image_id']
            if token in new_annotations:
                new_annotations[token].append(anno)
            else:
                new_annotations[token] = [anno]
    return new_annotations


def is_rotation_matrix(matrix):
    """
    检查给定的矩阵是否是一个有效的旋转矩阵。

    参数:
        matrix (numpy.ndarray): 一个3x3的NumPy数组。

    返回:
        bool: 如果是有效的旋转矩阵则返回True，否则返回False。
    """
    # 确保输入是一个3x3矩阵
    if matrix.shape != (3, 3):
        return False
    # 检查矩阵是否接近正交
    identity = np.identity(3)
    product = np.dot(matrix.T, matrix)
    if not np.allclose(product, identity, atol=1e-10):
        return False
    # 检查行列式是否为+1
    det = np.linalg.det(matrix)
    if not np.isclose(det, 1.0, atol=1e-8):
        return False
    return True


def nearest_rotation_matrix(matrix):
    """
    使用SVD找到最近的有效旋转矩阵。

    参数:
        matrix (numpy.ndarray): 一个3x3的NumPy数组。

    返回:
        numpy.ndarray: 最接近的有效旋转矩阵。
    """
    U, _, Vt = np.linalg.svd(matrix)
    R = np.dot(U, Vt)
    # 确保行列式为+1
    if np.linalg.det(R) < 0:
        R = -R
    return R


def convert(lidar_pkl_data, camera_annotations):
    for lidar_info in lidar_pkl_data['infos']:
        # update basic info
        lidar_info.update({
            'sweeps': [0],
            'lidar2ego_translation': [0, 0, 0] if len(lidar_info.get('lidar2cam_rotation', [])) == 0 else lidar_info[
                'lidar2cam_rotation'],
            'lidar2ego_rotation': [1, 0, 0, 0] if len(lidar_info.get('lidar2ego_rotation', [])) == 0 else lidar_info[
                'lidar2ego_rotation'],
            'ego2global_translation': [0, 0, 0],
            'ego2global_rotation': [1, 0, 0, 0],
            'camera_ranks': camera_ranks
        })
        cams2del = set(lidar_info['cams'].keys()) - set(valid_cameras)
        for invalid_cam in cams2del:
            lidar_info['cams'].pop(invalid_cam)
        # update cams
        for cam_type, cam_info in lidar_info['cams'].items():
            lidar2cam_rotation = cam_info.pop('lidar2cam_rotation', None) or cam_info.pop('vcs2cam_rotation', None)
            lidar2cam_translation = cam_info.pop('lidar2cam_translation', None) or cam_info.pop('vcs2cam_translation',
                                                                                                None)
            assert len(lidar2cam_rotation) == 9 and len(lidar2cam_translation) == 3
            lidar2cam_rotation_np = np.array(lidar2cam_rotation, dtype=np.double).reshape(3, 3)
            try:
                sensor2ego_rotation = list(Quaternion(matrix=lidar2cam_rotation_np).inverse)
            except:
                lidar2cam_rotation_np = nearest_rotation_matrix(lidar2cam_rotation_np)
                sensor2ego_rotation = list(Quaternion(matrix=lidar2cam_rotation_np).inverse)

            cam_info.update({
                'type': cam_type,
                'data_path': cam_info.pop('file_path'),
                'sample_data_token': cam_info.pop('token'),
                'timestamp': int(cam_info.pop('timestamp')),
                'sensor2ego_translation': list(map(lambda x: -x, lidar2cam_translation)),  # 默认传感器的信息和自车系统处于一个位置
                'sensor2ego_rotation': sensor2ego_rotation,
                'ego2global_translation': [0, 0, 0],
                'ego2global_rotation': [1, 0, 0, 0],
                'cam_intrinsic': cam_info.pop('intrinsic'),
                'intrinsic_raw': np.array(cam_info['intrinsic_raw']).reshape(3, 3)
            })
        # update annotations
        # get corresponding 2d annotation by lidar annotations
        lidar_instance_ids = lidar_info['gt_ids'].tolist()
        camera_instances = dict()
        for cam_info in lidar_info['cams'].values():
            single_cam_annos = camera_annotations.get(cam_info['sample_data_token'], [])
            for anno in single_cam_annos:
                if anno['category_name'] in class_names:
                    if anno['id'] in camera_instances:
                        camera_instances[anno['id']].append(anno)
                    else:
                        camera_instances[anno['id']] = [anno]
        # get valid instance ids
        keeps_idx = [idx for idx, id in enumerate(lidar_instance_ids) if id in camera_instances.keys()]

        # fill 3d annotations
        gt_names = lidar_info['gt_names'][keeps_idx]
        lidar_info.update({
            'gt_boxes': lidar_info['gt_boxes'][keeps_idx],
            'gt_names': gt_names,
            'gt_labels': [class_names.index(label_name) for label_name in gt_names],
            'gt_ids': lidar_info['gt_ids'][keeps_idx],
            'num_lidar_pts': lidar_info['num_lidar_pts'][keeps_idx],
            'valid_flag': lidar_info['valid_flag'][keeps_idx],
            'gt_velocity': lidar_info['gt_velocity'][keeps_idx]
        })

        # record image info for inference
        cam_infos_new = [dict() for _ in range(len(lidar_info['cams']))]
        assert len(cam_infos_new) == 4
        for cam_type, cam_info in lidar_info['cams'].items():
            cam_infos_new[camera_ranks[cam_type]] = cam_info

            # fill 2d annotations
            gt_2d_boxes = np.zeros((len(keeps_idx), len(cam_infos_new), 8))
            # num_obj, num_cam, (outer_exists_this_cam, inner_exists_this_cam, outer_x, outer_y, outer_w, outer_h, inner_x1, inner_x2)
            gt_wheel_kps = np.zeros((len(keeps_idx), len(cam_infos_new), 4))  # num_obj, num_cam, (x1, y1, x2, y2)
            for row_idx, lidar_idx in enumerate(keeps_idx):
                instance_annos = camera_instances[lidar_instance_ids[lidar_idx]]
                for single_cam_anno in instance_annos:
                    for cam_rank, sensor_type in enumerate(cam_infos_new):
                        if sensor_type == single_cam_anno['sensor']:
                            gt_2d_boxes[row_idx][cam_rank][0] = 1
                            gt_2d_boxes[row_idx][cam_rank][2] = single_cam_anno['bbox'][0]
                            gt_2d_boxes[row_idx][cam_rank][3] = single_cam_anno['bbox'][1]
                            gt_2d_boxes[row_idx][cam_rank][4] = single_cam_anno['bbox'][2]
                            gt_2d_boxes[row_idx][cam_rank][5] = single_cam_anno['bbox'][3]
                            if single_cam_anno['category_name'] in vehicle_classes:
                                inner_boxes = [box for box in single_cam_anno['stage_two_bbox2d'] if
                                               box['category_name'] in ['head', 'rear']]
                                kps = [point for point in single_cam_anno['stage_two_kps'] if
                                       point['category_name'] == 'gnd_kps']
                                assert len(kps) <= 2 and len(inner_boxes) <= 1
                                if len(inner_boxes) == 1:
                                    gt_2d_boxes[row_idx][cam_rank][1] = 1
                                    gt_2d_boxes[row_idx][cam_rank][6] = inner_boxes[0]['bbox'][0]
                                    gt_2d_boxes[row_idx][cam_rank][7] = inner_boxes[0]['bbox'][0] + \
                                                                        inner_boxes[0]['bbox'][2]
                                if len(kps) == 1:
                                    gt_wheel_kps[row_idx][cam_rank][0] = kps[0]['bbox'][0]
                                    gt_wheel_kps[row_idx][cam_rank][1] = kps[0]['bbox'][1]
                                if len(kps) == 2:
                                    gt_wheel_kps[row_idx][cam_rank][0] = kps[0]['bbox'][0]
                                    gt_wheel_kps[row_idx][cam_rank][1] = kps[0]['bbox'][1]
                                    gt_wheel_kps[row_idx][cam_rank][2] = kps[1]['bbox'][0]
                                    gt_wheel_kps[row_idx][cam_rank][3] = kps[1]['bbox'][1]
            anno_infos_2d = dict({
                'cam_infos': cam_infos_new,
                'gt_2d_boxes': gt_2d_boxes,
                'gt_wheel_kps': gt_wheel_kps
            })
            lidar_info['anno_infos_2d'] = anno_infos_2d

    if __name__ == '__main__':
        with open('./data/nuscenes/bevdetv3-nuscenes_infos_train.pkl', 'rb') as f:
            bevdet_data = pickle.load(f)

        with open('/mnt/storage/parking_dataset/cyl_mono_infos_train_parking.pkl', 'rb') as f:
            parking_pkl_data = pickle.load(f)

        with open('/mnt/storage/parking_dataset/cyl_mono_infos_train_parking.coco.json', 'r') as f:
            parking_json_data = json.load(f)

        new_annos = filter_and_reorganize_camera_annos(parking_json_data)
        convert(parking_pkl_data, new_annos)
        with open('/mnt/storage/parking_dataset/bevdetv3-nuscenes_info_train.pkl', 'wb') as f:
            pickle.dump(parking_pkl_data, f)
