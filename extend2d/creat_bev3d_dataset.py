# Copyright (c) OpenMMLab. All rights reserved.
import pickle

import numpy as np
# from nuscenes import NuScenes
# from common.evaluation_tools.eval.detection.data_classes import Box
from pyquaternion import Quaternion
import mmcv
import os
import glob
import json
import argparse
import math
# from nuscenes_converter import nuscenes_converter

# 这个主要按照cyl中的pkl文件来进行修改的,此时生成的信息也主要是pkl文件中的信息
map_name_from_general_to_detection = {
    'sedan': 'vehicle',
    'suv': 'vehicle',
    'bus': 'vehicle',
    'tiny_car': 'vehicle',
    'van': 'vehicle',
    'lorry': 'vehicle',
    'truck': 'vehicle',
    'engineering_vehicle': 'vehicle',
    'tanker': 'vehicle',
    'cementing_unit': 'vehicle',
    'semi-trailer': 'vehicle',
    'fire_trucks': 'vehicle',
    'ambulances': 'vehicle',
    'police_vehicle': 'vehicle',
    'bicycle': 'vehicle',
    'motor': 'vehicle',
    'pedestrian': 'pedestrian',
    'rider': 'rider',
    'rear': 'ignore',
    'head': 'ignore',
    'trolley': 'ignore',
    'animal': 'ignore',
    'stacked_trolleys': 'ignore',
    'non-motor_vehicle_group': 'ignore',
    'crowd': 'pedestrian',
    'cement_column': 'ignore',
    'bollard': 'ignore',
    'folding_warning_sign': 'ignore',
    'traffic_cone': 'ignore',
    'parking_lock_open': 'ignore',
    'parking_lock_close': 'ignore',
    'fire_hydrant_air': 'ignore',
    'fire_hydrant_gnd': 'ignore',
    'cement_sphere': 'ignore',
    'water_barrier': 'ignore',
    'charging_pile': 'ignore',
    'reflector': 'ignore'
}
map_name_from_general_to_detection = {
    k.lower(): v.lower() for k, v in map_name_from_general_to_detection.items()
}
# 将其中的类别进行更改


classes = (
        'sedan', 'suv', 'bus', 'tiny_car', 'van', 'lorry', 'truck',
        'engineering_vehicle', 'tanker', 'cementing_unit', 'semi-trailer',
        'fire_trucks', 'ambulances', 'police_vehicle', 'bicycle', 'motor',
        'pedestrian', 'rider', 'rear', 'head', 'trolley', 'animal', 'stacked_trolleys',
        'non-motor_vehicle_group', 'crowd', 'cement_column', 'bollard',
        'folding_warning_sign', 'traffic_cone', 'parking_lock_open',
        'parking_lock_close', 'fire_hydrant_air', 'fire_hydrant_gnd',
        'cement_sphere', 'water_barrier',
        'charging_pile', 'reflector'
    )
CLASSES_SIM = ['None','vehicle', 'pedestrian', 'rider']
def get_cameras_calib(package_root, sync_info):
    path = os.path.join(package_root, 'calib', 'cameras.json')
    with open(path, 'r') as file:
        calib_infos = json.load(file)
    # 此时有四条数据分别对应着前后左右的数据形式，此时是列表格式中嵌套着字典形式的数据
    if len(calib_infos) == 0:
        raise LookupError("No camera calibs found!")
    cams = {}
    for cam_info in calib_infos:
        # sensor用来确定相机固定的位置，是哪一个情况 首先在cam_info将其除去
        sensor = cam_info.pop('sensor')  
        # 是根据一个总体（一个雷达信息，四个相机信息）的情况，得到相机的具体信息
        token = sync_info['cameras'].get(sensor, None)
        if not token:
            # print(f"[Warning]{sensor} has no png specified!")
            continue
        # 按照第二个维度来确定截取的信息是多少
        timestamp = token.split('_')[1]
        # 根据分割出的情况来确定图像的路径在哪里
        image_path = os.path.join(package_root, 'cameras', sensor,
                                token + '.png')
        cam_info.update({
            "type": sensor,
            "data_path": image_path,
            "timestamp": timestamp, # 是由token截取得到
            "sample_data_token": token,
            "package": os.path.basename(package_root),# 获取我们采用的是哪一个数据
            # 'sensor2ego_translation': list(map(lambda x:x, cam_info['lidar2cam_translation'])),# 默认传感器的信息和自车系统处于一个位置
            'sensor2ego_translation': list(map(lambda x:-x, cam_info['lidar2cam_translation'])),# 默认传感器的信息和自车系统处于一个位置
            'sensor2ego_rotation': list(
                # Quaternion(matrix=np.array(cam_info['lidar2cam_rotation']).reshape(3, 3))),
                Quaternion(matrix=np.array(cam_info['lidar2cam_rotation']).reshape(3, 3)).inverse),
            'ego2global_translation': [0, 0, 0],
            'ego2global_rotation': [1, 0, 0, 0],  
            'cam_intrinsic':np.array(cam_info['intrinsic']).reshape(3, 3).tolist()
            })
        cams.update({sensor: cam_info})
        # 此时的sensor不是一个字符串，而是上述的角度，每个角度包含原始json文件除了sensor信息的信息并且还包含
        # file_path等情况的信息
    return cams


class BEV3Ddataset():

    def __init__(self,
                #  dataset_cfg='hyper_data/configs/CylinderMono/cylinder_mono3d.py',
                 compensation_path=None,
                 dataset_type='detection',
                 root_path='/home/gpu/works/fk/task/hyper_dl/data/samples/bev3d' ,
                 version='train',
                 scale_ratio=1.0,
                 sensor_type='fisheye',
                 # save_image=False,
                 out_dir='/home/gpu/works/fk/task/hyper_dl/data/samples/bev3d' ):
        # self.cfg = dataset_cfg
        self.dataset_type = dataset_type
        self.root_path = root_path
        self.version = version
        self.scale_ratio = scale_ratio
        self.sensor_type = sensor_type
        # self.save_image = save_image
        self.out_dir = out_dir
        self.vir_camera = None
        self.lidar_infos = dict(
            {'metadata': {'version': version},
             'infos': []
             })
        self.cam_infos = []

    
    @staticmethod
    def get_lidar_calib(package_root):

        path = os.path.join(package_root, 'calib', 'lidars.json')
        with open(path, 'r') as file:
            calib_info = json.load(file)
        lidar_top_calib_list = [item for item in calib_info if item['sensor'] == "LIDAR_TOP"]
        if len(lidar_top_calib_list) != 1:
            print(f"lidar top has {len(lidar_top_calib_list)} calib info")
            raise AssertionError("LIDAR_TOP's calibs num wrong!")
        # 此时lidar_calib 就是获得LIDAR_TOP 这个信息，表示雷达具体放置的位置
        lidar_calib = lidar_top_calib_list[0]
        # 此时获得矩阵，矩阵里面包含的信息是旋转和平移矩阵的情况
        lidar2ego_translation = lidar_calib['lidar2ego_translation'] if \
            lidar_calib['lidar2ego_translation'] else []
        lidar2ego_rotation = lidar_calib['lidar2ego_rotation'] if \
            lidar_calib['lidar2ego_rotation'] else []
        return lidar2ego_translation, lidar2ego_rotation
    

    
    def create_cyl_data(self):
        train_nusc_infos = self.prepare_data()
        self.lidar_infos.update(infos=train_nusc_infos)
        print("********** Saving .pkl and coco.json... **********")
        mmcv.dump(self.lidar_infos, os.path.join(self.out_dir, f'cyl_infos_{self.version}.pkl'))


    def prepare_data(self):
        train_nusc_infos = []
        i = 0
        for item in mmcv.track_iter_progress(os.listdir(self.root_path)): 
            i=i+1
            if i%16 !=0:
                continue
            # if i>10:
            #     continue
            package_root = os.path.join(self.root_path, item)
            # if item.find('2023-03-03-')==-1:
            #     continue
            # print(item)
            if os.path.isdir(package_root):
                sync_folder = os.path.join(package_root, 'sync_info')
                sync_jsons = glob.glob(os.path.join(sync_folder, '*.json'))
                for file in sync_jsons:
                    with open(file, 'r') as f:
                        sync_info = json.load(f)
                    lidar_info = self.collect_lidar_data_with_2d_box(sync_info, package_root)#save image path and lidar gt;
                    if lidar_info is not None:
                        train_nusc_infos.append(lidar_info)
        return train_nusc_infos
          
    def collect_lidar_data_with_2d_box(self, sync_info, package_root):
        if len(sync_info['cameras'].keys())!=4:
            return None
        lidar_token = sync_info['lidars']['LIDAR_TOP']
        lidar_json = os.path.join(package_root, 'anno', self.dataset_type, '3d', f'{lidar_token}.json')
        lidar_path = os.path.join(os.path.basename(package_root), 'lidars', 'LIDAR_TOP', f'{lidar_token}.pcd')
        
        if not os.path.exists(lidar_json):
            return None
        
        with open(lidar_json, 'r') as f:
            json_data = json.load(f)
        
        record_path = json_data.pop('file_name', None)
        if record_path and record_path != lidar_path:
            raise AssertionError("lidar path not same")
        
        lidar2ego_translation, lidar2ego_rotation = self.get_lidar_calib(package_root)
        json_data['token'] = lidar_token  
        timestamp = lidar_token.split('_')[-1]

        
        info = {
            'lidar_path': os.path.join(package_root, 'lidars', 'LIDAR_TOP', f'{lidar_token}.pcd'),
            'lidar2ego_translation': lidar2ego_translation,
            'lidar2ego_rotation': lidar2ego_rotation,
            'ego2global_translation': [0, 0, 0],
            'ego2global_rotation': [1, 0, 0, 0],
            'sweeps': [0],
            'token': lidar_json,
            'package': os.path.basename(package_root),
            'timestamp': float(timestamp),
        }
        info['cams'] = get_cameras_calib(package_root, sync_info)
        info['ann_infos'] = dict()
        info['ann_infos']['gt_3d_labels'] = []
        info['ann_infos']['gt_3d_boxes'] = []
        info['ann_infos']['gt_3d_ids'] = []
        cam_2d_info = dict()
        cam_dict = ['FISHEYE_CAMERA_BACK', 'FISHEYE_CAMERA_FRONT', 'FISHEYE_CAMERA_LEFT', 'FISHEYE_CAMERA_RIGHT']
        cam_index = 0
        for cam_type in cam_dict:
            cam_token = sync_info['cameras'][cam_type]
            one_gt_boxes = []
            cam_info = dict()
            camera_json_path = os.path.join(package_root, 'anno', self.dataset_type, '2d',
                                            cam_token + '.json')
            camera_img_path = os.path.join(os.path.basename(package_root), 'cameras', cam_type,
                                           cam_token + '.png')
            if not os.path.exists(camera_json_path):
                # print(f"[Warning] {camera_json_path} not exists!")
                continue
            with open(camera_json_path, 'r') as f:
                data = json.load(f)
                # 只更新token和package用于后面通过lidar_infos下cams找到annotations并更新相机的内外参等信息
                data['token'] = cam_token
                data['package'] = os.path.basename(package_root)
                record_path = data.pop('file_name', None)
                if record_path and record_path != camera_img_path:
                    print(
                        f"record img path: {record_path} is not as same as dir path:{camera_img_path}")
                    raise AssertionError("lidar path not same")
                annotations = data['annotations']
                box_scale = 1.0
                if 'scale' in data.keys():
                    box_scale = 2.0

                for anno in annotations:
                    one_box = []

                    cls = anno['category']
                    if cls not in ['sedan',
                                   'suv',
                                   'bus',
                                   'tiny_car',
                                   'van',
                                   'lorry',
                                   'truck',
                                   'engineering_vehicle',
                                   'tanker',
                                   'cementing_unit',
                                   'semi-trailer',
                                   'fire_trucks',
                                   'ambulances',
                                   'police_vehicle']:
                        continue
                    occ_state = anno['occlusion']
                    if occ_state not in ['fully-visible', 'slightly-occluded', 'partially-occluded']:
                        continue
                    if 'id' not in anno.keys():  # FIXME: without id?
                        continue
                    ignore = anno['ignore']
                    if type(ignore) == str:
                        if ignore == 'true':
                            continue
                    elif ignore:
                        continue
                    truncation = anno['truncation']
                    if truncation == 'High':
                        continue
                    id = anno['id']
                    one_box.append(id)
                    box2d = anno['box2d']
                    x1 = box2d['x1'] * box_scale
                    y1 = box2d['y1'] * box_scale
                    x2 = box2d['x2'] * box_scale
                    y2 = box2d['y2'] * box_scale
                    xywh = [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1]
                    if xywh[2] <= 0 and xywh[3] <= 0:
                        continue
                    one_box = one_box + xywh

                    if anno.get('inner')==None:
                        one_box = one_box + [x1, x2]
                    else:
                        inner_box = anno['inner']
                        if len(inner_box.keys()) ==0:
                            one_box = one_box + [x1, x2]
                        else:
                            if inner_box.get('attr')==None:
                                one_box = one_box + [x1, x2]
                            else:
                                ibbox = inner_box['box2d']
                                lft  = ibbox['x1']*box_scale
                                rgt = ibbox['x2']*box_scale
                                one_box = one_box+[lft,rgt]
                    one_gt_boxes.append(one_box)
                cam_info['box2d'] = one_gt_boxes
                cam_info['package_root'] = os.path.dirname(package_root)
                cam_info['camera_img_path'] = camera_img_path
            cam_2d_info[cam_type] = cam_info

        annotations = json_data.pop('annotations', None)
        annotations = [anno for anno in annotations if
                       map_name_from_general_to_detection.get(anno['category'].lower(), 'ignore') != 'ignore']
        new_annotations = [anno for anno in annotations if anno['category'] in ['sedan',
                                    'suv',
                                    'bus',
                                    'tiny_car',
                                    'van',
                                    'lorry',
                                    'truck',
                                    'engineering_vehicle',
                                    'tanker',
                                    'cementing_unit',
                                    'semi-trailer',
                                    'fire_trucks',
                                    'ambulances',
                                    'police_vehicle'] ]
        annotations = []
        for anno in new_annotations:  # lidar annotations
            try:
                lidar_visible = False
                if anno['visible_cls'] == 'all_visible':
                    lidar_visible=True
                id = anno['id']
                cam_visible = False
                for cam_type in cam_dict:
                    cam_info = cam_2d_info[cam_type]
                    one_gt_boxes = cam_info['box2d']
                    for one_box in one_gt_boxes:
                        box_id = one_box[0]
                        if box_id == id:
                            cam_visible = True
                            break
                if lidar_visible and cam_visible:
                    annotations.append(anno)



            except:
                return None

        print(lidar_json)
        if len(annotations)!=0:
            locs = np.array([[anno['center3d']['x'], anno['center3d']['y'], anno['center3d']['z']] for anno in annotations]).reshape(-1, 3)
            dims = np.array([[anno['dim']['length'], anno['dim']['width'], anno['dim']['height']] for anno in annotations]).reshape(-1, 3)
            rots = np.array([anno['yaw'] for anno in annotations]).reshape(-1, 1)
            velocity = np.zeros((len(annotations), 2))
            gt_boxes = np.concatenate([locs, dims, rots,velocity], axis=1)
            assert len(gt_boxes) == len(annotations), f'{len(gt_boxes)}, {len(annotations)}'

            info['ann_infos']['gt_3d_labels']=[CLASSES_SIM.index(map_name_from_general_to_detection[anno['category']].lower()) for anno in annotations]
            info['ann_infos']['gt_3d_boxes'] = gt_boxes

            info['gt_velocity'] = velocity
            info['num_lidar_pts'] = np.array([anno['num_lidar_pts'] for anno in annotations])
            # info['valid_flag'] = np.array([anno['visible_cls'] == 'all_visible' for anno in annotations], dtype=bool).reshape(-1)
            info['ann_infos']['gt_3d_ids']=[anno['id'] for anno in annotations]
        #match 2d&3d;


        info['ann_infos']['cam_2d_info'] =cam_2d_info
        if len( info['ann_infos']['cam_2d_info'].keys()) !=4:
            info = None
            print("not find 4 cams")

        return info



parser = argparse.ArgumentParser(description='Data converter arg parser')
parser.add_argument(
    '--root-path',
    type=str,
    default='/media/gpu/HDD/fk/bev_fisheyey_part0',
    help='specify the root path of dataset')
parser.add_argument(
    '--dataset-config',
    default='configs/det_3d/bev3d/bevdet-cylin.py',
    type=str,
    help='specify the root path of dataset')
parser.add_argument(
    '--dataset-type',
    default='detection',
    type=str,
    help='detection or tracking')
parser.add_argument(
    '--version',
    type=str,
    default='train',
    required=False,
    help='specify the dataset version, no need for kitti')
parser.add_argument(
    '--out-dir',
    type=str,
    default='/media/gpu/HDD/fk/bev_fisheyey_part0',
    required=False,
    help='path to save coco.json and .pkl')
args = parser.parse_args()


if __name__ == '__main__':
    """
    USAGE: 
    python create_cyl_mono_dataset.py
    # dataset config
    --dataset-config configs/det_3d/bev3d/bevdet-cylin.py    
    # dataset type
    --dataset-type detection
    # dataset folder 
    --root-path data/samples/bev3d
    # dataset version currently only support train
    --version train 
    # output path to save coco.json & pkl files
    --out-dir data/samples/bev3d
    """

    dataset = 'nuscenes'
    version = 'v1.0'
    train_version = f'{version}-mini'
    dataset = BEV3Ddataset(compensation_path=None,
                                      dataset_type='detection',
                                      root_path=args.root_path,
                                      scale_ratio=1.0,
                                      sensor_type='fisheye',
                                      # save_image=args.save_image,
                                      version='train',
                                      out_dir=args.out_dir)
    
    dataset.create_cyl_data()



    