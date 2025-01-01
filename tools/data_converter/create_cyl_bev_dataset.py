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


class CylBevDatasetConverter:
    def __init__(self,
                 root_path='hyper_data/data/cyl_detection',
                 sensor_type='fisheye',
                 sub_type='train',
                 version=None,
                 out_dir='hyper_data/data/cyl_detection',
                 ):
        self.root_path = root_path
        self.sub_type = sub_type
        self.version = version
        self.out_dir = out_dir
        self.dataset_type = 'detection'
        self.sensor_type = sensor_type
        self.valid_cameras = ['FISHEYE_CAMERA_BACK', 'FISHEYE_CAMERA_FRONT',
                              'FISHEYE_CAMERA_LEFT', 'FISHEYE_CAMERA_RIGHT',
                              'FISHEYE_CAMERA_REAR']
        self.bev_data = dict(
            {'metadata': {'version': sub_type},
             'infos': []
             })
        self.cam_infos = {}

    def prepare_data(self):
        for item in tqdm(os.listdir(self.root_path)):
            # 单包数据
            package_root = os.path.join(self.root_path, item)
            if os.path.isdir(package_root):
                sync_folder = os.path.join(package_root, 'sync_info')
                sync_jsons = glob.glob(os.path.join(sync_folder, '*.json'))
                if not os.path.exists(sync_folder):
                    continue
                for file in sync_jsons:
                    with open(file, 'r') as f:
                        sync_info = json.load(f)
                        cams2del = set()
                        for cam, token in sync_info['cameras'].items():
                            if cam not in self.valid_cameras:
                                cams2del.add(cam)
                        for key in cams2del:
                            sync_info['cameras'].pop(key)
                    self.collect_info(sync_info, package_root)


    def collect_info(self, sync_info, package_root):
        lidar_token = str(sync_info['lidars']['LIDAR_TOP'])
        lidar_json = os.path.join(package_root, 'anno', self.dataset_type, '3d',
                                  lidar_token + '.json')
        if not os.path.exists(lidar_json):
            return
        with open(lidar_json, 'r') as f:
            json_data = json.load(f)
        record_path = json_data.pop('file_name', None)
        if record_path and os.path.splitext(os.path.basename(record_path))[0] != \
                lidar_token:
            print(f"record pcd path: {record_path} is not as same as dir path: {lidar_token}")
        lidar2ego_translation, lidar2ego_rotation = self.get_lidar_calib(package_root)
        cams = self.get_cameras_calib(package_root, sync_info)
        if len(cams.keys()) == 0:
            return
        vir_camera_intrinsic = self.get_vir_camera_intrinsic(package_root)
        for cam in cams.keys():
            cams[cam]['vir_camera_intrinsic'] = vir_camera_intrinsic
        info = {
            'lidar_path': record_path,
            'token': lidar_token,
            'timestamp': json_data['timestamp'],
            'sweeps': [0],
            'package': os.path.relpath(package_root, self.root_path),
            'lidar2ego_translation': lidar2ego_translation,
            'lidar2ego_rotation': lidar2ego_rotation,
            'ego2global_translation': [0, 0, 0],
            'ego2global_rotation': [1, 0, 0, 0],
            'cams': cams
        }
        bev_annotations = [ann for ann in json_data['annotations'] if ann['category'].lower() in class_names]

        # collect camera 2d annotations
        for sensor in self.valid_cameras:
            cam_token = str(sync_info['cameras'][sensor])
            camera_json_path = os.path.join(package_root, 'anno', self.dataset_type, '2d',
                                            cam_token + '.json')
            if not os.path.exists(camera_json_path):
                continue
            pack_rel_path = os.path.relpath(package_root, self.root_path)
            camera_img_path = None
            if os.path.exists(os.path.join(package_root, 'cameras', sensor,
                                           cam_token + '.png')):
                camera_img_path = os.path.join(pack_rel_path, 'cameras', sensor,
                                               cam_token + '.png')
            elif os.path.exists(os.path.join(package_root, 'cameras', sensor,
                                             cam_token + '.jpg')):
                camera_img_path = os.path.join(pack_rel_path, 'cameras', sensor,
                                               cam_token + '.jpg')
        self.bev_data['infos'].append(info)

    @staticmethod
    def get_lidar_calib(package_root):
        path = os.path.join(package_root, 'calib', 'lidars.json')
        if not os.path.exists(path):
            calib_info = [
                {
                    "sensor": "LIDAR_TOP",
                    "lidar2ego_translation": [],
                    "lidar2ego_rotation": []
                }
            ]
        else:
            with open(path, 'r') as file:
                calib_info = json.load(file)
        lidar_top_calib_list = [item for item in calib_info if item['sensor'] == "LIDAR_TOP"]
        if len(lidar_top_calib_list) != 1:
            print(f"lidar top has {len(lidar_top_calib_list)} calib info")
            raise AssertionError("LIDAR_TOP's calibs num wrong!")
        lidar_calib = lidar_top_calib_list[0]
        lidar2ego_translation = lidar_calib['lidar2ego_translation'] if \
            lidar_calib['lidar2ego_translation'] else []
        lidar2ego_rotation = lidar_calib['lidar2ego_rotation'] if \
            lidar_calib['lidar2ego_rotation'] else []
        return lidar2ego_translation, lidar2ego_rotation

    def get_vir_camera_intrinsic(self, package_root):
        path = os.path.join(package_root, 'calib', 'cyl_camera.json')
        if not os.path.exists(path):
            path = os.path.join(package_root, 'calib', 'cyl_camera_' + self.sensor_type + '.json')
        with open(path, 'r') as file:
            calib_info = json.load(file)
            return np.array([calib_info['focal_length'], 0, calib_info['cx'],
                             0, calib_info['focal_length'], calib_info['cy'],
                             0, 0, 1], dtype=np.float32).reshape(3, 3)

    def get_cameras_calib(self, package_root, sync_info):
        path = os.path.join(package_root, 'calib', 'cameras.json')
        with open(path, 'r') as file:
            calib_infos = json.load(file)
        if len(calib_infos) == 0:
            raise LookupError("No camera calibs found!")
        cams = {}
        for cam_info in calib_infos:
            sensor = cam_info.pop('sensor')
            if sensor not in self.valid_cameras:
                continue
            token = sync_info['cameras'].get(sensor, None)
            if not token:
                continue
            token = str(token)
            if '_' in token:
                token_lower = token.lower()
                if ('front' in token_lower or 'back' in token_lower or 'rear' in token_lower or
                        'left' in token_lower or 'right' in token_lower):
                    timestamp = token.split('_')[0]
                else:
                    timestamp = token.split('_')[1]
            else:
                timestamp = token
            image_path = None
            if os.path.exists(os.path.join(package_root, 'cameras', sensor, token + '.png')):
                image_path = os.path.join(os.path.relpath(package_root, self.root_path), 'cameras', sensor,
                                          token + '.png')
            elif os.path.exists(os.path.join(package_root, 'cameras', sensor, token + '.jpg')):
                image_path = os.path.join(os.path.relpath(package_root, self.root_path), 'cameras', sensor,
                                          token + '.jpg')
            if image_path is None:
                # sync info内的cam是全量的，送标的不一定全
                continue
            lidar2cam_rotation = cam_info.pop('lidar2cam_rotation', None) or cam_info.pop('vcs2cam_rotation', None)
            lidar2cam_translation = cam_info.pop('lidar2cam_translation', None) or cam_info.pop('vcs2cam_translation',
                                                                                                None)

            cam_info.update({
                "type": sensor,
                "data_path": image_path,
                "timestamp": timestamp,
                "sample_data_token": token,
                "package": os.path.basename(package_root),
                'sensor2ego_translation': list(map(lambda x: -x, lidar2cam_translation)),  # 默认传感器的信息和自车系统处于一个位置
                'sensor2ego_rotation': list(
                    Quaternion(matrix=np.array(lidar2cam_rotation).reshape(3, 3)).inverse),
                'ego2global_translation': [0, 0, 0],
                'ego2global_rotation': [1, 0, 0, 0],
                'cam_intrinsic': np.array(cam_info.pop('intrinsic')).reshape(3, 3).tolist()
            })
            cams.update({sensor: cam_info})
        return cams


if __name__ == '__main__':
    with open('/home/lby/Projects/DeepLearning/BEVDet/data/nuscenes/bevdetv3-nuscenes_infos_train.pkl', 'rb') as f:
        bevdet_data = pickle.load(f)

    print("completed")
    convertor = CylBevDatasetConverter(root_path='/home/lby/Data/parking_dataset/',
                                       version='parking',
                                       out_dir='/home/lby/Data/parking_dataset/',)
    convertor.prepare_data()