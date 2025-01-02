# Copyright (c) OpenMMLab. All rights reserved.
import tempfile
import os
import os.path as osp
import mmcv
import numpy as np
import pyquaternion
import argparse
from mmcv import Config
from hyper_data.common.core import show_result
from hyper_data.common.core.bbox import Box3DMode, Coord3DMode, LiDARInstance3DBoxes
from common.datasets.builder import DATASETS
from common.datasets.compose import Compose
from common.evaluation_tools.utils.data_classes import Box as NuScenesBox
from common.evaluation_tools.utils.data_classes import Real3dBox
from common.evaluation_tools.eval.detection.config import config_factory
from common.datasets.custom_3d import Custom3DDataset


@DATASETS.register_module(force=True)
class Cyl_BEV3D_Dataset(Custom3DDataset):
    """BEV3D Dataset.

    This class serves as the API for experiments on the Hongjing Dataset.

    Args:
        ann_file (str): Path of annotation file.
        pipeline (list[dict], optional): Pipeline used for data processing.
            Defaults to None.
        data_root (str): Path of dataset root.
        classes (tuple[str], optional): Classes used in the dataset.
            Defaults to None.
        load_interval (int, optional): Interval of loading the dataset. It is
            used to uniformly sample the dataset. Defaults to 1.
        with_velocity (bool, optional): Whether include velocity prediction
            into the experiments. Defaults to True.
        modality (dict, optional): Modality to specify the sensor data used
            as input. Defaults to None.
        box_type_3d (str, optional): Type of 3D box of this dataset.
            Based on the `box_type_3d`, the dataset will encapsulate the box
            to its original format then converted them to `box_type_3d`.
            Defaults to 'LiDAR' in this dataset. Available options includes.
            - 'LiDAR': Box in LiDAR coordinates.
            - 'Depth': Box in depth coordinates, usually for indoor dataset.
            - 'Camera': Box in camera coordinates.
        filter_empty_gt (bool, optional): Whether to filter empty GT.
            Defaults to True.
        test_mode (bool, optional): Whether the dataset is in test mode.
            Defaults to False.
        eval_version (bool, optional): Configuration version of evaluation.
            Defaults to  'detection_cvpr_2019'.
        use_valid_flag (bool, optional): Whether to use `use_valid_flag` key
            in the info file as mask to filter gt_boxes and gt_names.
            Defaults to False.
        img_info_prototype (str, optional): Type of img information.
            Based on 'img_info_prototype', the dataset will prepare the image
            data info in the type of 'mmcv' for official image infos,
            'bevdet' for BEVDet, and 'bevdet4d' for BEVDet4D.
            Defaults to 'mmcv'.
        multi_adj_frame_id_cfg (tuple[int]): Define the selected index of
            reference adjcacent frames.
        ego_cam (str): Specify the ego coordinate relative to a specified
            camera by its name defined in NuScenes.
            Defaults to None, which use the mean of all cameras.
    """
    # https://github.com/nutonomy/nuscenes-devkit/blob/57889ff20678577025326cfc24e57424a829be0a/python-sdk/nuscenes/eval/detection/evaluate.py#L222 # noqa
    ErrNameMapping = {
        'trans_err': 'mATE',
        'scale_err': 'mASE',
        'orient_err': 'mAOE',
        'vel_err': 'mAVE',
        'attr_err': 'mAAE'
    }
    # default
    CLASSES = ['vehicle', 'pedestrian', 'rider']
    CLASS_NAMES = (
        'sedan', 'suv', 'bus', 'tiny_car', 'van', 'lorry', 'truck', 'engineering_vehicle', 'tanker', 'cementing_unit', 'semi-trailer',
        'fire_trucks', 'ambulances', 'police_vehicle', 'bicycle', 'motor', 'fork_truck', 'travel_trailer','pedestrian', 'rider', 'rear', 'head', 'wheels',
        'trolley', 'animal', 'stacked_trolleys', 'non-motor_vehicle_group', 'crowd', 'cement_column', 'bollard',
        'folding_warning_sign', 'traffic_cone', 'parking_lock_open','parking_lock_close', 'fire_hydrant_air', 'fire_hydrant_gnd',
        'cement_sphere', 'water_barrier', 'charging_pile', 'reflector', 'distribution_box', 'single_fence', 'multi_fence', 'trash_can',
        'parking_sign', 'triangle_warning'
    )
    VEHICLE_CLASS_NAMES = ['sedan', 'suv', 'bus', 'tiny_car', 'van', 'lorry', 'truck', 'engineering_vehicle', 'tanker', 'cementing_unit',
                           'semi-trailer','fire_trucks', 'ambulances', 'police_vehicle', 'bicycle', 'motor', 'fork_truck', 'travel_trailer']
    PEDESTRIAN_CLASS_NAMES = ['pedestrian']
    RIDER_CLASS_NAMES = ['rider']


    SOD_CLASSES = ('cement_column', 'bollard', 'folding_warning_sign', 'traffic_cone', 'parking_lock_open','parking_lock_close')
    ErrNameMapping = {
        'trans_err': 'mATE',
        'scale_err': 'mASE',
        'orient_err': 'mAOE',
        'vel_err': 'mAVE',
        'attr_err': 'mAAE'
    }

    REAL3D_CLASS_NAME = {
        'vehicle', 'person', 'rider'
    }

    def __init__(self,
                 ann_file,
                 pipeline=None,
                 data_root=None,
                 classes=None,
                 load_interval=1,
                 with_velocity=True,
                 modality=None,
                 box_type_3d='LiDAR',
                 filter_empty_gt=True,
                 test_mode=False,
                 eval_version='real3d',
                 use_valid_flag=False,
                 img_info_prototype='mmcv',
                 multi_adj_frame_id_cfg=None,
                 ego_cam='FISHEYE_CAMERA_FRONT',
                 PALETTE = None,
                 
                 stereo=False):
        self.load_interval = load_interval
        self.use_valid_flag = use_valid_flag
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            classes=classes,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode)

        self.with_velocity = with_velocity
        self.eval_version = eval_version
        # from nuscenes.eval.detection.config import config_factory
        self.eval_detection_configs = config_factory(self.eval_version)
        if self.modality is None:
            self.modality = dict(
                use_camera=True,
                use_lidar=False,
                use_radar=False,
                use_map=False,
                use_external=False,
            )

        self.img_info_prototype = img_info_prototype
        self.multi_adj_frame_id_cfg = multi_adj_frame_id_cfg
        self.ego_cam = ego_cam
        self.CLASSES = self.get_classes(classes)
        self.PALETTE= PALETTE
        self.CLASS_NAMES = tuple(name.lower() for name in self.CLASS_NAMES)
        self.VEHICLE_CLASS_NAMES = [name.lower() for name in self.VEHICLE_CLASS_NAMES]
        self.PEDESTRIAN_CLASS_NAMES = [name.lower() for name in self.PEDESTRIAN_CLASS_NAMES]
        self.RIDER_CLASS_NAMES = [name.lower() for name in self.RIDER_CLASS_NAMES]

        self.cat_ids  = range(0, self.CLASS_NAMES.index('single_fence') + 1)
        self.cat2_label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        for i in range(self.CLASS_NAMES.index('travel_trailer') + 1):
            self.cat2_label[i] = 0
        self.cat2_label[self.CLASS_NAMES.index('pedestrian')] = 1
        self.cat2_label[self.CLASS_NAMES.index('rider')] = 2
        self.cat2_label[self.CLASS_NAMES.index('rear')] = 3
        self.cat2_label[self.CLASS_NAMES.index('head')] = 3
        self.cat2_label[self.CLASS_NAMES.index('wheels')] = 3

        self.cat2_label_3d = {}
        for i, k in enumerate(self.VEHICLE_CLASS_NAMES):
            self.cat2_label_3d[k] = i
        for i, k in enumerate(self.PEDESTRIAN_CLASS_NAMES):
            self.cat2_label_3d[k] = i
        for i, k in enumerate(self.RIDER_CLASS_NAMES):
            self.cat2_label_3d[k] = i

        self.stereo = stereo 

    def get_cat_ids(self, idx):
        """Get category distribution of single scene.

        Args:
            idx (int): Index of the data_info.

        Returns:
            dict[list]: for each category, if the current scene
                contains such boxes, store a list containing idx,
                otherwise, store empty list.
        """
        info = self.data_infos[idx]
        if self.use_valid_flag and 'valid_flag' in info:
            mask = info['valid_flag']
            gt_names = set(info.get('gt_names', [])[mask])
        else:
            gt_names = set(info.get('gt_names', []))

        cat_ids = []
        for name in gt_names:
            if name in self.CLASSES:
                cat_ids.append(self.cat2id[name])
        return cat_ids

    def load_annotations(self, ann_file):# 2
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations sorted by timestamps.
        """
        data = mmcv.load(ann_file, file_format='pkl')
        data_infos = list(sorted(data['infos'], key=lambda e: e['timestamp']))
        data_infos = data_infos[::self.load_interval]
        self.metadata = data['metadata']
        self.version = self.metadata['version']
        return data_infos

    def get_data_info(self, index):#5
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        info = self.data_infos[index]
        # standard protocol modified from SECOND.Pytorch
        input_dict = dict(
            sample_idx=info['token'],
            pts_filename=info['lidar_path'],
            sweeps=info['sweeps'],
            timestamp=info['timestamp'] / 1e6,
        )
        if 'ann_infos' in info:
            input_dict['ann_infos'] = info['ann_infos']
        if self.modality['use_camera']:
            if self.img_info_prototype == 'mmcv':
                image_paths = []
                lidar2img_rts = []
                for cam_type, cam_info in info['cams'].items():
                    image_paths.append(cam_info['data_path'])
                    # obtain lidar to image transformation matrix
                    lidar2cam_r = np.linalg.inv(
                        cam_info['sensor2lidar_rotation'])
                    lidar2cam_t = cam_info[
                        'sensor2lidar_translation'] @ lidar2cam_r.T
                    lidar2cam_rt = np.eye(4)
                    lidar2cam_rt[:3, :3] = lidar2cam_r.T
                    lidar2cam_rt[3, :3] = -lidar2cam_t
                    intrinsic = cam_info['cam_intrinsic']
                    viewpad = np.eye(4)
                    viewpad[:intrinsic.shape[0], :intrinsic.
                            shape[1]] = intrinsic
                    lidar2img_rt = (viewpad @ lidar2cam_rt.T)
                    lidar2img_rts.append(lidar2img_rt)

                input_dict.update(
                    dict(
                        img_filename=image_paths,
                        lidar2img=lidar2img_rts,
                    ))

                if not self.test_mode:
                    annos = self.get_ann_info(index)
                    input_dict['ann_info'] = annos
            else:
                assert 'bevdet' in self.img_info_prototype
                input_dict.update(dict(curr=info))
                if '4d' in self.img_info_prototype:
                    info_adj_list = self.get_adj_info(info, index)
                    input_dict.update(dict(adjacent=info_adj_list))
        return input_dict

    def get_adj_info(self, info, index):
        info_adj_list = []
        adj_id_list = list(range(*self.multi_adj_frame_id_cfg))
        if self.stereo:
            assert self.multi_adj_frame_id_cfg[0] == 1
            assert self.multi_adj_frame_id_cfg[2] == 1
            adj_id_list.append(self.multi_adj_frame_id_cfg[1])
        for select_id in adj_id_list:
            select_id = max(index - select_id, 0)
            if not self.data_infos[select_id]['scene_token'] == info[
                    'scene_token']:
                info_adj_list.append(info)
            else:
                info_adj_list.append(self.data_infos[select_id])
        return info_adj_list

    def get_ann_info(self, index):
        """Get annotation info according to the given index.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: Annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`):
                    3D ground truth bboxes
                - gt_labels_3d (np.ndarray): Labels of ground truths.
                - gt_names (list[str]): Class names of ground truths.
        """
        info = self.data_infos[index]
        # filter out bbox containing no points
        if self.use_valid_flag:
            mask = info['valid_flag']
        else:
            mask = info['num_lidar_pts'] > 0
        gt_bboxes_3d = info['gt_boxes'][mask]
        gt_names_3d = info['gt_names'][mask]
        gt_labels_3d = []
        for cat in gt_names_3d:
            if cat in self.VEHICLE_CLASS_NAMES:
                gt_labels_3d.append(self.CLASSES.index('vehicle'))
            elif cat in self.PEDESTRIAN_CLASS_NAMES:
                gt_labels_3d.append(self.CLASSES.index('pedestrian'))
            elif cat in self.RIDER_CLASS_NAMES:
                gt_labels_3d.append(self.CLASSES.index('rider'))
            else:
                gt_labels_3d.append(-1)
        gt_labels_3d = np.array(gt_labels_3d)

        if self.with_velocity:
            gt_velocity = info['gt_velocity'][mask]
            nan_mask = np.isnan(gt_velocity[:, 0])
            gt_velocity[nan_mask] = [0.0, 0.0]
            gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocity], axis=-1)

        # the nuscenes box center is [0.5, 0.5, 0.5], we change it to be
        # the same as KITTI (0.5, 0.5, 0)
        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d,
            box_dim=gt_bboxes_3d.shape[-1],
            origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_names=gt_names_3d)
        return anns_results     
    
    def _format_bbox(self, results, jsonfile_prefix=None):
        """Convert the results to the standard format.

        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str): The prefix of the output jsonfile.
                You can specify the output directory/filename by
                modifying the jsonfile_prefix. Default: None.

        Returns:
            str: Path of the output json file.
        """
        nusc_annos = {}
        mapped_class_names = self.CLASSES

        print('Start to convert detection format...')
        for sample_id, det in enumerate(mmcv.track_iter_progress(results)):
            boxes = det['boxes_3d'].tensor.numpy()
            scores = det['scores_3d'].numpy()
            labels = det['labels_3d'].numpy()
            sample_token = self.data_infos[sample_id]['token']

            trans = self.data_infos[sample_id]['cams'][
                self.ego_cam]['ego2global_translation']
            rot = self.data_infos[sample_id]['cams'][
                self.ego_cam]['ego2global_rotation']
            rot = pyquaternion.Quaternion(rot)
            annos = list()
            for i, box in enumerate(boxes):
                name = mapped_class_names[labels[i]]
                center = box[:3]
                wlh = box[[4, 3, 5]]
                box_yaw = box[6]
                box_vel = box[7:].tolist()
                box_vel.append(0)
                quat = pyquaternion.Quaternion(axis=[0, 0, 1], radians=box_yaw)
                nusc_box = NuScenesBox(center, wlh, quat, velocity=box_vel)
                nusc_box.rotate(rot)
                nusc_box.translate(trans)
                if np.sqrt(nusc_box.velocity[0]**2 +
                           nusc_box.velocity[1]**2) > 0.2:
                    if name in [
                            'car',
                            'construction_vehicle',
                            'bus',
                            'truck',
                            'trailer',
                    ]:
                        attr = 'vehicle.moving'
                    elif name in ['bicycle', 'motorcycle']:
                        attr = 'cycle.with_rider'
                    else:
                        attr = self.DefaultAttribute[name]
                else:
                    if name in ['pedestrian']:
                        attr = 'pedestrian.standing'
                    elif name in ['bus']:
                        attr = 'vehicle.stopped'
                    else:
                        attr = self.DefaultAttribute[name]
                nusc_anno = dict(
                    sample_token=sample_token,
                    translation=nusc_box.center.tolist(),
                    size=nusc_box.wlh.tolist(),
                    rotation=nusc_box.orientation.elements.tolist(),
                    velocity=nusc_box.velocity[:2],
                    detection_name=name,
                    detection_score=float(scores[i]),
                    attribute_name=attr,
                )
                annos.append(nusc_anno)
            # other views results of the same frame should be concatenated
            if sample_token in nusc_annos:
                nusc_annos[sample_token].extend(annos)
            else:
                nusc_annos[sample_token] = annos
        nusc_submissions = {
            'meta': self.modality,
            'results': nusc_annos,
        }

        mmcv.mkdir_or_exist(jsonfile_prefix)
        res_path = osp.join(jsonfile_prefix, 'results_nusc.json')
        print('Results writes to', res_path)
        mmcv.dump(nusc_submissions, res_path)
        return res_path

    def _format_bbox_real3d(self, results, jsonfile_prefix=None):
        classes = ['vehicle', 'pedestrian', 'rider']
        real3d_annos = {}
        mapped_class_names = classes
        for sample_id, det in enumerate(mmcv.track_iter_progress(results)):
            annos = []
            boxes = output_to_real3d_box(det)
            anno_filepath = self.data_infos[sample_id]['token']
            sample_token = anno_filepath.split('/')[-1].replace('.json', '')
            img_filepath = anno_filepath.split('/')[-5]
            # cam_boxes3d, scores, labels = real3d_box_to_cam_box3d(boxes)
            # det = bbox3d2result(cam_boxes3d, scores, labels)

            for i, box in enumerate(boxes):
                name = mapped_class_names[box.label]
                real3d_anno = dict(
                    sample_token=sample_token,
                    translation=box.center.tolist(),
                    size=box.wlh.tolist(),
                    rotation=box.orientation.elements.tolist(),
                    detection_score=box.score,
                    detection_name=name,
                    anno_filepath=anno_filepath,
                    img_filepath=img_filepath
                )
                annos.append(real3d_anno)
            if sample_token in real3d_annos:
                real3d_annos[sample_token].extend(annos)
            else:
                real3d_annos[sample_token] = annos
        real3d_submissions = {
            'meta' : self.modality,
            'results' : real3d_annos
        }
        mmcv.mkdir_or_exist(jsonfile_prefix)
        res_path = osp.join(jsonfile_prefix, 'results_real3d.json')
        print('Results writes to', res_path)
        mmcv.dump(real3d_submissions, res_path)
        return res_path
    
    def _evaluate_single(self,
                         result_path,
                         logger=None,
                         metric='bbox',
                         result_name='pts_bbox'):
        """Evaluation for a single model in nuScenes protocol.

        Args:
            result_path (str): Path of the result file.
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Default: None.
            metric (str, optional): Metric name used for evaluation.
                Default: 'bbox'.
            result_name (str, optional): Result name in the metric prefix.
                Default: 'pts_bbox'.

        Returns:
            dict: Dictionary of evaluation details.
        """
        from hyper_data.common.evaluation_tools.real3d import Real3d
        from hyper_data.common.evaluation_tools.bev3d import BEV3dEval

        output_dir = osp.join(*osp.split(result_path)[:-1])
        real3d = Real3d(
            dataroot=self.ann_file.replace(self.ann_file.split('/')[-1], ''),   
            train_file="cyl_infos_train.pkl",
            val_file="cyl_infos_val.pkl",
            test_file="cyl_infos_val.pkl"
        )
        real3d_eval = BEV3dEval(
            real3d=real3d,
            config=self.eval_detection_configs,
            result_path=result_path,
            output_dir=output_dir
        )
        real3d_eval.main()

        # record metrics
        metrics = mmcv.load(osp.join(output_dir, 'metrics_summary.json'))
        detail = dict()
        metric_prefix = f'{result_name}_real3d'
        for name in ['vehicle', 'pedestrian', 'rider']:
            for k, v in metrics['label_aps'][name].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}_AP_dist_{}'.format(metric_prefix, name, k)] = val
            for k, v in metrics['label_tp_errors'][name].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}_{}'.format(metric_prefix, name, k)] = val
            for k, v in metrics['tp_errors'].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}'.format(metric_prefix,
                                      self.ErrNameMapping[k])] = val

        detail['{}/NDS'.format(metric_prefix)] = metrics['nd_score']
        detail['{}/mAP'.format(metric_prefix)] = metrics['mean_ap']
        return detail

    def format_results(self, results, jsonfile_prefix=None):
        """Format the results to json (standard format for COCO evaluation).

        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: Returns (result_files, tmp_dir), where `result_files` is a
                dict containing the json filepaths, `tmp_dir` is the temporal
                directory created for saving json files when
                `jsonfile_prefix` is not specified.
        """
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: {} != {}'.
            format(len(results), len(self)))

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None

        # currently the output prediction results could be in two formats
        # 1. list of dict('boxes_3d': ..., 'scores_3d': ..., 'labels_3d': ...)
        # 2. list of dict('pts_bbox' or 'img_bbox':
        #     dict('boxes_3d': ..., 'scores_3d': ..., 'labels_3d': ...))
        # this is a workaround to enable evaluation of both formats on nuScenes
        # refer to https://github.com/open-mmlab/mmdetection3d/issues/449
        if not ('pts_bbox' in results[0] or 'img_bbox' in results[0]):
            result_files = self._format_bbox(results, jsonfile_prefix)
        else:
            # should take the inner dict out of 'pts_bbox' or 'img_bbox' dict
            result_files = dict()
            for name in results[0]:
                print(f'\nFormating bboxes of {name}')
                results_ = [out[name] for out in results]
                tmp_file_ = osp.join(jsonfile_prefix, name)
                result_files.update(
                    {name: self._format_bbox(results_, tmp_file_)})
        return result_files, tmp_dir

    def format_results_real3d(self, results, jsonfile_prefix=None, **kwargs):
        # assert (results, list)
        # assert len(results) == len(self)

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None

        results_files = dict()
        for name in results[0]:
            if '2d' in name:
                continue
            results_ = [out[name] for out in results]
            tmp_file_ = osp.join(jsonfile_prefix, name)
            results_files.update({
                name: self._format_bbox_real3d(results_, tmp_file_)
            })
        return results_files, tmp_dir
    
    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 result_names=['pts_bbox'],
                 show=False,
                 out_dir=None,
                 pipeline=None):
        """Evaluation in nuScenes protocol.

        Args:
            results (list[dict]): Testing results of the dataset.
            metric (str | list[str], optional): Metrics to be evaluated.
                Default: 'bbox'.
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str, optional): The prefix of json files including
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            show (bool, optional): Whether to visualize.
                Default: False.
            out_dir (str, optional): Path to save the visualization results.
                Default: None.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.

        Returns:
            dict[str, float]: Results of each evaluation metric.
        """
        # result_files, tmp_dir = self.format_results(results, jsonfile_prefix)   # 存储推理结果
        result_files, tmp_dir = self.format_results_real3d(results, jsonfile_prefix)

        if isinstance(result_files, dict):
            results_dict = dict()
            for name in result_names:
                print('Evaluating bboxes of {}'.format(name))
                ret_dict = self._evaluate_single(result_files[name])
            results_dict.update(ret_dict)
        elif isinstance(result_files, str):
            results_dict = self._evaluate_single(result_files)

        if tmp_dir is not None:
            tmp_dir.cleanup()

        if show or out_dir:
            self.show(results, out_dir, show=show, pipeline=pipeline)
        return results_dict

    def _build_default_pipeline(self):
        """Build the default pipeline for this dataset."""
        pipeline = [
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=5,
                use_dim=5,
                file_client_args=dict(backend='disk')),
            dict(
                type='LoadPointsFromMultiSweeps',
                sweeps_num=10,
                file_client_args=dict(backend='disk')),
            dict(
                type='DefaultFormatBundle3D',
                class_names=self.CLASSES,
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ]
        return Compose(pipeline)
    
    def show(self, results, out_dir, show=False, pipeline=None):
        """Results visualization.

        Args:
            results (list[dict]): List of bounding boxes results.
            out_dir (str): Output directory of visualization result.
            show (bool): Whether to visualize the results online.
                Default: False.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.
        """
        assert out_dir is not None, 'Expect out_dir, got none.'
        pipeline = self._get_pipeline(pipeline)
        for i, result in enumerate(results):
            if 'pts_bbox' in result.keys():
                result = result['pts_bbox']
            data_info = self.data_infos[i]
            pts_path = data_info['lidar_path']
            file_name = osp.split(pts_path)[-1].split('.')[0]
            points = self._extract_data(i, pipeline, 'points').numpy()
            # for now we convert points into depth mode
            points = Coord3DMode.convert_point(points, Coord3DMode.LIDAR,
                                               Coord3DMode.DEPTH)
            inds = result['scores_3d'] > 0.1
            gt_bboxes = self.get_ann_info(i)['gt_bboxes_3d'].tensor.numpy()
            show_gt_bboxes = Box3DMode.convert(gt_bboxes, Box3DMode.LIDAR,
                                               Box3DMode.DEPTH)
            pred_bboxes = result['boxes_3d'][inds].tensor.numpy()
            show_pred_bboxes = Box3DMode.convert(pred_bboxes, Box3DMode.LIDAR,
                                                 Box3DMode.DEPTH)
            show_result(points, show_gt_bboxes, show_pred_bboxes, out_dir,
                        file_name, show)


def output_to_nusc_box(detection, with_velocity=True):
    """Convert the output to the box class in the nuScenes.

    Args:
        detection (dict): Detection results.

            - boxes_3d (:obj:`BaseInstance3DBoxes`): Detection bbox.
            - scores_3d (torch.Tensor): Detection scores.
            - labels_3d (torch.Tensor): Predicted box labels.

    Returns:
        list[:obj:`NuScenesBox`]: List of standard NuScenesBoxes.
    """
    box3d = detection['boxes_3d']
    scores = detection['scores_3d'].numpy()
    labels = detection['labels_3d'].numpy()

    box_gravity_center = box3d.gravity_center.numpy()
    box_dims = box3d.dims.numpy()
    box_yaw = box3d.yaw.numpy()

    # our LiDAR coordinate system -> nuScenes box coordinate system
    nus_box_dims = box_dims[:, [1, 0, 2]]

    box_list = []
    for i in range(len(box3d)):
        quat = pyquaternion.Quaternion(axis=[0, 0, 1], radians=box_yaw[i])
        if with_velocity:
            velocity = (*box3d.tensor[i, 7:9], 0.0)
        else:
            velocity = (0, 0, 0)
        # velo_val = np.linalg.norm(box3d[i, 7:9])
        # velo_ori = box3d[i, 6]
        # velocity = (
        # velo_val * np.cos(velo_ori), velo_val * np.sin(velo_ori), 0.0)
        box = NuScenesBox(
            box_gravity_center[i],
            nus_box_dims[i],
            quat,
            label=labels[i],
            score=scores[i],
            velocity=velocity)
        box_list.append(box)
    return box_list

def output_to_real3d_box(detection):
    box3d = detection['boxes_3d']
    scores = detection['scores_3d'].numpy()
    labels = detection['labels_3d'].numpy()

    box_gravity_center = box3d.gravity_center.numpy()
    box_dims = box3d.dims.numpy()
    box_yaw = box3d.yaw.numpy()

    # convert the dim/rot to nuscbox convention
    box_dims[:, [0, 1, 2]] = box_dims[:, [2, 0, 1]]
    box_yaw = -box_yaw

    box_list = []
    for i in range(len(box3d)):
        q1 = pyquaternion.Quaternion(axis=[0, 0, 1], radians=box_yaw[i])
        q2 = pyquaternion.Quaternion(axis=[1, 0, 0], radians=np.pi / 2)
        quat = q2 * q1
        box = Real3dBox(
            box_gravity_center[i],
            box_dims[i],
            quat,
            label=labels[i],
            score=scores[i]
        )
        box_list.append(box)
    return box_list

def lidar_nusc_box_to_global(info,
                             boxes,
                             classes,
                             eval_configs,
                             eval_version='real3d'):
    """Convert the box from ego to global coordinate.

    Args:
        info (dict): Info for a specific sample data, including the
            calibration information.
        boxes (list[:obj:`NuScenesBox`]): List of predicted NuScenesBoxes.
        classes (list[str]): Mapped classes in the evaluation.
        eval_configs (object): Evaluation configuration object.
        eval_version (str, optional): Evaluation version.
            Default: 'detection_cvpr_2019'

    Returns:
        list: List of standard NuScenesBoxes in the global
            coordinate.
    """
    box_list = []
    for box in boxes:
        # Move box to ego vehicle coord system
        box.rotate(pyquaternion.Quaternion(info['lidar2ego_rotation']))
        box.translate(np.array(info['lidar2ego_translation']))
        # filter det in ego.
        cls_range_map = eval_configs.class_range
        radius = np.linalg.norm(box.center[:2], 2)
        det_range = cls_range_map[classes[box.label]]
        if radius > det_range:
            continue
        # Move box to global coord system
        box.rotate(pyquaternion.Quaternion(info['ego2global_rotation']))
        box.translate(np.array(info['ego2global_translation']))
        box_list.append(box)
    return box_list

def parse_args():
    parser = argparse.ArgumentParser(description='Cyl_BEV3D_Dataset debug')
    parser.add_argument('config', help='config file path')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    dataset = Cyl_BEV3D_Dataset(
        ann_file=cfg.data.train.ann_file,
        pipeline=cfg.data.train.pipeline,
        data_root=cfg.data.train.data_root,
        classes=cfg.class_names,
        load_interval=cfg.data.train.load_interval,
        with_velocity=cfg.with_velocity,
        modality=cfg.modality,
        box_type_3d=cfg.box_type_3d,
        filter_empty_gt=cfg.filter_empty_gt,
        test_mode=cfg.data.test_mode,
        eval_version=cfg.eval_version,
        use_valid_flag=cfg.use_valid_flag,
        img_info_prototype=cfg.img_info_prototype,
        multi_adj_frame_id_cfg=cfg.multi_adj_frame_id_cfg,
        ego_cam=cfg.ego_cam,
        stereo=cfg.stereo
    )

    for i in range(len(dataset)):
        data = dataset[i]
        print(f"Index: {i}")
        print(f"Sample ID: {data['sample_idx']}")
        print(f"Points shape: {data['points'].shape}")
        if 'img' in data:
            print(f"Image shape: {data['img'].shape}")
        if 'gt_bboxes_3d' in data:
            print(f"GT bboxes 3D: {data['gt_bboxes_3d']}")
            print(f"GT labels 3D: {data['gt_labels_3d']}")
        print("------------------------")

if __name__ == '__main__':
    local = True
    main(local)
