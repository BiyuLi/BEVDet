## Get Started

#### Installation and Data Preparation

step 1. Please prepare environment as follows.
```shell
conda create --name bevdet python=3.8 -y
conda activate bevdet
conda install [pytorch]
pip install -U openmim
mim install mmcv-full==1.6.0
mim install mmdet==2.26.0
mim install mmsegmentation==0.30.0
pip install numpy==1.23.5 pycuda lyft_dataset_sdk networkx==2.2 numba==0.53.0 nuscenes-devkit plyfile scikit-image tensorboard trimesh==2.35.39 setuptools==58.2.0
pip install yapf==0.40.1
pip install spconv-cu116 [根据cuda版本选择cu***]
```

```shell

step 2. Install mmdet3d dependencies
cd BEVDet
pip install -v -e .
```

step 3. Prepare nuScenes dataset as introduced in [nuscenes_det.md](docs/en/datasets/nuscenes_det.md) and create the pkl for BEVDet by running:
```shell
python tools/create_data_bevdet.py
```
step 4. For Occupancy Prediction task, download (only) the 'gts' from [CVPR2023-3D-Occupancy-Prediction](https://github.com/CVPR2023-3D-Occupancy-Prediction/CVPR2023-3D-Occupancy-Prediction) and arrange the folder as:
```shell script
└── nuscenes
    ├── v1.0-trainval (existing)
    ├── sweeps  (existing)
    ├── samples (existing)
    └── gts (new)
```

#### Train model
```shell
# single gpu
python tools/train.py $config
# multiple gpu
./tools/dist_train.sh $config num_gpu
```

#### Test model
```shell
# single gpu
python tools/test.py $config $checkpoint --eval mAP
# multiple gpu
./tools/dist_test.sh $config $checkpoint num_gpu --eval mAP
```