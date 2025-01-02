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
print(CUR_DIR)
sys.path.append(str(CUR_DIR))
import hyper_dl
import hyper_data
from common.datasets import build_dataset,build_dataloader
from common.core.visualizer.image_vis import draw_camera_bbox3d_on_img
from common.core.utils.utils import get_box_type
from copy import deepcopy
from common.datasets.compose import Compose
import pickle
from hyper_dl.models import build_model


mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
h = 576
w = 704
bev_h = h*2
bev_w = h*2
scale_h = bev_h/120
scale_w= bev_w/120
cx = bev_w//2
cy = bev_h//2

colors = []
for i in range(50):
    b = np.random.randint(10, 255)
    g = np.random.randint(10, 255)
    r = np.random.randint(10, 255)
    colors.append([b, g, r])


link_idx = [0,1,7,2]
link_idx_pred = [3,6,4,5]
cam_gt_order = ['FISHEYE_CAMERA_BACK', 'FISHEYE_CAMERA_FRONT', 'FISHEYE_CAMERA_LEFT', 'FISHEYE_CAMERA_RIGHT']
scale = 0.5
def get_corner_pred(bbox3d):
    center = bbox3d[:3]
    l,h,w  = bbox3d[3:6]
    size = [l,w,h]
    rotation = bbox3d[6]
    obb = o3d.geometry.OrientedBoundingBox(center=center,
                                           extent=size,
                                           R=o3d.geometry.get_rotation_matrix_from_axis_angle(
                                               [0,0,rotation]))
    corners = np.asarray(obb.get_box_points())

    return corners


def parse_args():
    parser = argparse.ArgumentParser(description='show gt')
    parser = argparse.ArgumentParser(description='show gt')
    parser.add_argument('--config',type = str,default='/home/gpu/works/fk/task/hyper_dl/configs/det_3d/bev3d/bevdet-cylin.py',
                        help='inference config file path, using test_config')
    parser.add_argument(
        '--checkpoint', default = "/home/gpu/works/zzq/task/hyper_dl/work_dirs/iter_40000.pth",help='the checkpoint file to load')

    args = parser.parse_args()
    return args
def get_corner(bbox3d):
    center = bbox3d[:3]
    size = bbox3d[3:6]
    rotation = bbox3d[6]
    obb = o3d.geometry.OrientedBoundingBox(center=center,
                                           extent=size,
                                           R=o3d.geometry.get_rotation_matrix_from_axis_angle(
                                               [0, 0, rotation]))
    corners = np.asarray(obb.get_box_points())

    return corners
def main():
    args = parse_args()
    config = mmcv.Config.fromfile(args.config)
    config_train = config.data.train
    test_config = config.model.test_cfg.pts
    dataset = build_dataset(config_train)

    config_test = mmcv.Config.fromfile(args.config)
    config_test.model.pretrained = None
    config_test.model.train_cfg = None
    model = build_model(config_test.model,
                        test_cfg=config_test.get('test_cfg'))
    mmcv.runner.load_checkpoint(model, args.checkpoint, map_location='cuda')

    model.cfg = config_test  # save the config in the model for convenience
    device = "cuda"

    model.to(device)
    model.eval()

 ## dataloader
    runner_type = 'EpochBasedRunner' if 'runner' not in config else config.runner[
        'type']

    train_dataloader_default_args = dict(
            samples_per_gpu=1,
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
    data_loaders = build_dataloader(dataset, **train_loader_cfg) 
    iter_loader = iter(data_loaders)

    with open('/home/gpu/works/fk/task/hyper_dl/data/samples/bev3d/anchor.pkl','rb') as fr:
        anchor = pickle.load(fr)
    for i in range(50):
        #show_gt
        print(f'index-----{i}')
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
        data = next(iter_loader)
        # if i !=0:
        #     continue
        name = f'bev_test_{i}.png'

        bboxes3d = data['gt_bboxes_3d'].data[0][0]
        for ii, box in enumerate(bboxes3d.tensor):
            center3d = box[:3]
            x, y, z = center3d.numpy()
            v = int(cy + y * scale_h)
            u = int(x * scale_w + cx)
            cv2.circle(bev_map, (u, v), 2, (0, 255, 255), 2)
            corners = get_corner(box)
            bev_pt = []
            for idx in link_idx:
                x, y, z = corners[idx]

                v = int(cy +y * scale_h )
                u = int(x*scale_w + cx)
                bev_pt.append((u, v))
                cv2.circle(bev_map, (u, v), 2, (255, 255, 0), 2)
                if idx ==0:
                    cv2.putText(bev_map, str(ii), (u, v), 1, 2, (0, 255, 255), 2)
            for pi in range(4):
                if pi == 3:
                    cv2.line(bev_map, bev_pt[pi], bev_pt[0], (0, 0, 255), 2)
                else:
                    cv2.line(bev_map, bev_pt[pi], bev_pt[pi + 1], (0, 0, 255), 2)
        canvas = np.zeros((h*2,w*2,3),dtype =np.uint8)
        img = data['img_inputs']
        img = img[0][0].cpu().numpy()
        bboxes2d = data['gt_bboxes_2d'].data[0][0]
        imgs = []
        for id in range(img.shape[0]):#back,front, left,right;
            im = img[id].copy()
            im = im.transpose(2,1,0)
            im = im*std+mean
            im = im.astype(np.uint8)
            im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
            cv2.putText(im, cam_gt_order[id], (300, 100), 1, 2, (0, 255, 255), 2)
            cam_id = id
            for nb in range(bboxes2d.shape[0]):
                if bboxes2d[nb][cam_id][0] !=0:
                    _, bcx, bcy, bw, bh,_,_ = bboxes2d[nb][cam_id]
                    bcx = bcx * scale
                    bcy = bcy * scale
                    bw = bw * scale
                    bh = bh * scale
                    box = bboxes3d.tensor[nb]
                    center3d = box[:3]
                    x, y, z = center3d.numpy()
                    coor_x = (
                        x - test_config.pc_range[0]
                        ) / test_config.voxel_size[0] / test_config['out_size_factor']
                    coor_x = int(coor_x)
                    coor_y = (
                        y - test_config.pc_range[1]
                        ) / test_config.voxel_size[1] / test_config['out_size_factor']
                    coor_y= int(coor_y)
                    one_anchor = anchor[cam_gt_order[cam_id]]
                    ax,ay,awid = one_anchor[coor_x,coor_y]
                    # print(cam_id,x, y,ax,ay,awid,(bcx-ax)/awid, (bcy-ay)/awid,torch.log(bw/awid),torch.log(bh/awid))
           

                    x1 = bcx - bw / 2
                    x2 = bcx + bw / 2
                    y1 = bcy - bh / 2
                    y2 = bcy + bh / 2
                    cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)), colors[nb], 2)
                    cv2.putText(im, str(nb), (int(x1), int(y1)), 1, 3, colors[nb], 2)
            imgs.append(im)



        # data = dataset[id]
        # # import pickle
        # # with open('data0.pkl','wb') as fw:
        # #     pickle.dump(rst,fw)
        # # exit(0)

        with torch.no_grad():
            data['img_metas'] = data['img_metas'].data
            data['img_inputs'] = [x.to('cuda') for x in data['img_inputs']]
            data['img_inputs']=[data['img_inputs']]
            
            result = model(return_loss=False, rescale=True, **data)
            det_rst = result[0]['pts_bbox']
            scores = det_rst['scores_3d']
            bboxes = det_rst['boxes_3d']
            bboxes2d = det_rst['boxes_2d']
            for ii in range(len(bboxes)):
                if scores[ii] < 0.35:
                    continue
                bbox = bboxes[ii]
                bbox_data = bbox.tensor[0]
                x, y, z = bbox_data[:3].cpu().numpy()

                coor_x = (
                    x - test_config.pc_range[0]
                ) / test_config.voxel_size[0] / test_config['out_size_factor']
                coor_x = int(coor_x)
                coor_y = (
                    y - test_config.pc_range[1]
                ) / test_config.voxel_size[1] / test_config['out_size_factor']
                coor_y= int(coor_y)
                by = y * scale_h
                bx = x* scale_w
                u = int(cx + bx )
                v = int(cy + by)
                cv2.circle(bev_map, (u, v), 3, (255, 255,255), 4)
                corner = get_corner_pred(bbox_data)
                data = corner
                bev_pt = []
                for idx in link_idx_pred:
                    x, y, z = data[idx]
                    y = y * scale_h
                    x = x*scale_w
                    u = int(cx + x )
                    v = int(cy + y)
                    bev_pt.append((u, v))
                for i in range(4):
                    if i == 3:
                        cv2.line(bev_map, bev_pt[i], bev_pt[0], (0, 255, 0), 2)
                    else:
                        cv2.line(bev_map, bev_pt[i], bev_pt[i + 1], (0, 255, 0), 2)
                #decode box in 2d
                # continue
                cam_box = bboxes2d[ii][4:]
                conf_box = bboxes2d[ii][:4]
                for ci in range(4):
                    if conf_box[ci]>0.45:
                        one_anchor = anchor[cam_gt_order[ci]]
            
                        # exit(0)
                        tx,ty,tw,th,lft,rgt = cam_box[6*ci:6*ci+6]
                        # tx,ty,tw,th= [0.02154921,  0.03598525, -0.03153007,  0.01876494]
                        ix,iy,iw = one_anchor[coor_x,coor_y]
                        # print( ci,bbox_data[:2].cpu().numpy(),ix,iy,iw,cam_box[ci:ci+4])
                        # cv2.circle(imgs[ci], (int(ix), int(iy)), 4, (255, 255, 255), 3)
                        #decode
                            # one_box[i,0,int_coor_y,int_coor_x] = (bx-ax)/awid
                            # one_box[i,1,int_coor_y,int_coor_x]  = (by-ay)/awid
                            # one_box[i,2,int_coor_y,int_coor_x]  = torch.log(bw/awid)
                            # one_box[i,3,int_coor_y,int_coor_x]  = torch.log(bh/awid)
                        tx = (iw*tx+ix)
                        ty = (iw*ty+iy)
                        lft = (iw*lft+ix)
                        rgt = (iw*rgt+ix)

                        tw = np.exp(tw)*iw
                        th = np.exp(th)*iw
                        top_x= int(tx-tw/2)
                        top_y= int(ty-th/2)
                        bot_x= int(tx+tw/2)
                        bot_y = int(ty+th/2)
                        lft = int(lft)
                        rgt = int(rgt)
                        # if top_x<0 or bot_x>w-1:
                        #     continue
                        # if top_y<0 or bot_y>h-1:
                        #     continue
                        cv2.circle(imgs[ci], (int(tx), int(ty)), 4, (255, 255, 255), 3)
                        cv2.rectangle(imgs[ci], (top_x, top_y), (bot_x,bot_y), (255,255,255), 2)
                        cv2.rectangle(imgs[ci], (lft, top_y), (rgt,bot_y), (255,0,0), 2)

                # exit(0)

        for id,im in enumerate(imgs):
            wi = id%2
            hi = int(id/2)
            canvas[h*hi:h*hi+h,w*wi:w*wi+w]=im
        save_im = np.hstack([canvas, bev_map])
        cv2.imwrite(name,save_im)
        # cv2.imshow('test',im)
        # cv2.imshow('front',show_im)
        # cv2.waitKey(0)
    print("hello world")
if __name__=="__main__":
    main()