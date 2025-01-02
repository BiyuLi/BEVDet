# -------------------------------------------------------------------------------
# Author:       biyu_lee
# Date:         2023/4/28
# Description:  convert lidar 3d info to cylinder 3d/2d
# -------------------------------------------------------------------------------

import math
from abc import abstractmethod
import cv2
import numpy as np
import torch

def limit_yaw(yaw):
    while yaw > math.pi:
        yaw -= 2 * math.pi
    while yaw <= -math.pi:
        yaw += 2 * math.pi
    return yaw


class VirtualCamera:
    def __init__(self, cfg):
        w2w_4 = np.matrix([0, -1, 0, 0, 0, 0, -1, 0, 1, 0, 0, 0, 0, 0,
                           0, 1], dtype=np.float32).reshape(4, 4)
        self.w2w_4_inv = w2w_4.I
        w2w = np.array([0, -1, 0, 0, 0, -1, 1, 0, 0],
                       dtype=np.float32).reshape(3, 3)
        self.w2w_3_inv = np.matrix(w2w).I
        self.c2c_3 = np.array([0, -1, 0, 0, 0, -1, 1, 0, 0],
                              dtype=np.float32).reshape(3, 3)
        self.c2c_3_inv = np.matrix(self.c2c_3).I

        self.vir_fu_ = cfg["focal_length"]
        self.vir_fv_ = cfg["focal_length"]
        self.vir_cu_ = cfg["cx"]
        self.vir_cv_ = cfg["cy"]
        self.vir_hfov_ = cfg["hfov"]
        self.vir_vfov_ = cfg["vfov"]
        image_w_h = cfg.get("gdc_inputs_szie", False) or cfg.get("gdc_inputs_size", False)
        if not image_w_h:
            raise AttributeError("gdc_inputs_szie or gdc_inputs_size key not found!")
        self.vir_img_width_ = image_w_h[0]
        self.vir_img_height_ = image_w_h[1]
        self.vir_intrinsic_ = np.array([[self.vir_fu_, 0, self.vir_cu_],
                                        [0, self.vir_fv_, self.vir_cv_],
                                        [0, 0, 1]], dtype=np.float32)
        # calc virtual camera intrinsics
        self.vir_K_ = np.array([self.vir_fu_, 0, self.vir_cu_,
                                0, self.vir_fv_, self.vir_cv_,
                                0, 0, 1], dtype=np.float32).reshape(3, 3)
        self.vir_K_inv_ = np.matrix(self.vir_K_).I
        self.camera_type = None
        self.R_ = None
        self.t_ = None
        self.lidar2cam_ = None
        self.rot_vir2cam_ = None
        self.rot_cam2vir_ = None

    def init_local_extrinsic(self, cam_type, lidar2cam_rotation, lidar2cam_translation):
        self.camera_type = cam_type
        if lidar2cam_rotation is None or lidar2cam_translation is None:
            raise ValueError("Extrinsic not valid")
        self.R_ = np.array(lidar2cam_rotation, dtype=np.float32).reshape(3, 3)
        self.t_ = np.array(lidar2cam_translation, dtype=np.float32)[:, None]
        lidar2cam = np.concatenate((self.R_, self.t_), axis=-1)
        self.lidar2cam_ = np.concatenate((lidar2cam, np.array([0, 0, 0, 1],
                                                              dtype=lidar2cam.dtype)[None]), axis=0)

    def get_vir_camera_intrinsics(self):
        return self.vir_K_

    # convert point(3d) from lidar to raw camera
    def convert_lidar_to_raw_camera(self, lidar_point):
        lidar_pt = np.matrix([lidar_point[0], lidar_point[1], lidar_point[2]],
                             dtype=np.float32).reshape(3, 1)
        R = self.lidar2cam_[0:3, 0:3]
        t = self.lidar2cam_[0:3, 3:4]
        return np.matrix(np.matmul(R, lidar_pt)) + t
    def batch_convert_lidar_to_raw_camera(self, lidar_point):
        lidar_pt = lidar_point.reshape(-1,3,1)
        lidar_pt= lidar_pt.astype(np.double)
        R = self.lidar2cam_[0:3, 0:3]
        t = self.lidar2cam_[0:3, 3:4]
        return R@lidar_pt+t


    # convert point(3d) from fisheye camera to cyl camera
    def convert_raw_camera_to_vir_camera(self, cam_point):
        vir_pt = np.matmul(
            np.matmul(self.rot_cam2vir_, self.c2c_3_inv), cam_point)
        return vir_pt
    def batch_convert_raw_camera_to_vir_camera(self, cam_point):
        rot = self.rot_cam2vir_@self.c2c_3_inv
        return rot@cam_point


    # convert point(3d) from cyl camera to fisheye camera
    def convert_vir_camera_to_raw_camera(self, vir_point):
        cam_point = np.matmul(
            np.matmul(self.c2c_3, self.rot_vir2cam_), vir_point)
        return cam_point

    # convert point(3d) from lidar to cyl camera
    def convert_lidar_to_vir_camera(self, lidar_point):
        cam_pt = self.convert_lidar_to_raw_camera(lidar_point)
        vir_pt = self.convert_raw_camera_to_vir_camera(cam_pt)

        # print(f'{self.camera_type} : lidar ====> cyl')
        # print(f"lidar: {lidar_point}\ncam: {np.array(cam_pt).reshape(3, ).tolist()}\nvir_pt: {np.array(vir_pt).reshape(3, ).tolist()}")
        return vir_pt
    def batch_convert_lidar_to_vir_camera(self, lidar_point):
        cam_pt = self.batch_convert_lidar_to_raw_camera(lidar_point)
        vir_pt = self.batch_convert_raw_camera_to_vir_camera(cam_pt)
        # print(f'{self.camera_type} : lidar ====> cyl')
        # print(f"lidar: {lidar_point}\ncam: {np.array(cam_pt).reshape(3, ).tolist()}\nvir_pt: {np.array(vir_pt).reshape(3, ).tolist()}")
        return vir_pt

    # convert point(3d) from raw camera to lidar
    def convert_raw_camera_to_lidar(self, cam_point):
        cam_point = cam_point.reshape(3, 1)
        R_ = np.matrix(self.lidar2cam_[0:3, 0:3]).I
        t_ = -np.matrix(self.lidar2cam_[0:3, 3:4], dtype=np.float32).reshape(3,
                                                                             1)
        return np.matmul(R_, (cam_point + t_))

    # convert point(3d) from vir camera to lidar
    def convert_vir_camera_to_lidar(self, vir_point):
        cam_pt = self.convert_vir_camera_to_raw_camera(vir_point)
        lidar_pt = self.convert_raw_camera_to_lidar(cam_pt)
        return lidar_pt

    @abstractmethod
    def convert_vir_camera_to_vir_img(self, vir_pt):
        pass

    # convert point(3d) from lidar to vir img(2d)
    def convert_lidar_to_vir_img(self, lidar_pt):
        vir_pt = self.convert_lidar_to_vir_camera(lidar_pt)
        vir_img_pt = self.convert_vir_camera_to_vir_img(vir_pt)
        return vir_img_pt

    def batch_convert_lidar_to_vir_img(self,lidar_pt):
        vir_pt = self.batch_convert_lidar_to_vir_camera(lidar_pt)
        vir_pt =vir_pt.reshape(-1,3,1)
        np.save("vir_pt.npy",vir_pt)
        vir_img_pt = self.batch_convert_vir_camera_to_vir_img(vir_pt)
        return vir_img_pt

    def convert_lidar_rot_to_vir_rot(self, lidar_rot):
        if self.camera_type == 'CAMERA_FRONT' or self.camera_type == 'FISHEYE_CAMERA_FRONT':
            rot = limit_yaw(lidar_rot)
        elif self.camera_type == 'CAMERA_BACK' or self.camera_type == 'FISHEYE_CAMERA_BACK':
            rot = limit_yaw(lidar_rot - math.pi)
        elif self.camera_type == 'CAMERA_LEFT' or self.camera_type == 'FISHEYE_CAMERA_LEFT':
            rot = limit_yaw(lidar_rot - math.pi / 2)
        elif self.camera_type == 'CAMERA_RIGHT' or self.camera_type == 'FISHEYE_CAMERA_RIGHT':
            rot = limit_yaw(lidar_rot + math.pi / 2)
        else:
            print(f"[ERROR] wrong camera type: {self.camera_type}")
            raise TypeError("Wrong camera type")
        return rot

    def convert_vir_rot_to_lidar_rot(self, vir_rot):
        if self.camera_type == 'CAMERA_FRONT' or self.camera_type == 'FISHEYE_CAMERA_FRONT':
            rot = limit_yaw(vir_rot)
        elif self.camera_type == 'CAMERA_BACK' or self.camera_type == 'FISHEYE_CAMERA_BACK':
            rot = limit_yaw(vir_rot + math.pi)
        elif self.camera_type == 'CAMERA_LEFT' or self.camera_type == 'FISHEYE_CAMERA_LEFT':
            rot = limit_yaw(vir_rot + math.pi / 2)
        elif self.camera_type == 'CAMERA_RIGHT' or self.camera_type == 'FISHEYE_CAMERA_RIGHT':
            rot = limit_yaw(vir_rot - math.pi / 2)
        else:
            print(f"[ERROR] wrong camera type: {self.camera_type}")
            raise TypeError("Wrong camera type")
        return rot


class VirCylinderCamera(VirtualCamera):
    def __init__(self, cylinder_cfg):
        super().__init__(cylinder_cfg)

    def init_local_camera(self, cam_type, lidar2cam_rotation, lidar2cam_translation):
        super().init_local_extrinsic(cam_type, lidar2cam_rotation, lidar2cam_translation)
        if cam_type == 'FISHEYE_CAMERA_FRONT':
            self.R_ = np.matmul(self.w2w_3_inv, self.R_)
        elif cam_type == 'FISHEYE_CAMERA_BACK':
            R_rear = np.array((0, 0, math.pi), dtype=np.float32)
            R_rear = cv2.Rodrigues(R_rear)[0]
            self.R_ = np.matmul(np.matmul(self.w2w_3_inv, self.R_),
                                R_rear)
        elif cam_type == 'FISHEYE_CAMERA_LEFT':
            R_left = np.array((0, 0, math.pi / 2), dtype=np.float32)
            R_left = cv2.Rodrigues(R_left)[0]
            self.R_ = np.matmul(np.matmul(self.w2w_3_inv, self.R_),
                                R_left)
        elif cam_type == 'FISHEYE_CAMERA_RIGHT':
            R_right = np.array((0, 0, -math.pi / 2), dtype=np.float32)
            R_right = cv2.Rodrigues(R_right)[0]
            self.R_ = np.matmul(np.matmul(self.w2w_3_inv, self.R_),
                                R_right)
        else:
            print(f"[ERROR] wrong camera type: {self.camera_type}")
            raise TypeError("Wrong camera type")
        # update camera2cyl translation matrix
        self.rot_vir2cam_ = self.R_.copy()
        self.rot_cam2vir_ = np.matrix(self.rot_vir2cam_).I

    # convert point from cyl camera to cyl image
    def convert_vir_camera_to_vir_img(self, vir_pt):
        r = np.sqrt(vir_pt[0] * vir_pt[0] + vir_pt[1] * vir_pt[1])
        vir_x = r * math.atan2(-vir_pt[1], vir_pt[0])
        vir_y = -vir_pt[2]
        vir_z = r
        if math.fabs(vir_z) < 1e-6:
            vir_z = 1e-6
        vir_pt_p = np.matrix(
            [vir_x.item(0, 0), vir_y.item(0, 0), vir_z.item(0, 0)],
            dtype=np.double).reshape(3, 1)
        img_pt = np.matmul(self.vir_K_, vir_pt_p)

        # print(self.vir_K_)
        
        pt = [img_pt[0] / img_pt[2],
              img_pt[1] / img_pt[2]]
        return [int(pt[0].item(0, 0)), int(pt[1].item(0, 0))]

    def batch_convert_vir_camera_to_vir_img(self, vir_pt):#[n,3]
        vir_pt = np.array(vir_pt)
        rad = np.sqrt(vir_pt[:,0] * vir_pt[:,0] + vir_pt[:,1] * vir_pt[:,1])
        vir_x = rad * np.arctan2(-vir_pt[:,1], vir_pt[:,0])
        vir_y = -vir_pt[:,2]
        vir_z = rad
        mask = vir_z > 1e-6
        vir_x= vir_x[mask]
        vir_y= vir_y[mask]
        vir_z= vir_z[mask]
        vir_x =vir_x.reshape(-1,1)
        vir_y =vir_y.reshape(-1,1)
        vir_z =vir_z.reshape(-1,1)
        vir_pts = np.concatenate((vir_x,vir_y,vir_z),axis = 1)
        vir_pts = vir_pts.reshape(-1,3,1)
        pts_2d = self.vir_K_@vir_pts
        pts_2d[:, :2] /= pts_2d[:, 2:3]
        return pts_2d

 
    def convert_vir_img_to_vir_camera(self, img_pt, z):
        pt = np.matrix([img_pt[0], img_pt[1], 1.0], dtype=np.float32).reshape(3, 1)

        vir_cam_pt = np.matmul(self.vir_K_inv_, pt)
        vir_cam_X = vir_cam_pt[0]
        vir_cam_Y = vir_cam_pt[1]

        # 需要将期望的raw坐标系下的深度变成vir下的
        a, b, c = np.matmul(self.c2c_3, self.rot_vir2cam_)[-1, :].tolist()[0]  # 三个系数
        z_i = z / ((b * np.tan(vir_cam_X) - a) / (vir_cam_Y * np.sqrt(1 + np.tan(vir_cam_X) * np.tan(vir_cam_X))) + c)

        cyl_x = - z_i / (vir_cam_Y * np.sqrt(1 + np.tan(vir_cam_X) * np.tan(vir_cam_X)))
        cyl_y = - np.tan(vir_cam_X) * cyl_x

        return [cyl_x.item(0, 0), cyl_y.item(0, 0), z_i.item(0, 0)]

    def convert_vir_img_to_vir_camera_by_virdepth(self, img_pt, z):
        vir_pt = np.zeros(3)
        pt = np.matrix([img_pt[0]*z, img_pt[1]*z, z], dtype=np.double).reshape(3, 1)

        vir_cam_pt = np.matmul(self.vir_K_inv_, pt)
        # print('after', pt, vir_cam_pt)
        # print(np.matrix(self.vir_K_inv_).I)
        vir_z = vir_cam_pt[2,0]
        vir_y = vir_cam_pt[1,0]
        vir_x = vir_cam_pt[0,0]

        vir_pt[2]=-vir_y
        r = vir_z
        theta = vir_x/r
        vir_pt[1] = -r*np.sin(theta)
        vir_pt[0]=r*np.cos(theta)
        return vir_pt

    def CvtCylImgPt2CylCam(self, img_pt, r):
        pt = np.array([[img_pt[0]], [img_pt[1]], [1.0]], dtype=np.float32)
        pt = pt.reshape(3,1)
        # virtual coordinates
        vir_cam_pt = np.dot(self.vir_K_inv_, pt)
        vir_cam_Y = vir_cam_pt[1, 0] * r
        vir_cam_Z = vir_cam_pt[2, 0] * r
     # cylinder camera coordinates
        cyl_cam_X = vir_cam_Z * np.cos(vir_cam_pt[0, 0])
        cyl_cam_Y = -vir_cam_Z * np.sin(vir_cam_pt[0, 0])
        cyl_cam_Z = -vir_cam_Y
        return [cyl_cam_X, cyl_cam_Y, cyl_cam_Z]
        # return np.array([cyl_cam_X, cyl_cam_Y, cyl_cam_Z], dtype=np.float32)

    def batch_convert_vir_img_to_vir_camera_by_vir_r(self, img_pt):
        # img_pt: [B x N x D x H x W x 3 x 1]
        B, N, _, _, _, _, _ = img_pt.shape

        r = img_pt[..., 2, :].clone().unsqueeze(-1).double()  # 先把z拿出来，需要先经过raw-vir的映射才能得到vir视角下的z_i
        img_pt[..., 2, :] = 1.0  # 再补充图像像素坐标最后一维为1
        img_pt = img_pt.double()

        vir_cam_pt = torch.tensor(self.vir_K_inv_).double().to(img_pt).repeat(B, N, 1, 1, 1, 1, 1).matmul(img_pt)
        vir_cam_Y =  vir_cam_pt[..., 1:2, :]*r
        vir_cam_Z = vir_cam_pt[..., 2:3, :]*r

        cyl_cam_X = vir_cam_Z * torch.cos(vir_cam_pt[..., 0:1, :])
        cyl_cam_Y = -vir_cam_Z * torch.sin(vir_cam_pt[..., 0:1, :])
        cyl_cam_Z = -vir_cam_Y
        return torch.cat([cyl_cam_X, cyl_cam_Y, cyl_cam_Z], dim=-2)

    def batch_convert_vir_img_to_vir_camera(self, img_pt):
        # img_pt: [B x N x D x H x W x 3 x 1]
        B, N, _, _, _, _, _ = img_pt.shape

        z = img_pt[..., 2, :].clone().unsqueeze(-1).double()  # 先把z拿出来，需要先经过raw-vir的映射才能得到vir视角下的z_i
        img_pt[..., 2, :] = 1.0  # 再补充图像像素坐标最后一维为1
        img_pt = img_pt.double()

        vir_cam_pt = torch.tensor(self.vir_K_inv_).double().to(img_pt).repeat(B, N, 1, 1, 1, 1, 1).matmul(img_pt)

        vir_cam_X = vir_cam_pt[..., 0, :].unsqueeze(-1)
        tan_vir_cam_X = torch.tan(vir_cam_X)  # 提前计算tan，节省时间，必须double类型
        vir_cam_Y = vir_cam_pt[..., 1, :].unsqueeze(-1)

        a, b, c = np.matmul(self.c2c_3, self.rot_vir2cam_)[-1, :].tolist()[0]  # 三个系数
        z_i = z / ((b * tan_vir_cam_X - a) / (vir_cam_Y * torch.sqrt(1 + tan_vir_cam_X * tan_vir_cam_X)) + c)

        cyl_x = - z_i / (vir_cam_Y * torch.sqrt(1 + tan_vir_cam_X * tan_vir_cam_X))
        cyl_y = - tan_vir_cam_X * cyl_x

        # # 限制范围
        # cyl_x = torch.clamp(cyl_x, min=-100, max=100)
        # cyl_y = torch.clamp(cyl_y, min=-100, max=100)

        return torch.cat([cyl_x, cyl_y, z_i], dim=-2)

    def batch_convert_vir_camera_to_raw_camera(self, vir_pt):
        B, N, _, _, _, _, _ = vir_pt.shape
        rot_tensor = torch.tensor(np.matmul(self.c2c_3, self.rot_vir2cam_)).to(vir_pt)

        return rot_tensor.repeat([B, N, 1, 1, 1, 1, 1]).double().matmul(vir_pt).float()


class VirPinholeCamera(VirtualCamera):
    def __init__(self, vir_cam_cfg):
        super().__init__(vir_cam_cfg)
        # 图像转换需要的参数
        self.focal_u_ = None
        self.focal_v_ = None
        self.optical_center_x = None
        self.optical_center_y = None
        self.distorts_ = None
        self.u_map_ = None
        self.v_map_ = None
        self.raw_intrinsic = None

    def init_local_camera(self, cam_type, lidar2cam_rotation, lidar2cam_translation):
        super().init_local_extrinsic(cam_type, lidar2cam_rotation, lidar2cam_translation)
        if self.camera_type == 'CAMERA_BACK':
            R_rear = np.array((0, 0, math.pi), dtype=np.float32)
            R_rear = cv2.Rodrigues(R_rear)[0]
            self.R_ = np.matmul(np.matmul(self.w2w_3_inv, self.R_), R_rear)
        elif self.camera_type == 'CAMERA_FRONT':
            self.R_ = np.matmul(self.w2w_3_inv, self.R_)
        else:
            raise NotImplementedError(f"Init pinhole failed! Not support for {cam_type}")
        self.rot_vir2cam_ = self.R_.copy()
        self.rot_cam2vir_ = np.matrix(self.rot_vir2cam_).I

    def set_distortion_param(self, cfg):
        self.focal_u_ = cfg["focal_u"]
        self.focal_v_ = cfg["focal_v"]
        self.optical_center_x = cfg["center_u"]
        self.optical_center_y = cfg["center_v"]
        self.distorts_ = np.array(cfg["distort"], dtype=np.float32)
        self.raw_intrinsic = np.array([self.focal_u_, 0, self.optical_center_x, 0, self.focal_v_,
                                   self.optical_center_y, 0, 0, 1], dtype=np.float32).reshape(3, 3)

    def build_raw_img_to_vir_img_map(self):
        locations = np.zeros((2, self.vir_img_height_ * self.vir_img_width_), dtype=np.float32)
        u_ptr = locations[0, :]
        v_ptr = locations[1, :]

        for y in range(self.vir_img_height_):
            for x in range(self.vir_img_width_):
                idx = x + y * self.vir_img_width_
                u_ptr[idx] = float(x)
                v_ptr[idx] = float(y)

        pinhole_img_points = self.fill_vir_img_map(locations)
        u_map_ = pinhole_img_points[0].reshape(
            self.vir_img_height_, -1)
        v_map_ = pinhole_img_points[1].reshape(
            self.vir_img_height_, -1)

        self.u_map_ = u_map_.astype(np.float32)
        self.v_map_ = v_map_.astype(np.float32)

    def fill_vir_img_map(self, img_pt):
        img_pt_t = img_pt
        last_row = np.ones((1, img_pt_t.shape[1]), dtype=img_pt_t.dtype)
        img_pt_t = np.vstack((img_pt_t, last_row))

        vir_cam_pt = np.dot(self.vir_K_inv_, img_pt_t)
        vir_cam_X = vir_cam_pt[0, :]
        vir_cam_Y = vir_cam_pt[1, :]
        vir_cam_Z = vir_cam_pt[2, :]

        vir_cam = np.zeros((3, vir_cam_X.shape[1]), dtype=vir_cam_X.dtype)
        for i in range(img_pt_t.shape[1]):
            vir_cam[0, i] = vir_cam_Z[0, i]
            vir_cam[1, i] = -vir_cam_Z[0, i] * vir_cam_X[0, i]
            vir_cam[2, i] = -vir_cam_Y[0, i]

        vir_cam_pt = vir_cam
        cam_pt = np.dot(self.rot_vir2cam_, vir_cam_pt)
        pinhole_points = self.cvt_vir_camera_pts_to_vir_img_map(cam_pt)

        return pinhole_points

    def cvt_vir_camera_pts_to_vir_img_map(self, pt):
        pin_x = pt[1, :]
        temp = np.multiply(pin_x, pin_x) + np.multiply(pt[2, :], pt[2, :])
        r = np.sqrt(temp)
        r2 = np.multiply(r, r)
        r4 = np.multiply(r2, r2)
        r6 = np.multiply(r4, r2)

        distortion_factor = 1.0 + self.distorts_[0] * r2 + self.distorts_[1] * r4 + self.distorts_[
            4] * r6

        if len(self.distorts_) == 8:
            distortion_factor_57 = 1.0 + self.distorts_[7] * r6 + self.distorts_[6] * r4 + self.distorts_[5]* r2
            distortion_factor = distortion_factor / distortion_factor_57

        pinhole_points_x = self.focal_u_ * (
                np.multiply(distortion_factor, -pt[1, :]) +
                2 * self.distorts_[2] * np.multiply(-pt[1, :], -pt[2, :]) +
                self.distorts_[3] * r2 + 2 * self.distorts_[3] * np.multiply(-pt[1, :], -pt[1, :])
        ) + self.optical_center_x

        pinhole_points_y = self.focal_v_ * (
                np.multiply(distortion_factor, -pt[2, :]) +
                self.distorts_[2] * r2 + 2 * self.distorts_[2] * np.multiply(-pt[2, :], -pt[2, :]) +
                2 * self.distorts_[3] * np.multiply(-pt[1, :], -pt[2, :])
        ) + self.optical_center_y

        pinhole_points = np.vstack((pinhole_points_x, pinhole_points_y))
        return pinhole_points

    def cvt_raw_img2virtual_img(self, src_im):
        pj_image_ = cv2.remap(src_im, self.u_map_, self.v_map_, cv2.INTER_LINEAR)
        return pj_image_

    def convert_vir_camera_to_vir_img(self, vir_pt):
        if self.camera_type == 'CAMERA_BACK':
            # only for pinhole camera model
            vir_x = -vir_pt[1] / vir_pt[0]
            vir_y = -vir_pt[2] / vir_pt[0]
            vir_z = np.matrix(1)
        elif self.camera_type == 'CAMERA_FRONT':
            # only for pinhole camera model
            vir_x = -vir_pt[1] / vir_pt[0]
            vir_y = -vir_pt[2] / vir_pt[0]
            vir_z = np.matrix(1)
        else:
            raise NotImplementedError(f"convert_vir_camera_to_vir_img failed! Not support for "
                                      f"{self.camera_type}")
        if math.fabs(vir_z) < 1e-6:
            vir_z = 1e-6
        vir_pt = np.matrix(
            [vir_x.item(0, 0), vir_y.item(0, 0), vir_z.item(0, 0)],
            dtype=np.float32).reshape(3, 1)
        img_pt = np.matmul(self.vir_K_, vir_pt)

        pt = [img_pt[0] / img_pt[2],
              img_pt[1] / img_pt[2]]
        return [int(pt[0].item(0, 0)), int(pt[1].item(0, 0))]

    def convert_raw_img_to_vir_img(self, img_pt):
        distorted_point = np.array([[img_pt[0], img_pt[1]]], dtype=np.float32)
        undistorted_point = cv2.undistortPoints(distorted_point, self.raw_intrinsic, self.distorts_)
        x_undistorted = undistorted_point[0, 0, 0]
        y_undistorted = undistorted_point[0, 0, 1]
        # x_projected = self.focal_u_ * x_undistorted + self.optical_center_x
        # y_projected = self.focal_v_ * y_undistorted + self.optical_center_y
        homogeneous_point = np.array([x_undistorted, y_undistorted, 1], dtype=np.float32)
        homo_pt_cam = np.array([homogeneous_point[2], -homogeneous_point[0], -homogeneous_point[1]])
        homo_pt_vir = np.squeeze(np.array(np.dot(self.rot_cam2vir_, homo_pt_cam)))
        homo_pt_vir_pit_removed = np.array([-homo_pt_vir[1], -homo_pt_vir[2], homo_pt_vir[0]])
        pixel_coords = np.dot(self.vir_intrinsic_, homo_pt_vir_pit_removed)
        u = pixel_coords[0] / pixel_coords[2]
        v = pixel_coords[1] / pixel_coords[2]
        return u, v
