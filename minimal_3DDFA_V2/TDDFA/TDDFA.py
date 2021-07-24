import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn

from .mobilenet import MobileNet
from .utils import (
    _to_ctype,
    make_abs_path,
    _load,
    crop_img,
    parse_roi_box_from_bbox,
    load_model,
    _parse_param,
    similar_transform,
)


class BFMModel(object):
    def __init__(self, bfm_fp, bfm_tri_fp, shape_dim=40, exp_dim=10):
        bfm = _load(bfm_fp)

        self.u = bfm.get('u').astype(np.float32)  # fix bug
        self.w_shp = bfm.get('w_shp').astype(np.float32)[..., :shape_dim]
        self.w_exp = bfm.get('w_exp').astype(np.float32)[..., :exp_dim]
        w = np.concatenate((self.w_shp, self.w_exp), axis=1)
        self.w_norm = np.linalg.norm(w, axis=0)

        self.keypoints = bfm.get('keypoints').astype(np.long)  # fix bug
        self.u_base = self.u[self.keypoints].reshape(-1, 1)
        self.w_shp_base = self.w_shp[self.keypoints]
        self.w_exp_base = self.w_exp[self.keypoints]

        self.tri = _load(bfm_tri_fp)
        self.tri = _to_ctype(self.tri.T).astype(np.int32)


class TDDFA(object):
    """TDDFA: named Three-D Dense Face Alignment (TDDFA)"""

    def __init__(self, weights, bfm_path, bfm_tri_path, _3DDM_mean_std_path, gpu_mode):
        torch.set_grad_enabled(False)

        # Config
        self.gpu_mode = gpu_mode
        self.gpu_id = 0
        self.size = 120

        # Load BFM
        self.bfm = BFMModel(
            bfm_fp=bfm_path,
            bfm_tri_fp=bfm_tri_path,
            shape_dim=40,
            exp_dim=10,
        )
        self.tri = self.bfm.tri

        # Load TDDFA, wrapping MobileNet default output is 62 dimensional
        # 12 (pose) + 40 (shape) +10 (expression)
        self.model = MobileNet(num_classes=62, widen_factor=1)
        self.model = load_model(self.model, weights)
        self.model.eval()

        if self.gpu_mode:
            cudnn.benchmark = True
            self.model = self.model.cuda(device=self.gpu_id)

        # params normalization config
        r = _load(_3DDM_mean_std_path)
        self._3DDM_mean = r.get('mean')
        self._3DDM_std = r.get('std')


    def __call__(self, img_ori, objs):
        """
        The main call of TDDFA, given image and box / landmark, return 3DMM params and roi_box
        :param img_ori: the input image
        :param objs: the list of box or landmarks
        :return: param list and roi_box list
        """
        # Crop image, forward to get the param
        param_lst = []
        roi_box_lst = []

        for obj in objs:
            roi_box = parse_roi_box_from_bbox(obj)
            roi_box_lst.append(roi_box)

            img = crop_img(img_ori, roi_box)
            img = cv2.resize(img, dsize=(self.size, self.size), interpolation=cv2.INTER_LINEAR)

            inp = torch.Tensor(img).permute(2, 0, 1).unsqueeze(0).float()
            inp = (inp - 127.5) / 128.0

            if self.gpu_mode:
                inp = inp.cuda(device=self.gpu_id)

            param = self.model(inp)
            param = param.squeeze().cpu().numpy().flatten().astype(np.float32)
            param = param * self._3DDM_std + self._3DDM_mean  # re-scale

            param_lst.append(param)

        return param_lst, roi_box_lst


    def recon_vers(self, param_lst, roi_box_lst):
        size = self.size

        ver_lst = []
        for param, roi_box in zip(param_lst, roi_box_lst):
            R, offset, alpha_shp, alpha_exp = _parse_param(param)
            pts3d = R @ (self.bfm.u_base + self.bfm.w_shp_base @ alpha_shp + self.bfm.w_exp_base @ alpha_exp). \
                reshape(3, -1, order='F') + offset
            pts3d = similar_transform(pts3d, roi_box, size)

            ver_lst.append(pts3d)

        return ver_lst
