import os
import cv2
from glob import glob
import numpy as np
import datetime

import matplotlib.pyplot as plt

import torch 

import mmcv

from mmdet.apis import init_detector, inference_detector
from mmdet.apis.inference import LoadImage
from mmdet.datasets.pipelines import Compose
from mmcv.parallel import collate, scatter

import pycocotools.mask as maskUtils


class BillboardInferenceDetector():

    def __init__(self, video_file_name, inference_image_path, checkpoint_file, config_file, fps = 1, score_draw_img=0.9, score_draw_box=0.9, device='cpu'):
        self.video_file_name = video_file_name
        self.inference_image_path = inference_image_path
        self.fps = fps
        self.score_draw_img = score_draw_img
        self.score_draw_box = score_draw_box
        self.checkpoint_file = checkpoint_file
        self.config_file = config_file
        self.device = device

    def find_true_bboxes(self, result, score_thr=0.7):
        bboxes = np.vstack(result)
        if len(bboxes) == 0:
            return False
        for bbox in bboxes:
            if bbox[4] >= score_thr:
                return True
        return False

    def imshow_det_bboxes(self, 
        img, bboxes, labels, class_names=None, score_thr=0,
        bbox_color='green', text_color='green', thickness=2, font_scale=2,
        show=True, win_name='', wait_time=0, out_file=None):
        
        assert bboxes.ndim == 2
        assert labels.ndim == 1
        assert bboxes.shape[0] == labels.shape[0]
        assert bboxes.shape[1] == 4 or bboxes.shape[1] == 5
        img = mmcv.imread(img)

        if score_thr > 0:
            assert bboxes.shape[1] == 5
            scores = bboxes[:, -1]
            inds = scores > score_thr
            bboxes = bboxes[inds, :]
            labels = labels[inds]

        bbox_color = mmcv.color_val(bbox_color)
        text_color = mmcv.color_val(text_color)

        for bbox, label in zip(bboxes, labels):
            bbox_int = bbox.astype(np.int32)
            left_top = (bbox_int[0], bbox_int[1])
            right_bottom = (bbox_int[2], bbox_int[3])
            cv2.rectangle(
                img, left_top, right_bottom, bbox_color, thickness=thickness)
            label_text = class_names[
                label] if class_names is not None else 'cls {}'.format(label)
            if len(bbox) > 4:
                label_text += '|{:.02f}'.format(bbox[-1])
            cv2.putText(img, label_text, (bbox_int[0], bbox_int[1] - 2),
                       cv2.FONT_HERSHEY_COMPLEX, font_scale, text_color, thickness)

        if show:
            plt.figure(figsize=(20,20))
            #plt.axis('off')
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.show()
        if out_file is not None:
            mmcv.imwrite(img, out_file)

    def show_result(self, img, result, class_names, score_thr=0.7, wait_time=0, 
        out_file=None, font_scale=2, thickness=8, show_mask=True):
        assert isinstance(class_names, (tuple, list))
        img = mmcv.imread(img)
        if isinstance(result, tuple):
            bbox_result, segm_result = result
        else:
            bbox_result, segm_result = result, None
        bboxes = np.vstack(bbox_result)
        # draw segmentation masks
        if show_mask and segm_result is not None:
            print(len(segm_result[0]))
            segms = mmcv.concat_list(segm_result)
            inds = np.where(bboxes[:, -1] > score_thr)[0]
            for i in inds:
                color_mask = np.random.randint(0, 256, (1, 3), dtype=np.uint8)
                mask = maskUtils.decode(segms[i]).astype(np.bool)
                img[mask] = img[mask] * 0.5 + color_mask * 0.5
        # draw bounding boxes
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)
        self.imshow_det_bboxes(img.copy(), bboxes,
            labels, class_names=class_names, score_thr=score_thr,
            show=out_file is None, wait_time=wait_time,
            out_file=out_file, font_scale=font_scale,
            thickness=thickness)

    def cpu_inference_detector(self, model, img):
        cfg = model.cfg
        # build the data pipeline
        test_pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]
        test_pipeline = Compose(test_pipeline)
        # prepare data
        data = dict(img=img)
        data = test_pipeline(data)
        data = collate([data], samples_per_gpu=1)
        data['img_meta'] =  data['img_meta'][0].data
        # forward the model
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        return result
    
    def inference_detector_from_video(self):
        model = init_detector(self.config_file, self.checkpoint_file, device=self.device)
        vidcap = cv2.VideoCapture(self.video_file_name)
        duration = (int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT)) / vidcap.get(cv2.CAP_PROP_FPS)) * 1000
        sec = 0
        count = 0
        vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
        success,image = vidcap.read()

        ts = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
        dest_path = os.path.join(self.inference_image_path, ts)
        os.mkdir(dest_path)

        is_cpu = self.device.startswith('cpu')
        while success:
            result = self.cpu_inference_detector(model, image) if is_cpu else inference_detector(model, image)
            is_find = (self.score_draw_img == 0) or self.find_true_bboxes(result, score_thr=self.score_draw_img)
            if is_find:
                self.show_result(image, result, ('БАННЕР', 'none'), score_thr=self.score_draw_box, out_file = os.path.join(dest_path, "img_%s_%d.jpg" % (ts, count)))
            count += 1
            sec +=  self.fps
            pos_msec = sec*1000
            if pos_msec > duration:
                break
            vidcap.set(cv2.CAP_PROP_POS_MSEC, pos_msec)
            success,image = vidcap.read()
        return ts