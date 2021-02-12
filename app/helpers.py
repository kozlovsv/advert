from glob import glob
import os
from app import app
from flask import url_for 
from app.inference import BillboardInferenceDetector

def get_filenames(path, ext='jpg'):
    fname_all = glob(os.path.join(app.root_path, 'static', path, '*.' + ext))
    fname_all.sort()
    return [url_for('static', filename = path + '/' + os.path.basename(fname)) for fname in fname_all]

def get_detector(video_file_name, image_path = 'predict', show_all_images = False, accurate_calc = False, one_fps = False):
    if not accurate_calc:
        checkpoint_file = os.path.join(app.root_path, 'model', 'ssd512-6af640a5.pth')
        config_file = os.path.join(app.root_path, 'model', 'ssd512.py')
    else:
        checkpoint_file = os.path.join(app.root_path, 'model', 'faster_rcnn_r50_all_ds-154ab07e.pth')
        config_file = os.path.join(app.root_path, 'model', 'faster_rcnn_r50_fpn_1x_1.py')
        
    inference_image_path = os.path.join(app.root_path, 'static', image_path) 
    score_draw_box = 0.9
    score_draw_img = 0 if show_all_images else score_draw_box
    fps = 3 if not one_fps else 1
    
    detector = BillboardInferenceDetector(
        video_file_name, 
        inference_image_path, 
        checkpoint_file, 
        config_file, 
        fps = fps, 
        score_draw_img=score_draw_img, 
        score_draw_box=score_draw_box)
    return detector