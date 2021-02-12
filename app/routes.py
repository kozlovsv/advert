# -*- coding: utf-8 -*- 
from flask import render_template, redirect, url_for
from app import app
from app.forms import UploadVideoForm
from app.helpers import get_filenames, get_detector
from werkzeug.utils import secure_filename
import os

@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def index():
    form = UploadVideoForm()
    if form.validate_on_submit():
        f = form.video.data
        #file
        filename = secure_filename(f.filename)
        video_file_name = os.path.join(app.instance_path, filename)
        f.save(video_file_name)
        #params
        accurate_calc = form.accurate_calc.data
        one_fps = form.one_fps.data
        show_all_images = form.images_all.data
        detector = get_detector(video_file_name, image_path = 'predict', show_all_images = show_all_images, accurate_calc = accurate_calc, one_fps = one_fps)
        path = detector.inference_detector_from_video()
        return redirect(url_for('result', path = path))
    return render_template('index.html', form=form)

@app.route('/result/<path>')
def result(path):
    return render_template('result.html', title = 'Результаты работы нейронной сети', images = get_filenames(os.path.join('predict', path)))  

