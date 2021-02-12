# -*- coding: utf-8 -*- 

from flask_wtf import FlaskForm
from wtforms import BooleanField
from flask_wtf.file import FileField, FileRequired, FileAllowed

class UploadVideoForm(FlaskForm):
    video = FileField(label='Видео файл', validators=[FileRequired('Обязательное поле'), FileAllowed(['mp4', 'avi'], 'Только видео файлы!')]) 
    images_all = BooleanField('Показывать все картинки')
    accurate_calc = BooleanField('Точный рассчет')
    one_fps = BooleanField('Один кадр в секунду')