REM docker run -it --rm -v %cd%:/notebooks -w /notebooks -e HOME=/notebooks/home -p 4545:8888 -e JUPYTER_RUNTIME_DIR=/tmp -e JUPYTER_DATA_DIR=./ mmdetection jupyter lab --ip='0.0.0.0' --port=8888 --allow-root --password_required=False --token=''
docker run -it --rm -p 8000:5000 -e FLASK_ENV=development -e FLASK_DEBUG=1 kozlovsv78/mmdetection flask run --host=0.0.0.0 
PAUSE



