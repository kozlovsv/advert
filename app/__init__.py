import os
from flask import Flask
from flask_bootstrap import Bootstrap
from config import Config

def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__)
    

    if test_config is None:
        app.config.from_object(Config)
    else:
        app.config.from_mapping(test_config)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass
        
    bootstrap = Bootstrap(app)
    
    # a simple page that says hello
    @app.route('/hello')
    def hello():
        return 'Hello, World!'

    return app
    
app = create_app()
    
from app import routes, errors