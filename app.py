from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from config import Config

# db = SQLAlchemy()

def create_app(config_class=Config):
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_object(config_class)

    # 初始化数据库
    # db.init_app(app)

    # 注册 Blueprints
    from views.main import bp as main_bp
    app.register_blueprint(main_bp)

    from views.visualization import bp as viz_bp
    app.register_blueprint(viz_bp, url_prefix='/viz')

    from views.news import bp as news_bp
    app.register_blueprint(news_bp, url_prefix='/news')

    from views.forum import bp as forum_bp
    app.register_blueprint(forum_bp, url_prefix='/forum')

    from views.auth import bp as auth_bp
    app.register_blueprint(auth_bp, url_prefix='/auth')

    # 配置日志
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    return app

# 在脚本主入口创建应用实例 (方便直接运行 python app.py 启动开发服务器)
if __name__ == '__main__':

    app = create_app()
    app.run(debug=True, host="127.0.0.1", port=8080)