import logging
from flask import Blueprint, render_template

# 设置日志记录
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

bp = Blueprint('inv', __name__)

@bp.route('/')
def index():
    return render_template('404.html')