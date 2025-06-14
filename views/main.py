from flask import Blueprint, render_template, redirect, url_for, session, jsonify
import os
import pandas as pd
import logging
from views.auth import user_required, login_required
from views.data_utils import reset_data_file_path, get_full_data_path, load_data
import random

# 设置日志记录
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

bp = Blueprint('main', __name__)

@bp.route('/')
@login_required
def index():
    if 'user_id' not in session:
        return redirect(url_for('auth.login'))

    # 加载产品数据
    product_data = [p for p in load_data('../data/forum/topics.json') if p['category'] != 'announcement']

    return render_template(
        'index.html', 
        latest_news_1=random.choices(product_data, k=3), 
        latest_news_2=random.choices(product_data, k=3)
    )

@bp.route('/api/latest-market-data')
def get_latest_market_data():
    try:
        # 获取数据文件路径
        full_data_path = get_full_data_path()

        if not os.path.exists(full_data_path):
            logger.error(f"数据文件不存在: {full_data_path}")
            return jsonify({"error": "数据文件不存在"}), 404

        # 读取CSV文件
        df = pd.read_csv(full_data_path)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date', ascending=True)
        latest_data = df.iloc[-1]
        if len(df) > 1:
            previous_close = df.iloc[-2]['close']
            change_percent = ((latest_data['close'] - previous_close) / previous_close * 100)
        else:
            change_percent = 0

        date_str = latest_data['date'].strftime('%Y-%m-%d')
        response_data = {
            "date": date_str,
            "open": float(latest_data['open']),
            "high": float(latest_data['high']),
            "low": float(latest_data['low']),
            "close": float(latest_data['close']),
            "volume": int(latest_data['volume']),
            "change_percent": float(change_percent)
        }

        return jsonify(response_data)

    except Exception as e:
        logger.error(f"获取最新市场数据时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

