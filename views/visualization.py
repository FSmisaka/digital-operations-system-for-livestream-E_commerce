from flask import Blueprint, render_template, current_app, jsonify, request
import json
import os
import logging
from views.auth import login_required, user_required
import requests
from config import DEEPSEEK_API_KEY
from openai import OpenAI

bp = Blueprint('visualization', __name__)

# 设置 DEEPSEEK_API_KEY 环境变量
os.environ["DEEPSEEK_API_KEY"] = DEEPSEEK_API_KEY

# 设置 DEEPSEEK_BASE_URL 环境变量
os.environ["DEEPSEEK_BASE_URL"] = "https://api.deepseek.com/v1"

client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url=os.environ["DEEPSEEK_BASE_URL"],
)

@bp.route('/')
@user_required
def view_data():
    return render_template('visualization/view_data.html')

@bp.route('/api/generate-transcript', methods=['POST'])
@user_required
def generate_transcript():
    try:
        data = request.get_json()
        product_description = data.get('product_description')
        
        if not product_description:
            return jsonify({'error': '请提供商品描述'}), 400
        
        # 使用 OpenAI 客户端调用 DeepSeek API 生成直播逐字稿
        completion = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是一位专业的主播助手，请根据商品描述生成适合直播时念的逐字稿。逐字稿应该生动有趣，突出商品特点，包含互动话术，时长约3-5分钟。"},
                {"role": "user", "content": f"请为以下商品生成直播逐字稿：\n{product_description}"}
            ]
        )
        
        transcript = completion.choices[0].message.content
        return jsonify({'transcript': transcript})
    except Exception as e:
        logging.error(f'生成逐字稿时出错: {str(e)}')
        return jsonify({'error': str(e)}), 500

@bp.route('/api/streamer-data')
@user_required
def get_streamer_data():
    try:
        data_file = os.path.join(current_app.root_path, 'data', 'streamer_data.json')
        if not os.path.exists(data_file):
            return jsonify({'error': '数据文件不存在'}), 404
            
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        logging.error(f'获取主播数据时出错: {str(e)}')
        return jsonify({'error': str(e)}), 500

@bp.route('/api/streamer-stats')
@user_required
def get_streamer_stats():
    try:
        import pandas as pd
        stats_file = os.path.join(current_app.root_path, 'data', 'streamer_stats.csv')
        if not os.path.exists(stats_file):
            return jsonify({'error': '统计数据文件不存在'}), 404
            
        df = pd.read_csv(stats_file)
        
        # 确保数据列存在，如果不存在则提供默认值
        stats = {
            'total_streams': int(len(df)),
            'total_duration': float(df['duration'].sum()) if 'duration' in df.columns else 0,
            'avg_viewers': float(df['viewers'].mean()) if 'viewers' in df.columns else 0,
            'total_revenue': float(df['revenue'].sum()) if 'revenue' in df.columns else 0,
            'streams_by_date': df.astype({
                'duration': float,
                'viewers': float,
                'revenue': float
            }).to_dict('records')
        }
        
        return jsonify(stats)
    except Exception as e:
        logging.error(f'获取主播统计数据时出错: {str(e)}')
        return jsonify({'error': str(e)}), 500