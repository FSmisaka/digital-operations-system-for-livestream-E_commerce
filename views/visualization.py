from flask import Blueprint, render_template, current_app, jsonify, request
import json
import os
import logging
import pandas as pd
from views.auth import login_required, user_required, load_data
from views.live_draw import get_live_analysis
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

# 定义主播风格模板
STREAMER_STYLES = {
    "董宇辉": {
        "system_prompt": "你是一位富有文化底蕴和诗意的带货主播，擅长用优美的语言和典故来介绍商品。你的风格温和儒雅，善于用诗词典故来打动观众。",
        "style_tips": "请用优美的语言描述商品，可以适当引用诗词典故，突出商品的文化内涵和品质。"
    },
    "李佳琦": {
        "system_prompt": "你是一位充满激情和感染力的带货主播，擅长用夸张的语气和生动的描述来介绍商品。你的风格热情奔放，善于制造购买氛围。",
        "style_tips": "请用夸张的语气描述商品，多用感叹句，突出商品的独特卖点和优惠力度。"
    },
    "疯狂小杨哥": {
        "system_prompt": "你是一位幽默风趣的带货主播，擅长用搞笑的方式和夸张的表演来介绍商品。你的风格轻松活泼，善于用段子来吸引观众。",
        "style_tips": "请用幽默的方式描述商品，可以加入一些搞笑的段子，突出商品的趣味性和实用性。"
    }
}

@bp.route('/')
@user_required
def view_data():
    return render_template('visualization/view_data.html')

PRODUCTS = '../data/forum/topics.json'

@bp.route('/api/products')
@user_required
def get_products():
    try:
        topics = load_data(PRODUCTS)
        
        # 确保跳过第一个公告项，并只包含有效的商品数据
        formatted_products = []
        for topic in topics:  # 跳过第一个公告
            if topic.get('id') and topic.get('category') != 'announcement':  # 确保有 id
                formatted_products.append({
                    'id': topic.get('id'),
                    'name': topic.get('title'),
                    'price': topic.get('price', 0),
                    'category': topic.get('category', '未分类'),
                    'supplier': topic.get('supplier', '未知供应商'),
                    'description': topic.get('content', '')
                })
        
        return jsonify(formatted_products)
    except Exception as e:
        logging.error(f'获取商品数据时出错: {str(e)}')
        return jsonify({'error': str(e)}), 500

@bp.route('/api/streamer-styles')
@user_required
def get_streamer_styles():
    return jsonify(list(STREAMER_STYLES.keys()))

@bp.route('/api/generate-transcript', methods=['POST'])
@user_required
def generate_transcript():
    try:
        data = request.get_json()
        product_id = data.get('product_id')
        streamer_style = data.get('streamer_style', '李佳琦')
        
        if not product_id:
            return jsonify({'error': '请提供商品ID'}), 400
            
        # 获取商品信息
        products = load_data(PRODUCTS)
            
        product = next((p for p in products if p['id'] == product_id), None)
        if not product:
            return jsonify({'error': '商品不存在'}), 404
            
        # 构建商品描述
        product_description = f"""
商品名称：{product['title']}
商品类别：{product['category_name']}
价格：{product['price']}元
供应商：{product['supplier']}
商品描述：{product['details']['specs']}
产品特点：{product['details']['features']}
销售亮点：{product['details']['sale_points']}
"""
        
        # 获取主播风格配置
        style_config = STREAMER_STYLES.get(streamer_style, STREAMER_STYLES['李佳琦'])
        
        # 使用 OpenAI 客户端调用 DeepSeek API 生成直播逐字稿
        completion = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": style_config['system_prompt']},
                {"role": "user", "content": f"{style_config['style_tips']}\n\n请为以下商品生成直播逐字稿：\n{product_description}"}
            ]
        )
        
        transcript = completion.choices[0].message.content
        return jsonify({
            'transcript': transcript,
            'product': product,
            'streamer_style': streamer_style
        })
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

@bp.route('/api/product-analysis', methods=['POST'])
@user_required
def product_analysis():
    try:
        data = request.get_json()
        product_id = data.get('product_id')
        streamer_style = data.get('streamer_style', '李佳琦')
        
        if not product_id:
            return jsonify({'error': '请提供商品ID'}), 400
            
        # 获取商品信息
        data_file = 'C:/Users/16390/Documents/GitHub/digital-operations-system-for-livestream-E_commerce/data/forum/topics.json'
        logging.info(f'尝试读取文件: {data_file}')
        with open(data_file, 'r', encoding='utf-8') as f:
            products = json.load(f)
            
        product = next((p for p in products if p['id'] == product_id), None)
        if not product:
            return jsonify({'error': '商品不存在'}), 404
            
        # 获取主播风格配置
        style_config = STREAMER_STYLES.get(streamer_style, STREAMER_STYLES['李佳琦'])
        
        # 使用 DeepSeek API 进行实时分析
        analysis_response = requests.post(
            f"{os.environ['DEEPSEEK_BASE_URL']}/live-analysis",
            headers={
                "Authorization": f"Bearer {os.environ['DEEPSEEK_API_KEY']}",
                "Content-Type": "application/json"
            },
            json={
                "product_id": product_id,
                "streamer_style": streamer_style
            }
        )
        
        if analysis_response.status_code != 200:
            return jsonify({'error': '实时分析请求失败'}), 500
            
        analysis_result = analysis_response.json()
        
        return jsonify({
            'analysis': analysis_result,
            'product': product,
            'streamer_style': streamer_style
        })
    except Exception as e:
        logging.error(f'实时分析时出错: {str(e)}')
        return jsonify({'error': str(e)}), 500

@bp.route('/api/live-analysis/<live_number>')
@user_required
def live_analysis(live_number):
    """获取直播分析数据"""
    try:
        # 如果live_number是日期格式，从日期中提取直播场次
        if '-' in live_number:
            # 这里可以根据需要添加日期到场次的转换逻辑
            live_id = live_number.replace('-', '')  # 临时处理，实际应该根据业务逻辑来转换
        else:
            live_id = live_number
            
        return jsonify(get_live_analysis(live_id))
    except Exception as e:
        logging.error(f'获取直播分析数据时出错: {str(e)}')
        return jsonify({'error': str(e)}), 500

@bp.route('/live-detail/<live_number>')
@user_required
def live_detail(live_number):
    """渲染直播详情页面"""
    return render_template('visualization/live_detail.html', live_number=live_number)


