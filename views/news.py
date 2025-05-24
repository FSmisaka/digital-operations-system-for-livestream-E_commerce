from flask import Blueprint, render_template, request, jsonify, redirect, url_for
from views.auth import login_required
import os
import json
import logging
from datetime import datetime
from views.data_utils import reset_data_file_path

# 获取日志记录器
logger = logging.getLogger(__name__)

# 创建一个名为 'news' 的 Blueprint
bp = Blueprint('news', __name__)

# 新闻数据文件路径 (相对于当前文件)
NEWS_FILE_PATH = '../data/news.json'

# 新闻分类
NEWS_CATEGORIES = [
    {"id": "all", "name": "全部", "active": True},
    {"id": "market", "name": "市场动态", "active": False},
    {"id": "policy", "name": "政策法规", "active": False},
    {"id": "industry", "name": "行业分析", "active": False},
    {"id": "international", "name": "国际资讯", "active": False},
    {"id": "company", "name": "企业新闻", "active": False}
]

# 默认空新闻列表，实际数据从文件加载
SAMPLE_NEWS = []

# 市场日历数据
MARKET_CALENDAR = [
    {
        "id": 1,
        "title": "USDA大豆产量报告",
        "organization": "美国农业部",
        "date": "2023-05-10"
    },
    {
        "id": 2,
        "title": "豆粕期货交割日",
        "organization": "大连商品交易所",
        "date": "2023-05-15"
    },
    {
        "id": 3,
        "title": "全球油脂油料峰会",
        "organization": "北京",
        "date": "2023-05-20"
    }
]

def get_views(x):
    return x.get('views', 0)

def load_news_data():
    try:
        # 获取当前文件所在目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        news_file_path = os.path.join(current_dir, NEWS_FILE_PATH)

        # 检查文件是否存在
        if os.path.exists(news_file_path):
            with open(news_file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            logger.warning(f"新闻数据文件不存在: {news_file_path}，使用示例数据")
            return SAMPLE_NEWS
    except Exception as e:
        logger.error(f"加载新闻数据时出错: {e}")
        return SAMPLE_NEWS

def save_news_data(news_data):
    try:
        # 获取当前文件所在目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        news_file_path = os.path.join(current_dir, NEWS_FILE_PATH)

        # 确保目录存在
        os.makedirs(os.path.dirname(news_file_path), exist_ok=True)

        with open(news_file_path, 'w', encoding='utf-8') as f:
            json.dump(news_data, f, ensure_ascii=False, indent=4)

        logger.info(f"新闻数据已保存到: {news_file_path}")
        return True
    except Exception as e:
        logger.error(f"保存新闻数据时出错: {e}")
        return False

@bp.route('/')
@login_required
def index():
    # 重置数据文件路径为默认值
    reset_data_file_path()

    # 加载新闻数据
    news_data = load_news_data()

    # 获取头条新闻（标记为featured的新闻）
    featured_news = next((news for news in news_data if news.get('is_featured')), news_data[0] if news_data else None)

    # 获取热门新闻（按浏览量排序）
    popular_news = sorted(news_data, key=get_views, reverse=True)[:5]

    return render_template(
        'news/index.html',
        news_list=news_data,
        featured_news=featured_news,
        popular_news=popular_news,
        market_calendar=MARKET_CALENDAR
    )

@bp.route('/detail/<int:news_id>')
@login_required
def detail(news_id):
    news_data = load_news_data()
    news_item = next((news for news in news_data if news['id'] == news_id), None)

    if not news_item:
        return render_template('404.html'), 404

    news_item['views'] = news_item.get('views', 0) + 1
    save_news_data(news_data)

    related_news = [
        news for news in news_data
        if news.get('category') == news_item.get('category') and news['id'] != news_id
    ][:3]

    if 'url' in news_item and news_item['url']:
        news_item['external_url'] = news_item['url']

    return render_template(
        'news/detail.html',
        news=news_item,
        related_news=related_news
    )

@bp.route('/search')
def search():
    # 获取搜索关键词
    keyword = request.args.get('keyword', '')

    if not keyword:
        return jsonify({'error': '请输入搜索关键词'}), 400

    # 加载新闻数据
    news_data = load_news_data()

    # 搜索标题和内容中包含关键词的新闻
    search_results = [
        news for news in news_data
        if keyword.lower() in news.get('title', '').lower() or keyword.lower() in news.get('content', '').lower()
    ]

    return render_template(
        'news/search.html',
        news_list=search_results,
        keyword=keyword,
        count=len(search_results)
    )

@bp.route('/api/subscribe', methods=['POST'])
def subscribe():
    email = request.form.get('email')

    if not email:
        return jsonify({'success': False, 'message': '请输入有效的邮箱地址'}), 400

    return jsonify({
        'success': True,
        'message': f'感谢订阅！我们会将最新资讯发送到 {email}'
    })
