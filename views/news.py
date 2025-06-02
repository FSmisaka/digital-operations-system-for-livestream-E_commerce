from flask import Blueprint, render_template, request, jsonify, redirect, url_for, session
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

# 商品数据文件路径
PRODUCTS_FILE = '../data/news.json'

def load_news_data():
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        products_path = os.path.join(current_dir, PRODUCTS_FILE)
        
        if os.path.exists(products_path):
            with open(products_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            logger.warning(f"商品数据文件不存在: {products_path}")
            return []
    except Exception as e:
        logger.error(f"加载商品数据时出错: {str(e)}")
        return []

def calculate_recommendations(user_id):
    products = load_news_data()
    
    # 模拟用户数据（实际应从数据库获取）
    user_selections = {
        "22377002": [1001, 1003, 1005],  # 老用户
        "22377005": [1002, 1004],       # 老用户
        "0": []                          # 新用户
    }
    
    # 判断是否为新用户
    is_new_user = str(user_id) not in user_selections
    user_product_ids = user_selections.get(str(user_id), [])
    
    # 计算推荐分数
    for product in products:
        # 用户历史选品次数
        user_count = user_product_ids.count(product['id'])
        
        # 全体用户选品次数
        total_selected = product['total_selected']
        
        # 滞销情况（滞销次数越多，分数越低）
        unsold_penalty = product['unsold_count']
        
        # 计算最终推荐分数
        if is_new_user:
            # 新用户: 全体选品次数 - 滞销惩罚
            product['recommend_score'] = total_selected - unsold_penalty
        else:
            # 老用户: 用户选品次数 + 全体选品次数 - 滞销惩罚
            # 可以调整权重，例如 user_count * 0.5 降低历史选品的影响
            product['recommend_score'] = user_count + total_selected - unsold_penalty
        
        # 记录用户是否已选过此商品
        product['user_selected'] = user_count > 0
    
    # 按推荐分数排序，分数相同则按总选品次数排序
    sorted_products = sorted(
        products, 
        key=lambda x: (x['recommend_score'], x['total_selected']), 
        reverse=True
    )
    
    return sorted_products

@bp.route('/')
@login_required
def index():
    # 获取用户ID
    user_id = session.get('user_id', '0')
    
    # 获取推荐商品
    recommended_products = calculate_recommendations(user_id)
    
    # 获取热门商品（按全体选择次数排序）
    hot_products = sorted(load_news_data(), key=lambda x: x['total_selected'], reverse=True)[:5]
    
    return render_template(
        'news/index.html',
        featured_product=recommended_products[0] if recommended_products else None,
        recommended_products=recommended_products[1:6],
        hot_products=hot_products
    )

# 修改路由参数名从 <int:news_id> 改为 <int:product_id>
@bp.route('/detail/<int:product_id>')  # 修改这里
@login_required
def detail(product_id):  # 函数参数名也修改
    products = load_news_data()
    product = next((p for p in products if p['id'] == product_id), None)
    
    if not product:
        return render_template('404.html'), 404
    
    # 获取当前用户的推荐商品（排除当前商品）
    user_id = session.get('user_id', '0')
    recommended_products = [
        p for p in calculate_recommendations(user_id) 
        if p['id'] != product_id
    ][:4]  # 取前4个推荐
    
    # 获取相关商品（同分类）
    related_products = [
        p for p in products 
        if p['category'] == product['category'] and p['id'] != product_id
    ][:5]
    
    return render_template(
        'news/detail.html',
        product=product,
        recommended_products=recommended_products,  # 新增推荐商品
        related_products=related_products
    )

@bp.route('/search')
def search():
    """商品搜索功能（需修改模板）"""
    keyword = request.args.get('keyword', '')
    if not keyword:
        return jsonify({'error': '请输入搜索关键词'}), 400
    
    products = load_news_data()
    search_results = [
        p for p in products
        if keyword.lower() in p['name'].lower() or keyword.lower() in p['description'].lower()
    ]
    
    return render_template(
        'news/search.html',
        products=search_results,
        keyword=keyword,
        count=len(search_results)
    )