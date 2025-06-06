from flask import Blueprint, render_template, request, jsonify, redirect, url_for, session
from views.auth import login_required, user_required
import os
import json
import logging
from datetime import datetime
from views.data_utils import save_data, load_data

# 获取日志记录器
logger = logging.getLogger(__name__)

# 创建一个名为 'news' 的 Blueprint
bp = Blueprint('news', __name__)

# 商品数据文件路径
PRODUCTS_FILE = '../data/forum/topics.json'
SELECTED = '../data/selected.json'

def load_news_data():
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        products_path = os.path.join(current_dir, PRODUCTS_FILE)
        if os.path.exists(products_path):
            with open(products_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # 只保留 id>=2 的商品
                return [item for item in data if isinstance(item.get('id'), int) and item['id'] >= 2]
        else:
            logger.warning(f"商品数据文件不存在: {products_path}")
            return []
    except Exception as e:
        logger.error(f"加载商品数据时出错: {str(e)}")
        return []

def calculate_recommendations(user_id):
    products = load_data(PRODUCTS_FILE)
    
    user_selections = load_data(SELECTED)
    l = len(user_selections)

    for s in user_selections:
        if s['user_id'] == user_id:
            user_product_ids = s['selected']

    for product in products[1:]:
        # 用户历史选品次数
        user_count = user_product_ids.get(str(product['id']), 0)
        
        # 全体用户选品次数
        total_selected = product['total_selected']
        
        # 计算最终推荐分数: 用户选品次数 + 全体选品次数 - 滞销惩罚
        # 可以调整权重，例如 user_count * 0.5 降低历史选品的影响
        product['recommend_score'] = user_count + total_selected//l
    
    # 按推荐分数排序，分数相同则按总选品次数排序
    sorted_products = sorted(
        products[1:], 
        key=lambda x: (x['recommend_score'], x['total_selected']), 
        reverse=True
    )
    
    return sorted_products

@bp.route('/')
@user_required
def index():
    # 获取用户ID
    user_id = session.get('user_id')
    
    # 获取推荐商品
    recommended_products = calculate_recommendations(user_id)
    
    # 获取热门商品（按全体选择次数排序）
    hot_products = sorted(load_data(PRODUCTS_FILE)[1:], key=lambda x: x['total_selected'], reverse=True)[:5]
    
    return render_template(
        'news/index.html',
        featured_product=recommended_products[0] if recommended_products else None,
        recommended_products=recommended_products[1:4],
        hot_products=hot_products
    )

@bp.route('/detail/<int:product_id>') 
@login_required
def detail(product_id): 
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
@user_required
def search():
    keywords = request.args.get('keyword', '')
    if not keywords:
        return jsonify({'error': '请输入搜索关键词'}), 400
    
    products = load_data(PRODUCTS_FILE)
    search_results = []
    keyword_ = []

    for product in products:
        flag = False
        keyword_.append('')
        for keyword in keywords.split():
            if keyword.lower() in product['title'].lower() + product['description'].lower():
                flag = True
                keyword_[-1] = keyword.lower()
        if flag: search_results.append(product)
    
    print(len(search_results))
    
    return render_template(
        'news/search.html',
        products=search_results,
        keywords=keyword_,
        keyword=keywords,
        count=len(search_results)
    )

@bp.route('/batch_select', methods=['POST'])
@login_required
def batch_select():
    data = request.get_json()
    product_ids = data.get('product_ids', [])
    
    try:
        # 加载topics.json
        products = load_data(PRODUCTS_FILE)
        print(products)
        
        # 更新选品次数
        for topic in products:
            if topic['id'] in product_ids:
                print(topic['id'])
                topic['total_selected'] = topic.get('total_selected', 0) + 1
        
        # 保存更新后的数据
        save_data(PRODUCTS_FILE, products)
        
        return jsonify({
            'success': True,
            'message': '选品成功'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        })