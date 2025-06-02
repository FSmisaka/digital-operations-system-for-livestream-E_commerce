import os
import json
import logging
from datetime import datetime
import time
from flask import Blueprint, render_template, request, jsonify, redirect, url_for, flash, session
from views.auth import login_required, supplier_required
from views.data_utils import reset_data_file_path, load_data, save_data

# 设置日志记录
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建一个名为 'forum' 的 Blueprint
bp = Blueprint('forum', __name__)

# 论坛数据文件路径 (相对于当前文件)
FORUM_DATA_PATH = '../data/forum'
TOPICS_FILE = f'{FORUM_DATA_PATH}/topics.json'
REPLIES_FILE = f'{FORUM_DATA_PATH}/replies.json'
USERS_FILE = f'{FORUM_DATA_PATH}/users.json'

# 论坛版块
# 修改后的论坛类别数据
FORUM_CATEGORIES = [
    {"id": "electronics", "name": "电子产品", "icon": "bi-phone", "color": "primary", "topics": 125, "replies": 1234},
    {"id": "food", "name": "食品饮料", "icon": "bi-cup-hot", "color": "warning", "topics": 98, "replies": 876},
    {"id": "daily_necessities", "name": "生活用品", "icon": "bi-house", "color": "success", "topics": 76, "replies": 654},
    {"id": "cosmetics", "name": "美妆产品", "icon": "bi-palette", "color": "info", "topics": 145, "replies": 1876},
    {"id": "purchase_order", "name": "采购订单", "icon": "bi-cart-check", "color": "purple", "topics": 60, "replies": 180}
]

# 修改后的样本主题数据
SAMPLE_TOPICS = [
    {
        "id": 1,
        "title": "电子产品市场动态",
        "content": "近期电子产品市场价格波动较大，特别是智能手机领域。大家有何看法？",
        "category": "electronics",
        "category_name": "电子产品",
        "user_id": 1,
        "username": "管理员",
        "avatar": "https://via.placeholder.com/40",
        "created_at": "2023-04-20",
        "updated_at": "2023-04-20",
        "views": 1256,
        "replies": 24,
        "last_reply_at": "2023-05-01",
        "last_reply_user": "交易达人",
        "is_sticky": True,
        "is_announcement": False,
        "is_hot": False,
        "tags": ["智能手机", "价格波动"]
    },
    {
        "id": 2,
        "title": "食品饮料行业分析",
        "content": "最近食品行业受到原料价格上涨的影响，大家对未来几个月的趋势有什么看法？",
        "category": "food",
        "category_name": "食品饮料",
        "user_id": 2,
        "username": "交易达人",
        "avatar": "https://via.placeholder.com/40",
        "created_at": "2023-05-02",
        "updated_at": "2023-05-02",
        "views": 876,
        "replies": 32,
        "last_reply_at": "2023-05-03 14:30",
        "last_reply_user": "市场分析师",
        "is_sticky": False,
        "is_announcement": False,
        "is_hot": True,
        "tags": ["食品价格", "原料"]
    },
    {
        "id": 3,
        "title": "生活用品市场价格变化",
        "content": "生活用品中的洗衣液、纸巾等产品价格也有明显波动，大家如何看待这个趋势？",
        "category": "daily_necessities",
        "category_name": "生活用品",
        "user_id": 3,
        "username": "AI研究者",
        "avatar": "https://via.placeholder.com/40",
        "created_at": "2023-05-01",
        "updated_at": "2023-05-01",
        "views": 654,
        "replies": 18,
        "last_reply_at": "2023-05-03 11:45",
        "last_reply_user": "技术派",
        "is_sticky": False,
        "is_announcement": False,
        "is_hot": True,
        "tags": ["洗衣液", "生活用品", "价格波动"]
    },
    {
        "id": 4,
        "title": "美妆行业新兴品牌推荐",
        "content": "近期美妆行业的某些新兴品牌发展势头强劲，大家对这些品牌的前景怎么看？",
        "category": "cosmetics",
        "category_name": "美妆产品",
        "user_id": 4,
        "username": "期货老手",
        "avatar": "https://via.placeholder.com/40",
        "created_at": "2023-05-03",
        "updated_at": "2023-05-03",
        "views": 432,
        "replies": 45,
        "last_reply_at": "2023-05-03 15:20",
        "last_reply_user": "交易达人",
        "is_sticky": False,
        "is_announcement": False,
        "is_hot": False,
        "tags": ["美妆品牌", "行业发展"]
    },
    {
        "id": 5,
        "title": "采购订单处理与优化",
        "content": "采购订单管理在供应链中的重要性不言而喻，如何提高采购订单的处理效率，降低成本呢？",
        "category": "purchase_order",
        "category_name": "采购订单",
        "user_id": 5,
        "username": "市场分析师",
        "avatar": "https://via.placeholder.com/40",
        "created_at": "2023-05-03",
        "updated_at": "2023-05-03",
        "views": 345,
        "replies": 12,
        "last_reply_at": "2023-05-03 13:15",
        "last_reply_user": "基本面研究",
        "is_sticky": False,
        "is_announcement": False,
        "is_hot": False,
        "tags": ["采购管理", "订单优化"]
    }
]

# 修改后的样本回复数据
SAMPLE_REPLIES = [
    {
        "id": 1,
        "topic_id": 2,
        "user_id": 5,
        "username": "市场分析师",
        "avatar": "https://via.placeholder.com/40",
        "content": "从基本面看，这波上涨主要受以下因素驱动：\n\n1. 国内饲料需求恢复，特别是生猪养殖规模扩大\n2. 豆粕库存处于近五年同期较低水平\n3. 美豆种植进度低于预期，市场担忧供应\n\n短期来看，价格可能继续震荡上行，但需关注美豆种植进度和南美大豆产量情况。",
        "created_at": "2023-05-02 10:30",
        "updated_at": "2023-05-02 10:30",
        "likes": 15
    },
    {
        "id": 2,
        "topic_id": 2,
        "user_id": 7,
        "username": "技术派",
        "avatar": "https://via.placeholder.com/40",
        "content": "从技术面看，豆粕期货突破了前期高点，MACD指标金叉，KDJ指标向上发散，短期有望继续上行。但RSI指标已接近超买区域，后市可能面临回调压力。建议逢低做多，关注3480元/吨压力位。",
        "created_at": "2023-05-02 11:45",
        "updated_at": "2023-05-02 11:45",
        "likes": 8
    },
    {
        "id": 3,
        "topic_id": 2,
        "user_id": 4,
        "username": "期货老手",
        "avatar": "https://via.placeholder.com/40",
        "content": "我认为这波上涨主要是资金推动，机构大幅增仓做多。从持仓数据看，主力多头持仓增加明显。但需要注意的是，价格上涨过快，短期可能面临回调风险。建议设置合理止损，控制仓位。",
        "created_at": "2023-05-02 14:20",
        "updated_at": "2023-05-02 14:20",
        "likes": 12
    }
]


# 模拟用户数据
SAMPLE_USERS = [
    {
        "id": 1,
        "username": "管理员",
        "avatar": "https://via.placeholder.com/40",
        "role": "admin",
        "posts": 56,
        "replies": 234,
        "join_date": "2023-01-01",
        "last_active": "2023-05-03"
    },
    {
        "id": 2,
        "username": "交易达人",
        "avatar": "https://via.placeholder.com/40",
        "role": "user",
        "posts": 87,
        "replies": 342,
        "join_date": "2023-01-15",
        "last_active": "2023-05-03"
    },
    {
        "id": 3,
        "username": "AI研究者",
        "avatar": "https://via.placeholder.com/40",
        "role": "user",
        "posts": 45,
        "replies": 156,
        "join_date": "2023-02-05",
        "last_active": "2023-05-02"
    },
    {
        "id": 4,
        "username": "期货老手",
        "avatar": "https://via.placeholder.com/40",
        "role": "user",
        "posts": 123,
        "replies": 567,
        "join_date": "2023-01-10",
        "last_active": "2023-05-03"
    },
    {
        "id": 5,
        "username": "市场分析师",
        "avatar": "https://via.placeholder.com/40",
        "role": "user",
        "posts": 67,
        "replies": 289,
        "join_date": "2023-02-20",
        "last_active": "2023-05-03"
    },
    {
        "id": 6,
        "username": "期货新手",
        "avatar": "https://via.placeholder.com/40",
        "role": "user",
        "posts": 12,
        "replies": 45,
        "join_date": "2023-04-10",
        "last_active": "2023-05-02"
    },
    {
        "id": 7,
        "username": "技术派",
        "avatar": "https://via.placeholder.com/40",
        "role": "user",
        "posts": 34,
        "replies": 123,
        "join_date": "2023-03-15",
        "last_active": "2023-05-02"
    },
    {
        "id": 8,
        "username": "基本面研究",
        "avatar": "https://via.placeholder.com/40",
        "role": "user",
        "posts": 56,
        "replies": 234,
        "join_date": "2023-02-28",
        "last_active": "2023-05-01"
    }
]



def get_created_at(x):
    return x.get('created_at', '')

def get_replies(x):
    return x.get('replies', 0)

@bp.route('/')
@login_required
def index():
    reset_data_file_path()

    category = request.args.get('category', 'all')
    topics = load_data(TOPICS_FILE, SAMPLE_TOPICS)

    if category != 'all':
        filtered_topics = [topic for topic in topics if topic['category'] == category]
    else:
        filtered_topics = topics

    announcements = [topic for topic in topics if topic.get('is_announcement')]
    hot_topics = [topic for topic in filtered_topics if topic.get('is_hot')]
    latest_topics = [
        topic for topic in filtered_topics
        if not topic.get('is_announcement') and not topic.get('is_hot')
    ]
    latest_topics.sort(key=get_created_at, reverse=True)

    users = load_data(USERS_FILE, SAMPLE_USERS)
    active_users = sorted(users, key=get_replies, reverse=True)[:6]

    forum_stats = {
        "topics": len(topics),
        "replies": sum(topic.get('replies', 0) for topic in topics),
        "users": len(users),
        "latest_user": users[-1]['username'] if users else ""
    }

    return render_template(
        'forum/index.html',
        categories=FORUM_CATEGORIES,
        announcements=announcements,
        hot_topics=hot_topics,
        latest_topics=latest_topics,
        active_users=active_users,
        forum_stats=forum_stats,
        current_category=category
    )

@bp.route('/topic/<int:topic_id>')
@login_required
def topic(topic_id):
    topics = load_data(TOPICS_FILE, SAMPLE_TOPICS)
    topic = next((t for t in topics if t['id'] == topic_id), None)

    if not topic:
        return render_template('404.html'), 404

    topic['views'] = topic.get('views', 0) + 1
    save_data(TOPICS_FILE, topics)

    replies = load_data(REPLIES_FILE, SAMPLE_REPLIES)
    topic_replies = [reply for reply in replies if reply['topic_id'] == topic_id]
    topic_replies.sort(key=get_created_at)

    related_topics = [
        t for t in topics
        if t['category'] == topic['category'] and t['id'] != topic_id
    ][:3]

    users = load_data(USERS_FILE, SAMPLE_USERS)
    user = next((u for u in users if u['id'] == topic['user_id']), {
        'posts': 0,
        'replies': 0,
        'join_date': '未知'
    })

    return render_template(
        'forum/topic.html',
        topic=topic,
        replies=topic_replies,
        related_topics=related_topics,
        user=user
    )

@bp.route('/new')
@supplier_required
def new_topic():
    return render_template('forum/new_topic.html')

@bp.route('/create_topic', methods=['POST'])
@supplier_required
def create_topic():
    # 获取表单数据
    category = request.form.get('category')
    title = request.form.get('title')
    content = request.form.get('content')
    tags = request.form.get('tags', '[]')

    # 验证数据
    if not category or not title or not content:
        return jsonify({'success': False, 'message': '请填写完整信息'}), 400

    # 加载主题数据
    topics = load_data(TOPICS_FILE, SAMPLE_TOPICS)

    # 生成新主题ID
    new_id = max([topic['id'] for topic in topics], default=0) + 1

    # 获取分类名称
    category_name = next((cat['name'] for cat in FORUM_CATEGORIES if cat['id'] == category), '未分类')

    # 解析标签
    try:
        tag_list = json.loads(tags)
    except:
        tag_list = []

    # 创建新主题
    new_topic = {
        'id': new_id,
        'title': title,
        'content': content,
        'category': category,
        'category_name': category_name,
        'user_id': 2,  
        'username': '交易达人',  
        'avatar': 'https://via.placeholder.com/40',
        'created_at': datetime.now().strftime('%Y-%m-%d'),
        'updated_at': datetime.now().strftime('%Y-%m-%d'),
        'views': 0,
        'replies': 0,
        'last_reply_at': datetime.now().strftime('%Y-%m-%d'),
        'last_reply_user': '交易达人',
        'is_sticky': False,
        'is_announcement': False,
        'is_hot': False,
        'tags': tag_list
    }

    # 添加新主题
    topics.append(new_topic)

    # 保存数据
    save_data(TOPICS_FILE, topics)

    # 更新用户发帖数
    users = load_data(USERS_FILE, SAMPLE_USERS)
    for user in users:
        if user['id'] == 2:  
            user['posts'] = user.get('posts', 0) + 1
            user['last_active'] = datetime.now().strftime('%Y-%m-%d')
            break
    save_data(USERS_FILE, users)

    # 返回成功消息
    return jsonify({'success': True, 'message': '主题发布成功', 'topic_id': new_id})

@bp.route('/reply/<int:topic_id>', methods=['POST'])
@login_required
def reply(topic_id):
    # 获取表单数据
    content = request.form.get('content')

    # 验证数据
    if not content:
        return jsonify({'success': False, 'message': '回复内容不能为空'}), 400

    # 加载主题数据
    topics = load_data(TOPICS_FILE, SAMPLE_TOPICS)

    # 查找指定ID的主题
    topic = next((t for t in topics if t['id'] == topic_id), None)

    if not topic:
        return jsonify({'success': False, 'message': '主题不存在'}), 404

    # 加载回复数据
    replies = load_data(REPLIES_FILE, SAMPLE_REPLIES)

    # 生成新回复ID
    new_id = max([reply['id'] for reply in replies], default=0) + 1

    # 创建新回复
    new_reply = {
        'id': new_id,
        'topic_id': topic_id,
        'user_id': 2,  
        'username': '交易达人',  
        'avatar': 'https://via.placeholder.com/40',
        'content': content,
        'created_at': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'updated_at': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'likes': 0
    }

    # 添加新回复
    replies.append(new_reply)

    # 保存回复数据
    save_data(REPLIES_FILE, replies)

    # 更新主题回复数和最后回复信息
    topic['replies'] = topic.get('replies', 0) + 1
    topic['last_reply_at'] = datetime.now().strftime('%Y-%m-%d %H:%M')
    topic['last_reply_user'] = '交易达人'
    save_data(TOPICS_FILE, topics)

    # 更新用户回复数
    users = load_data(USERS_FILE, SAMPLE_USERS)
    for user in users:
        if user['id'] == 2:  
            user['replies'] = user.get('replies', 0) + 1
            user['last_active'] = datetime.now().strftime('%Y-%m-%d')
            break
    save_data(USERS_FILE, users)

    # 返回成功消息
    return jsonify({'success': True, 'message': '回复成功', 'reply_id': new_id})

@bp.route('/search')
@login_required
def search():
    keyword = request.args.get('keyword', '')

    if not keyword:
        return jsonify({'error': '请输入搜索关键词'}), 400

    topics = load_data(TOPICS_FILE, SAMPLE_TOPICS)
    search_results = [
        topic for topic in topics
        if keyword.lower() in topic['title'].lower() or keyword.lower() in topic['content'].lower()
        or (topic.get('tags') and any(keyword.lower() in tag.lower() for tag in topic['tags']))
    ]

    return render_template(
        'forum/search.html',
        topics=search_results,
        keyword=keyword
    )
