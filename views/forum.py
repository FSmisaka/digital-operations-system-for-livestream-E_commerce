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
    topics = load_data(TOPICS_FILE)

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

    users = load_data(USERS_FILE)
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
    topics = load_data(TOPICS_FILE)
    topic = next((t for t in topics if t['id'] == topic_id), None)

    if not topic:
        return render_template('404.html'), 404

    topic['views'] = topic.get('views', 0) + 1
    save_data(TOPICS_FILE, topics)

    replies = load_data(REPLIES_FILE)
    topic_replies = [reply for reply in replies if reply['topic_id'] == topic_id]
    topic_replies.sort(key=get_created_at)

    related_topics = [
        t for t in topics
        if t['category'] == topic['category'] and t['id'] != topic_id
    ][:3]

    users = load_data(USERS_FILE)
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
    topics = load_data(TOPICS_FILE)

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
    users = load_data(USERS_FILE)
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
    topics = load_data(TOPICS_FILE)

    # 查找指定ID的主题
    topic = next((t for t in topics if t['id'] == topic_id), None)

    if not topic:
        return jsonify({'success': False, 'message': '主题不存在'}), 404

    # 加载回复数据
    replies = load_data(REPLIES_FILE)

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
    users = load_data(USERS_FILE)
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

    topics = load_data(TOPICS_FILE)
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
