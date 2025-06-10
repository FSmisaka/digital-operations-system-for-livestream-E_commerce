import os
import json
import logging
from datetime import datetime
import time
from flask import Blueprint, render_template, request, jsonify, redirect, url_for, flash, session
from views.auth import login_required, supplier_required
from views.data_utils import reset_data_file_path, load_data, save_data
from flask import Blueprint, render_template, request, redirect, url_for

bp = Blueprint('forum', __name__)
# 设置日志记录
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 论坛数据文件路径
FORUM_DATA_PATH = '../data/forum'
TOPICS_FILE = f'{FORUM_DATA_PATH}/topics.json'
REPLIES_FILE = f'{FORUM_DATA_PATH}/replies.json'
USERS_FILE = f'{FORUM_DATA_PATH}/users.json'
MESSAGES_FILE = f'{FORUM_DATA_PATH}/messages.json'
FORUM_CATEGORIES = load_data('../data/forum/categories.json')

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

    #筛选出纯净的产品
    filtered_topics = [topic for topic in topics if topic['category'] != "announcement"]
    
    forum_stats = {
        "topics": len(topics),
        "filtered_topics":len(filtered_topics),
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
        'username': session.get('username'),  
        'avatar': 'https://via.placeholder.com/40',
        'created_at': datetime.now().strftime('%Y-%m-%d'),
        'updated_at': datetime.now().strftime('%Y-%m-%d'),
        'views': 0,
        'replies': 0,
        'last_reply_at': datetime.now().strftime('%Y-%m-%d'),
        'last_reply_user': session.get('username'),
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
    'role_id': 0 if session.get('user_role') == 'admin' else (1 if session.get('user_role') == 'supplier' else 2),
    'username': session.get('username'),  
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
    topic['last_reply_user'] = session.get('username')
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

#实现用户之间的私聊
def load_users():
    return load_data(USERS_FILE)

# 读取聊天记录
def load_messages():
    return load_data(MESSAGES_FILE)

# 保存聊天记录
def save_messages(messages):
    save_data(MESSAGES_FILE, messages)

# 在会话中获取user的名字
def get_username(user_id):
    users = load_users()  # 假设这是加载所有用户的函数
    for user in users:
        if user['id'] == user_id:
            return user['username']
    return "未知用户"  # 如果找不到用户，则返回默认值

# 获取私聊记录
@bp.route('/private_message/<int:receiver_id>')
@login_required
def private_message(receiver_id):
    user_id = session.get('user_id')  # 获取当前登录用户的ID
    user_name = session.get('username')  # 获取当前登录用户的用户名
    
    # 使用 receiver_id 查找接收者的名字
    receiver_name = request.args.get('receiver_name')  # 获取查询参数中的 receiver_name
    messages = load_messages()

    # 构造聊天记录的key
    key = f"{min(user_id, receiver_id)}_{max(user_id, receiver_id)}"
    
    # 获取用户间的聊天记录
    chat_history = messages.get(key, [])

    # 将每条消息的 sender_id 和 receiver_id 替换为用户名
    for message in chat_history:
        # 根据 sender_id 获取发送者的名字，而不是统一设置为当前登录用户的名字
        message['sender_name'] = get_username(message['sender_id'])
        
        # receiver_name 通过查询参数传递过来，已经是接收者的用户名
        message['receiver_name'] = receiver_name

    return render_template('forum/private_message.html', chat_history=chat_history, receiver_id=receiver_id, receiver_name=receiver_name,user_id=user_id)

#发送消息
@bp.route('/send_message', methods=['POST'])
@login_required
def send_message():
    user_id = session.get('user_id')  # 获取当前登录用户的ID
    receiver_id = int(request.form['receiver_id'])  # 获取接收者的ID
    user_name = session.get('username')  # 获取当前登录用户的用户名
    
    # 使用 receiver_id 查找接收者的用户名
    receiver_name = get_username(receiver_id)  # 通过 receiver_id 获取接收者的用户名
    
    message_content = request.form['message']  # 获取消息内容
    
    # 获取聊天记录
    messages = load_messages()
    
    # 构造聊天记录的key
    key = f"{min(user_id, receiver_id)}_{max(user_id, receiver_id)}"
    chat_history = messages.get(key, [])
    
    # 新消息
    new_message = {
        "sender_id": user_id,
        "receiver_id": receiver_id,
        "message": message_content,
        "timestamp": datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
    }
    
    # 将新消息添加到记录中
    chat_history.append(new_message)
    
    # 保存回json
    messages[key] = chat_history
    save_messages(messages)
    
    # 返回发送者和接收者的用户名
    return jsonify({
        "success": True, 
        "message": "消息已发送", 
        "sender_name": user_name,  # 发送者的用户名
        "receiver_name": receiver_name  # 接收者的用户名
    })

# 获取与所有用户的聊天列表
@bp.route('/messages_center')
@login_required
def messages_center():
    user_id = session.get('user_id')  # 获取当前登录用户的ID
    messages = load_messages()  # 加载所有的聊天记录

    chat_list = []  # 存储所有与当前用户的聊天记录
    for key, chat_history in messages.items():
        sender_id, receiver_id = map(int, key.split('_'))
        
        # 如果聊天记录涉及到当前用户，则添加到聊天列表中
        if sender_id == user_id or receiver_id == user_id:
            last_message = chat_history[-1]  # 获取最后一条消息
            last_msg = last_message['message']
            timestamp = last_message['timestamp']
            
            other_user_id = receiver_id if sender_id == user_id else sender_id
            other_user_name = get_username(other_user_id)  # 获取对方用户名
            
            chat_list.append({
                'receiver_id': other_user_id,
                'receiver_name': other_user_name,
                'last_message': last_msg,
                'timestamp': timestamp,
            })

    return render_template('forum/messages_center.html', chat_list=chat_list)

# 查看与某个用户的聊天记录
@bp.route('/chat/<int:receiver_id>')
def chat(receiver_id):
    user_id = session.get('user_id')  # 获取当前登录用户的ID
    messages = load_messages()  # 加载所有的聊天记录
    receiver_name = get_username(receiver_id)

    key = f"{min(user_id, receiver_id)}_{max(user_id, receiver_id)}"  # 构造聊天记录的key
    chat_history = messages.get(key, [])  # 获取该聊天记录的历史消息

    return render_template('forum/private_message.html', chat_history=chat_history, receiver_id=receiver_id,receiver_name=receiver_name,user_id=user_id)







