import os
import json
import logging
from datetime import datetime
import time
from flask import Blueprint, render_template, request, jsonify, redirect, url_for, flash, session
from views.auth import login_required
from views.data_utils import reset_data_file_path

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
FORUM_CATEGORIES = [
    {"id": "market_analysis", "name": "市场分析", "icon": "bi-graph-up", "color": "primary", "topics": 125, "replies": 1234},
    {"id": "trading_strategy", "name": "交易策略", "icon": "bi-lightbulb", "color": "warning", "topics": 98, "replies": 876},
    {"id": "news_analysis", "name": "新闻解读", "icon": "bi-newspaper", "color": "success", "topics": 76, "replies": 654},
    {"id": "beginner", "name": "新手交流", "icon": "bi-people", "color": "info", "topics": 145, "replies": 1876}
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

# 模拟主题数据
SAMPLE_TOPICS = [
    {
        "id": 1,
        "title": "论坛规则更新：请文明发言，禁止发布广告内容",
        "content": "为了维护良好的论坛环境，特制定以下规则：\n\n1. 请文明发言，禁止人身攻击和侮辱性言论\n2. 禁止发布广告和垃圾信息\n3. 禁止发布违法内容\n4. 请勿重复发帖\n5. 尊重原创，转载请注明出处\n\n违反规则的用户将视情节轻重给予警告、禁言或封号处理。",
        "category": "announcement",
        "category_name": "公告",
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
        "is_announcement": True,
        "is_hot": False,
        "tags": ["规则", "公告"]
    },
    {
        "id": 2,
        "title": "如何看待近期豆粕价格波动？",
        "content": "近期豆粕期货价格波动较大，主力合约从3300元/吨上涨至3500元/吨，涨幅超过6%。这波上涨主要受哪些因素驱动？后市如何看待？欢迎各位分享观点。",
        "category": "market_analysis",
        "category_name": "市场分析",
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
        "tags": ["价格波动", "市场分析", "豆粕"]
    },
    {
        "id": 3,
        "title": "深度学习模型在期货预测中的应用",
        "content": "近年来，深度学习技术在金融领域的应用越来越广泛。本帖分享一些我在豆粕期货价格预测中使用深度学习模型的经验和心得。\n\n我主要使用了LSTM和Transformer模型，结合技术指标和基本面数据进行预测。在回测中，模型准确率达到了65%左右。\n\n大家有没有其他好的模型推荐？或者有什么特别有效的特征工程方法？",
        "category": "trading_strategy",
        "category_name": "交易策略",
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
        "tags": ["深度学习", "价格预测", "LSTM", "Transformer"]
    },
    {
        "id": 4,
        "title": "豆粕期货交易策略分享",
        "content": "分享一下我近期使用的豆粕期货交易策略，主要基于价格突破和成交量变化。\n\n策略要点：\n1. 关注日线级别的价格突破\n2. 结合60分钟周期的MACD指标\n3. 成交量放大确认\n4. 设置合理的止损位\n\n过去三个月，这个策略的胜率在60%左右，盈亏比约为1.8。欢迎大家讨论和提出改进建议。",
        "category": "trading_strategy",
        "category_name": "交易策略",
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
        "tags": ["交易策略", "技术分析", "突破", "成交量"]
    },
    {
        "id": 5,
        "title": "解读：中国大豆进口量创新高对豆粕价格的影响",
        "content": "近日，海关总署公布的数据显示，今年第一季度中国大豆进口量同比增长15.2%，创历史新高。这一数据对豆粕期货价格有何影响？\n\n从供需角度分析，进口量增加意味着国内大豆供应充足，豆粕产量有保障，短期内可能对豆粕价格形成压制。但从另一方面看，进口量增加也反映了下游需求的增长，特别是养殖业的恢复，这对豆粕价格形成支撑。\n\n综合来看，短期内豆粕价格可能震荡调整，但中长期仍有上涨空间。各位怎么看？",
        "category": "news_analysis",
        "category_name": "新闻解读",
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
        "tags": ["大豆进口", "新闻解读", "价格影响"]
    },
    {
        "id": 6,
        "title": "新手求助：如何开始豆粕期货交易？",
        "content": "我是期货市场的新手，对豆粕期货比较感兴趣，想请教各位前辈：\n\n1. 开始交易前需要学习哪些基础知识？\n2. 有哪些推荐的书籍或课程？\n3. 如何控制风险？\n4. 初始资金需要准备多少比较合适？\n\n感谢各位指导！",
        "category": "beginner",
        "category_name": "新手交流",
        "user_id": 6,
        "username": "期货新手",
        "avatar": "https://via.placeholder.com/40",
        "created_at": "2023-05-02",
        "updated_at": "2023-05-02",
        "views": 567,
        "replies": 28,
        "last_reply_at": "2023-05-02 23:45",
        "last_reply_user": "期货老手",
        "is_sticky": False,
        "is_announcement": False,
        "is_hot": False,
        "tags": ["新手入门", "学习资料", "风险控制"]
    },
    {
        "id": 7,
        "title": "技术分析：豆粕期货突破前高后的操作建议",
        "content": "豆粕期货主力合约昨日突破前期高点3450元/吨，收盘站稳该位置。从技术面看，这是一个重要的突破信号。\n\n目前MACD指标金叉，KDJ指标进入超买区域，短期有望继续上行。但需注意，RSI指标已达到70以上，接近超买区域，后市可能面临回调压力。\n\n操作建议：\n1. 短线可考虑逢低做多策略\n2. 关注3480元/吨压力位和3420元/吨支撑位\n3. 设置合理止损，控制仓位\n\n各位怎么看？",
        "category": "trading_strategy",
        "category_name": "交易策略",
        "user_id": 7,
        "username": "技术派",
        "avatar": "https://via.placeholder.com/40",
        "created_at": "2023-05-02",
        "updated_at": "2023-05-02",
        "views": 432,
        "replies": 15,
        "last_reply_at": "2023-05-02 22:30",
        "last_reply_user": "交易达人",
        "is_sticky": False,
        "is_announcement": False,
        "is_hot": False,
        "tags": ["技术分析", "突破", "操作建议"]
    },
    {
        "id": 8,
        "title": "基本面分析：饲料需求增长对豆粕价格的支撑",
        "content": "随着生猪养殖规模扩大和禽类养殖恢复，国内饲料需求持续增长。据农业农村部数据，今年一季度全国饲料产量同比增长8.5%，其中蛋白饲料增长更为明显。\n\n豆粕作为主要的蛋白饲料原料，需求增长将对价格形成支撑。从库存数据看，截至上周五，全国豆粕库存为65.2万吨，环比减少5.3%，同比减少12.8%，处于近五年同期较低水平。\n\n综合来看，基本面因素对豆粕价格形成支撑，中长期仍有上涨空间。",
        "category": "market_analysis",
        "category_name": "市场分析",
        "user_id": 8,
        "username": "基本面研究",
        "avatar": "https://via.placeholder.com/40",
        "created_at": "2023-05-01",
        "updated_at": "2023-05-01",
        "views": 321,
        "replies": 9,
        "last_reply_at": "2023-05-01 21:15",
        "last_reply_user": "市场分析师",
        "is_sticky": False,
        "is_announcement": False,
        "is_hot": False,
        "tags": ["基本面", "饲料需求", "库存"]
    }
]

# 模拟回复数据
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
    },
    {
        "id": 4,
        "topic_id": 3,
        "user_id": 7,
        "username": "技术派",
        "avatar": "https://via.placeholder.com/40",
        "content": "我也尝试过使用深度学习模型预测期货价格，但发现一个问题：模型在历史数据上表现很好，但在实盘中准确率下降明显。你是如何解决这个过拟合问题的？",
        "created_at": "2023-05-01 15:30",
        "updated_at": "2023-05-01 15:30",
        "likes": 5
    },
    {
        "id": 5,
        "topic_id": 3,
        "user_id": 3,
        "username": "AI研究者",
        "avatar": "https://via.placeholder.com/40",
        "content": "关于过拟合问题，我采取了以下措施：\n\n1. 增加正则化（L1和L2）\n2. 使用Dropout层\n3. 采用交叉验证\n4. 特征工程中加入更多基本面数据\n\n此外，我发现模型预测的不是具体价格，而是价格变化方向（涨/跌）效果更好。",
        "created_at": "2023-05-01 16:45",
        "updated_at": "2023-05-01 16:45",
        "likes": 10
    }
]

def load_data(file_path, default_data=None):
    try:
        # 获取当前文件所在目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.join(current_dir, file_path)

        # 确保目录存在
        os.makedirs(os.path.dirname(full_path), exist_ok=True)

        # 检查文件是否存在
        if os.path.exists(full_path):
            with open(full_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            logger.warning(f"数据文件不存在: {full_path}，使用默认数据")
            # 如果文件不存在且提供了默认数据，则保存默认数据
            if default_data:
                save_data(file_path, default_data)
            return default_data or []
    except Exception as e:
        logger.error(f"加载数据时出错: {e}")
        return default_data or []

def save_data(file_path, data):
    try:
        # 获取当前文件所在目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.join(current_dir, file_path)

        # 确保目录存在
        os.makedirs(os.path.dirname(full_path), exist_ok=True)

        with open(full_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

        logger.info(f"数据已保存到: {full_path}")
        return True
    except Exception as e:
        logger.error(f"保存数据时出错: {e}")
        return False

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
    hot_topics = [topic for topic in topics if topic.get('is_hot')]
    latest_topics = [
        topic for topic in topics
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
@login_required
def new_topic():
    return render_template('forum/new_topic.html')

@bp.route('/create', methods=['POST'])
@login_required
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
