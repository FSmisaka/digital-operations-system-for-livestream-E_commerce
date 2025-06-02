# 售后
import logging
from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify
import pandas as pd
import os
from datetime import datetime
from views.auth import login_required, user_required

# 设置日志记录
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

bp = Blueprint('ass', __name__)

# 模拟数据库操作
@user_required
def read_csv(file_path):
    try:
        # 确保文件存在
        if not os.path.exists(file_path):
            logger.error(f"文件不存在: {file_path}")
            return pd.DataFrame()
        # 读取CSV文件
        df = pd.read_csv(file_path)
        logger.info(f"成功读取文件: {file_path}, 共 {len(df)} 条记录")
        return df
    except Exception as e:
        logger.error(f"读取文件出错: {file_path}, 错误: {str(e)}")
        return pd.DataFrame()

@user_required
def write_csv(df, file_path):
    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        # 写入CSV文件
        df.to_csv(file_path, index=False)
        logger.info(f"成功写入文件: {file_path}, 共 {len(df)} 条记录")
    except Exception as e:
        logger.error(f"写入文件出错: {file_path}, 错误: {str(e)}")

@bp.route('/')
@user_required
def index():
    return render_template('ass/index.html')

# 客服聊天功能
@bp.route('/customer_service')
@user_required
def customer_service():
    user_id = request.args.get('user_id')
    chat_records = read_csv('data/ass/chat_records.csv')
    
    if chat_records.empty:
        logger.warning("聊天记录为空")
        flash('暂无聊天记录', 'warning')
        return render_template('ass/customer_service.html', chat_records=pd.DataFrame())
    
    # 如果指定了用户ID，只显示该用户的聊天记录
    if user_id:
        try:
            user_id = int(user_id)
            user_records = chat_records[chat_records['user_id'] == user_id]
            if user_records.empty:
                flash('未找到该用户的聊天记录', 'warning')
        except ValueError:
            flash('无效的用户ID', 'error')
            user_records = pd.DataFrame()
    else:
        user_records = pd.DataFrame()
    
    return render_template('ass/customer_service.html', chat_records=chat_records)

# 发送客服消息
@bp.route('/api/send_message', methods=['POST'])
@user_required
def send_message():
    user_id = request.form.get('user_id')
    message = request.form.get('message')
    
    if not user_id or not message:
        return jsonify({'status': 'error', 'message': '参数不完整'})
    
    try:
        # 读取现有聊天记录
        chat_records = read_csv('data/ass/chat_records.csv')
        
        # 创建新消息记录
        new_message = pd.DataFrame({
            'user_id': [int(user_id)],
            'message': [message],
            'is_customer': [False],
            'timestamp': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
        })
        
        # 添加新消息到记录中
        chat_records = pd.concat([chat_records, new_message], ignore_index=True)
        
        # 保存到CSV文件
        write_csv(chat_records, 'data/ass/chat_records.csv')
        
        logger.info(f"成功发送消息: 用户={user_id}, 消息={message}")
        return jsonify({'status': 'success'})
    except Exception as e:
        logger.error(f"发送消息失败: {str(e)}")
        return jsonify({'status': 'error', 'message': '发送失败，请重试'})

# 退货处理功能
@bp.route('/return_processing')
@user_required
def return_processing():
    search = request.args.get('search', '')
    return_records = read_csv('data/ass/return_records.csv')
    
    if search:
        # 将order_id转换为字符串后再进行搜索
        return_records['order_id'] = return_records['order_id'].astype(str)
        # 使用字符串包含匹配进行搜索
        mask = return_records['order_id'].str.contains(search, na=False)
        return_records = return_records[mask]
    
    if return_records.empty:
        logger.warning("退货记录为空")
        flash('暂无退货记录', 'warning')
    return render_template('ass/return_processing.html', return_records=return_records)

# 处理退货申请
@bp.route('/api/process_return', methods=['POST'])
@user_required
def process_return():
    try:
        data = request.get_json()
        return_id = data.get('return_id')
        action = data.get('action')
        
        if not return_id or not action:
            return jsonify({'status': 'error', 'message': '参数不完整'})
        
        # 读取退货记录
        return_records = read_csv('data/ass/return_records.csv')
        
        # 确保return_id是整数
        return_id = int(return_id)
        return_records['return_id'] = return_records['return_id'].astype(int)
        
        # 查找对应的记录
        mask = return_records['return_id'] == return_id
        if not any(mask):
            return jsonify({'status': 'error', 'message': '退货申请不存在'})
        
        # 更新状态
        return_records.loc[mask, 'status'] = f'已{action}'
        
        # 保存更新后的记录
        write_csv(return_records, 'data/ass/return_records.csv')
        
        logger.info(f"处理退货申请: ID={return_id}, 动作={action}")
        return jsonify({'status': 'success'})
    except Exception as e:
        logger.error(f"处理退货申请失败: {str(e)}")
        return jsonify({'status': 'error', 'message': '处理失败，请重试'})

# 物流管理功能
@bp.route('/logistics')
@user_required
def logistics():
    search = request.args.get('search', '')
    # 每次访问页面时都重新读取数据库
    logistics_records = read_csv('data/ass/logistics_records.csv')
    
    if search:
        # 将数值列转换为字符串后再进行搜索
        logistics_records['order_id'] = logistics_records['order_id'].astype(str)
        logistics_records['tracking_number'] = logistics_records['tracking_number'].astype(str)
        
        # 使用字符串包含匹配进行搜索
        mask = (
            logistics_records['order_id'].str.contains(search, na=False) |
            logistics_records['tracking_number'].str.contains(search, na=False)
        )
        logistics_records = logistics_records[mask]
    
    if logistics_records.empty:
        logger.warning("物流记录为空")
        flash('暂无物流记录', 'warning')
    
    return render_template('ass/logistics.html', logistics_records=logistics_records)

# 更新物流状态
@bp.route('/api/update_logistics', methods=['POST'])
@user_required
def update_logistics():
    try:
        # 重新读取数据库获取最新状态
        logistics_records = read_csv('data/ass/logistics_records.csv')
        
        if logistics_records.empty:
            return jsonify({'status': 'error', 'message': '暂无物流记录'})
        
        # 返回最新记录数
        return jsonify({
            'status': 'success', 
            'message': f'已更新到最新状态，共{len(logistics_records)}条记录'
        })
    except Exception as e:
        logger.error(f"更新物流状态失败: {str(e)}")
        return jsonify({'status': 'error', 'message': '更新失败，请重试'})

def get_latest_logistics_status(tracking_number):
    """
    获取物流最新状态
    这里应该实现真实的物流状态查询逻辑，比如调用物流公司API
    目前使用模拟数据
    """
    try:
        # 模拟从物流API获取数据
        # 实际应该替换为真实的API调用
        import random
        statuses = ['待发货', '已发货', '运输中', '已签收', '已退回']
        current_status = random.choice(statuses)
        
        return {
            'status': current_status,
            'update_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'tracking_info': f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}|{current_status}"
        }
    except Exception as e:
        logger.error(f"获取物流状态失败: {str(e)}")
        return None