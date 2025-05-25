import logging
import json
from views.auth import login_required
from views.data_utils import load_data, save_data
from flask import Blueprint, render_template, jsonify, request, session
from datetime import date, datetime

# 设置日志记录
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

bp = Blueprint('inv', __name__)

INV_FILE = f'../data/inv.json'

@bp.route('/')
@login_required
def index():
    user_id = session.get('user_id')
    inv_list = []
    for inv in load_data(INV_FILE):
        if inv['user_id'] == user_id:
            inv_list = inv['inventory']
            break
    
    for i in inv_list:
        exp_str = i.get('expiration_date', None)
        if exp_str:
            i['expiration_date'] = datetime.strptime(exp_str, '%Y-%m-%d').date()
        else:
            i['expiration_date'] = None

    inv_list.sort(key=lambda x: (
        x['expiration_date'] if x['expiration_date'] else date.today(),
        x['quantity'] / x['capacity'] if x['capacity'] else 0
    ))

    return render_template(
        'inv/index.html',
        inv_list=inv_list,
        today=date.today()
    )

@bp.route('/delete/<int:inv_id>', methods=['DELETE'])
@login_required
def delete_inventory(inv_id):
    user_id = session.get('user_id')
    inventory = load_data(INV_FILE)
    idx_1 = None
    for k1, inv in enumerate(inventory):
        if inv['user_id'] == user_id:
            inv_list = inv['inventory']
            idx_1 = k1
            break
    
    idx_2 = None
    for k2, inv in enumerate(inv_list):
        if inv['id'] == inv_id: 
            idx_2 = k2
            break

    if idx_1 is None or idx_2 is None:
        return jsonify({'success': False, 'message': "仓库不存在"}), 404
    
    try: 
        del inventory[idx_1]['inventory'][idx_2]
        save_data(INV_FILE, inventory)
        return jsonify({'success': True, 'message': '仓库删除成功'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@bp.route('/detail/<int:inv_id>')
@login_required
def detail(inv_id):
    user_id = session.get('user_id')
    inv_data = None
    
    for inv in load_data(INV_FILE):
        if inv['user_id'] == user_id:
            for item in inv['inventory']:
                if item['id'] == inv_id:
                    inv_data = item
                    break
            break
    
    if not inv_data:
        return "仓库不存在", 404
        
    exp_str = inv_data.get('expiration_date', None)
    if exp_str:
        inv_data['expiration_date'] = datetime.strptime(exp_str, '%Y-%m-%d').date()
    
    return render_template('inv/detail.html', inv=inv_data)

@bp.route('/stock/<int:inv_id>', methods=['POST'])
@login_required
def update_stock(inv_id):
    data = request.get_json()
    operation = data.get('operation')  # 'in' or 'out'
    quantity = int(data.get('quantity', 0))
    
    if not operation or quantity <= 0:
        return jsonify({'success': False, 'message': '无效的操作参数'}), 400

    user_id = session.get('user_id')
    inventory = load_data(INV_FILE)
    
    for inv in inventory:
        if inv['user_id'] == user_id:
            for item in inv['inventory']:
                if item['id'] == inv_id:
                    if operation == 'out' and item['quantity'] < quantity:
                        return jsonify({'success': False, 'message': '库存不足'}), 400
                    
                    if operation == 'in' and item['quantity'] + quantity > item['capacity']:
                        return jsonify({'success': False, 'message': '库存有限'}), 400

                    if operation == 'in':
                        item['quantity'] += quantity
                    else:
                        item['quantity'] -= quantity
                    
                    # 可以在这里添加记录操作历史的代码
                    
                    save_data(INV_FILE, inventory)
                    return jsonify({
                        'success': True, 
                        'message': '操作成功',
                        'new_quantity': item['quantity']
                    })
    
    return jsonify({'success': False, 'message': '仓库不存在'}), 404

@bp.route('/create_inventory', methods=['POST'])
@login_required
def create_inventory():
    # 获取表单数据
    category = request.form.get('category')
    title = request.form.get('title')
    capacity = request.form.get('capacity')
    expiration = request.form.get('expiration', None)

    # 验证数据
    if not category or not title or not capacity:
        return jsonify({'success': False, 'message': '请填写完整信息'}), 400
    
    try:
        capacity = int(capacity)
    except:
        return jsonify({'success': False, 'message': '仓库上限只能为整数'}), 400

    # 加载库存数据
    inventory = load_data(INV_FILE)
    user_id = session.get('user_id')  # 默认用户ID为1

    # 生成新ID
    for inv in inventory:
        if not inv['user_id'] == user_id: continue
        new_id = max([i['id'] for i in inv['inventory']], default=0) + 1

    # 创建新仓库
    new_inv = {
        "id": new_id,
        "name": title,
        "category": category,
        "capacity": int(capacity),
        "quantity": 0,
        "expiration_date": expiration
    }

    # 添加新仓库
    for inv in inventory:
        if inv['user_id'] == user_id:
            inv['inventory'].append(new_inv)
            break

    # 保存数据
    save_data(INV_FILE, inventory)

    # 返回成功消息
    return jsonify({'success': True, 'message': '仓库创建成功', 'user_id': user_id, 'id': new_id})