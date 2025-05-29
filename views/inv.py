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

                    # 历史
                    history_data = inv_data.get('history', [])
                    history_data.sort(key=lambda x: x['date'])
                    inv_data['chart_labels'] = [h['date'] for h in history_data][:5]
                    inv_data['chart_data'] = [h['quantity'] for h in history_data][:5]

                    # 操作
                    inv_data['operations'] = inv_data['operations'][:5]

                    # 备注
                    inv_data['notes'] = inv_data['notes'][:2]
                    break
            break
    
    if not inv_data:
        return "仓库不存在", 404
        
    exp_str = inv_data.get('expiration_date', None)
    if exp_str:
        inv_data['expiration_date'] = datetime.strptime(exp_str, '%Y-%m-%d').date()
    
    return render_template('inv/detail.html', inv=inv_data)

@bp.route('/note/<int:inv_id>', methods=['POST'])
@login_required
def add_note(inv_id):
    data = request.get_json()
    content = data.get('content')
    
    if not content:
        return jsonify({'success': False, 'message': '备注内容不能为空'}), 400

    user_id = session.get('user_id')
    inventory = load_data(INV_FILE)
    
    for inv in inventory:
        if inv['user_id'] == user_id:
            for item in inv['inventory']:
                if item['id'] == inv_id:
                    # 新备注
                    new_note = {
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                        "content": content
                    }
                    
                    item['notes'].insert(0, new_note)
                    
                    save_data(INV_FILE, inventory)
                    
                    return jsonify({
                        'success': True,
                        'message': '备注添加成功',
                        'note': new_note
                    })
    
    return jsonify({'success': False, 'message': '仓库不存在'}), 404

@bp.route('/stock/<int:inv_id>', methods=['POST'])
@login_required
def update_stock(inv_id):
    data = request.get_json()
    operation = data.get('operation')  # 'in' or 'out'
    quantity = int(data.get('quantity', 0))
    remark = data.get('remark', '')
    
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
                    
                    # 操作记录
                    new_operation = {
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                        "type": operation,
                        "amount": quantity if operation == 'in' else -quantity,
                        "operator": session.get('username'),
                        "remark": remark
                    }
                    item['operations'].insert(0, new_operation)
                    
                    today = date.today().strftime("%Y-%m-%d")
                    if item['history'] and item['history'][0]["date"] == today:
                        item['history'][0]["quantity"] = item['quantity']
                    else:
                        history_entry = {
                            "date": today,
                            "quantity": item['quantity']
                        }
                        item['history'].insert(0, history_entry)
                    
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
    location = request.form.get('location')
    manager = request.form.get('manager')

    if not all([category, title, capacity, location, manager]):
        return jsonify({'success': False, 'message': '请填写完整信息'}), 400
    
    try: capacity = int(capacity)
    except:
        return jsonify({'success': False, 'message': '仓库上限只能为整数'}), 400

    # 加载库存数据
    inventory = load_data(INV_FILE)
    user_id = session.get('user_id')  # 默认用户ID为1

    # 新ID
    for inv in inventory:
        if not inv['user_id'] == user_id: continue
        new_id = max([i['id'] for i in inv['inventory']], default=0) + 1

    # 新仓库
    new_inv = {
        "id": new_id,
        "name": title,
        "category": category,
        "capacity": int(capacity),
        "quantity": 0,
        "expiration_date": expiration,
        "location": location,
        "manager": manager,
        "history": [],
        "operations": [],
        "notes": []
    }

    # 添加新仓库
    for inv in inventory:
        if inv['user_id'] == user_id:
            inv['inventory'].append(new_inv)
            break

    save_data(INV_FILE, inventory)
    return jsonify({'success': True, 'message': '仓库创建成功', 'user_id': user_id, 'id': new_id})