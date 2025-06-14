from flask import Blueprint, render_template, request, redirect, url_for, flash, session, jsonify
import os
import json
import logging
from functools import wraps
from datetime import datetime
from views.data_utils import reset_data_file_path, load_data, save_data

# 设置日志记录
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建一个名为 'auth' 的 Blueprint
bp = Blueprint('auth', __name__)

# 用户数据文件路径 (相对于当前文件)
USERS_FILE = '../data/forum/users.json'
SELECTED = '../data/selected.json'

def get_users_file_path():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_dir, USERS_FILE)

def load_users():
    try:
        full_path = get_users_file_path()

        if os.path.exists(full_path):
            with open(full_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            logger.warning(f"用户数据文件不存在: {full_path}")
            return []
    except Exception as e:
        logger.error(f"加载用户数据时出错: {str(e)}")
        return []

def save_users(users):
    try:
        full_path = get_users_file_path()

        # 确保目录存在
        os.makedirs(os.path.dirname(full_path), exist_ok=True)

        with open(full_path, 'w', encoding='utf-8') as f:
            json.dump(users, f, indent=4, ensure_ascii=False)

        logger.info(f"用户数据已保存到: {full_path}")
        return True
    except Exception as e:
        logger.error(f"保存用户数据时出错: {str(e)}")
        return False

def login_required(view):
    @wraps(view)
    def wrapped_view(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('auth.login'))
        return view(*args, **kwargs)
    return wrapped_view

# 直播电商
def user_required(view):
    @wraps(view)
    def wrapped_view(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('auth.login'))

        if session.get('user_role') != 'user':
            flash('您没有权限访问该页面', 'danger')
            return redirect(url_for('main.index'))

        return view(*args, **kwargs)
    return wrapped_view

# 供应商
def supplier_required(view):
    @wraps(view)
    def wrapped_view(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('auth.login'))

        if session.get('user_role') != 'supplier':
            flash('您没有权限访问该页面', 'danger')
            return redirect(url_for('main.index'))

        return view(*args, **kwargs)
    return wrapped_view

# 管理员
def admin_required(view):
    @wraps(view)
    def wrapped_view(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('auth.login'))

        if session.get('user_role') != 'admin':
            flash('您没有权限访问该页面', 'danger')
            return redirect(url_for('main.index'))

        return view(*args, **kwargs)
    return wrapped_view

@bp.route('/user_center')
@login_required
def user_center():
    user_id = session['user_id']
    users = load_users()
    user = next((u for u in users if u['id'] == user_id), None)
    if not user:
        return render_template('404.html')
    
    # 转换注册时间为更友好的格式
    join_date = datetime.strptime(user['join_date'], '%Y-%m-%d').strftime('%Y年%m月%d日')
    
    return render_template('auth/user_center.html',
                           user_id=user['id'],
                           username=user['username'],
                           join_date=join_date,
                           avatar=user['avatar'])

@bp.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')  
        confirm_password = request.form.get('confirm_password')
        role = request.form.get('role', 'user')  

        # 验证输入
        if not username or not password:
            flash('用户名和密码不能为空', 'danger')
            return render_template('auth/register.html')

        if password != confirm_password:
            flash('两次输入的密码不一致', 'danger')
            return render_template('auth/register.html')

        # 检查用户名是否已存在
        users = load_users()
        if any(u['username'] == username for u in users):
            flash('用户名已存在，请选择其他用户名', 'danger')
            return render_template('auth/register.html')

        new_id = max([u['id'] for u in users], default=0) + 1

        # 创建新用户
        new_user = {
            "id": new_id,
            "username": username,
            "password": password,
            "avatar": "https://via.placeholder.com/40",
            "role": role,
            "posts": 0,
            "replies": 0,
            "join_date": datetime.now().strftime('%Y-%m-%d'),
            "last_active": datetime.now().strftime('%Y-%m-%d')
        }

        selected = load_data(SELECTED)
        selected.append({
            "user_id": new_id,
            "selected": {}
        })

        # 添加新用户并保存
        users.append(new_user)
        if save_users(users):
            flash('注册成功，请登录', 'success')
            return redirect(url_for('auth.login'))
        else:
            flash('注册失败，请稍后再试', 'danger')

    return render_template('auth/register.html')

@bp.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')  
        role = request.form.get('role')  

        # 加载用户数据
        users = load_users()

        # 根据角色筛选用户
        if role:
            filtered_users = [u for u in users if u['role'] == role]
        else:
            filtered_users = users

        # 查找用户
        user = next((u for u in filtered_users if u['username'] == username), None)

        if user and user['password'] == password:
            # 登录成功，保存用户信息到会话
            session['user_id'] = user['id']
            session['username'] = user['username']
            session['user_role'] = user['role']

            # 重置数据文件路径为默认值
            reset_data_file_path()

            flash(f'欢迎回来，{username}！', 'success')
            if role == 'supplier':
                return redirect(url_for('forum.index'))
            elif role == 'user':
                return redirect(url_for('main.index')) 
            else:
                return redirect(url_for('auth.manage_users'))
        else:
            if role:
                flash(f'未找到{role}角色的用户"{username}"或密码错误', 'danger')
            else:
                flash('用户名或密码错误', 'danger')

    return render_template('auth/login.html')

@bp.route('/logout')
@login_required
def logout():
    session.clear()
    flash('您已成功登出', 'success')
    return redirect(url_for('main.index'))

@bp.route('/api/check-auth')
def check_auth():
    if 'user_id' in session:
        return jsonify({
            'authenticated': True,
            'user': {
                'id': session['user_id'],
                'username': session['username'],
                'role': session['user_role']
            }
        })
    else:
        return jsonify({'authenticated': False})

# 用户管理功能
@bp.route('/manage-users')
@admin_required
def manage_users():
    users = load_users()
    return render_template('auth/manage_users.html', users=users)

@bp.route('/add-user', methods=['GET', 'POST'])
@admin_required
def add_user():
    if request.method == 'POST':
        username = request.form.get('username')
        role = request.form.get('role', 'user')
        passwd = request.form.get('passwd')
        
        # 验证输入
        if not username:
            flash('用户名不能为空', 'danger')
            return redirect(url_for('auth.add_user'))
        if not passwd:
            flash('密码不能为空', 'danger')
            return redirect(url_for('auth.add_user'))
            
        # 检查用户名是否已存在
        users = load_users()
        if any(u['username'] == username for u in users):
            flash('用户名已存在，请选择其他用户名', 'danger')
            return redirect(url_for('auth.add_user'))
            
        # 创建新用户
        new_user = {
            "id": max([u['id'] for u in users], default=0) + 1,
            "username": username,
            "password": passwd,
            "avatar": "https://via.placeholder.com/40",
            "role": role,
            "posts": 0,
            "replies": 0,
            "join_date": datetime.now().strftime('%Y-%m-%d'),
            "last_active": datetime.now().strftime('%Y-%m-%d')
        }
        
        # 添加新用户并保存
        users.append(new_user)
        if save_users(users):
            flash('添加用户成功', 'success')
            return redirect(url_for('auth.manage_users'))
        else:
            flash('添加用户失败，请稍后再试', 'danger')
            
    return render_template('auth/add_user.html')

@bp.route('/edit-user/<int:user_id>', methods=['GET', 'POST'])
@admin_required
def edit_user(user_id):
    users = load_users()
    user = next((u for u in users if u['id'] == user_id), None)
    
    if not user:
        flash('用户不存在', 'danger')
        return redirect(url_for('auth.manage_users'))
        
    if request.method == 'POST':
        username = request.form.get('username')
        role = request.form.get('role')
        passwd = request.form.get('passwd')
        
        # 验证输入
        if not username:
            flash('用户名不能为空', 'danger')
            return render_template('auth/edit_user.html', user=user)

        if not passwd:
            flash('密码不能为空', 'danger')
            return render_template('auth/edit_user.html', user=user)
            
        # 检查用户名是否已被其他用户使用
        if username != user['username'] and any(u['username'] == username for u in users if u['id'] != user_id):
            flash('用户名已存在，请选择其他用户名', 'danger')
            return render_template('auth/edit_user.html', user=user)
            
        # 更新用户信息
        user['username'] = username
        user['role'] = role
        user['password'] = passwd
        user['last_active'] = datetime.now().strftime('%Y-%m-%d')
        
        if save_users(users):
            flash('更新用户成功', 'success')
            return redirect(url_for('auth.manage_users'))
        else:
            flash('更新用户失败，请稍后再试', 'danger')
            
    return render_template('auth/edit_user.html', user=user)

@bp.route('/delete-user/<int:user_id>', methods=['POST'])
@admin_required
def delete_user(user_id):
    users = load_users()
    
    # 查找要删除的用户
    user_index = next((i for i, u in enumerate(users) if u['id'] == user_id), None)
    
    if user_index is None:
        flash('用户不存在', 'danger')
        return redirect(url_for('auth.manage_users'))
        
    # 不能删除自己
    if users[user_index]['id'] == session.get('user_id'):
        flash('不能删除当前登录的用户', 'danger')
        return redirect(url_for('auth.manage_users'))
        
    # 删除用户
    del users[user_index]
    
    if save_users(users):
        flash('删除用户成功', 'success')
    else:
        flash('删除用户失败，请稍后再试', 'danger')
        
    return redirect(url_for('auth.manage_users'))

# 在auth.py中添加以下路由

@bp.route('/change-password', methods=['GET', 'POST'])
@login_required
def change_password():
    if request.method == 'POST':
        current_password = request.form.get('current_password')
        new_password = request.form.get('new_password')
        confirm_password = request.form.get('confirm_password')
        
        # 验证输入
        if not current_password or not new_password or not confirm_password:
            flash('请填写所有字段', 'danger')
            return render_template('auth/change_key.html')
        
        if new_password != confirm_password:
            flash('两次输入的新密码不一致', 'danger')
            return render_template('auth/change_key.html')
        
        if len(new_password) < 8:
            flash('密码长度至少需要8位', 'danger')
            return render_template('auth/change_key.html')
        
        # 加载用户数据
        users = load_users()
        user_id = session['user_id']
        user = next((u for u in users if u['id'] == user_id), None)
        
        if not user:
            flash('用户不存在', 'danger')
            return redirect(url_for('auth.user_center'))
        
        # 验证当前密码
        if user['password'] != current_password:
            flash('当前密码不正确', 'danger')
            return render_template('auth/change_key.html')
        
        # 更新密码
        user['password'] = new_password
        
        # 保存用户数据
        if save_users(users):
            flash('密码修改成功，请重新登录', 'success')
            # 清除会话并重定向到登录页
            session.clear()
            return redirect(url_for('auth.login'))
        else:
            flash('密码修改失败，请稍后再试', 'danger')
    
    return render_template('auth/change_key.html')

@bp.route('/api/users')
@admin_required
def api_users():
    users = load_users()
    return jsonify(users)
