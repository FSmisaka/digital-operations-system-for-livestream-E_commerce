import os
import logging
import glob
import json

# 设置日志记录
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 默认数据文件路径
DEFAULT_DATA_FILE_PATH = '../model_data/date1.csv'

# 备用数据文件路径列表
BACKUP_DATA_PATHS = [
    '../model_data/date1.csv',
    'model_data/date1.csv',
    'date1.csv',
    'data.csv',
    'data/data.csv',
    '../data/data.csv'
]

# 当前使用的数据文件路径
current_data_file_path = DEFAULT_DATA_FILE_PATH

def reset_data_file_path():
    global current_data_file_path
    current_data_file_path = DEFAULT_DATA_FILE_PATH
    logger.info(f"已重置数据文件路径为默认值: {current_data_file_path}")
    return current_data_file_path

def set_data_file_path(path):
    global current_data_file_path
    current_data_file_path = path
    logger.info(f"已设置数据文件路径为: {current_data_file_path}")
    return current_data_file_path

def get_data_file_path():
    return current_data_file_path

def get_full_data_path():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(current_dir, current_data_file_path)
    if os.path.exists(full_path):
        logger.info(f"使用当前设置的数据文件路径: {full_path}")
        return full_path
    logger.warning(f"当前设置的数据文件路径不存在: {full_path}，尝试备用路径")

    # 尝试相对于当前目录的备用路径
    for backup_path in BACKUP_DATA_PATHS:
        temp_path = os.path.join(current_dir, backup_path)
        if os.path.exists(temp_path):
            logger.info(f"使用备用数据文件路径: {temp_path}")
            # 更新当前路径
            set_data_file_path(backup_path)
            return temp_path

    # 尝试在项目根目录中查找
    base_dir = os.path.dirname(os.path.dirname(current_dir))
    for backup_path in BACKUP_DATA_PATHS:
        temp_path = os.path.join(base_dir, backup_path)
        if os.path.exists(temp_path):
            logger.info(f"使用项目根目录下的备用数据文件路径: {temp_path}")
            # 更新当前路径
            set_data_file_path(os.path.relpath(temp_path, current_dir))
            return temp_path

    # 尝试查找任何CSV文件
    logger.warning("无法找到预定义的数据文件，尝试查找任何CSV文件")

    # 在当前目录及其子目录中查找CSV文件
    csv_files = glob.glob(os.path.join(current_dir, "**/*.csv"), recursive=True)
    if csv_files:
        csv_path = csv_files[0]
        logger.info(f"使用找到的CSV文件: {csv_path}")
        set_data_file_path(os.path.relpath(csv_path, current_dir))
        return csv_path

    # 在项目根目录及其子目录中查找CSV文件
    csv_files = glob.glob(os.path.join(base_dir, "**/*.csv"), recursive=True)
    if csv_files:
        csv_path = csv_files[0]
        logger.info(f"使用项目根目录下找到的CSV文件: {csv_path}")
        set_data_file_path(os.path.relpath(csv_path, current_dir))
        return csv_path

    # 如果仍然找不到，返回原始路径
    logger.error(f"无法找到任何数据文件，返回原始路径: {full_path}")
    return full_path

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
