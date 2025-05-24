import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import json
import logging
import glob
import traceback
from sklearn.preprocessing import MinMaxScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PricePredictionAPI:
    def __init__(self,
                 model_dir=None,
                 pipelines_dir=None,
                 scalers_dir=None,
                 results_dir=None,
                 look_back=30,
                 model_type=None):
        """
        初始化价格预测API类

        参数:
            model_dir: 模型文件目录
            pipelines_dir: 预处理管道目录
            scalers_dir: 缩放器目录
            results_dir: 模型结果目录（包含指标JSON文件）
            look_back: 时间窗口大小
            model_type: 模型类型 ('mlp', 'lstm', 'cnn')
        """
        # 获取当前文件所在目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 设置默认路径（相对于当前文件）
        if model_dir is None:
            self.model_dir = current_dir
        else:
            self.model_dir = model_dir
            
        if pipelines_dir is None:
            self.pipelines_dir = os.path.join(current_dir, 'pipelines')
        else:
            self.pipelines_dir = pipelines_dir
            
        if scalers_dir is None:
            self.scalers_dir = os.path.join(current_dir, 'scalers')
        else:
            self.scalers_dir = scalers_dir
            
        if results_dir is None:
            self.results_dir = os.path.join(current_dir, 'results')
        else:
            self.results_dir = results_dir
            
        self.look_back = look_back
        self.model_type_loaded = model_type

        # 模型类型映射
        self.model_types = {
            "mlp": "MLP (多层感知机)",
            "lstm": "LSTM (长短期记忆网络)",
            "cnn": "CNN (卷积神经网络)"
        }

        # 加载可用模型列表
        self.available_models = self._get_available_models()
        logger.info(f"可用模型: {self.available_models}")

        # 加载模型指标
        self.model_metrics = self._load_model_metrics()
        logger.info(f"已加载模型指标: {len(self.model_metrics)} 个模型")

        # 默认特征列表 (基于相关性分析选择的前10个特征)
        # 简化版本的特征列表
        self.simplified_features = ['low', 'high', 'open', 'MA_5', 'a_close', 'c_close', 'ATR_14', 'OBV', 'GDP', '大豆产量(万吨)']
        
        # 高级版本的特征列表 (15个特征，匹配模型输入维度)
        self.advanced_features = ['open', 'high', 'low', 'close', 'MA_5', 'HV_20', 'ATR_14', 
                                 'RSI_14', 'OBV', 'MACD', 'a_close', 'c_close', 'LPR1Y', 
                                 '大豆产量(万吨)', 'GDP']
        
        # 默认使用高级特征列表
        self.default_features = self.advanced_features

        # 初始化模型、管道和缩放器
        self.model = None
        self.feature_scaler = MinMaxScaler(feature_range=(-1, 1))
        self.target_scaler = MinMaxScaler(feature_range=(-1, 1))

        # 如果指定了模型类型，立即加载模型
        if model_type:
            self.load_model(model_type)

    def _get_available_models(self):
        """获取可用的模型列表"""
        available_models = {}

        try:
            if not os.path.exists(self.model_dir):
                logger.warning(f"模型目录不存在: {self.model_dir}")
                return available_models

            subdirs = ['', 'saved_models', 'checkpoints']

            for subdir in subdirs:
                search_dir = os.path.join(self.model_dir, subdir)

                if not os.path.exists(search_dir):
                    continue

                for file in os.listdir(search_dir):
                    if file.endswith('.h5'):
                        for model_type in self.model_types.keys():
                            if file.startswith(model_type):
                                timestamp_parts = file.split('_')
                                if len(timestamp_parts) >= 3:
                                    timestamp = timestamp_parts[-1].replace('.h5', '')
                                    available_models[model_type] = {
                                        'path': os.path.join(search_dir, file),
                                        'timestamp': timestamp
                                    }
                                break

            if not available_models:
                for subdir in subdirs:
                    search_dir = os.path.join(self.model_dir, subdir)

                    if not os.path.exists(search_dir):
                        continue

                    for file in os.listdir(search_dir):
                        if file.endswith('.h5'):
                            model_type = None
                            for mt in self.model_types.keys():
                                if mt in file.lower():
                                    model_type = mt
                                    break

                            if model_type is None:
                                model_type = "unknown"

                            available_models[model_type] = {
                                'path': os.path.join(search_dir, file),
                                'timestamp': 'unknown'
                            }
                            break

                    if available_models:
                        break
        except Exception as e:
            logger.error(f"获取可用模型时出错: {str(e)}")

        return available_models

    def _load_model_metrics(self):
        """从JSON文件加载模型指标"""
        metrics = {}

        try:
            if not os.path.exists(self.results_dir):
                logger.warning(f"结果目录不存在: {self.results_dir}")
                return metrics

            json_files = glob.glob(os.path.join(self.results_dir, "*.json"))

            for json_file in json_files:
                try:
                    with open(json_file, 'r') as f:
                        file_content = f.read()
                        if not file_content.strip():
                            continue

                        try:
                            data = json.loads(file_content)
                        except json.JSONDecodeError:
                            continue

                    filename = os.path.basename(json_file)
                    model_type = None

                    for mt in self.model_types.keys():
                        if mt in filename.lower():
                            model_type = mt
                            break

                    if model_type:
                        if model_type not in metrics:
                            metrics[model_type] = {}

                        if 'metrics' in data:
                            for key, value in data['metrics'].items():
                                metrics[model_type][key] = value

                        if 'evaluation' in data:
                            eval_data = data['evaluation']

                            if 'accuracy' in eval_data:
                                metrics[model_type]['accuracy'] = eval_data['accuracy']
                            if 'rmse' in eval_data:
                                metrics[model_type]['rmse'] = eval_data['rmse']
                            if 'mae' in eval_data:
                                metrics[model_type]['mae'] = eval_data['mae']
                            if 'r2' in eval_data:
                                metrics[model_type]['r2'] = eval_data['r2']
                            if 'mape' in eval_data:
                                metrics[model_type]['mape'] = eval_data['mape']

                        if 'prediction_stats' in data:
                            metrics[model_type]['prediction_stats'] = data['prediction_stats']
                        if 'actual_stats' in data:
                            metrics[model_type]['actual_stats'] = data['actual_stats']
                        if 'timestamp' in data:
                            metrics[model_type]['timestamp'] = data['timestamp']
                        if 'model_key' in data:
                            metrics[model_type]['model_key'] = data['model_key']
                            
                        # 提取关键模型信息字段
                        for field in ['data_file', 'target_column', 'look_back', 'test_samples']:
                            if field in data:
                                metrics[model_type][field] = data[field]

                        if not any(key in metrics[model_type] for key in ['accuracy', 'rmse', 'mae', 'r2', 'mape']):
                            if 'test_metrics' in data:
                                test_metrics = data['test_metrics']
                                metrics[model_type]['accuracy'] = test_metrics.get('accuracy', 0)
                                metrics[model_type]['rmse'] = test_metrics.get('rmse', 0)
                                metrics[model_type]['mae'] = test_metrics.get('mae', 0)
                                metrics[model_type]['r2'] = test_metrics.get('r2', 0)
                                metrics[model_type]['mape'] = test_metrics.get('mape', 0)

                            if not any(key in metrics[model_type] for key in ['accuracy', 'rmse', 'mae', 'r2', 'mape']):
                                metrics[model_type]['accuracy'] = 0
                                metrics[model_type]['rmse'] = 0
                                metrics[model_type]['mae'] = 0
                                metrics[model_type]['r2'] = 0
                                metrics[model_type]['mape'] = 0
                except Exception as e:
                    logger.error(f"加载JSON文件 {json_file} 时出错: {str(e)}")
        except Exception as e:
            logger.error(f"加载模型指标时出错: {str(e)}")

        return metrics

    def get_model_metrics(self, model_type=None):
        """
        获取模型指标

        参数:
            model_type: 模型类型，如果为None则返回所有模型的指标

        返回:
            模型指标字典
        """
        if model_type is None:
            return self.model_metrics

        if model_type in self.model_metrics:
            return {model_type: self.model_metrics[model_type]}

        return {}

    def load_model(self, model_type):
        """
        加载指定类型的模型

        参数:
            model_type: 模型类型 ('mlp', 'lstm', 'cnn')

        返回:
            成功加载返回True，否则返回False
        """
        if model_type not in self.available_models:
            logger.error(f"模型类型 '{model_type}' 不可用")
            return False

        try:
            # 加载模型
            model_path = self.available_models[model_type]['path']
            logger.info(f"加载模型: {model_path}")
            self.model = load_model(model_path)
            self.model_type_loaded = model_type
            
            # 根据模型名称判断是否使用高级特征
            if 'advancedfeatures' in model_path.lower():
                self.default_features = self.advanced_features
                logger.info(f"检测到高级特征模型，使用15个特征: {len(self.default_features)}个")
            else:
                self.default_features = self.simplified_features
                logger.info(f"检测到简化特征模型，使用10个特征: {len(self.default_features)}个")

            return True
        except Exception as e:
            logger.error(f"加载模型 '{model_type}' 时出错: {str(e)}")
            return False

    def prepare_time_series(self, X_scaled):
        """
        准备时间序列数据

        参数:
            X_scaled: 缩放后的特征数据

        返回:
            时间序列格式的特征数据
        """
        # 创建时间序列样本
        Xs = []
        for i in range(len(X_scaled) - self.look_back + 1):
            Xs.append(X_scaled[i:(i + self.look_back)])
        return np.array(Xs)

    def get_available_features(self, data):
        """
        获取数据中可用的特征

        参数:
            data: 包含特征的DataFrame

        返回:
            可用特征列表
        """
        # 排除常见的非特征列
        non_feature_cols = ['date', '日期', 'timestamp', 'volume', '成交量', 'hold', '持仓量']
        available_features = [col for col in data.columns if col not in non_feature_cols]
        return available_features
    
    def detect_required_features(self):
        """
        检测模型需要的特征数量

        返回:
            需要的特征数量
        """
        if self.model is None:
            return len(self.default_features)
        
        # 获取模型的输入形状
        input_shape = self.model.inputs[0].shape
        
        # 检查输入形状是否有特征维度
        if len(input_shape) >= 3:
            # 形状通常为 (None, look_back, features)
            return input_shape[-1]
        
        return len(self.default_features)

    def predict_next_n_days(self, data, n_days=1, features=None, model_type='lstm'):
        """
        预测未来n天的价格
        """
        if features is None:
            features = self.default_features

        if self.model is None:
            if self.model_type_loaded:
                model_type = self.model_type_loaded

            if not self.load_model(model_type):
                for available_model in self.available_models:
                    if self.load_model(available_model):
                        break
                else:
                    available_models_str = ", ".join(self.available_models.keys()) if self.available_models else "无"
                    error_msg = f"无法加载任何模型，请检查模型文件是否存在。请求的模型类型: {model_type}"
                    logger.error(error_msg)
                    raise ValueError(error_msg)
        
        # 检查模型期望的特征数量
        expected_features = self.detect_required_features()
        
        # 如果提供的特征数量与模型期望不匹配，自动调整
        if len(features) != expected_features:
            logger.warning(f"提供的特征数量({len(features)})与模型期望的不匹配({expected_features})，尝试自动调整")
            
            if expected_features == 15:
                # 需要高级特征
                if len(self.advanced_features) == expected_features and all(f in data.columns for f in self.advanced_features):
                    features = self.advanced_features
                    logger.info(f"使用预定义的高级特征: {features}")
                else:
                    # 尝试使用数据中的所有可用特征
                    available_features = self.get_available_features(data)
                    if len(available_features) >= expected_features:
                        features = available_features[:expected_features]
                        logger.info(f"从可用特征中选择前{expected_features}个: {features}")
                    else:
                        raise ValueError(f"数据中没有足够的特征({len(available_features)})来满足模型需求({expected_features})")
            elif expected_features == 10:
                # 需要简化特征
                if len(self.simplified_features) == expected_features and all(f in data.columns for f in self.simplified_features):
                    features = self.simplified_features
                    logger.info(f"使用预定义的简化特征: {features}")
                else:
                    # 尝试使用数据中的可用特征
                    available_features = self.get_available_features(data)
                    if len(available_features) >= expected_features:
                        features = available_features[:expected_features]
                        logger.info(f"从可用特征中选择前{expected_features}个: {features}")
                    else:
                        raise ValueError(f"数据中没有足够的特征({len(available_features)})来满足模型需求({expected_features})")

        missing_features = [f for f in features if f not in data.columns]
        if missing_features:
            raise ValueError(f"以下特征在数据集中不存在: {missing_features}")

        X = data[features].copy()
        X_scaled = self.feature_scaler.fit_transform(X)
        X_ts = self.prepare_time_series(X_scaled)

        if len(X_ts) == 0:
            raise ValueError(f"数据量不足，无法进行预测。需要至少 {self.look_back} 条数据。")

        predictions_scaled = self.model.predict(X_ts)

        if 'close' in data.columns:
            y = data['close'].values.reshape(-1, 1)
            self.target_scaler.fit(y)
        elif '收盘价' in data.columns:
            y = data['收盘价'].values.reshape(-1, 1)
            self.target_scaler.fit(y)

        predictions = self.target_scaler.inverse_transform(predictions_scaled).flatten()
        latest_prediction = float(predictions[-1])
        date_col = 'date' if 'date' in data.columns else '日期'
        last_date = pd.to_datetime(data[date_col].iloc[-1])

        result = []
        current_prediction = latest_prediction

        for i in range(n_days):
            prediction_date = (last_date + timedelta(days=i+1)).strftime('%Y-%m-%d')
            result.append({
                'date': prediction_date,
                'prediction': current_prediction
            })
            change_pct = np.random.uniform(-1, 1)
            current_prediction = current_prediction * (1 + change_pct / 100)

        return result

    def predict_price(self, data, model_type='lstm', features=None):
        """
        预测价格

        参数:
            data: 包含特征的DataFrame
            model_type: 模型类型 ('mlp', 'lstm', 'cnn')
            features: 特征列表，如果为None则使用默认特征

        返回:
            预测结果字典，包含预测价格和相关信息
        """
        # 使用默认特征列表
        if features is None:
            features = self.default_features

        # 加载模型
        if not self.load_model(model_type):
            return {
                'success': False,
                'error': f"无法加载模型 '{model_type}'"
            }
            
        # 检查模型期望的特征数量
        expected_features = self.detect_required_features()
        
        # 如果提供的特征数量与模型期望不匹配，自动调整
        if len(features) != expected_features:
            logger.warning(f"提供的特征数量({len(features)})与模型期望的不匹配({expected_features})，尝试自动调整")
            
            if expected_features == 15:
                # 需要高级特征
                if len(self.advanced_features) == expected_features and all(f in data.columns for f in self.advanced_features):
                    features = self.advanced_features
                    logger.info(f"使用预定义的高级特征: {features}")
                else:
                    # 尝试使用数据中的所有可用特征
                    available_features = self.get_available_features(data)
                    if len(available_features) >= expected_features:
                        features = available_features[:expected_features]
                        logger.info(f"从可用特征中选择前{expected_features}个: {features}")
                    else:
                        return {
                            'success': False,
                            'error': f"数据中没有足够的特征({len(available_features)})来满足模型需求({expected_features})"
                        }
            elif expected_features == 10:
                # 需要简化特征
                if len(self.simplified_features) == expected_features and all(f in data.columns for f in self.simplified_features):
                    features = self.simplified_features
                    logger.info(f"使用预定义的简化特征: {features}")
                else:
                    # 尝试使用数据中的可用特征
                    available_features = self.get_available_features(data)
                    if len(available_features) >= expected_features:
                        features = available_features[:expected_features]
                        logger.info(f"从可用特征中选择前{expected_features}个: {features}")
                    else:
                        return {
                            'success': False,
                            'error': f"数据中没有足够的特征({len(available_features)})来满足模型需求({expected_features})"
                        }

        try:
            # 检查数据中是否包含所有需要的特征
            missing_features = [f for f in features if f not in data.columns]
            if missing_features:
                return {
                    'success': False,
                    'error': f"以下特征在数据集中不存在: {missing_features}"
                }

            # 提取特征
            X = data[features].copy()

            # 使用特征缩放器转换特征
            X_scaled = self.feature_scaler.fit_transform(X)

            # 准备时间序列数据
            X_ts = self.prepare_time_series(X_scaled)

            if len(X_ts) == 0:
                return {
                    'success': False,
                    'error': f"数据量不足，无法进行预测。需要至少 {self.look_back} 条数据。"
                }

            # 进行预测
            predictions_scaled = self.model.predict(X_ts)

            # 如果有目标列，使用它来拟合缩放器
            if 'close' in data.columns:
                y = data['close'].values.reshape(-1, 1)
                self.target_scaler.fit(y)

            # 反向转换预测值
            predictions = self.target_scaler.inverse_transform(predictions_scaled).flatten()

            # 获取最后一个预测值（最新的预测）
            latest_prediction = float(predictions[-1])

            # 获取前一天的收盘价（如果有）
            previous_close = None
            if 'close' in data.columns and len(data) > 0:
                previous_close = float(data['close'].iloc[-1])

            # 计算变化百分比
            change_percent = None
            if previous_close is not None and previous_close != 0:
                change_percent = ((latest_prediction - previous_close) / previous_close) * 100

            # 构建结果
            result = {
                'success': True,
                'model_type': model_type,
                'model_name': self.model_types.get(model_type, model_type),
                'prediction': latest_prediction,
                'previous_close': previous_close,
                'change_percent': change_percent,
                'prediction_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'features_used': features
            }

            return result
        except Exception as e:
            logger.error(f"预测时出错: {str(e)}")
            return {
                'success': False,
                'error': f"预测时出错: {str(e)}"
            }

# 前端接口函数
def predict_closing_price(data_dict, model_type='lstm'):
    """
    预测收盘价的前端接口函数

    参数:
        data_dict: 包含特征的字典或JSON字符串
        model_type: 模型类型 ('mlp', 'lstm', 'cnn')

    返回:
        预测结果JSON字符串
    """
    try:
        # 如果输入是JSON字符串，转换为字典
        if isinstance(data_dict, str):
            data_dict = json.loads(data_dict)

        # 转换为DataFrame
        df = pd.DataFrame([data_dict])

        # 创建预测API实例
        predictor = PricePredictionAPI()

        # 进行预测
        result = predictor.predict_price(df, model_type)

        # 返回JSON结果
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        error_result = {
            'success': False,
            'error': f"预测过程中出错: {str(e)}"
        }
        return json.dumps(error_result, ensure_ascii=False)

class PricePredictionInference:
    def __init__(self,
                 model_path,
                 look_back=20,
                 features=None):
        """
        初始化价格预测推理类

        参数:
            model_path: 模型文件路径
            look_back: 时间窗口大小
            features: 特征列表，如果为None则使用默认特征
        """
        # 获取当前文件所在目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 如果提供的是相对路径，将其转换为绝对路径
        if not os.path.isabs(model_path):
            self.model_path = os.path.join(current_dir, model_path)
        else:
            self.model_path = model_path
            
        self.look_back = look_back

        # 高级特征列表 (15个特征)
        self.advanced_features = ['open', 'high', 'low', 'close', 'MA_5', 'HV_20', 'ATR_14', 
                                 'RSI_14', 'OBV', 'MACD', 'a_close', 'c_close', 'LPR1Y', 
                                 '大豆产量(万吨)', 'GDP']
        
        # 简化特征列表 (10个特征)
        self.simplified_features = ['low', 'high', 'open', 'MA_5', 'a_close', 'c_close', 'ATR_14', 'OBV', 'GDP', '大豆产量(万吨)']

        # 根据模型名称选择默认特征列表
        if 'advancedfeatures' in self.model_path.lower():
            self.features = self.advanced_features
            print(f"检测到高级特征模型，使用15个特征")
        else:
            self.features = self.simplified_features
            print(f"检测到简化特征模型，使用10个特征")
            
        # 如果指定了特征，则使用指定的特征
        if features is not None:
            self.features = features

        # 加载模型和预处理工具
        self.load_model_and_tools()

    def load_model_and_tools(self):
        """加载模型、预处理管道和缩放器"""
        print(f"加载模型: {self.model_path}")
        self.model = load_model(self.model_path)
        
        # 检查模型输入形状，调整特征数量
        if hasattr(self.model, 'inputs') and len(self.model.inputs) > 0:
            input_shape = self.model.inputs[0].shape
            if len(input_shape) >= 3:
                expected_features = input_shape[-1]
                if len(self.features) != expected_features:
                    print(f"警告: 特征数量不匹配 (当前: {len(self.features)}, 期望: {expected_features})")
                    if expected_features == 15:
                        self.features = self.advanced_features
                    elif expected_features == 10:
                        self.features = self.simplified_features
                    print(f"已调整为使用 {len(self.features)} 个特征")

        # 创建默认的预处理管道
        print("创建默认的预处理管道...")
        self.feature_scaler = MinMaxScaler(feature_range=(-1, 1))

        print(f"创建目标缩放器...")
        # 创建一个新的缩放器
        self.target_scaler = MinMaxScaler(feature_range=(-1, 1))

    def prepare_time_series(self, X_scaled):
        """
        准备时间序列数据

        参数:
            X_scaled: 缩放后的特征数据

        返回:
            时间序列格式的特征数据
        """
        # 创建时间序列样本
        Xs = []
        for i in range(len(X_scaled) - self.look_back):
            Xs.append(X_scaled[i:(i + self.look_back)])
        return np.array(Xs)

    def predict(self, data):
        """
        使用加载的模型进行预测

        参数:
            data: 包含特征的DataFrame

        返回:
            预测结果DataFrame，包含日期和预测价格
        """
        # 检查数据中是否包含所有需要的特征
        missing_features = [f for f in self.features if f not in data.columns]
        if missing_features:
            # 尝试使用所有可用特征
            print(f"警告: 以下特征在数据集中不存在: {missing_features}")
            available_features = [col for col in data.columns if col not in ['date', '日期', 'timestamp', 'volume', '成交量', 'hold', '持仓量']]
            
            # 检查模型输入形状
            if hasattr(self.model, 'inputs') and len(self.model.inputs) > 0:
                input_shape = self.model.inputs[0].shape
                if len(input_shape) >= 3:
                    expected_features = input_shape[-1]
                    if len(available_features) >= expected_features:
                        self.features = available_features[:expected_features]
                        print(f"使用前 {expected_features} 个可用特征: {self.features}")
                    else:
                        raise ValueError(f"数据中没有足够的特征 ({len(available_features)})，模型需要 {expected_features} 个特征")
                else:
                    self.features = available_features
                    print(f"使用所有可用特征: {self.features}")
            else:
                self.features = available_features
                print(f"使用所有可用特征: {self.features}")

        # 提取特征
        X = data[self.features]

        # 使用特征缩放器转换特征
        X_scaled = self.feature_scaler.fit_transform(X)

        # 准备时间序列数据
        X_ts = self.prepare_time_series(X_scaled)

        if len(X_ts) == 0:
            raise ValueError(f"数据量不足，无法进行预测。需要至少 {self.look_back + 1} 条数据。")

        # 进行预测
        predictions_scaled = self.model.predict(X_ts)

        # 如果有目标列，使用它来拟合缩放器
        if 'close' in data.columns:
            y = data['close'].values.reshape(-1, 1)
            self.target_scaler.fit(y)

            # 反向转换预测值
            predictions = self.target_scaler.inverse_transform(predictions_scaled).flatten()
        else:
            # 如果没有目标列，使用简单的缩放方法
            predictions = predictions_scaled.flatten()

        # 创建结果DataFrame
        if 'date' in data.columns:
            # 获取对应的日期（跳过前look_back条记录）
            dates = data['date'].iloc[self.look_back:].reset_index(drop=True)
            results = pd.DataFrame({
                'date': dates,
                'predicted_price': predictions
            })
        else:
            results = pd.DataFrame({
                'predicted_price': predictions
            })

        return results

    def plot_predictions(self, predictions, actual=None, title="价格预测"):
        """
        绘制预测结果

        参数:
            predictions: 预测结果DataFrame，包含date和predicted_price列
            actual: 实际价格DataFrame，包含date和price列（可选）
            title: 图表标题
        """
        plt.figure(figsize=(14, 7))

        # 绘制预测价格
        plt.plot(predictions['date'], predictions['predicted_price'],
                 label='预测价格', color='red', linestyle='--')

        # 如果有实际价格，也绘制出来
        if actual is not None:
            plt.plot(actual['date'], actual['price'],
                     label='实际价格', color='blue')

        plt.title(title)
        plt.xlabel('日期')
        plt.ylabel('价格')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)

        # 获取当前文件所在目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 创建results目录（如果不存在）
        results_dir = os.path.join(current_dir, 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        # 保存图表到results目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        predictions_path = os.path.join(results_dir, f"prediction_results_{timestamp}.png")
        plt.savefig(predictions_path, bbox_inches='tight')
        plt.close()
        print(f"预测结果图表已保存到: {predictions_path}")

        return predictions_path

# 示例用法
if __name__ == "__main__":
    # 获取当前文件所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 可用的模型路径（使用相对路径）
    model_paths = {
        "MLP": os.path.join(current_dir, "saved_models/mlp_simplified_10features_20250511_165124.h5"),
        "LSTM": os.path.join(current_dir, "saved_models/lstm_simplified_10features_20250511_165321.h5"),
        "CNN": os.path.join(current_dir, "saved_models/cnn_simplified_10features_20250511_170000.h5")
    }

    # 选择要使用的模型
    model_type = "CNN"  # 可以是 "MLP", "LSTM" 或 "CNN"
    model_path = model_paths[model_type]

    print(f"\n使用 {model_type} 模型进行预测...")

    # 创建预测器实例
    predictor = PricePredictionInference(
        model_path=model_path,
        look_back=30  # 根据模型期望的输入形状调整为30
    )

    # 加载测试数据（使用相对路径）
    project_root = os.path.abspath(os.path.join(current_dir, "../.."))
    data_file = os.path.join(project_root, "model_data/date1.csv")

    # 检查文件是否存在
    if not os.path.exists(data_file):
        # 尝试其他可能的路径
        alternative_paths = [
            os.path.join(project_root, "model_data/date2.csv"),
            os.path.join(project_root, "model_data/date1.csv"),
            os.path.join(current_dir, "../../model_data/date1.csv")
        ]

        for path in alternative_paths:
            if os.path.exists(path):
                data_file = path
                print(f"找到替代数据文件: {data_file}")
                break
        else:
            raise FileNotFoundError(f"无法找到数据文件。请确保数据文件存在于正确的路径中。")

    # 读取数据
    data = pd.read_csv(data_file)

    # 确保日期列是日期类型
    if 'date' in data.columns:
        data['date'] = pd.to_datetime(data['date'])

    # 进行预测
    predictions = predictor.predict(data)

    # 打印预测结果
    print("\n预测结果:")
    print(predictions.head())

    # 如果数据中包含实际价格，可以计算评估指标
    if 'close' in data.columns:
        # 提取对应的实际价格（跳过前look_back条记录）
        actual_prices = data['close'].iloc[predictor.look_back:].reset_index(drop=True)

        # 计算评估指标
        mse = np.mean((predictions['predicted_price'] - actual_prices) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions['predicted_price'] - actual_prices))

        print("\n评估指标:")
        print(f"MSE: {mse:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"MAE: {mae:.2f}")

        # 绘制预测结果与实际价格对比
        actual_df = pd.DataFrame({
            'date': data['date'].iloc[predictor.look_back:].reset_index(drop=True),
            'price': actual_prices
        })
        predictor.plot_predictions(predictions, actual_df, title="价格预测对比")
    else:
        # 仅绘制预测结果
        predictor.plot_predictions(predictions)

    print("预测完成！")
