import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import joblib
import os
import time
import random
from datetime import datetime
import platform
import json
import seaborn as sns
import sys
from sklearn.pipeline import Pipeline

# 删除特征转换器的导入
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# 设置GPU内存增长，避免一次性占用所有GPU内存
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # 对每个GPU设置内存增长
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"找到 {len(gpus)} 个GPU设备，已设置内存动态增长")

        # 设置可见的GPU设备
        tf.config.set_visible_devices(gpus, 'GPU')

        # 启用混合精度训练 (如果GPU支持)
        from tensorflow.keras import mixed_precision
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_global_policy(policy)
        print(f"混合精度策略已设置为: {policy.name}")

        # 检查是否在Apple Silicon芯片上运行
        is_apple_silicon = platform.processor() == 'arm'

        # 如果是Apple Silicon，尝试启用Metal支持
        if is_apple_silicon:
            try:
                # 导入Metal支持（前提是已安装tensorflow-metal）
                from tensorflow.python.compiler.mlcompute import mlcompute
                print("ML Compute加速已启用")
                # 使用GPU进行加速
                mlcompute.set_mlc_device(device_name='gpu')
                print(f"ML Compute设备: {mlcompute.get_mlc_device()}")
            except ImportError:
                print("未检测到tensorflow-metal，将使用标准GPU加速")
    except RuntimeError as e:
        print(f"GPU设置错误: {e}")
else:
    print("未检测到GPU设备，将使用CPU运行")

from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Conv1D, MaxPooling1D, Flatten, BatchNormalization, Activation
from tensorflow.keras.layers import Input, Concatenate, GlobalAveragePooling1D, GlobalMaxPooling1D, Attention, Add, TimeDistributed
from tensorflow.keras.layers import Bidirectional, GRU, LeakyReLU, PReLU, Reshape, RepeatVector, Lambda
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.regularizers import l2
from tensorflow.keras.metrics import RootMeanSquaredError, MeanAbsoluteError
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
from config import TRAINING_CONFIG, MLP_CONFIG, LSTM_CONFIG, CNN_CONFIG

# 设置随机种子，保证结果可复现
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)  # tensorflow 2.9.0使用这种方式设置随机种子

print(f"TensorFlow版本: {tf.__version__}")
# print(f"Keras版本: {tf.keras.__version__}") # Keras版本通常与TensorFlow版本一致，且此属性可能在新版中移除

# 检测可用的GPU设备
print("可用设备:")
for device in tf.config.list_physical_devices():
    print(f" - {device.name} ({device.device_type})")

class DeepLearningModels:
    def __init__(self, data_file, target_col='收盘价', look_back=TRAINING_CONFIG['look_back']):
        self.data_file = data_file
        self.target_col = target_col
        self.look_back = look_back
        self.models = {}
        self.history = {}
        self.predictions = {}
        self.metrics = {}

        # 创建必要的目录
        self.create_directories()

        # 加载和准备数据
        self.load_and_prepare_data()

    def create_directories(self):
        directories = ['saved_models', 'logs', 'checkpoints', 'results', 'scalers', 'correlation_analysis']
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
        print("已创建必要的目录，包括相关性分析目录")

    def load_and_prepare_data(self):
        print(f"正在加载数据: {self.data_file}")
        self.df = pd.read_csv(self.data_file)
        print(f"数据加载后，DataFrame 形状: {self.df.shape}")

        # 检查数据集格式并进行必要的映射
        if 'date' in self.df.columns and '日期' not in self.df.columns:
            print("检测到新数据集格式，进行列名映射...")
            # 列名映射（新数据集到旧数据集的映射）
            self.column_mapping = {
                'date': '日期',
                'open': '开盘价',
                'high': '最高价',
                'low': '最低价',
                'close': '收盘价',
                'volume': '成交量',
                'hold': '持仓量',
                'MA_5': 'MA5',
                'HV_20': '波动率',
                'ATR_14': 'ATR',
                'RSI_14': 'RSI',
                'OBV': 'OBV',
                'MACD': 'MACD'
            }

            # 将日期列转换为datetime
            self.df['date'] = pd.to_datetime(self.df['date'])

            # 更新目标列名
            if self.target_col == '收盘价' and 'close' in self.df.columns:
                self.target_col = 'close'
                print(f"目标列已更新为: {self.target_col}")

            # 不进行列名映射，直接使用新的列名
            self.date_col = 'date'
        else:
            # 原始数据集格式
            self.df['日期'] = pd.to_datetime(self.df['日期'])
            self.date_col = '日期'

        # 按日期升序排序
        self.df = self.df.sort_values(self.date_col)

        # 显示数据的基本信息
        print(f"原始数据行数: {len(self.df)}")
        print(f"原始数据列数 (包括日期和目标列): {len(self.df.columns)}")
        print(f"时间范围: {self.df[self.date_col].min()} 至 {self.df[self.date_col].max()}")

        # 选择要使用的特征
        self.features = self.select_features()  # 选择相关性最高的10个特征
        print(f"基于相关性分析选择的特征列表 ({len(self.features)} 个特征): {self.features}")

        # 创建简化的预处理管道，只包含MinMaxScaler
        print("创建简化的预处理管道...")
        self.preprocessing_pipeline = Pipeline([
            ('scaler', MinMaxScaler(feature_range=(-1, 1)))
        ])

        # 分割数据集
        self.split_data()

        # 使用Pipeline处理数据
        self.process_data_with_pipeline()

        # 准备时间序列数据
        self.prepare_time_series()

    def analyze_feature_correlation(self):
        print("\n开始进行特征相关性分析（仅计算皮尔逊相关系数）...")

        # 确保数据已加载
        if self.df is None or self.df.empty:
            print("错误: 数据未加载或为空，无法进行相关性分析")
            return

        # 如果已经计算过相关性，直接返回结果
        if hasattr(self, 'correlation_results') and self.correlation_results:
            print("使用已计算的相关性分析结果")
            return self.correlation_results

        # 创建相关性分析的目录
        correlation_dir = 'correlation_analysis'
        os.makedirs(correlation_dir, exist_ok=True)

        # 获取特征和目标变量的数据
        # 使用原始数据而不是缩放后的数据，以便更直观地解释结果
        features_df = self.df[self.features]
        target_series = self.df[self.target_col]

        # 计算皮尔逊相关系数 (线性相关性)
        pearson_corr = features_df.corrwith(target_series, method='pearson').sort_values(ascending=False)

        # 保存相关性结果到CSV文件
        correlation_results = pd.DataFrame({
            '特征': self.features,
            '皮尔逊相关系数': [pearson_corr[f] for f in self.features]
        })

        # 按皮尔逊相关系数的绝对值排序
        correlation_results['皮尔逊相关系数_绝对值'] = correlation_results['皮尔逊相关系数'].abs()
        correlation_results = correlation_results.sort_values('皮尔逊相关系数_绝对值', ascending=False)
        correlation_results = correlation_results.drop('皮尔逊相关系数_绝对值', axis=1)

        # 保存到CSV
        csv_path = os.path.join(correlation_dir, 'feature_correlation.csv')
        correlation_results.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"相关性分析结果已保存到: {csv_path}")

        # 打印相关性分析结果
        print("\n特征与目标变量 ({}) 的皮尔逊相关系数:".format(self.target_col))
        print(pearson_corr)

        # 可视化相关性结果

        # 皮尔逊相关系数热图
        plt.figure(figsize=(12, 10))

        # 计算所有特征之间的相关性矩阵
        correlation_matrix = features_df.corr(method='pearson')

        # 使用seaborn绘制热图
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
        plt.title('特征之间的皮尔逊相关系数热图', fontsize=16)
        plt.tight_layout()
        heatmap_path = os.path.join(correlation_dir, 'correlation_heatmap.png')
        plt.savefig(heatmap_path)
        plt.close()
        print(f"相关性热图已保存到: {heatmap_path}")

        # 特征与目标变量的相关性条形图
        plt.figure(figsize=(12, 8))

        # 按绝对值排序，保留符号
        abs_pearson = pearson_corr.abs().sort_values(ascending=False)
        sorted_pearson = pearson_corr[abs_pearson.index]

        # 绘制条形图
        bars = plt.barh(range(len(sorted_pearson)), sorted_pearson, color=['red' if x < 0 else 'blue' for x in sorted_pearson])
        plt.yticks(range(len(sorted_pearson)), sorted_pearson.index)
        plt.xlabel('皮尔逊相关系数')
        plt.title(f'特征与{self.target_col}的相关性 (按绝对值排序)', fontsize=16)

        # 添加数值标签
        for i, bar in enumerate(bars):
            width = bar.get_width()
            label_x_pos = width + 0.01 if width >= 0 else width - 0.08
            plt.text(label_x_pos, i, f'{width:.2f}', va='center')

        plt.axvline(x=0, color='gray', linestyle='--')
        plt.tight_layout()
        barplot_path = os.path.join(correlation_dir, 'feature_correlation_barplot.png')
        plt.savefig(barplot_path)
        plt.close()
        print(f"特征相关性条形图已保存到: {barplot_path}")

        # 散点图矩阵 (仅使用相关性最高的几个特征)
        # 选择相关性最高的5个特征
        top_features = abs_pearson.index[:5].tolist()
        scatter_df = features_df[top_features].copy()
        scatter_df[self.target_col] = target_series

        plt.figure(figsize=(15, 12))
        # 绘制散点图矩阵并忽略返回值
        pd.plotting.scatter_matrix(scatter_df, alpha=0.8, figsize=(15, 12), diagonal='kde')
        plt.suptitle(f'相关性最高的5个特征与{self.target_col}的散点图矩阵', fontsize=16, y=0.95)
        plt.tight_layout()
        scatter_path = os.path.join(correlation_dir, 'top_features_scatter_matrix.png')
        plt.savefig(scatter_path)
        plt.close()
        print(f"顶级特征散点图矩阵已保存到: {scatter_path}")

        print("\n相关性分析完成。所有结果已保存到 '{}'目录。".format(correlation_dir))

        # 保存相关性结果，以便其他方法可能使用
        self.correlation_results = {
            'pearson': pearson_corr,
            'correlation_matrix': correlation_matrix,
            'correlation_df': correlation_results
        }

        return self.correlation_results

    def select_features(self):
        print("信息: 正在从原始加载数据中选择特征，将基于相关性分析选择特征。")
        if self.df is None or self.df.empty:
            raise ValueError("错误: DataFrame未加载或为空，无法选择特征。")

        all_loaded_columns = self.df.columns.tolist()

        # 这些属性应该在调用此方法之前已经被正确设置
        columns_to_exclude = []
        if hasattr(self, 'date_col') and self.date_col:
            columns_to_exclude.append(self.date_col)
        else:
            print("警告: 'self.date_col' 未定义或为空，可能导致日期列未被正确排除。")

        if hasattr(self, 'target_col') and self.target_col:
            columns_to_exclude.append(self.target_col)
        else:
            print("警告: 'self.target_col' 未定义或为空，可能导致目标列未被正确排除。")

        # 确保排除列表中的列名确实存在于DataFrame中，避免KeyError
        columns_to_exclude = [col for col in columns_to_exclude if col in all_loaded_columns]

        potential_feature_names = [col for col in all_loaded_columns if col not in columns_to_exclude]

        # 首先获取所有有效的数值型特征
        valid_numeric_features = []
        for feature_name in potential_feature_names:
            if feature_name in self.df.columns:
                if not pd.api.types.is_numeric_dtype(self.df[feature_name]):
                    print(f"信息: 特征 '{feature_name}' 非数值类型，尝试转换为数值类型...")
                    try:
                        # 原地转换列类型，并将无法转换的值设为NaN
                        self.df[feature_name] = pd.to_numeric(self.df[feature_name], errors='coerce')
                        # 检查转换后是否所有值都变为NaN (例如，如果列是纯文本)
                        if self.df[feature_name].isnull().all():
                            print(f"警告: 特征 '{feature_name}' 转换为数值后全为NaN。此特征将不被使用。")
                        else:
                            valid_numeric_features.append(feature_name)
                            print(f"信息: 特征 '{feature_name}' 已成功转换为数值类型。")
                    except Exception as e:
                        print(f"警告: 转换特征 '{feature_name}' 为数值类型失败: {e}。此特征将不被使用。")
                else:
                    valid_numeric_features.append(feature_name)
            else:
                # 这个情况理论上不应该发生，因为是从all_loaded_columns开始的
                print(f"警告: 在特征选择过程中，预期中的列 '{feature_name}' 未在DataFrame中找到。")

        # 进行相关性分析并选择特征
        print("信息: 进行相关性分析以选择最相关的特征...")

        # 获取特征和目标变量的数据
        features_df = self.df[valid_numeric_features]
        target_series = self.df[self.target_col]

        # 计算皮尔逊相关系数 (线性相关性)
        pearson_corr = features_df.corrwith(target_series, method='pearson').sort_values(ascending=False)

        # 按皮尔逊相关系数的绝对值排序，排除volume特征
        abs_pearson = pearson_corr.abs().sort_values(ascending=False)
        if 'volume' in abs_pearson.index:
            print(f"信息: 排除 'volume' 特征")
            abs_pearson = abs_pearson[abs_pearson.index != 'volume']

        # 选择所有剩余特征，不再限制固定数量
        final_selected_features = abs_pearson.index.tolist()
        top_n = len(final_selected_features)

        print(f"信息: 基于相关性分析，选择了除volume外的所有 {top_n} 个特征: {final_selected_features}")

        # 保存相关性结果，以便其他方法可能使用
        self.correlation_results = {
            'pearson': pearson_corr,
            'correlation_df': pd.DataFrame({
                '特征': valid_numeric_features,
                '皮尔逊相关系数': [pearson_corr[f] for f in valid_numeric_features]
            })
        }

        # 打印所有特征的相关性排名，帮助调试
        print("\n所有特征的相关性排名 (已排除volume):")
        for i, (feature, corr) in enumerate(abs_pearson.items(), 1):
            print(f"{i}. {feature}: {corr:.4f}")

        # 如果特征数量过少，发出警告
        if len(final_selected_features) < 5:
            print(f"警告: 特征数量较少 ({len(final_selected_features)} 个)，可能影响模型性能。")
            print(f"       当前所有加载的列: {all_loaded_columns}")
            print(f"       被定义为日期/目标而排除的列: {columns_to_exclude}")
            print(f"       最终选择的特征: {final_selected_features}")

        return final_selected_features

    def split_data(self, test_size=0.1, val_size=0.1):
        # 确保数据按时间排序
        data = self.df.sort_values(self.date_col)

        # 总数据量
        n = len(data)

        # 计算测试集和验证集的大小
        test_samples = int(n * test_size)
        val_samples = int(n * val_size)

        # 分割数据框
        train_df = data.iloc[:n - test_samples - val_samples].copy()
        val_df = data.iloc[n - test_samples - val_samples:n - test_samples].copy()
        test_df = data.iloc[n - test_samples:].copy()

        # 保存数据框，以便后续使用Pipeline处理
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        # 保存目标变量
        self.y_train = train_df[self.target_col].values
        self.y_val = val_df[self.target_col].values
        self.y_test = test_df[self.target_col].values

        # 保存日期信息以便可视化
        self.train_dates = train_df[self.date_col]
        self.val_dates = val_df[self.date_col]
        self.test_dates = test_df[self.date_col]

        print(f"训练集大小: {len(train_df)}")
        print(f"验证集大小: {len(val_df)}")
        print(f"测试集大小: {len(test_df)}")

        if len(train_df) == 0:
            print("错误：训练集为空！请检查数据预处理步骤和原始数据量。")
            raise ValueError("训练集在split_data后为空，无法继续。")

    def process_data_with_pipeline(self):
        print("使用简化Pipeline处理数据...")

        # 创建目标变量缩放器
        self.target_scaler = MinMaxScaler(feature_range=(-1, 1))

        # 不再限制特征数量
        print(f"使用相关性分析选择的所有 {len(self.features)} 个特征: {self.features}")

        # 从训练数据中提取选定的特征
        train_df_selected = self.train_df[self.features].copy()
        val_df_selected = self.val_df[self.features].copy()
        test_df_selected = self.test_df[self.features].copy()

        # 打印数据形状，帮助调试
        print(f"训练数据形状: {train_df_selected.shape}")
        print(f"验证数据形状: {val_df_selected.shape}")
        print(f"测试数据形状: {test_df_selected.shape}")

        # 简化预处理管道，只使用MinMaxScaler
        self.preprocessing_pipeline = Pipeline([
            ('scaler', MinMaxScaler(feature_range=(-1, 1)))
        ])

        # 拟合并转换训练数据
        print("拟合并转换训练数据...")
        self.X_train_scaled = self.preprocessing_pipeline.fit_transform(train_df_selected)
        self.y_train_scaled = self.target_scaler.fit_transform(self.y_train.reshape(-1, 1)).ravel()

        # 转换验证数据
        print("转换验证数据...")
        self.X_val_scaled = self.preprocessing_pipeline.transform(val_df_selected)
        self.y_val_scaled = self.target_scaler.transform(self.y_val.reshape(-1, 1)).ravel()

        # 转换测试数据
        print("转换测试数据...")
        self.X_test_scaled = self.preprocessing_pipeline.transform(test_df_selected)
        self.y_test_scaled = self.target_scaler.transform(self.y_test.reshape(-1, 1)).ravel()

        # 打印转换后的数据形状，确认特征数量
        print(f"转换后的训练数据形状: {self.X_train_scaled.shape}")
        print(f"转换后的验证数据形状: {self.X_val_scaled.shape}")
        print(f"转换后的测试数据形状: {self.X_test_scaled.shape}")

        # 保存Pipeline和目标缩放器
        pipeline_dir = 'pipelines'
        scaler_dir = 'scalers'
        os.makedirs(pipeline_dir, exist_ok=True)
        os.makedirs(scaler_dir, exist_ok=True)

        pipeline_path = os.path.join(pipeline_dir, 'preprocessing_pipeline.pkl')
        target_scaler_path = os.path.join(scaler_dir, 'target_scaler.pkl')

        joblib.dump(self.preprocessing_pipeline, pipeline_path)
        joblib.dump(self.target_scaler, target_scaler_path)

        print(f"预处理管道已保存到: {pipeline_path}")
        print(f"目标缩放器已保存到: {target_scaler_path}")

    def select_features_by_correlation(self, correlation_type='pearson', top_n=None, threshold=None):
        # 确保已经进行了相关性分析
        if not hasattr(self, 'correlation_results') or not self.correlation_results:
            print("警告: 未找到相关性分析结果，将进行相关性分析...")
            self.analyze_feature_correlation()

        # 只支持皮尔逊相关系数
        if correlation_type != 'pearson':
            print(f"警告: 只支持皮尔逊相关系数，忽略指定的相关性类型 '{correlation_type}'")
            correlation_type = 'pearson'

        # 获取相关性数据
        corr_series = self.correlation_results['pearson']

        # 根据相关性绝对值排序
        abs_corr = corr_series.abs().sort_values(ascending=False)
        sorted_features = abs_corr.index.tolist()

        # 应用选择条件
        selected_features = []

        # 如果设置了阈值
        if threshold is not None:
            selected_features = [f for f in sorted_features if abs(corr_series[f]) >= threshold]
            print(f"根据皮尔逊相关系数阈值 {threshold} 选择了 {len(selected_features)} 个特征")

        # 如果设置了top_n
        if top_n is not None:
            # 如果已经通过阈值筛选了特征，从中选择top_n个
            if selected_features:
                selected_features = selected_features[:min(top_n, len(selected_features))]
            else:
                selected_features = sorted_features[:min(top_n, len(sorted_features))]

            print(f"选择了相关性最高的 {len(selected_features)} 个特征")

        # 如果没有设置任何条件，使用所有特征
        if not selected_features:
            selected_features = sorted_features
            print(f"未设置选择条件，将使用所有 {len(selected_features)} 个特征")

        # 打印选择的特征及其相关性
        print(f"\n根据皮尔逊相关系数选择的特征:")
        for feature in selected_features:
            print(f"  - {feature}: {corr_series[feature]:.4f}")

        return selected_features

    def prepare_time_series(self):
        # 创建时间序列样本
        def create_time_series(X, y, time_steps=1):
            Xs, ys = [], []
            for i in range(len(X) - time_steps):
                Xs.append(X[i:(i + time_steps)])
                ys.append(y[i + time_steps])
            return np.array(Xs), np.array(ys)

        # 不再限制特征数量
        print(f"使用所有选定的特征，特征数量: {self.X_train_scaled.shape[1]}")

        # 创建训练集时间序列
        self.X_train_ts, self.y_train_ts = create_time_series(
            self.X_train_scaled, self.y_train_scaled, self.look_back
        )

        # 创建验证集时间序列
        self.X_val_ts, self.y_val_ts = create_time_series(
            self.X_val_scaled, self.y_val_scaled, self.look_back
        )

        # 创建测试集时间序列
        self.X_test_ts, self.y_test_ts = create_time_series(
            self.X_test_scaled, self.y_test_scaled, self.look_back
        )

        # 打印时间序列数据的形状，确认特征数量
        print(f"时间序列训练集形状: {self.X_train_ts.shape}")
        print(f"时间序列验证集形状: {self.X_val_ts.shape}")
        print(f"时间序列测试集形状: {self.X_test_ts.shape}")

        # 确认特征维度
        feature_dim = self.X_train_ts.shape[2]
        print(f"时间序列数据的特征维度: {feature_dim}")

    def _create_dataset(self, X, y, batch_size, shuffle=False):
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(X))
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset

    def create_mlp_model(self, params=None):
        if params is None:
            params = {
                'units1': 512,
                'units2': 256,
                'units3': 128,
                'units4': 64,
                'dropout': 0.3,
                'learning_rate': 0.001,
                'l2_reg': 0.0005,
                'use_correlation': True,
                'batch_norm': True,
                'activation': 'elu',
                'selected_feature_indices': None
            }

        # 确保使用相关性分析
        batch_norm = params.get('batch_norm', True)
        activation_fn = params.get('activation', 'elu')
        
        # 获取实际特征维度
        feature_dim = self.X_train_ts.shape[2]
        print(f"MLP模型使用的实际特征维度: {feature_dim}")
        
        # 创建模型输入
        inputs = Input(shape=(self.look_back, feature_dim))
        
        # 应用批归一化到输入
        normalized = BatchNormalization()(inputs)

        # 使用多尺度特征提取 - 使用不同尺寸的卷积核
        # 注意：我们已经在数据处理阶段选择了相关性最高的特征，这里直接使用
        print(f"MLP模型使用相关性分析选择的 {feature_dim} 个特征")

        # 使用RNN层捕获时间序列信息
        rnn_layer = Bidirectional(LSTM(128, return_sequences=True))(normalized)

        # 添加注意力机制
        attention_layer = Attention()([rnn_layer, rnn_layer])

        # 使用多尺度卷积提取不同尺度的特征
        conv1 = Conv1D(filters=128, kernel_size=2, padding='same')(normalized)
        conv1 = BatchNormalization()(conv1)
        conv1 = Activation(activation_fn)(conv1)

        conv2 = Conv1D(filters=128, kernel_size=3, padding='same')(normalized)
        conv2 = BatchNormalization()(conv2)
        conv2 = Activation(activation_fn)(conv2)

        conv3 = Conv1D(filters=128, kernel_size=5, padding='same')(normalized)
        conv3 = BatchNormalization()(conv3)
        conv3 = Activation(activation_fn)(conv3)

        # 全局特征提取
        global_avg_rnn = GlobalAveragePooling1D()(attention_layer)
        global_avg1 = GlobalAveragePooling1D()(conv1)
        global_avg2 = GlobalAveragePooling1D()(conv2)
        global_avg3 = GlobalAveragePooling1D()(conv3)

        # 创建全局特征提取分支 - 直接扁平化处理
        global_flat = Flatten()(normalized)

        # 合并所有特征
        merged = Concatenate()([global_avg_rnn, global_avg1, global_avg2, global_avg3, global_flat])

        # 打印合并层的形状，用于调试
        print(f"合并层形状: {merged.shape}")

        # 构建深度全连接网络
        # 根据参数确定单元数
        units1 = params['units1']
        units2 = params['units2']
        units3 = params['units3']
        units4 = params['units4']
        dropout_rate = params['dropout']

        # 第一个深度块
        x = Dense(units1, kernel_regularizer=l2(params['l2_reg']), kernel_initializer='he_normal')(merged)
        if batch_norm:
            x = BatchNormalization()(x)
        x = Activation(activation_fn)(x)
        x = Dropout(dropout_rate)(x)

        # 第一个残差连接 - 注意确保维度匹配
        # 我们需要将merged转换为与x相同的维度
        res1 = Dense(units1, kernel_initializer='he_normal')(merged)  # 调整为units1而不是units2
        if batch_norm:
            res1 = BatchNormalization()(res1)
        # 添加激活函数，确保与x有相同的处理
        res1 = Activation(activation_fn)(res1)

        # 第二个深度块
        x = Dense(units2, kernel_regularizer=l2(params['l2_reg']), kernel_initializer='he_normal')(x)
        if batch_norm:
            x = BatchNormalization()(x)
        x = Activation(activation_fn)(x)
        x = Dropout(dropout_rate)(x)

        # 应用残差连接 - 需要先将res1映射到相同维度
        res1_mapped = Dense(units2, kernel_initializer='he_normal')(res1)  # 映射到units2维度
        if batch_norm:
            res1_mapped = BatchNormalization()(res1_mapped)
        res1_mapped = Activation(activation_fn)(res1_mapped)
        # 现在两者应该有相同的形状
        x = Add()([x, res1_mapped])
        if batch_norm:
            x = BatchNormalization()(x)

        # 第三个深度块
        x_identity = x  # 保存当前状态用于第二个残差连接
        x = Dense(units3, kernel_regularizer=l2(params['l2_reg']), kernel_initializer='he_normal')(x)
        if batch_norm:
            x = BatchNormalization()(x)
        x = Activation(activation_fn)(x)
        x = Dropout(dropout_rate)(x)

        # 第二个残差连接 - 将x_identity映射到units3维度
        res2 = Dense(units3, kernel_initializer='he_normal')(x_identity)
        # 确保res2和x有相同的形状
        if batch_norm:
            res2 = BatchNormalization()(res2)
        # 添加激活函数，确保与x有相同的处理
        res2 = Activation(activation_fn)(res2)
        # 现在两者应该有相同的形状
        x = Add()([x, res2])
        if batch_norm:
            x = BatchNormalization()(x)

        # 第四个深度块
        x = Dense(units4, kernel_regularizer=l2(params['l2_reg']), kernel_initializer='he_normal')(x)
        if batch_norm:
            x = BatchNormalization()(x)
        x = Activation(activation_fn)(x)
        x = Dropout(dropout_rate)(x)

        # 添加更多的层以增加模型深度
        x = Dense(32, kernel_regularizer=l2(params['l2_reg']))(x)
        if batch_norm:
            x = BatchNormalization()(x)
        x = Activation(activation_fn)(x)

        # 输出层前的额外层，确保输出稳定性
        x = Dense(16, activation='linear', kernel_initializer='glorot_normal')(x)

        # 输出层
        outputs = Dense(1, kernel_initializer='glorot_normal')(x)

        # 创建模型
        model_name = "MLP_AdvancedFeatures"  # 更改名称以反映使用所有特征
        model = Model(inputs=inputs, outputs=outputs, name=model_name)

        # 编译模型 - 使用固定学习率，依靠ReduceLROnPlateau回调进行学习率调整
        optimizer = Adam(
            learning_rate=params['learning_rate'],
            clipnorm=1.0,
            clipvalue=0.5,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07
        )

        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=[RootMeanSquaredError(name='rmse'), MeanAbsoluteError(name='mae')]
        )

        return model

    def create_lstm_model(self, params=None):
        if params is None:
            params = {
                'lstm_units1': 256,
                'lstm_units2': 128,
                'dense_units1': 128,
                'dense_units2': 64,
                'dropout': 0.25,
                'recurrent_dropout': 0.1,
                'learning_rate': 0.0005,
                'l2_reg': 0.0002,
                'bidirectional': True,
                'stacked_layers': 2,
                'attention': True,
                'batch_norm': True,
                'use_correlation': True,
                'selected_feature_indices': None
            }

        # 确保使用相关性分析
        bidirectional = params.get('bidirectional', True)
        stacked_layers = params.get('stacked_layers', 2)
        use_attention = params.get('attention', True)
        batch_norm = params.get('batch_norm', True)
        recurrent_dropout = params.get('recurrent_dropout', 0.1)
        
        # 获取实际特征维度
        feature_dim = self.X_train_ts.shape[2]
        print(f"LSTM模型使用的实际特征维度: {feature_dim}")
        
        # 使用函数式API构建模型
        inputs = Input(shape=(self.look_back, feature_dim))
        
        # 应用批归一化到输入
        normalized = BatchNormalization()(inputs)
        
        print(f"LSTM模型使用相关性分析选择的 {feature_dim} 个特征")
        x = normalized
        lstm_units1 = params['lstm_units1']
        lstm_units2 = params['lstm_units2']

        # 构建LSTM网络 - 根据配置使用单向或双向LSTM
        if bidirectional:
            # 多层堆叠的双向LSTM
            for i in range(stacked_layers - 1):
                x = Bidirectional(LSTM(
                    lstm_units1,
                    return_sequences=True,
                    kernel_regularizer=l2(params['l2_reg']/2),
                    recurrent_dropout=recurrent_dropout,
                    dropout=params['dropout']/2,
                    kernel_initializer='glorot_uniform',
                    recurrent_initializer='orthogonal',
                    bias_initializer='zeros',
                    unroll=False, # 长序列设为False更高效
                    activation='tanh'
                ))(x)
                if batch_norm:
                    x = BatchNormalization()(x)

            # 最后一层LSTM
            lstm_out = Bidirectional(LSTM(
                lstm_units2,
                return_sequences=use_attention,  # 如果使用注意力机制则返回序列
                kernel_regularizer=l2(params['l2_reg']/4),
                recurrent_dropout=recurrent_dropout,
                dropout=params['dropout']/2,
                kernel_initializer='glorot_uniform',
                recurrent_initializer='orthogonal',
                bias_initializer='zeros'
            ))(x)
        else:
            # 多层堆叠的单向LSTM
            for i in range(stacked_layers - 1):
                x = LSTM(
                    lstm_units1,
                    return_sequences=True,
                    kernel_regularizer=l2(params['l2_reg']/2),
                    recurrent_dropout=recurrent_dropout,
                    dropout=params['dropout']/2,
                    kernel_initializer='glorot_uniform',
                    recurrent_initializer='orthogonal',
                    bias_initializer='zeros',
                    unroll=False,
                    activation='tanh'
                )(x)
                if batch_norm:
                    x = BatchNormalization()(x)

            # 最后一层LSTM
            lstm_out = LSTM(
                lstm_units2,
                return_sequences=use_attention,  # 如果使用注意力机制则返回序列
                kernel_regularizer=l2(params['l2_reg']/4),
                recurrent_dropout=recurrent_dropout,
                dropout=params['dropout']/2,
                kernel_initializer='glorot_uniform',
                recurrent_initializer='orthogonal',
                bias_initializer='zeros'
            )(x)

        # 添加注意力机制
        if use_attention:
            # 自注意力机制
            attention_output = Attention()([lstm_out, lstm_out])

            # 全局平均池化
            x = GlobalAveragePooling1D()(attention_output)

            # 可选: 添加额外的全局最大池化并合并
            max_pool = GlobalMaxPooling1D()(attention_output)
            x = Concatenate()([x, max_pool])
        else:
            # 如果不使用注意力，lstm_out已经是最后一个时间步的输出
            x = lstm_out

        # 添加时间卷积支线 - 多尺度特征提取
        conv1 = Conv1D(filters=64, kernel_size=2, padding='same', activation='relu')(normalized)
        conv1 = BatchNormalization()(conv1)
        conv1 = MaxPooling1D(pool_size=2)(conv1)

        conv2 = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(normalized)
        conv2 = BatchNormalization()(conv2)
        conv2 = MaxPooling1D(pool_size=2)(conv2)

        # 全局池化并合并卷积特征
        conv_feat1 = GlobalAveragePooling1D()(conv1)
        conv_feat2 = GlobalAveragePooling1D()(conv2)

        # 合并LSTM和CNN特征
        x = Concatenate()([x, conv_feat1, conv_feat2])

        # 根据是否使用相关性分析调整网络规模
        dense_units1 = params['dense_units1']
        dense_units2 = params['dense_units2']
        dropout_rate = params['dropout']

        # 全连接层 1
        x = Dense(dense_units1, kernel_regularizer=l2(params['l2_reg']))(x)
        if batch_norm:
            x = BatchNormalization()(x)
        x = PReLU()(x)
        x = Dropout(dropout_rate)(x)

        # 全连接层 2 - 添加残差连接
        identity = x
        x = Dense(dense_units1, kernel_regularizer=l2(params['l2_reg']))(x)
        if batch_norm:
            x = BatchNormalization()(x)
        x = PReLU()(x)
        x = Dropout(dropout_rate)(x)

        # 残差连接
        identity_mapping = Dense(dense_units1, activation='linear')(identity)
        x = Add()([x, identity_mapping])

        # 全连接层 3
        x = Dense(dense_units2, kernel_regularizer=l2(params['l2_reg']))(x)
        if batch_norm:
            x = BatchNormalization()(x)
        x = PReLU()(x)
        x = Dropout(dropout_rate/2)(x)

        # 深度调节层 - 非线性变换
        x = Dense(32, kernel_regularizer=l2(params['l2_reg']))(x)
        x = PReLU()(x)

        # 输出层前的最终处理 - 16个单元的线性层
        x = Dense(16, activation='linear')(x)

        # 输出层 - 单个值预测
        outputs = Dense(1,
                       kernel_initializer='glorot_uniform',
                       bias_initializer='zeros'
                      )(x)

        # 创建模型
        model_name = "LSTM_AdvancedFeatures"  # 更改名称以反映使用所有特征
        model = Model(inputs=inputs, outputs=outputs, name=model_name)

        # 编译模型 - 使用固定学习率，依靠ReduceLROnPlateau回调进行学习率调整
        optimizer = Adam(
            learning_rate=params['learning_rate'],
            clipnorm=1.0,  # 梯度裁剪，防止梯度爆炸
            clipvalue=0.5,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07
        )

        # 使用Huber损失(结合MSE和MAE的优点)和自定义指标
        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.Huber(delta=1.0),  # Huber损失对异常值更鲁棒
            metrics=[
                RootMeanSquaredError(name='rmse'),
                MeanAbsoluteError(name='mae'),
                tf.keras.metrics.MeanAbsolutePercentageError(name='mape')
            ]
        )

        return model

    def create_cnn_model(self, params=None):
        if params is None:
            params = {
                'filters1': 128,
                'filters2': 64,
                'filters3': 32,
                'kernel_size': 3,
                'pool_size': 2,
                'dense_units1': 64,
                'dense_units2': 32,
                'dropout': 0.15,
                'learning_rate': 0.0008,
                'l2_reg': 0.0001,
                'use_correlation': True,
                'selected_feature_indices': None
            }

        # 获取实际特征维度
        feature_dim = self.X_train_ts.shape[2]
        print(f"CNN模型使用的实际特征维度: {feature_dim}")
        
        # 使用函数式API构建模型
        inputs = Input(shape=(self.look_back, feature_dim))
        
        print(f"CNN模型使用相关性分析选择的 {feature_dim} 个特征")

        # 批归一化输入
        x = BatchNormalization()(inputs)
        filters1 = params['filters1']
        filters2 = params['filters2']
        filters3 = params['filters3']

        # 多分支卷积网络 - 调整为适应实际特征数量
        # 分支1: 较小卷积核
        conv1 = Conv1D(
            filters=filters1,
            kernel_size=2,
            padding='same',
            kernel_regularizer=l2(params['l2_reg']/4),
            kernel_initializer='he_normal'
        )(x)
        conv1 = BatchNormalization()(conv1)
        conv1 = Activation('relu')(conv1)

        # 分支2: 中等卷积核
        conv2 = Conv1D(
            filters=filters1,
            kernel_size=3,
            padding='same',
            kernel_regularizer=l2(params['l2_reg']/4),
            kernel_initializer='he_normal'
        )(x)
        conv2 = Activation('relu')(conv2)

        # 分支3: 较大卷积核
        conv3 = Conv1D(
            filters=filters1,
            kernel_size=5,
            padding='same',
            kernel_regularizer=l2(params['l2_reg']/4),
            kernel_initializer='he_normal'
        )(x)
        conv3 = BatchNormalization()(conv3)
        conv3 = Activation('relu')(conv3)

        # 合并分支
        merged = Concatenate()([conv1, conv2, conv3])

        # 添加额外卷积层精炼特征
        refined_conv_output1 = Conv1D(
            filters=filters2,
            kernel_size=3,
            padding='same',
            kernel_regularizer=l2(params['l2_reg']/4),
            kernel_initializer='he_normal'
        )(merged)
        refined = BatchNormalization()(refined_conv_output1)
        refined = Activation('relu')(refined)
        refined = MaxPooling1D(pool_size=params['pool_size'], strides=None)(refined)

        # 使用默认的dropout率
        dropout_rate = params['dropout']
        
        refined = Dropout(dropout_rate)(refined)
        
        refined_conv_output2 = Conv1D(
            filters=filters3,
            kernel_size=3,
            padding='same',
            kernel_regularizer=l2(params['l2_reg']/4)
        )(refined)
        x = BatchNormalization()(refined_conv_output2)
        x = Activation('relu')(x)
        
        attention_output = Attention()([x, x])
        
        # 全局特征
        global_feature = GlobalAveragePooling1D()(attention_output)
        
        # 使用默认的全连接层单元数
        dense_units1 = params['dense_units1']
        dense_units2 = params['dense_units2']
        
        # 增强后端网络
        x = Dense(dense_units1, activation='relu')(global_feature)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
        x = Dense(dense_units2, activation='relu')(x)
        x = Dense(32, kernel_regularizer=l2(params['l2_reg']))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(dropout_rate)(x)
        
        # 输出层
        outputs = Dense(1,
                       kernel_initializer='glorot_uniform',
                       bias_initializer='glorot_uniform'
                      )(x)
        
        # 创建模型
        model_name = "CNN_AdvancedFeatures"  # 更改名称以反映使用所有特征
        model = Model(inputs=inputs, outputs=outputs, name=model_name)
        
        # 编译模型
        optimizer = Adam(
            learning_rate=params['learning_rate'],
            clipnorm=1.0,
            clipvalue=0.5,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07
        )
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=[RootMeanSquaredError(name='rmse'), MeanAbsoluteError(name='mae')]
        )
        
        return model

    def test_models(self, epochs=5, batch_size=16):
        print("\n开始小规模测试...")

        models_to_test = ['MLP', 'LSTM', 'CNN']
        results = {}

        # 创建小规模数据
        test_size = min(100, len(self.X_train_ts))
        X_test_small_np = self.X_train_ts[:test_size]
        y_test_small_np = self.y_train_ts[:test_size]

        # 创建 tf.data.Dataset
        test_dataset_small = self._create_dataset(X_test_small_np, y_test_small_np, batch_size)

        print(f"使用{test_size}个样本进行小规模测试")

        for model_type in models_to_test:
            print(f"\n测试 {model_type} 模型...")

            try:
                # 创建简化版模型用于测试
                if model_type == 'MLP':
                    # 创建一个简化的MLP模型，不使用残差连接
                    feature_dim = self.X_train_ts.shape[2]
                    inputs = Input(shape=(self.look_back, feature_dim))
                    x = Flatten()(inputs)
                    x = Dense(128, activation='relu')(x)
                    x = Dense(64, activation='relu')(x)
                    x = Dense(32, activation='relu')(x)
                    outputs = Dense(1)(x)
                    model = Model(inputs=inputs, outputs=outputs)
                    model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mse'])
                elif model_type == 'LSTM':
                    model = self.create_lstm_model()
                elif model_type == 'CNN':
                    model = self.create_cnn_model()

                print(f"{model_type} 模型摘要:")
                model.summary()

                # 简单训练几个epoch
                model.fit(
                    test_dataset_small,
                    epochs=epochs,
                    verbose=1
                )

                # 评估模型
                loss, mae, mse = model.evaluate(test_dataset_small, verbose=0)
                rmse = np.sqrt(mse)
                results[model_type] = {'loss': loss, 'rmse': rmse, 'mae': mae}

                # 试一下预测
                preds = model.predict(test_dataset_small.take(1))
                actuals_batch = list(test_dataset_small.take(1).as_numpy_iterator())[0][1]
                print(f"样本预测结果 vs 实际值 (来自一个小批次):")
                for i in range(min(5, len(preds))):
                    print(f"预测: {preds[i][0]:.4f}, 实际: {actuals_batch[i]:.4f}")
            except Exception as e:
                print(f"测试 {model_type} 模型时出错: {str(e)}")
                results[model_type] = {'loss': float('nan'), 'rmse': float('nan'), 'mae': float('nan')}

        print("\n小规模测试结果摘要:")
        for model_type, metrics in results.items():
            print(f"{model_type}: 损失={metrics['loss']:.4f}, RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}")

        return results

    def train_model(self, model_type, params=None, epochs=TRAINING_CONFIG['epochs'],
                   batch_size=TRAINING_CONFIG['batch_size'], resume_training=False,
                   run_small_test=False, correlation_params=None):
        # 打印特征数量信息，帮助调试
        print(f"\n训练模型前的特征信息:")
        print(f"特征数量: {len(self.features)}")
        print(f"特征列表: {self.features}")
        print(f"时间序列训练数据形状: {self.X_train_ts.shape}")
        if self.X_train_ts.shape[2] != len(self.features):
            print(f"警告: 时间序列数据的特征维度 ({self.X_train_ts.shape[2]}) 与特征列表长度 ({len(self.features)}) 不匹配!")
        print(f"目标变量形状: {self.y_train_ts.shape}")
        print(f"使用的look_back值: {self.look_back}")
        # 设置默认的相关性分析参数
        if correlation_params is None:
            correlation_params = {
                'correlation_type': 'pearson',
            }

        # 如果需要先进行小规模测试
        if run_small_test:
            self.test_models(epochs=5, batch_size=batch_size//2)

        # 生成时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        base_model_type = model_type
        # 永远使用相关性分析，设置模型键名
        current_model_key = f"{base_model_type}_AdvancedFeatures"  # 更改名称以反映使用所有特征
        
        # 设置模型保存路径
        model_path = f'saved_models/{current_model_key.lower()}_{timestamp}'
        checkpoint_path = f'checkpoints/{current_model_key.lower()}_{timestamp}'
        log_path = f'logs/{current_model_key.lower()}_{timestamp}'

        # 确定模型类型和创建/加载模型
        if resume_training and os.path.exists(f'saved_models/{current_model_key.lower()}_{timestamp}.h5'):
            print(f"正在加载已有模型: saved_models/{current_model_key.lower()}_{timestamp}.h5")
            model = load_model(f'saved_models/{current_model_key.lower()}_{timestamp}.h5')
        else:
            if base_model_type.lower() == 'mlp':
                # 确保使用相关性分析
                if params:
                    params['use_correlation'] = True
                model = self.create_mlp_model(params or MLP_CONFIG)
            elif base_model_type.lower() == 'lstm':
                # 确保使用相关性分析
                if params:
                    params['use_correlation'] = True
                model = self.create_lstm_model(params or LSTM_CONFIG)
            elif base_model_type.lower() == 'cnn':
                # 确保使用相关性分析
                if params:
                    params['use_correlation'] = True
                model = self.create_cnn_model(params or CNN_CONFIG)
            else:
                raise ValueError(f"不支持的模型类型: {base_model_type}")
        
        # 创建回调函数 - 优化配置
        callbacks = [
            # 早停策略 - 增加min_delta参数，避免过早停止
            EarlyStopping(
                monitor='val_loss',
                patience=TRAINING_CONFIG['early_stopping_patience'],
                restore_best_weights=True,
                min_delta=0.0001,
                mode='min',
                verbose=1
            ),
            # 学习率调度器 - 优化配置
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=TRAINING_CONFIG['reduce_lr_patience'],
                min_lr=TRAINING_CONFIG['min_lr'],
                min_delta=0.0001,
                cooldown=1,
                mode='min',
                verbose=1
            ),
            # 模型检查点 - 增加save_weights_only选项，减少存储空间
            ModelCheckpoint(
                filepath=f'{checkpoint_path}.h5',
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=False,
                mode='min',
                verbose=1
            ),
            # 训练日志记录
            CSVLogger(f'{log_path}.csv', append=resume_training, separator=',')
        ]

        # 创建 tf.data.Dataset
        train_dataset = self._create_dataset(self.X_train_ts, self.y_train_ts, batch_size, shuffle=True)
        val_dataset = self._create_dataset(self.X_val_ts, self.y_val_ts, batch_size)

        # 训练开始时间
        start_time = time.time()

        # 训练模型
        history = model.fit(
            train_dataset,
            epochs=epochs,
            validation_data=val_dataset,
            callbacks=callbacks,
            verbose=1
        )

        # 训练结束时间
        end_time = time.time()
        training_time = end_time - start_time

        # 保存最终模型
        model.save(f'{model_path}.h5')

        # 输出训练时间和GPU使用情况
        print(f"\n{current_model_key} 模型训练完成，耗时 {training_time:.2f} 秒")
        print("GPU使用情况:")
        try:
            gpu_devices = tf.config.list_physical_devices('GPU')
            for device in gpu_devices:
                print(f" - {device.name}")
        except:
            print("未检测到GPU设备")

        # 保存模型和训练历史, using current_model_key
        self.models[current_model_key] = model
        self.history[current_model_key] = history.history

        # 在测试集上进行预测, pass current_model_key
        self.evaluate_model(current_model_key, timestamp)

        # 绘制训练历史, pass current_model_key
        self.plot_training_history(current_model_key, history, timestamp)

        model_info_to_save = {
            'model_identifier': current_model_key,
            'base_model_type': base_model_type,
            'timestamp': timestamp,
            'data_file_used': self.data_file,
            'look_back': self.look_back,
            'model_parameters': params,
            'training_time_seconds': round(training_time, 2),
            'evaluation_metrics': self.metrics.get(current_model_key, {}),
            'saved_model_path': f'{model_path}.h5',
            'checkpoint_path': f'{checkpoint_path}.h5',
            'log_csv_path': f'{log_path}.csv',
            'results_metrics_path': f'results/{current_model_key.lower()}_{timestamp}_metrics.txt',
            'results_metrics_json_path': f'results/{current_model_key.lower()}_{timestamp}_metrics.json',
            'training_history_plot_path': f'results/{current_model_key.lower()}_{timestamp}_training_history.png',
            'prediction_details_plot_path': f'results/{current_model_key.lower()}_{timestamp}_prediction_details.png'
        }

        results_dir = 'results'
        os.makedirs(results_dir, exist_ok=True)
        json_output_path = os.path.join(results_dir, 'all_models_training_summary.json')

        all_models_data = {}
        if os.path.exists(json_output_path):
            try:
                with open(json_output_path, 'r') as f:
                    content = f.read()
                    if content.strip():
                        all_models_data = json.loads(content)
                    else:
                        print(f"信息: JSON文件 '{json_output_path}' 为空，将创建新内容。")
            except json.JSONDecodeError:
                print(f"警告: JSON文件 '{json_output_path}' 损坏或格式不正确，将创建新文件。")
            except Exception as e:
                print(f"警告: 读取JSON文件 '{json_output_path}' 时发生错误: {e}，将创建新文件。")

        if current_model_key not in all_models_data:
            all_models_data[current_model_key] = []
        all_models_data[current_model_key].append(model_info_to_save)

        return model, history

    def evaluate_model(self, current_model_key, timestamp):
        model = self.models[current_model_key]

        # 创建测试集的 tf.data.Dataset
        test_dataset = self._create_dataset(self.X_test_ts, self.y_test_ts, batch_size=TRAINING_CONFIG['batch_size'] * 2)

        # 在测试集上进行预测
        predictions_scaled = model.predict(test_dataset)

        # 反向转换预测值
        predictions = self.target_scaler.inverse_transform(predictions_scaled)
        actual = self.target_scaler.inverse_transform(self.y_test_ts.reshape(-1, 1))

        # 保存预测结果
        self.predictions[current_model_key] = predictions

        # 计算评估指标
        mse = mean_squared_error(actual, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actual, predictions)
        r2 = r2_score(actual, predictions)
        mape = np.mean(np.abs((actual - predictions) / actual)) * 100

        # 计算准确率 - 定义为预测值与实际值的相对误差在5%以内的比例
        tolerance = 0.05  # 5%的容忍度
        correct_predictions = np.sum(np.abs((predictions - actual) / actual) <= tolerance)
        accuracy = (correct_predictions / len(actual)) * 100

        # 保存评估指标
        metrics = {
            'MSE': float(mse),
            'RMSE': float(rmse),
            'MAE': float(mae),
            'R2': float(r2),
            'MAPE': float(mape),
            'Accuracy': float(accuracy)
        }
        self.metrics[current_model_key] = metrics

        # 保存评估结果到文本文件, use current_model_key in filename
        results_file = f'results/{current_model_key.lower()}_{timestamp}_metrics.txt'
        with open(results_file, 'w') as f:
            f.write(f"{current_model_key} 模型评估结果:\n")
            f.write(f"均方误差 (MSE): {mse:.4f}\n")
            f.write(f"均方根误差 (RMSE): {rmse:.4f}\n")
            f.write(f"平均绝对误差 (MAE): {mae:.4f}\n")
            f.write(f"决定系数 (R2): {r2:.4f}\n")
            f.write(f"平均绝对百分比误差 (MAPE): {mape:.4f}%\n")
            f.write(f"准确率 (Accuracy): {accuracy:.2f}%\n")

        # 保存评估结果到JSON文件
        json_results_file = f'results/{current_model_key.lower()}_{timestamp}_metrics.json'

        # 创建包含更多详细信息的评估结果字典
        evaluation_results = {
            'model_key': current_model_key,
            'timestamp': timestamp,
            'data_file': self.data_file,
            'target_column': self.target_col,
            'look_back': self.look_back,
            'test_samples': len(actual),
            'metrics': {
                'mse': float(mse),
                'rmse': float(rmse),
                'mae': float(mae),
                'r2': float(r2),
                'mape': float(mape),
                'accuracy': float(accuracy)
            },
            # 添加预测统计信息
            'prediction_stats': {
                'min': float(np.min(predictions)),
                'max': float(np.max(predictions)),
                'mean': float(np.mean(predictions)),
                'std': float(np.std(predictions)),
                'median': float(np.median(predictions))
            },
            # 添加实际值统计信息
            'actual_stats': {
                'min': float(np.min(actual)),
                'max': float(np.max(actual)),
                'mean': float(np.mean(actual)),
                'std': float(np.std(actual)),
                'median': float(np.median(actual))
            }
        }

        # 将评估结果保存到JSON文件
        with open(json_results_file, 'w', encoding='utf-8') as f:
            json.dump(evaluation_results, f, ensure_ascii=False, indent=4)

        print(f"\n评估结果已保存到: {results_file}")
        print(f"评估结果JSON已保存到: {json_results_file}")

        # 输出评估指标, use current_model_key
        print(f"\n{current_model_key} 模型评估结果:")
        print(f"均方误差 (MSE): {mse:.4f}")
        print(f"均方根误差 (RMSE): {rmse:.4f}")
        print(f"平均绝对误差 (MAE): {mae:.4f}")
        print(f"决定系数 (R2): {r2:.4f}")
        print(f"平均绝对百分比误差 (MAPE): {mape:.4f}%")
        print(f"准确率 (Accuracy): {accuracy:.2f}%")

    def plot_training_history(self, current_model_key, history, timestamp):
        plt.figure(figsize=(15, 10))

        # 绘制损失曲线
        plt.subplot(2, 2, 1)
        plt.plot(history.history['loss'], label='训练损失')
        plt.plot(history.history['val_loss'], label='验证损失')
        plt.title(f'{current_model_key} 模型损失曲线')
        plt.xlabel('Epoch')
        plt.ylabel('损失')
        plt.legend()

        # 绘制MAE曲线
        plt.subplot(2, 2, 2)
        plt.plot(history.history['mae'], label='训练MAE')
        plt.plot(history.history['val_mae'], label='验证MAE')
        plt.title(f'{current_model_key} 模型MAE曲线')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()

        # 绘制学习率曲线（如果存在）
        if 'lr' in history.history:
            plt.subplot(2, 2, 3)
            plt.plot(history.history['lr'])
            plt.title(f'{current_model_key} 学习率调整')
            plt.xlabel('Epoch')
            plt.ylabel('学习率')
            plt.yscale('log')

        # 绘制预测与实际值对比（使用测试集的一部分）
        plt.subplot(2, 2, 4)

        # 获取预测结果, use current_model_key
        if current_model_key in self.predictions:
            predictions = self.predictions[current_model_key]
            # 反向转换测试集目标值
            actual = self.target_scaler.inverse_transform(self.y_test_ts.reshape(-1, 1))

            # 只显示前30个点以避免图表过于拥挤
            display_points = min(30, len(predictions))

            plt.plot(range(display_points), actual[:display_points], 'b-', label='实际值')
            plt.plot(range(display_points), predictions[:display_points], 'r--', label='预测值')
            plt.title(f'{current_model_key} 预测 vs 实际')
            plt.xlabel('样本')
            plt.ylabel('价格')
            plt.legend()

        plt.tight_layout()
        plt.savefig(f'results/{current_model_key.lower()}_{timestamp}_training_history.png')
        plt.close()

        print(f"\n训练历史图表已保存到: results/{current_model_key.lower()}_{timestamp}_training_history.png")

        # 额外保存一个预测结果的详细图表
        if current_model_key in self.predictions:
            plt.figure(figsize=(12, 6))

            predictions = self.predictions[current_model_key]
            actual = self.target_scaler.inverse_transform(self.y_test_ts.reshape(-1, 1))

            # 计算预测误差
            errors = predictions - actual

            # 绘制预测与实际值
            plt.subplot(2, 1, 1)
            plt.plot(actual, 'b-', label='实际值')
            plt.plot(predictions, 'r--', label='预测值')
            plt.title(f'{current_model_key} 预测结果详细图')
            plt.ylabel('价格')
            plt.legend()

            # 绘制误差
            plt.subplot(2, 1, 2)
            plt.bar(range(len(errors)), errors.flatten())
            plt.axhline(y=0, color='r', linestyle='-')
            plt.title('预测误差')
            plt.xlabel('样本')
            plt.ylabel('误差')

            plt.tight_layout()
            plt.savefig(f'results/{current_model_key.lower()}_{timestamp}_prediction_details.png')
            plt.close()

            print(f"预测详细图表已保存到: results/{current_model_key.lower()}_{timestamp}_prediction_details.png")

    def compare_models(self, include_correlation=True, group_by_type=True):
        if not self.metrics:
            print("没有可比较的模型，请先训练模型")
            return

        # 创建比较结果表格
        model_keys = list(self.metrics.keys())

        # 解析描述性键名以获取基础模型类型和相关性信息
        parsed_model_info = []
        for key in model_keys:
            base_type = key
            correlation_status = '否'
            correlation_method = '-'

            if "_Corr_" in key:
                parts = key.split("_Corr_", 1)
                base_type = parts[0]
                correlation_status = '是'
                if len(parts) > 1 and parts[1]:
                    correlation_method = parts[1]
                else:
                    correlation_method = '未知类型'

            parsed_model_info.append({
                'key': key,
                'base_type': base_type,
                'correlation_status': correlation_status,
                'correlation_method': correlation_method
            })

        # 创建比较DataFrame
        comparison_data = {
            '模型原始键': model_keys,
            '模型类型': [info['base_type'] for info in parsed_model_info],
            '使用相关性分析': [info['correlation_status'] for info in parsed_model_info],
            '相关性类型': [info['correlation_method'] for info in parsed_model_info],
            'MSE': [self.metrics[key]['MSE'] for key in model_keys],
            'RMSE': [self.metrics[key]['RMSE'] for key in model_keys],
            'MAE': [self.metrics[key]['MAE'] for key in model_keys],
            'R2': [self.metrics[key]['R2'] for key in model_keys],
            'MAPE (%)': [self.metrics[key]['MAPE'] for key in model_keys],
            '准确率 (%)': [self.metrics[key]['Accuracy'] for key in model_keys]
        }
        comparison_df = pd.DataFrame(comparison_data)

        # 按RMSE排序
        comparison_df = comparison_df.sort_values('RMSE')

        print("\n所有模型比较结果 (按RMSE排序):")
        display_columns = ['模型类型', '使用相关性分析', '相关性类型', 'RMSE', '准确率 (%)', 'MSE', 'MAE', 'R2', 'MAPE (%)']
        print(comparison_df[display_columns])

        # 找出最佳模型
        if not comparison_df.empty:
            best_model_info_rmse = comparison_df.iloc[0]
            print(f"\n根据RMSE指标，最佳模型配置为: {best_model_info_rmse['模型原始键']} (类型: {best_model_info_rmse['模型类型']}, 相关性: {best_model_info_rmse['相关性类型']}) ")

            # 按准确率排序找出最佳模型
            comparison_by_acc = comparison_df.sort_values('准确率 (%)', ascending=False)
            best_model_info_acc = comparison_by_acc.iloc[0]
            print(f"根据准确率指标，最佳模型配置为: {best_model_info_acc['模型原始键']} (类型: {best_model_info_acc['模型类型']}, 相关性: {best_model_info_acc['相关性类型']})")
        else:
            print("没有可比较的模型结果。")

        # 如果需要按模型类型分组比较
        if group_by_type and not comparison_df.empty:
            print("\n按基础模型类型分组比较:")
            for base_model_type_group in set(comparison_df['模型类型']):
                type_comparison = comparison_df[comparison_df['模型类型'] == base_model_type_group]
                print(f"\n比较基础模型类型: {base_model_type_group}")
                print(type_comparison[display_columns])

                corr_models_in_group = type_comparison[type_comparison['使用相关性分析'] == '是']
                if len(corr_models_in_group) > 1 and len(set(corr_models_in_group['相关性类型'])) > 1:
                    print(f"  在 {base_model_type_group} 类型中，不同相关性方法的比较:")
                    sorted_corr_models_in_group = corr_models_in_group.sort_values('RMSE')
                    print(sorted_corr_models_in_group[display_columns])

        # 如果需要，加载并显示特征相关性分析结果
        if include_correlation:
            try:
                if not comparison_df.empty:
                    best_model_key_rmse = comparison_df.iloc[0]['模型原始键']
                    best_model_type_rmse = comparison_df.iloc[0]['模型类型']
                    best_model_key_acc = comparison_df.sort_values('准确率 (%)', ascending=False).iloc[0]['模型原始键']

                    print("\n基于特征相关性和模型性能的建议:")
                    print(f"1. 最佳模型 (RMSE): {best_model_key_rmse}")
                    print(f"2. 最佳模型 (准确率): {best_model_key_acc}")

                    if best_model_type_rmse == 'MLP':
                        print("3. MLP模型在处理线性关系时表现良好，建议重点关注皮尔逊相关系数高的特征")
                    elif best_model_type_rmse == 'LSTM':
                        print("3. LSTM模型在处理时序关系时表现良好，建议关注时间序列特征的相关性")
                    elif best_model_type_rmse == 'CNN':
                        print("3. CNN模型在提取局部特征时表现良好，建议关注皮尔逊相关系数高的特征")

                    print("4. 比较不同相关性策略对同一基础模型类型的影响，以选择最佳策略。")

                print("5. 考虑移除相关性最低的特征，可能有助于减少噪声并提高模型性能 (如果未使用所有特征)")

            except Exception as e:
                print(f"\n加载或解释相关性分析结果时出错: {e}")

        return comparison_df

    def plot_model_comparison(self, timestamp=None):
        """
        绘制三种模型(MLP、LSTM、CNN)在测试集上的预测值与真实值对比折线图
        
        Args:
            timestamp: 可选，时间戳字符串，用于保存文件名
        """
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 检查是否有训练好的模型
        model_types = ['MLP', 'LSTM', 'CNN']
        model_keys = []
        for model_type in model_types:
            model_key = f"{model_type}_AdvancedFeatures"
            if model_key in self.models:
                model_keys.append(model_key)
        
        if not model_keys:
            print("没有找到训练好的模型，无法绘制对比图")
            return
            
        plt.figure(figsize=(15, 10))
        
        # 获取测试集真实值
        actual = self.target_scaler.inverse_transform(self.y_test_ts.reshape(-1, 1))
        
        # 绘制真实值
        plt.plot(actual, 'k-', linewidth=2.5, label='真实值')
        
        # 颜色映射
        colors = {'MLP': 'r--', 'LSTM': 'b--', 'CNN': 'g--'}
        
        # 为每个模型绘制预测值
        for model_key in model_keys:
            if model_key in self.predictions:
                predictions = self.predictions[model_key]
                base_model_type = model_key.split('_')[0]  # 提取基本模型类型
                plt.plot(
                    predictions, 
                    colors.get(base_model_type, 'm--'), 
                    linewidth=1.5, 
                    alpha=0.8,
                    label=f'{base_model_type}预测值'
                )
            else:
                print(f"警告: 未找到模型 {model_key} 的预测结果")
        
        # 设置图表标题和标签
        plt.title('深度学习模型在测试集上的预测对比', fontsize=16)
        plt.xlabel('测试样本', fontsize=12)
        plt.ylabel('价格', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(loc='best', fontsize=12)
        
        # 添加模型性能指标到图表
        metrics_text = ""
        for model_key in model_keys:
            if model_key in self.metrics:
                base_type = model_key.split('_')[0]
                metrics = self.metrics[model_key]
                metrics_text += f"{base_type}: RMSE={metrics['RMSE']:.2f}, 准确率={metrics['Accuracy']:.2f}%\n"
        
        if metrics_text:
            plt.figtext(0.02, 0.02, metrics_text, fontsize=10, 
                       bbox=dict(facecolor='white', alpha=0.8))
        
        # 保存图表
        result_dir = 'results'
        os.makedirs(result_dir, exist_ok=True)
        plt.tight_layout()
        comparison_path = os.path.join(result_dir, f'model_comparison_{timestamp}.png')
        plt.savefig(comparison_path, dpi=300)
        plt.close()
        
        print(f"模型对比图已保存到: {comparison_path}")
        
        # 另外绘制一个放大后半部分的图表，以便更好地看到细节
        plt.figure(figsize=(15, 10))
        
        # 确定要展示的数据点范围（后50%的测试集数据）
        half_point = len(actual) // 2
        
        # 绘制后半部分数据
        plt.plot(range(half_point, len(actual)), actual[half_point:], 'k-', linewidth=2.5, label='真实值')
        
        for model_key in model_keys:
            if model_key in self.predictions:
                predictions = self.predictions[model_key]
                base_model_type = model_key.split('_')[0]
                plt.plot(
                    range(half_point, len(predictions)), 
                    predictions[half_point:], 
                    colors.get(base_model_type, 'm--'), 
                    linewidth=1.5, 
                    alpha=0.8,
                    label=f'{base_model_type}预测值'
                )
        
        plt.title('深度学习模型在测试集后半部分的预测对比（放大视图）', fontsize=16)
        plt.xlabel('测试样本', fontsize=12)
        plt.ylabel('价格', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(loc='best', fontsize=12)
        
        # 添加模型性能指标
        if metrics_text:
            plt.figtext(0.02, 0.02, metrics_text, fontsize=10, 
                       bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        zoomed_comparison_path = os.path.join(result_dir, f'model_comparison_zoomed_{timestamp}.png')
        plt.savefig(zoomed_comparison_path, dpi=300)
        plt.close()
        
        print(f"放大的模型对比图已保存到: {zoomed_comparison_path}")
        
        # 额外添加一个显示误差条形图的对比
        plt.figure(figsize=(15, 10))
        
        # 计算每个模型的误差
        for i, model_key in enumerate(model_keys):
            if model_key in self.predictions:
                predictions = self.predictions[model_key]
                errors = predictions - actual
                base_model_type = model_key.split('_')[0]
                
                plt.subplot(len(model_keys), 1, i+1)
                plt.bar(range(len(errors)), errors.flatten(), alpha=0.7)
                plt.axhline(y=0, color='r', linestyle='-')
                plt.title(f'{base_model_type}模型预测误差', fontsize=14)
                plt.ylabel('误差', fontsize=10)
                
                # 添加均方根误差和平均绝对误差标签
                if model_key in self.metrics:
                    metrics = self.metrics[model_key]
                    plt.text(
                        0.02, 0.80, 
                        f"RMSE: {metrics['RMSE']:.2f}\nMAE: {metrics['MAE']:.2f}\n准确率: {metrics['Accuracy']:.2f}%", 
                        transform=plt.gca().transAxes,
                        bbox=dict(facecolor='white', alpha=0.8)
                    )
        
        plt.xlabel('测试样本', fontsize=12)
        plt.tight_layout()
        error_comparison_path = os.path.join(result_dir, f'model_error_comparison_{timestamp}.png')
        plt.savefig(error_comparison_path, dpi=300)
        plt.close()
        
        print(f"模型误差对比图已保存到: {error_comparison_path}")
        
        return comparison_path, zoomed_comparison_path, error_comparison_path

if __name__ == "__main__":
    try:
        # 创建保存模型的目录
        os.makedirs('saved_models', exist_ok=True)

        # 定义数据文件路径
        current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        data_file_old_format = os.path.join(current_dir, "model_data/date1.csv")

        chosen_data_file = None
        chosen_target_col = None

        # 根据用户请求，优先尝试加载 date1.csv
        if os.path.exists(data_file_old_format):
            chosen_data_file = data_file_old_format
            chosen_target_col = '收盘价'
            print(f"使用用户指定的数据文件: {chosen_data_file}，目标列: {chosen_target_col}")
        else:
            error_message = f"错误: 数据文件 {data_file_old_format} 未找到。请检查文件路径。"
            print(error_message)
            raise FileNotFoundError(error_message)

        # 初始化模型类
        print(f"数据文件: {chosen_data_file}")
        temp_df = pd.read_csv(chosen_data_file)
        print(f"数据集形状: {temp_df.shape}, 列数: {len(temp_df.columns)}")
        print(f"数据集列名: {temp_df.columns.tolist()}")

        dl_models = DeepLearningModels(data_file=chosen_data_file, target_col=chosen_target_col, look_back=30)

        # 先进行小规模测试
        print("\n进行小规模测试...")
        # 适当增大测试时的batch_size以更好地利用GPU，减少epochs以便快速测试
        dl_models.test_models(epochs=3, batch_size=32)

        # 设置相关性分析参数
        correlation_params = {
            'correlation_type': 'pearson',
        }

        # 使用相同时间戳以便识别同批次训练的模型
        training_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 训练MLP模型 (默认使用相关性分析)
        print("\n训练优化的MLP模型 (使用相关性分析)...")
        dl_models.train_model('MLP', batch_size=64, correlation_params=correlation_params)

        # 训练LSTM模型 (默认使用相关性分析)
        print("\n训练优化的LSTM模型 (使用相关性分析)...")
        dl_models.train_model('LSTM', batch_size=64, correlation_params=correlation_params)

        # 训练CNN模型 (默认使用相关性分析)
        print("\n训练优化的CNN模型 (使用相关性分析)...")
        dl_models.train_model('CNN', batch_size=64, correlation_params=correlation_params)

        # 比较所有模型
        print("\n比较所有训练的模型...")
        dl_models.compare_models()

        # 绘制三种模型预测对比图
        print("\n绘制模型预测对比图...")
        comparison_paths = dl_models.plot_model_comparison(training_timestamp)
        print(f"模型预测对比图保存路径: {comparison_paths}")

        # 输出相关性分析的总结
        if hasattr(dl_models, 'correlation_results') and dl_models.correlation_results:
            # 获取皮尔逊相关系数最高的10个特征
            top_pearson = dl_models.correlation_results['pearson'].abs().sort_values(ascending=False).head(10)
            print("\n皮尔逊相关系数最高的10个特征 (这些特征将被用于模型训练):")
            for feature, corr in top_pearson.items():
                print(f"  - {feature}: {corr:.4f}")

            print("\n相关性分析结果已保存到 'correlation_analysis' 目录")

        print("\n--- 所有训练模型在测试集上的最终准确率总结 ---")
        if dl_models.metrics:
            sorted_model_metrics = sorted(dl_models.metrics.items(), key=lambda item: item[1].get('RMSE', float('inf')))
            for model_key, metrics_values in sorted_model_metrics:
                if 'Accuracy' in metrics_values:
                    accuracy = metrics_values['Accuracy']
                    print(f"模型: {model_key:<30} | 测试集准确率: {accuracy:>7.2f}%")
                else:
                    print(f"模型: {model_key:<30} | 未找到准确率指标。")
        else:
            print("没有可供总结的已训练模型指标。")

        print("\n测试完成！")

    except ValueError as ve:
        if "模型固定需要" in str(ve):
            print(f"错误: 特征数量不匹配。请修改代码中的select_features方法，移除特征数量检查。")
            print(f"详细错误: {str(ve)}")
            print("提示: 已经修改了代码，不再强制要求固定数量的特征。请重新运行。")
        else:
            print(f"发生ValueError错误: {str(ve)}")
            import traceback
            traceback.print_exc()
    except Exception as e:
        print(f"发生错误: {str(e)}")
        import traceback
        traceback.print_exc()