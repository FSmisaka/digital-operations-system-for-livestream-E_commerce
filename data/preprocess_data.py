import pandas as pd
import numpy as np

def preprocess_data(input_file, output_file=None):
    # 读取CSV文件
    df = pd.read_csv(input_file)

    try:
        df['日期'] = pd.to_datetime(df['日期'])
    except Exception as e:
        pass

    # 转换数值列并处理异常值
    numeric_cols = ['收盘价', '开盘价', '最高价', '最低价', '成交量']
    for col in numeric_cols:
        if col in df.columns:
            # 检查是否已经是数值类型
            if not pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].astype(str).str.replace(',', '').str.strip()
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # 处理异常值 (例如替换3倍标准差以外的值为NaN)
            mean = df[col].mean()
            std = df[col].std()
            def check_outlier(x, mean, std):
                return x < mean - 3 * std or x > mean + 3 * std
            def handle_nan_values(x):
                return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            outliers = df[col].apply(check_outlier, args=(mean, std))
            df[col] = df[col].apply(handle_nan_values)

    df = df.fillna(method='ffill')

    # 对于数值列，用列的平均值填充
    for col in numeric_cols:
        if col in df.columns and df[col].isna().any():
            col_mean = df[col].mean()
            df[col] = df[col].fillna(col_mean)

    # 计算涨跌幅
    df['涨跌幅'] = df['收盘价'].pct_change() * 100

    # 计算移动平均线
    df['MA5'] = df['收盘价'].rolling(window=5).mean()
    df['MA10'] = df['收盘价'].rolling(window=10).mean()
    df['MA20'] = df['收盘价'].rolling(window=20).mean()
    df['MA30'] = df['收盘价'].rolling(window=30).mean()
    df['MA60'] = df['收盘价'].rolling(window=60).mean()

    # 计算指数移动平均线
    df['EMA12'] = df['收盘价'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['收盘价'].ewm(span=26, adjust=False).mean()

    # 计算MACD
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

    # 计算RSI
    delta = df['收盘价'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # 计算成交量指标
    df['成交量MA5'] = df['成交量'].rolling(window=5).mean()
    df['成交量MA10'] = df['成交量'].rolling(window=10).mean()
    df['相对成交量'] = df['成交量'] / df['成交量MA10']

    # 添加时间特征
    df['年'] = df['日期'].dt.year
    df['月'] = df['日期'].dt.month
    df['日'] = df['日期'].dt.day
    df['季度'] = df['日期'].dt.quarter
    df['星期'] = df['日期'].dt.dayofweek + 1  

    # 计算布林带
    df['中轨线'] = df['MA20']
    df['标准差'] = df['收盘价'].rolling(window=20).std()
    df['上轨线'] = df['中轨线'] + 2 * df['标准差']
    df['下轨线'] = df['中轨线'] - 2 * df['标准差']

    # 计算KDJ指标
    low_min = df['最低价'].rolling(window=9).min()
    high_max = df['最高价'].rolling(window=9).max()
    df['RSV'] = 100 * ((df['收盘价'] - low_min) / (high_max - low_min))
    df['K'] = df['RSV'].ewm(com=2, adjust=False).mean()
    df['D'] = df['K'].ewm(com=2, adjust=False).mean()
    df['J'] = 3 * df['K'] - 2 * df['D']

    # 添加价格变动和日内波幅
    df['价格变动'] = df['收盘价'] - df['开盘价']
    df['日内波幅'] = (df['最高价'] - df['最低价']) / df['开盘价']

    #  添加技术指标交叉信号
    df['突破MA5'] = (df['收盘价'] > df['MA5']).astype(int)
    df['突破MA10'] = (df['收盘价'] > df['MA10']).astype(int)
    df['突破MA20'] = (df['收盘价'] > df['MA20']).astype(int)
    df['金叉'] = ((df['MACD'] > df['MACD_Signal']) & (df['MACD'].shift(1) <= df['MACD_Signal'].shift(1))).astype(int)
    df['死叉'] = ((df['MACD'] < df['MACD_Signal']) & (df['MACD'].shift(1) >= df['MACD_Signal'].shift(1))).astype(int)

    # 添加归一化特征
    df['日内波幅_scaled'] = (df['日内波幅'] - df['日内波幅'].min()) / (df['日内波幅'].max() - df['日内波幅'].min())
    df['RSI_scaled'] = df['RSI'] / 100
    df['相对成交量_scaled'] = (df['相对成交量'] - df['相对成交量'].min()) / (df['相对成交量'].max() - df['相对成交量'].min() + 1e-10)

    # 计算成交量变化率
    df['成交量变化率'] = df['成交量'].pct_change() * 100

    # 按日期排序（升序，旧日期在前）
    df = df.sort_values(by='日期', ascending=True)

    # 保存处理后的数据
    if output_file is None:
        output_file = input_file

    # 最终检查：确保没有NaN值
    for col in df.columns:
        if df[col].isna().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(0)
            else:
                df[col] = df[col].fillna('')

    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].apply(lambda x: np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0))

    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    return df

if __name__ == "__main__":
    input_file = "data.csv"
    # 将处理后的数据保存回原文件
    df_processed = preprocess_data(input_file)