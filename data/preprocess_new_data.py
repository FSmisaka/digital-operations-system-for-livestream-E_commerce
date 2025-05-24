import pandas as pd
import numpy as np

def preprocess_new_data(input_file, output_file=None):
    # 读取CSV文件
    df = pd.read_csv(input_file)
    
    df['date'] = pd.to_datetime(df['date'])
    
    numeric_cols = ['close', 'open', 'high', 'low', 'volume']
    for col in numeric_cols:
        if col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].astype(str).str.replace(',', '').str.strip()
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            mean = df[col].mean()
            std = df[col].std()
            def check_outlier(x, mean, std):
                return x < mean - 3 * std or x > mean + 3 * std
            def handle_nan_values(x):
                return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            outliers = df[col].apply(check_outlier, args=(mean, std))
            df[col] = df[col].apply(handle_nan_values)
    
    df = df.fillna(method='ffill')
    
    for col in numeric_cols:
        if col in df.columns and df[col].isna().any():
            col_mean = df[col].mean()
            df[col] = df[col].fillna(col_mean)
    
    if 'MA_5' not in df.columns:
        df['MA_5'] = df['close'].rolling(window=5).mean()
    if 'MA_10' not in df.columns:
        df['MA_10'] = df['close'].rolling(window=10).mean()
    if 'MA_20' not in df.columns:
        df['MA_20'] = df['close'].rolling(window=20).mean()
    if 'RSI_14' not in df.columns:
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI_14'] = 100 - (100 / (1 + rs))
    if 'MACD' not in df.columns:
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema12 - ema26
    df['price_change'] = df['close'] - df['open']
    df['daily_range'] = (df['high'] - df['low']) / df['open']
    df['price_change_pct'] = df['close'].pct_change() * 100
    df['volume_change_pct'] = df['volume'].pct_change() * 100
    df = df.sort_values(by='date', ascending=True)
    if output_file is None:
        output_file = input_file
    
    # 最终检查：确保没有NaN值
    for col in df.columns:
        if df[col].isna().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(0)
            else:
                df[col] = df[col].fillna('')
    
    # 使用numpy的nan_to_num函数处理inf和NaN
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].apply(lambda x: np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0))
    
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    return df

if __name__ == "__main__":
    input_file = "model_data/date1.csv"
    df_processed = preprocess_new_data(input_file)