import os

basedir = os.path.abspath(os.path.dirname(__file__))

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'a-hard-to-guess-string'
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or \
        'sqlite:///' + os.path.join(basedir, 'instance', 'app.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    NEWS_API_KEY = os.environ.get('NEWS_API_KEY')
    MARKET_API_KEY = os.environ.get('MARKET_API_KEY')

# DeepSeek API配置
DEEPSEEK_API_KEY = 'sk-84d110f0618f483984f74b1d3429947e'  # 请替换为实际的API密钥

LSTM_CONFIG = {
    'lstm_units1': 192,  
    'lstm_units2': 128,  
    'dense_units1': 128, 
    'dense_units2': 64,  
    'dropout': 0.3,    
    'learning_rate': 0.0005, 
    'l2_reg': 0.0001,
    'use_correlation': False, 
    'selected_feature_indices': None 
}

CNN_CONFIG = {
    'filters1': 192,    
    'filters2': 128,    
    'filters3': 64,     
    'kernel_size': 3,
    'pool_size': 2,
    'dense_units1': 128, 
    'dense_units2': 64,  
    'dropout': 0.25,   
    'learning_rate': 0.0005, 
    'l2_reg': 0.0001,
    'use_correlation': False, 
    'selected_feature_indices': None 
}