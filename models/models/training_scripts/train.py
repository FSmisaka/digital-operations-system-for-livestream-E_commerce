import os
import argparse
import traceback
from deep_learning_models import DeepLearningModels
from config import TRAINING_CONFIG, MLP_CONFIG, LSTM_CONFIG, CNN_CONFIG

def train_models(args):
    # 创建保存目录
    os.makedirs('saved_models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # 初始化模型类
    dl_models = DeepLearningModels(
        args.data_file,
        look_back=args.look_back or TRAINING_CONFIG['look_back']
    )
    
    # 如果只运行小规模测试
    if args.test_only:
        print("\n只运行小规模测试...")
        dl_models.test_models(epochs=args.test_epochs or 5, batch_size=args.batch_size or 16)
        return
    
    # 训练指定的模型
    if args.model_type == 'all':
        model_types = ['MLP', 'LSTM', 'CNN']
    else:
        model_types = [args.model_type.upper()]
    
    for model_type in model_types:
        print(f"\n开始训练 {model_type} 模型...")
        
        # 设置模型特定的参数
        if model_type == 'MLP':
            params = MLP_CONFIG
        elif model_type == 'LSTM':
            params = LSTM_CONFIG
        elif model_type == 'CNN':
            params = CNN_CONFIG
        
        # 训练模型
        dl_models.train_model(
            model_type=model_type,
            params=params,
            epochs=args.epochs or TRAINING_CONFIG['epochs'],
            batch_size=args.batch_size or TRAINING_CONFIG['batch_size'],
            resume_training=args.resume,
            run_small_test=args.run_small_test
        )
    
    # 比较模型性能
    if args.model_type == 'all':
        print("\n比较所有模型的性能:")
        comparison = dl_models.compare_models()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='深度学习模型训练脚本')
    
    parser.add_argument('--data_file', type=str, required=True,
                      help='训练数据文件路径')
    parser.add_argument('--model_type', type=str, default='all',
                      choices=['all', 'mlp', 'lstm', 'cnn'],
                      help='要训练的模型类型')
    parser.add_argument('--epochs', type=int,
                      help=f'训练轮数 (默认: {TRAINING_CONFIG["epochs"]})')
    parser.add_argument('--batch_size', type=int,
                      help=f'批次大小 (默认: {TRAINING_CONFIG["batch_size"]})')
    parser.add_argument('--look_back', type=int,
                      help=f'历史数据窗口大小 (默认: {TRAINING_CONFIG["look_back"]})')
    parser.add_argument('--resume', action='store_true',
                      help='是否从上次的检查点继续训练')
    parser.add_argument('--run_small_test', action='store_true',
                      help='训练前是否先进行小规模测试')
    parser.add_argument('--test_only', action='store_true',
                      help='是否仅运行小规模测试而不进行完整训练')
    parser.add_argument('--test_epochs', type=int, default=5,
                      help='小规模测试的训练轮数 (默认: 5)')
    
    args = parser.parse_args()
    
    try:
        train_models(args)
    except Exception as e:
        print(f"训练过程中发生错误: {str(e)}")
        traceback.print_exc() 