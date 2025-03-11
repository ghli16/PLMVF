import joblib
import torch
import pickle
from util import get_metrics, read_pkl
import numpy as np
from kan import KAN
import time


def predict_():
    x_tm = np.load('./data/GRU_ind_4.npy')
    # 加载独立测试数据集
    X_test, y_test = read_pkl('ind_576.pkl', 576, 576)
    # 加载最佳模型参数
    best_model_params_loaded = torch.load('Best_model.pt')
    best_model = KAN(width=[8, 4, 1], grid=3, k=3)
    best_model.load_state_dict(best_model_params_loaded)

    ml_model = joblib.load('model.pkl')

    # 使用最佳模型对独立测试数据集进行预测
    X_test = np.concatenate((X_test, x_tm), axis=1)
    test_input = ml_model.transform(X_test)  # 使用之前训练得到的MLModels对象进行转换
    test_input_tensor = torch.from_numpy(test_input)
    y_pred_test = (best_model(test_input_tensor)[:, 0]).detach().numpy()

    # 计算测试集上的评估指标
    test_metrics = get_metrics(y_test, y_pred_test)
    print("\nTest Metrics:")
    print(f"AUC: {test_metrics[0]}")
    print(f"AUPR: {test_metrics[1]}")
    print(f"F1 Score: {test_metrics[2]}")
    print(f"Accuracy: {test_metrics[3]}")
    print(f"Recall: {test_metrics[4]}")
    print(f"Specificity: {test_metrics[5]}")
    print(f"Precision: {test_metrics[6]}")

if __name__ == "__main__":
    start_time = time.time()
    predict_()
    end_time = time.time()
    final_time = (end_time - start_time) / 60
    print(f"predict  took {final_time:.2f} minutes")