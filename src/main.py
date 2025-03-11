from sklearn.model_selection import StratifiedKFold
import numpy as np
import pickle
import torch
from kan import KAN
import joblib
from util import read_pkl
from model import MLModels
import time
from util import get_metrics


def train():

    X_train,y_train = read_pkl('6000.pkl',3000,3000)
    #TM分数
    X_tm = np.load('./data/GRU_train_4.npy')
    X_train = np.concatenate((X_train, X_tm), axis=1)
    combined = list(zip(X_train, y_train))
    np.random.seed(42)
    np.random.shuffle(combined)
    X_train, y_train = zip(*combined)
    X = np.array(X_train)
    y = np.array(y_train)
    k_fold = 10
    skf = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=42)
    fold = 0
    best_model_params = None
    loss = 1
    # 使用交叉验证进行模型训练和评估
    for train_index, test_index in skf.split(X, y):
        print(f"第{fold + 1}轮交叉验证")
        X_train_fold, X_test_fold = X[train_index], X[test_index]
        y_train_fold, y_test_fold = y[train_index], y[test_index]
        # 训练MLModels模型
        models = MLModels()
        models.fit(X_train_fold, y_train_fold)
        # 使用MLModels进行预测
        train_input_fold = models.transform(X_train_fold)
        test_input_fold = models.transform(X_test_fold)
        # 将数据转换为PyTorch张量
        dataset_fold = {}
        dataset_fold['train_input'] = torch.from_numpy(train_input_fold)
        dataset_fold['test_input'] = torch.from_numpy(test_input_fold)
        dataset_fold['train_label'] = torch.from_numpy(y_train_fold[:, None])
        dataset_fold['test_label'] = torch.from_numpy(y_test_fold[:, None])

        # 使用KAN模型
        model = KAN(width=[8,4,1], grid=3, k=3, noise_scale=0.1, seed=42)
        # 训练KAN模型
        results_fold = model.train(dataset_fold, opt="LBFGS", steps=60)
        y_pred = model(dataset_fold['test_input'])[:, 0].detach().numpy()
        metrics_fold = get_metrics(y_test_fold, y_pred)
        print("Validation Metrics:")
        print(f"AUC: {metrics_fold[0]}")
        print(f"AUPR: {metrics_fold[1]}")
        print(f"F1 Score: {metrics_fold[2]}")
        print(f"Accuracy: {metrics_fold[3]}")
        print(f"Recall: {metrics_fold[4]}")
        print(f"Specificity: {metrics_fold[5]}")
        print(f"Precision: {metrics_fold[6]}")
        sum = 0
        for j in range(len(results_fold['test_loss'])):
            sum+=results_fold['test_loss'][j]

        meanloss = sum/len(results_fold['test_loss'])
        if meanloss < loss:
            loss = meanloss
            best_model_params = model.state_dict()
            joblib.dump(models, 'model.pkl')
    # 保存最佳模型参数到文件
    torch.save(best_model_params, 'Best_model.pt')

if __name__=="__main__" :
    start_time = time.time()
    train()
    end_time = time.time()
    final_time = (end_time - start_time) / 60
    print(f"train  took {final_time:.2f} minutes")