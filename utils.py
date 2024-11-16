import random
import torch
import numpy as np
import mlflow

from sklearn.metrics import roc_auc_score

def set_random_seed(seed, device):
    if device == 'cpu':
        pass
    else:
        device = device.split(':')[0]

    if seed != -1:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if device == 'cuda':
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

def logging_metric_with_mlflow(result_list):
    for idx in range(len(result_list["loss_list"])):
        mlflow.log_metric("loss_list", result_list["loss_list"][idx], step=idx, synchronous=False)
        mlflow.log_metric("loss_1_list", result_list["loss_1_list"][idx], step=idx, synchronous=False)
        mlflow.log_metric("loss_2_list", result_list["loss_2_list"][idx], step=idx, synchronous=False)
        mlflow.log_metric("recall_val_list", result_list["recall_val_list"][idx], step=idx, synchronous=False)
        mlflow.log_metric("recall_test_list", result_list["recall_test_list"][idx], step=idx, synchronous=False)
        mlflow.log_metric("prediction_val_list", result_list["prediction_val_list"][idx], step=idx, synchronous=False)
        mlflow.log_metric("prediction_test_list", result_list["prediction_test_list"][idx], step=idx, synchronous=False)
    mlflow.log_metric("best_recall_val", result_list["best_recall_val"])
    mlflow.log_metric("best_prediction_val", result_list["best_prediction_val"])
    mlflow.log_metric("best_recall_test", result_list["best_recall_test"])
    mlflow.log_metric("best_prediction_test", result_list["best_prediction_test"])
    mlflow.log_metric("best_recall_epoch", result_list["best_recall_epoch"])
    mlflow.log_metric("best_prediction_epoch", result_list["best_prediction_epoch"])
        
                                                                                                    

def collate_fn(batch):
    user, items = zip(*batch)
    user = torch.LongTensor(user)
    return user, items

def precision_recall(actual, recommended, num_users):
    """ 추천 시스템에서 Precision과 Recall을 계산하는 함수
    actual: 실제로 선호한 항목 (리스트)
    recommended: 추천된 항목 (리스트)
    k: 추천 항목의 개수 제한
    """
    recommended = recommended+num_users
    recommended_at_k = recommended.tolist()  # 상위 k개의 추천 항목만 고려
    # 교집합: 추천된 항목 중에서 실제 선호한 항목
    relevant_and_recommended = []
    len_actual = 0
    for ac, rec in zip(actual, recommended):
        inter = set(ac).intersection(set(rec.tolist()))
        relevant_and_recommended.extend(list(inter))
        len_actual += len(ac)
    # Precision: 추천된 항목 중에서 실제로 선호한 항목의 비율
    precision = len(relevant_and_recommended) / (len(recommended_at_k[0]) * len(actual)) 
    # Recall: 실제로 선호한 항목 중에서 추천된 항목의 비율
    recall = len(relevant_and_recommended) / len_actual
    
    return precision, recall

def select_top_k(user, pred, top_k, seen_items, num_users):
    """
    Select top k items from the predicted items.
    args:
        pred (torch.tensor): (num_user, all_items)
        top_k (int): top k items
    """
    if seen_items is not None:
        for idx, i in enumerate(user):
            user_idx = i.item()
            seen_item_idx = seen_items[user_idx]
            seen_item_idx = [i-num_users for i in seen_item_idx] 
            for item_idx in seen_item_idx:
                pred[idx][item_idx] = -1e9
    
    top_k_val, top_k_idx = torch.topk(pred, top_k, dim=1)
    return top_k_idx