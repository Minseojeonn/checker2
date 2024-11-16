#import
import mlflow
import torch 
import dotmap
import numpy as np

#from
from fire import Fire
from parser import parsing
from utils import set_random_seed, collate_fn, select_top_k, precision_recall, logging_metric_with_mlflow
from dataset.DataTemplate import DataTemplate
from torch.utils.data import DataLoader 
from model.LightGCN import LightGCN



def main():
    args_enviroments = dotmap.DotMap(vars(parsing()))
    #set env parameters
   
    # Set MLflow
    if args_enviroments.use_mlflow:
        remote_server_uri = "http://192.168.50.2:5001"
        mlflow.set_tracking_uri(remote_server_uri)
        experiment_name = f"undirected_settings-{args_enviroments.dataset_name}-{args_enviroments.seed}-{args_enviroments.aug}"
        mlflow.set_experiment(experiment_name)
        mlflow.start_run()

    # Step 0. Initialization
    args_enviroments.device = args_enviroments.device if torch.cuda.is_available() else 'cpu'
    set_random_seed(seed=args_enviroments.seed, device=args_enviroments.device)
    device = args_enviroments.device

    # Step 1. Preprocessing the dataset and load the dataset
    datatemplate = DataTemplate(args_enviroments.dataset_name, args_enviroments.seed, args_enviroments.split_ratio, args_enviroments.dataset_shuffle, args_enviroments.device, args_enviroments.direction, args_enviroments.input_dim, args_enviroments.aug, args_enviroments.iter_k, args_enviroments.alpha, args_enviroments.eps)
    train_dataset, valid_dataset, test_dataset, num_nodes = datatemplate.get_dataset()
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=0, collate_fn=collate_fn)
    seen_items = train_dataset.get_seen_nodes()
    
    
    # Step 2. Model definition
    model = LightGCN(args_enviroments, datatemplate).to(device)

    # Step 3. Optimzer definition
    opt = torch.optim.Adam(model.parameters(), lr=args_enviroments.lr)
    
    # Step 4. Training
    # metrics
    loss_list = []
    loss_1_list = []
    loss_2_list = []
    best_recall_val = float('-inf')
    best_prediction_val = float('-inf')
    best_recall_test = float('-inf')
    best_prediction_test = float('-inf')
    best_recall_epoch = -1
    best_prediction_epoch = -1
    recall_val_list = []
    recall_test_list = []
    prediction_val_list = []
    prediction_test_list = []
    indure_count = 0
    for epoch in range(args_enviroments.epochs):
        model.train()
        total_loss = 0
        total_loss_1 = 0
        total_loss_2 = 0    
        for batch in train_loader:
            #label is not used, cause it is unsigned model
            opt.zero_grad()
            user, pos, neg = batch
            user, pos, neg = user.to(device), pos.to(device), neg.to(device)
            loss_1, loss_2 = model.bpr_loss(user, pos, neg)
            loss = loss_1 + args_enviroments.wdc * loss_2
            loss.backward()
            opt.step()
            total_loss += loss.item()
            total_loss_1 += loss_1.item()
            total_loss_2 += loss_2.item()
        loss_list.append(total_loss / len(train_loader))
        loss_1_list.append(total_loss_1 / len(train_loader))
        loss_2_list.append(total_loss_2 / len(train_loader))
        print(f"Epoch {epoch} Loss: {total_loss / len(train_loader)}")
        if epoch % 1 == 0: #testing every 1 epochs (for debugging)
            model.eval()
            with torch.no_grad():
                val_precision, val_recall = [], []
                for batch in valid_loader:
                    user, items = batch
                    user, items = user.to(device), items
                    pred_rating = model.getUsersRating(user)
                    pred_items = select_top_k(user, pred_rating, args_enviroments.topk, seen_items, num_nodes[0])
                    batch_precision, batch_recall = precision_recall(items, pred_items, num_nodes[0])
                    val_precision.append(batch_precision)
                    val_recall.append(batch_recall)
                test_precision, test_recall = [], []
                for batch in test_loader:
                    user, items = batch
                    user, items = user.to(device), items
                    pred_rating = model.getUsersRating(user)
                    pred_items = select_top_k(user, pred_rating, args_enviroments.topk, seen_items, num_nodes[0])
                    batch_precision, batch_recall = precision_recall(items, pred_items, num_nodes[0])
                    test_precision.append(batch_precision)
                    test_recall.append(batch_recall)
                print(f"Epoch {epoch} Valid Precision: {np.mean(val_precision)} Recall: {np.mean(val_recall)}")
                print(f"Epoch {epoch} Test Precision: {np.mean(test_precision)} Recall: {np.mean(test_recall)}")
                recall_val_list.append(np.mean(val_recall))
                recall_test_list.append(np.mean(test_recall))
                prediction_val_list.append(np.mean(val_precision))
                prediction_test_list.append(np.mean(test_precision))
                if np.mean(val_recall) > best_recall_val:
                    best_recall_val = np.mean(val_recall)
                    best_recall_epoch = epoch
                    best_recall_test = np.mean(test_recall)
                    indure = 0
                if np.mean(val_precision) > best_prediction_val:
                    best_prediction_val = np.mean(val_precision)
                    best_prediction_epoch = epoch
                    best_prediction_test = np.mean(test_precision)
                    indure = 0
                indure += 1
                if indure_count >= args_enviroments.indure:
                    break
    # Step 5. mlflow logging
    if args_enviroments.use_mlflow:
        mlflow.log_params(dict(args_enviroments))
        logging_targets = {
            "best_recall_val": best_recall_val,
            "best_recall_test": best_recall_test,
            "best_recall_epoch": best_recall_epoch,
            "best_prediction_val": best_prediction_val,
            "best_prediction_test": best_prediction_test,
            "best_prediction_epoch": best_prediction_epoch,
            "loss_list": loss_list,
            "loss_1_list": loss_1_list,
            "loss_2_list": loss_2_list,
            "recall_val_list": recall_val_list,
            "recall_test_list": recall_test_list,
            "prediction_val_list": prediction_val_list,
            "prediction_test_list": prediction_test_list
        }
        logging_metric_with_mlflow(logging_targets)
    
    mlflow.end_run()

main()