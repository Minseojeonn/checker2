import argparse

# Parsing arguments
def parsing(): 
    parser = argparse.ArgumentParser(description='Python parser usage.')
    parser.add_argument('--device', default="cuda:0", type=str, help='device')
    parser.add_argument('--seed', default=0, type=int, help='seed')
    parser.add_argument('--use_mlflow', default=False, type=bool, help='use_mlflow')
    parser.add_argument('--dataset_name', default="ml-1m", type=str, help='dataset_name')
    parser.add_argument('--split_ratio', default=[0.8, 0.1, 0.1], type=list, help='split_ratio')
    parser.add_argument('--dataset_shuffle', default=True, type=bool, help='dataset_shuffle')
    parser.add_argument('--direction', default=False, type=bool, help='direction')
    parser.add_argument('--input_dim', default=64, type=int, help='input_dim')
    parser.add_argument('--batch_size', default=256, type=int, help='batch_size')
    parser.add_argument('--num_layers', default=3, type=int, help='num_layer')
    parser.add_argument('--epochs', default=100, type=int, help='epochs')
    parser.add_argument('--wdc', default=0.001, type=float, help='wdc')
    parser.add_argument('--lr', default=0.01, type=float, help='lr')
    parser.add_argument('--topk', default=40, type=int, help='topk')
    parser.add_argument('--iter_k', default=100, type=int, help='iter_k')
    parser.add_argument('--alpha', default=0.8, type=float, help='alpha')
    parser.add_argument('--sign', default=3, type=int, help='beta')
    parser.add_argument('--indure', default=5, type=int, help='indure')
    parser.add_argument('--eps', default=1e-9, type=float, help='eps')
    parser.add_argument('--sbpr', default=0, type=int, help='1 = sbpr loss , 0 = original bpr loss')
    args = parser.parse_args()
    return args