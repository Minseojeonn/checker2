import csv
from collections import defaultdict


file = open('gowalla.txt', 'r')
target_file = open('gowalla.tsv', 'w')
lines = file.readlines()
adj_dict = {}
for line in lines:
    splited = line.strip('\n').split(' ')
    src = int(splited[0])
    dst = splited[1:]
    adj_dict[src] = dst

num_user = max(adj_dict.keys())
start_idx_item = num_user + 1

for key, value in adj_dict.items():
    for v in value:
        v = int(v)
        target_file.write(f"{key}\t{v-start_idx_item}\t1\n")