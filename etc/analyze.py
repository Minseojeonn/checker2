from collections import defaultdict
raw_file = open('ml-1m.tsv', 'r')

data_dict = defaultdict(list)
tars = []
for i in raw_file.readlines():
    t = i.strip('\n').split('\t')
    src, tar, value = t
    tars.append(int(tar))
    data_dict[int(src)].append(int(tar))
    
num_user = max(data_dict.keys()) + 1
num_items = max(tars) + 1

print(num_user, num_items)