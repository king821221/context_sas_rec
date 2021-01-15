import argparse
from util import *
import random

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True)

args = parser.parse_args()
dataset = data_partition(args.dataset)
[user_train, user_valid, user_test, usernum, itemnum] = dataset

num_actions_per_u = []
item_2_user = dict()
for u in user_train:
    rated = set(user_train[u]).union(set(user_valid[u])).union(set(user_test[u]))
    num_actions_per_u.append(len(rated))
    for rated_item in rated:
        if rated_item not in item_2_user:
            item_2_user[rated_item] = set()
        item_2_user[rated_item].add(u)

num_actions_per_i = []
for item, users in item_2_user.iteritems():
    num_actions_per_i.append(len(users))

print('Number of actions per user stats. Avg: %f, min: %d, max: %d, 90-percentile: %d' %
      (np.mean(num_actions_per_u), np.min(num_actions_per_u), np.max(num_actions_per_u), np.percentile(num_actions_per_u, [90])[0]))
print("-------------------------")
print('Number of actions per item stats. Avg: %f, min: %d, max: %d, 90-percentile: %d' %
      (np.mean(num_actions_per_i), np.min(num_actions_per_i), np.max(num_actions_per_i), np.percentile(num_actions_per_i, [90])[0]))
