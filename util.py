import os
import sys
import copy
import random
import math
import numpy as np
from collections import defaultdict

def calculate_item_pop(user_train):
    Items = dict()
    for user, items in user_train.iteritems():
        item_set = set(items)
        for item in item_set:
            if item not in Items:
                Items[item] = 0
            Items[item] += 1
    return Items

def normalize_item_pop(Items, power = 1.0):
    pops = [math.pow(pop * 1.0, power) for item, pop in Items.iteritems()]
    tot_pops = np.sum(pops)
    norm_items = dict()
    for item, pop in Items.iteritems():
        norm_pop = math.pow(pop * 1.0, power) / tot_pops
        norm_items[item] = norm_pop
    return norm_items

def flatten_items(Items):
    items = [key for key, value in Items.iteritems()]
    item_pops = [value for key, value in Items.iteritems()]
    return items, item_pops

def random_pop_neg(items, item_weights, s):
    t = np.random.choice(items, 1, p=item_weights)[0]
    while t in s:
        t = np.random.choice(items, 1, p=item_weights)[0]
    return t

def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t

def data_partition(fname):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    # assume user/item index starting from 1
    f = open('data/%s.txt' % fname, 'r')
    for line in f:
        u, i = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append(i)

    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])

    user_test_negs = {}

    if os.path.exists('data/%s.negs' % fname):
        nf = open('data/%s.negs' % fname, 'r')
        for line in nf:
            u, i = line.rstrip().split(' ')
            u = int(u)
            i = int(i)
            if u not in user_test_negs:
               user_test_negs[u] = set()
            user_test_negs[u].add(i)
        nf.close()
    else:
        for user in User:
            negs = set()
            ts = set(User[u])
            while(len(negs) < 100):
                uts = ts | negs
                neg_elem = random_neq(1, itemnum + 1, uts)
                negs.add(neg_elem)
            user_test_negs[user] = negs

    return [user_train, user_valid, user_test, usernum, itemnum, user_test_negs]


def evaluate(model, dataset, args, sess):
    [train, valid, test, usernum, itemnum, user_test_negs] = copy.deepcopy(dataset)

    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0

    if usernum>100000000:
        users = random.sample(xrange(1, usernum + 1), 10000)
    else:
        users = xrange(1, usernum + 1)
    predict_summaries = []
    for u in users:

        if len(train[u]) < 1 or len(test[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        seq[idx] = valid[u][0]
        idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break
        rated = set(train[u])
        rated.add(0)
        item_idx = [test[u][0]]

        if args.use_fixed_test_set:
            item_idx.extend(user_test_negs[u])
            rated = rated.union(set(user_test_negs[u]))
        while len(item_idx) < 101:
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        predict_out = model.predict(sess, [u], [seq], item_idx) 
        predictions = -predict_out[0]
        predictions = predictions[0]

        #predict_summary = predict_out[1]

        #predict_summaries.append(predict_summary)

        rank = predictions.argsort().argsort()[0]

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print '.',
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user, predict_summaries


def evaluate_valid(model, dataset, args, sess):
    [train, valid, test, usernum, itemnum, user_test_negs] = copy.deepcopy(dataset)

    NDCG = 0.0
    valid_user = 0.0
    HT = 0.0
    if usernum>10000000:
        users = random.sample(xrange(1, usernum + 1), 10000)
    else:
        users = xrange(1, usernum + 1)
    predict_summaries = []
    for u in users:
        if len(train[u]) < 1 or len(valid[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break

        rated = set(train[u])
        rated.add(0)
        item_idx = [valid[u][0]]

        if args.use_fixed_test_set:
            item_idx.extend(user_test_negs[u])
            rated = rated.union(set(user_test_negs[u]))
	while len(item_idx) < 101:
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        predict_out = model.predict(sess, [u], [seq], item_idx)
        predictions = -predict_out[0]
        predictions = predictions[0]

        #predict_summary = predict_out[1]

        #predict_summaries.append(predict_summary)

        rank = predictions.argsort().argsort()[0]

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print '.',
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user, predict_summaries

def analyze_valid(model, dataset, args, sess):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    if usernum>10000:
        users = random.sample(xrange(1, usernum + 1), 10000)
    else:
        users = xrange(1, usernum + 1)
    avg_qf_dict = dict()
    avg_kf_dict = dict()
    for u in users:
        if len(train[u]) < 1 or len(valid[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break

        rated = set(train[u])
        rated.add(0)
        item_idx = [valid[u][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        output_dict = model.get_output_dict(sess, [u], [seq], item_idx)

        mask = output_dict['sequence_mask']

        mask = np.squeeze(mask, -1) # [B, L]

        for idx in xrange(args.num_blocks):
            if idx == 0:
               continue
            key = 'layer_wise_param_%d_Qf' % (idx + 0)
            qf = output_dict[key] # [B, L, 1]
            key = 'layer_wise_param_%d_Kf' % (idx + 0)
            kf = output_dict[key] # [B, L, 1]
            qf = np.squeeze(qf, -1) # [B, L]
            kf = np.squeeze(kf, -1) # [B, L]
            avg_qf = np.sum(qf * mask, -1) / np.sum(mask, -1)
            avg_kf = np.sum(kf * mask, -1) / np.sum(mask, -1)
            if idx not in avg_qf_dict:
                avg_qf_dict[idx] = list()
            avg_qf_dict[idx].extend(avg_qf)
            if idx not in avg_kf_dict:
                avg_kf_dict[idx] = list()
            avg_kf_dict[idx].extend(avg_kf)

    out_qf = dict()
    out_kf = dict()

    for idx in avg_qf_dict.keys():
        avg_qf = np.mean(avg_qf_dict[idx])
        avg_kf = np.mean(avg_kf_dict[idx])
        out_qf[idx] = avg_qf
        out_kf[idx] = avg_kf
    return out_qf, out_kf


