import os
import time
import argparse
import tensorflow as tf
from sampler import WarpSampler
from model import Model
from tqdm import tqdm
from util import *
import numpy as np

def str2bool(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True)
parser.add_argument('--train_dir', required=True)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--maxlen', default=50, type=int)
parser.add_argument('--hidden_units', default=50, type=int)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_epochs', default=301, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.2, type=float)
parser.add_argument('--l2_emb', default=0.0, type=float)
parser.add_argument('--apply_context', default='', type=str)
parser.add_argument('--context_combine_mode', default=0, type=int)
parser.add_argument('--project_context_mode', default=0, type=int)
parser.add_argument('--context_mode', default=0, type=int)
parser.add_argument('--apply_local', default=False, type=bool)
parser.add_argument('--apply_atten_logits', default=False, type=bool)
parser.add_argument('--apply_local_conv', default=False, type=bool)
parser.add_argument('--local_conv_kernel_size', default=3, type=int)
parser.add_argument('--apply_v2_atten', default=0, type=int)
parser.add_argument('--apply_layer_atten', default=0, type=int)
parser.add_argument('--num_block_sa_layers', default=0, type=int)
parser.add_argument('--block_len', default=0, type=int)
parser.add_argument('--num_conv_layers', default=0, type=int)
parser.add_argument('--kernel_size', default=0, type=int)
parser.add_argument('--summary_dir', default='', type=str)
parser.add_argument('--start_block', default=0, type=int)
parser.add_argument('--end_block', default=-1, type=int)
parser.add_argument('--loss', default=0, type=int)
parser.add_argument('--sq_reduction_ratio', default=0, type=int)
parser.add_argument('--num_cross_layers', default=1, type=int)
parser.add_argument('--multi_head_attn_head_combine', default=0, type=int)
parser.add_argument('--multi_head_feat_interaction', default=0, type=int)
parser.add_argument('--enable_rel_pos', default=False, type=bool)
parser.add_argument('--apply_user_emb', default=False, type=bool)
parser.add_argument('--enable_value_context', default=False, type=bool)
parser.add_argument('--context_dropout', default=0.0, type=float)
parser.add_argument('--use_multihead_context', default=False, type=bool)
parser.add_argument('--use_fixed_test_set', default=False, type=bool)
parser.add_argument('--multihead_disagreent_reg', default=0.0, type=float)
parser.add_argument('--apply_weighted_sampling', default=False, type=bool)
parser.add_argument('--include_current_item', default=0, type=int)


args = parser.parse_args()
if not os.path.isdir(args.dataset + '_' + args.train_dir):
    os.makedirs(args.dataset + '_' + args.train_dir)
with open(os.path.join(args.dataset + '_' + args.train_dir, 'args.txt'), 'w') as f:
    f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
f.close()

dataset = data_partition(args.dataset)
[user_train, user_valid, user_test, usernum, itemnum, user_test_negs] = dataset
num_batch = len(user_train) / args.batch_size
cc = 0.0
item_set = set()
user_set = set()
for u in user_train:
    cc += len(user_train[u])
    user_set.add(u)
for u in user_train:
    for i in user_train[u]:
        item_set.add(i)
print 'average sequence length: %.2f' % (cc / len(user_train))
print "total number of uesrs in train: %d" % len(user_set)
print "total number of items in train: %d" % len(item_set)
print "total number of user_test_negs: %d" % len(user_test_negs)
print "# of user_train: %d" % len(user_train)
print "# of batches: %d" % (num_batch)

item_pops = calculate_item_pop(user_train)
norm_item_pops = normalize_item_pop(item_pops)
print(np.histogram(item_pops.values()))
print(np.histogram(norm_item_pops.values()))

item_with_pops = None
if args.apply_weighted_sampling:
    item_with_pops = flatten_items(norm_item_pops)

f = open(os.path.join(args.dataset + '_' + args.train_dir, 'log.txt'), 'w')
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
sess = tf.Session(config=config)

sampler = WarpSampler(user_train, usernum, itemnum, user_test_negs, item_with_pops, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3, use_fixed_test_set=args.use_fixed_test_set)
model = Model(usernum, itemnum, args)

train_writer = None
test_writer = None
if args.summary_dir:
   train_writer = tf.summary.FileWriter(args.summary_dir + '/train',
                                     sess.graph)
   test_writer = tf.summary.FileWriter(args.summary_dir + '/test')
sess.run(tf.initialize_all_variables())

T = 0.0
t0 = time.time()

try:
    global_steps = 0
    Qf = dict()
    Kf = dict()
    for epoch in range(1, args.num_epochs + 1):
        for step in tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
            u, seq, pos, neg = sampler.next_batch()
            batch_size = np.shape(seq)[0]
            auc, loss, summary, _ = sess.run([model.auc, model.loss,model.merged, model.train_op],
                                    {model.u: u, model.input_seq: seq, model.pos: pos, model.neg: neg,
                                     model.is_training: True, model.batch_size:batch_size})
            if train_writer:
               train_writer.add_summary(summary, global_steps)
            global_steps+=1
        print("Finish train at epoch" + str(epoch))

        #Qf, Kf = analyze_valid(model, dataset, args, sess)

        if epoch % 20 == 0:
            t1 = time.time() - t0
            T += t1
            print 'Evaluating',
            t_test = evaluate(model, dataset, args, sess)
            print 'Evaluating valid',
            t_valid = evaluate_valid(model, dataset, args, sess)
            #t_valid_summary = t_valid[-1]
            print ''
            print 'epoch:%d, time: %f(s), valid (NDCG@10: %.4f, HR@10: %.4f), test (NDCG@10: %.4f, HR@10: %.4f)' % (
            epoch, T, t_valid[0], t_valid[1], t_test[0], t_test[1])
            #test_writer.add_summary(t_valid_summary, global_steps)
            f.write(str(t_valid) + ' ' + str(t_test) + '\n')
            f.flush()
            t0 = time.time()
  
            #print 'Analyzing'
            #print 'Query forget factor'
            #print Qf
            #print 'Key forget factor'
            #print Kf
            
            
except:
    sampler.close()
    f.close()
    exit(1)

f.close()
sampler.close()
print("Done")
