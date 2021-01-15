from modules import *


class Model():
    def __init__(self, usernum, itemnum, args, reuse=None):
        self.is_training = tf.placeholder(tf.bool, shape=())
        self.batch_size = tf.placeholder(tf.int32, shape=())
        self.u = tf.placeholder(tf.int32, shape=(None, ))
        self.input_seq = tf.placeholder(tf.int32, shape=(None, args.maxlen))
        self.pos = tf.placeholder(tf.int32, shape=(None, args.maxlen))
        self.neg = tf.placeholder(tf.int32, shape=(None, args.maxlen))
        pos = self.pos
        neg = self.neg
        # [B, L, 1]
        mask = tf.expand_dims(tf.to_float(tf.not_equal(self.input_seq, 0)), -1)
        print("sequence mask")
        print(mask)
        print("sequence batch size")
        print(self.batch_size)
        with tf.variable_scope("SASRec", reuse=reuse):
            # sequence embedding, item embedding table
            self.seq, item_emb_table = embedding(self.input_seq,
                                                 vocab_size=itemnum + 1,
                                                 num_units=args.hidden_units,
                                                 zero_pad=True,
                                                 scale=True,
                                                 l2_reg=args.l2_emb,
                                                 scope="input_embeddings",
                                                 with_t=True,
                                                 reuse=reuse
                                                 )

            rel_pos_emb_table = relative_position_embedding(self.seq,
                                                        l2_reg=args.l2_emb,
                                                        reuse=reuse)

            print('rel_pos_emb_table')
            print(rel_pos_emb_table)

            if args.apply_user_emb:
                user_seq, user_emb_table = user_embedding(self.u,
                             vocab_size=usernum + 1,
                             num_units=args.hidden_units,
                             zero_pad=True,
                             scale=True,
                             l2_reg=args.l2_emb,
                             scope="user_embeddings",
                             with_t=True,
                             reuse=reuse
                            )
            else:
                user_seq = None

            print('user_seq_embedding')
            print(user_seq)
 
            # Positional Encoding
            t, pos_emb_table = embedding(
                tf.tile(tf.expand_dims(tf.range(tf.shape(self.input_seq)[1]), 0), [tf.shape(self.input_seq)[0], 1]),
                vocab_size=args.maxlen,
                num_units=args.hidden_units,
                zero_pad=False,
                scale=False,
                l2_reg=args.l2_emb,
                scope="dec_pos",
                reuse=reuse,
                with_t=True
            )
            self.seq += t

            # Dropout
            self.seq = tf.layers.dropout(self.seq,
                                         rate=args.dropout_rate,
                                         training=tf.convert_to_tensor(self.is_training))
            seq_last_emb_unmask = self.seq[:, -1, :]
            self.seq *= mask

            # Build blocks

            multihead_atten_layers = []
            local_attention_layers = []
            conv_layers = []
            output_atten_layers = []
            attention_weights_per_layer = []
            multihead_dis_per_layer = []
            multihead_atten_layers.append(self.seq)
            normalized_seq = normalize(self.seq)
            #multihead_atten_layers.append(normalized_seq)
            apply_layer_atten = args.apply_layer_atten
            apply_v2_atten = args.apply_v2_atten
            num_block_sa_layers = args.num_block_sa_layers
            block_len = args.block_len
            num_conv_layers = args.num_conv_layers
            kernel_size = args.kernel_size

            sublayer = 1
            total_sublayers = (num_conv_layers + 2) * args.num_blocks

            output_dict = dict()

            output_dict['layer_wise_param'] = dict()

            output_dict['sequence_mask'] =  mask

            start_block = args.start_block
            if start_block < 0:
                start_block = args.num_blocks + start_block

            print("context sa start block")
            print(start_block)

            end_block = args.end_block
            if end_block < 0:
                end_block = args.num_blocks + end_block

            print("context sa end block")
            print(end_block)


            for i in range(args.num_blocks):
                layer_dict = dict()
                block_scope = "num_blocks_{}".format(i)
                print("block_scope at {}:{}".format(i, block_scope))
                with tf.variable_scope(block_scope):
                    apply_local_modeling = False
                    apply_context = (args.apply_context == 'True')
                    context_mode = args.context_mode
                    if i <= 0 :
                       apply_context = False
                    if i < 0 :
                       context_mode = 3
                    
                    context_layer_mask = 0
                    if i >= start_block and i <= end_block:
                       context_layer_mask = 1
                    if i < 1:
                       apply_local_modeling=args.apply_local
                    apply_local_conv = args.apply_local_conv
                    if i < 0:
                       apply_local_conv = False
                    if apply_local_conv:
                        conv_layer = conv_block(
                            normalized_seq,
                            kernel_size=kernel_size,
                            scope='conv_block_{}'.format(i),
                            is_training=self.is_training,
                            reuse=reuse,
                            dropout=args.dropout_rate)
                        print('conv_layer at block %d' % i)
                        print(conv_layer)
                        conv_layer = tf.Print(conv_layer, [conv_layer, tf.shape(conv_layer), tf.reduce_mean(conv_layer)], message = 'conv_layer_{}'.format(i))
                        local_attention_layers.append(conv_layer)

                    multihead_dis_reg = args.multihead_disagreent_reg
                    if i == args.num_blocks - 1:
                       multihead_dis_reg = 0.0 
                    if apply_v2_atten == 0:
                        scope="self_attention_block_{}".format(i)
                        self.seq, layer_output_dict = multihead_attention(queries=normalized_seq,
                                                   keys=self.seq,
                                                   num_units=args.hidden_units,
                                                   num_heads=args.num_heads,
                                                   dropout_rate=args.dropout_rate,
                                                   is_training=self.is_training,
                                                   batch_size=self.batch_size,
                                                   causality=True,
                                                   input_layers=multihead_atten_layers,
                                                   conv_layers=conv_layers,
                                                   mask=mask,
                                                   local_kernel_size=kernel_size,
                                                   attention_weights_per_layer = attention_weights_per_layer,
                                                   local_attention_layers=local_attention_layers,
                                                   apply_context=apply_context,
                                                   context_combine_mode=args.context_combine_mode,
                                                   project_context_mode=args.project_context_mode,
                                                   context_mode=context_mode,
                                                   context_layer_mask=context_layer_mask,
                                                   context_dropout=args.context_dropout,
                                                   enable_rel_pos=args.enable_rel_pos,
                                                   rel_pos_emb=rel_pos_emb_table,
                                                   user_seq=user_seq,
                                                   multi_head_attn_head_combine=args.multi_head_attn_head_combine,
                                                   apply_local_modeling=apply_local_modeling,
                                                   enable_value_context=args.enable_value_context,
                                                   use_multihead_context=args.use_multihead_context,
                                                   multihead_disagreent_reg=args.multihead_disagreent_reg,
                                                   include_current_item=args.include_current_item > 0,
                                                   scope=scope)
                        for k,v in layer_output_dict.iteritems():
                           layer_dict[k] = v
                        if 'attention_weights' in layer_output_dict:
                            attention_weights_per_layer.append(layer_output_dict['attention_weights'])
                        if 'multihead_disagreent' in layer_output_dict:
                            multihead_dis_per_layer.append(layer_output_dict['multihead_disagreent'])

                    normalized_seq = normalize(self.seq)
                    explicit_interactions = build_interaction_model(normalized_seq,
                                                                    args, 'explicit_interaction_%d' % (i))
                    # Feed forward
                    if args.num_heads == 1:
                       feedforward_scope = 'feed_foward_{}'.format(i)
                       self.seq = feedforward(normalized_seq, num_units=[args.hidden_units, args.hidden_units],
                                           dropout_rate=args.dropout_rate, is_training=self.is_training, reuse=reuse, scope=feedforward_scope)
                    else:
                       self.seq = feedforward_with_routing(normalized_seq, mask, args.num_heads, self.batch_size, num_units=[args.hidden_units, args.hidden_units], dropout_rate=args.dropout_rate, is_training=self.is_training, reuse=reuse)
                    print('feed_forward output at block %d' % i)
                    print(self.seq)
                    if explicit_interactions is not None:
                        explicit_interactions = tf.verify_tensor_all_finite(explicit_interactions, 'explicit_interactions verify')
                        #explicit_interactions = tf.Print(explicit_interactions, [explicit_interactions, tf.reduce_mean(explicit_interactions), tf.reduce_min(explicit_interactions), tf.reduce_max(explicit_interactions)], message = 'explicit_interactions')
                        #selseq = tf.Print(self.seq, [self.seq, tf.reduce_mean(self.seq), tf.reduce_min(self.seq), tf.reduce_max(self.seq)], message = 'implicit_interactions')
                        print("combined explicit and implicit projection")
                        print(explicit_interactions)
                        ex_im_c = tf.concat([self.seq, explicit_interactions], -1)
                        print("concat explicit and implicit projections")
                        print(ex_im_c)
                        with tf.variable_scope("combine_explicit_fc"):
                           self.seq = tf.layers.dense(ex_im_c, args.hidden_units)
                    seq_last_emb_unmask = self.seq[:, -1, :]
                    self.seq *= mask
                    normalized_seq = normalize(self.seq)
                    multihead_atten_layers.append(self.seq)
#                    multihead_atten_layers.append(normalized_seq)
#                    normalized_seq = tf.Print(normalized_seq, [normalized_seq, tf.shape(normalized_seq)], message = 'normalized_seq_%d' % i) 
                    output_atten_layers.append(normalized_seq)
                    variable_summaries(normalized_seq, 'self_attention_outputs_layer_{}'.format(i))
                    if len(layer_dict) > 0:
                       if 'local_attention_outputs' in layer_dict:
                           local_attention_layers.append(layer_dict['local_attention_outputs'])
                       output_dict['layer_wise_param'][i] = layer_dict

#            for i in xrange(num_block_sa_layers):
#                self.seq = simple_block_attention(self.seq, tf.cast(tf.to_int32(tf.squeeze(mask, -1)), tf.bool), block_len,
#                                  scope = 'block_sa_atten_%d' % i, is_train=self.is_training)  
#                self.seq *= mask
#                normalized_seq = normalize(self.seq)
#                output_atten_layers.append(normalized_seq)
#
            self.seq = normalized_seq

            if apply_layer_atten > 0:
                print('apply layer wise attention upon final output')
                if apply_layer_atten == 1:
                   self.seq = fuse_layer_attention_v1(output_atten_layers)
                elif apply_layer_atten == 2:
                   self.seq = fuse_layer_attention_v2(output_atten_layers)
                elif apply_layer_atten == 3:
                   self.seq = fuse_layer_attention_v3(output_atten_layers, {'batch_size': self.batch_size, 'num_iterations': 3, 'num_out_caps': 5})
                elif apply_layer_atten == 4:
                   self.seq = fuse_layer_attention_v4(output_atten_layers)
                self.seq *= mask
                self.seq = normalize(self.seq)
                #variable_summaries(self.seq, 'fused_self_atten_output')
                #self.seq = tf.Print(self.seq, [self.seq, tf.shape(self.seq)], message = 'fused_seq_' + str(self.is_training))

        pos = tf.reshape(pos, [tf.shape(self.input_seq)[0] * args.maxlen])
        neg = tf.reshape(neg, [tf.shape(self.input_seq)[0] * args.maxlen])
        pos_emb = tf.nn.embedding_lookup(item_emb_table, pos)
        neg_emb = tf.nn.embedding_lookup(item_emb_table, neg)
        seq_emb = tf.reshape(self.seq, [tf.shape(self.input_seq)[0] * args.maxlen, args.hidden_units])
        print('loss seq_emb')
        print(seq_emb)
        #seq_emb = tf.Print(seq_emb, [seq_emb, tf.shape(seq_emb)], message = 'seq_emb_' + str(self.is_training))
        apply_atten_logits = args.apply_atten_logits

        self.test_item = tf.placeholder(tf.int32, shape=(101))
        test_item_emb = tf.nn.embedding_lookup(item_emb_table, self.test_item)

        output_predict_logits = 2

        l2_reg = 1.0

        if apply_atten_logits:
            test_item_emb_exp = tf.expand_dims(test_item_emb, 0) # [1, 102, D]
            test_item_emb_tiled = tf.tile(test_item_emb_exp, [tf.shape(self.input_seq)[0], 1, 1]) # [B, 102, D]
            print('test_item_emb_tiled')
            print(test_item_emb_tiled)
            print('seq_last_emb_prev')
            print(seq_last_emb_unmask)
            seq_last_emb = tf.expand_dims(seq_last_emb_unmask, -2) # [B, 1, D]
            seq_last_emb = tf.tile(seq_last_emb, [1, 101, 1])
            print('seq_last_emb')
            print(seq_last_emb)
            self.test_logits = prediction_logits(seq_last_emb, test_item_emb_tiled, l2_reg,  output_logits_mode=output_predict_logits)
            print("test_logits")
            print(self.test_logits)
            self.test_logits = tf.reshape(self.test_logits, [-1])
        else:
            self.test_logits = tf.matmul(seq_emb, tf.transpose(test_item_emb))
#            self.test_logits = tf.Print(self.test_logits, [self.test_logits, tf.shape(self.test_logits)], message = 'test_logits_matmul')
            self.test_logits = tf.reshape(self.test_logits, [tf.shape(self.input_seq)[0], args.maxlen, 101])
            print("test_logits")
            print(self.test_logits)
            self.test_logits = self.test_logits[:, -1, :]
#            self.test_logits = tf.Print(self.test_logits, [self.test_logits, tf.shape(self.test_logits)], message = 'test_logits')

        # prediction layer
        if apply_atten_logits:
            print("pos_emb")
            pos_emb_shape_list = pos_emb.get_shape().as_list()
            pos_emb_reshaped = tf.reshape(pos_emb, [tf.shape(self.input_seq)[0], args.maxlen, pos_emb_shape_list[-1]])
            print(pos_emb_reshaped)
            print("neg_emb")
            neg_emb_shape_list = neg_emb.get_shape().as_list()
            neg_emb_reshaped = tf.reshape(neg_emb, [tf.shape(self.input_seq)[0], args.maxlen, neg_emb_shape_list[-1]])
            print(neg_emb_reshaped)
            self.pos_logits = prediction_logits(seq_emb, pos_emb_reshaped, l2_reg, output_logits_mode=output_predict_logits)
            #self.pos_logits = tf.Print(self.pos_logits, [self.pos_logits, tf.shape(self.pos_logits), tf.reduce_min(self.pos_logits), tf.reduce_max(self.pos_logits), tf.reduce_mean(self.pos_logits)], message= 'pos_logits', summarize=30)
            self.pos_logits = tf.reshape(self.pos_logits, [-1])
            print('pos_logits')
            print(self.pos_logits)
            self.neg_logits = prediction_logits(seq_emb, neg_emb_reshaped, l2_reg, output_logits_mode=output_predict_logits)
            #self.neg_logits = tf.Print(self.neg_logits, [self.neg_logits, tf.shape(self.neg_logits), tf.reduce_min(self.neg_logits), tf.reduce_max(self.neg_logits), tf.reduce_mean(self.neg_logits)], message = 'neg_logits', summarize=30)
            self.neg_logits = tf.reshape(self.neg_logits, [-1])
            print('neg_logits')
            print(self.neg_logits)
        else:
            print("pos_emb")
            print(pos_emb)
            print("seq_emb")
            print(seq_emb)
            self.pos_logits = tf.reduce_sum(pos_emb * seq_emb, -1)
#            self.pos_logits = tf.Print(self.pos_logits, [self.pos_logits, tf.shape(self.pos_logits)], message = 'pos_logits')
            print('pos_logits')
            print(self.pos_logits)
            self.neg_logits = tf.reduce_sum(neg_emb * seq_emb, -1)
#            self.neg_logits = tf.Print(self.neg_logits, [self.neg_logits, tf.shape(self.neg_logits)], message = 'neg_logits')
            print("neg_emb")
            print(neg_emb)
            print('neg_logits')
            print(self.neg_logits)


        # ignore padding items (0)
        istarget = tf.reshape(tf.to_float(tf.not_equal(pos, 0)), [tf.shape(self.input_seq)[0] * args.maxlen])
        self.loss = tf.reduce_sum(
            - tf.log(tf.sigmoid(self.pos_logits) + 1e-24) * istarget -
            tf.log(1 - tf.sigmoid(self.neg_logits) + 1e-24) * istarget
        ) / tf.reduce_sum(istarget)
        if args.loss == 1:
            self.loss = aux_loss(self.pos_logits, self.neg_logits, istarget)
        elif args.loss == 2:
            self.loss = bpr_loss(self.pos_logits, self.neg_logits, istarget)
        print("Train loss")
        #print(self.loss)
#        self.loss = tf.Print(self.loss, [self.loss], message = 'model loss')
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.loss += sum(reg_losses)
#        self.loss = tf.Print(self.loss, [self.loss], message = 'model reg loss')
        if len(multihead_dis_per_layer) > 0:
            dis_reg_loss = tf.add_n(multihead_dis_per_layer)
            dis_reg_loss = tf.Print(dis_reg_loss, [dis_reg_loss], message = 'dis_reg_loss')
            self.loss += dis_reg_loss

#        self.loss = tf.Print(self.loss, [self.loss], message = 'loss')

        tf.summary.scalar('loss', self.loss)
        self.auc = tf.reduce_sum(
            ((tf.sign(self.pos_logits - self.neg_logits) + 1) / 2) * istarget
        ) / tf.reduce_sum(istarget)

#        self.auc= tf.Print(self.auc, [self.auc], message = 'auc')
        if reuse is None:
            tf.summary.scalar('auc', self.auc)
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=args.lr, beta2=0.98)
            self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)
        else:
            tf.summary.scalar('test_auc', self.auc)

        self.merged = tf.summary.merge_all()

        self.output_dict = dict()
        layer_dict = output_dict['layer_wise_param']
        for key, value in layer_dict.iteritems():
            okey = "layer_wise_param_" + str(key)
            print("okey")
            print(okey)
            print(value)
            for vkey, vval in value.iteritems():
               ovkey = okey + "_" + vkey
               self.output_dict[ovkey] = vval

        self.output_dict['sequence_mask'] = output_dict['sequence_mask']
        print('get_output_dict')
        print(self.output_dict)

        self.output_dict['multihead_dis_per_layer'] = multihead_dis_per_layer

    def predict(self, sess, u, seq, item_idx):
        return sess.run([self.test_logits],
                        {self.u: u, self.input_seq: seq, self.test_item: item_idx, self.is_training: False, self.batch_size: 1})

    def get_output_dict(self, sess, u, seq, item_idx):
        return sess.run(self.output_dict,
                        {self.u: u, self.input_seq: seq, self.test_item: item_idx, self.is_training: False, self.batch_size: 1})
