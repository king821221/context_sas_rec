# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
June 2017 by kyubyong park. 
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer
'''

from __future__ import print_function
import tensorflow as tf
import numpy as np
from operator import mul

context_l2_reg = 0.1

initializer_relu = lambda: tf.contrib.layers.variance_scaling_initializer(factor=2.0,
                                                             mode='FAN_IN',
                                                             uniform=False,
                                                             dtype=tf.float32)
regularizer = tf.contrib.layers.l2_regularizer(scale = 3e-7)

def variable_summaries(var, name_scope="summaries"):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope(name_scope):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)

def positional_encoding(dim, sentence_length, dtype=tf.float32):

    encoded_vec = np.array([pos/np.power(10000, 2*i/dim) for pos in range(sentence_length) for i in range(dim)])
    encoded_vec[::2] = np.sin(encoded_vec[::2])
    encoded_vec[1::2] = np.cos(encoded_vec[1::2])

    return tf.convert_to_tensor(encoded_vec.reshape([sentence_length, dim]), dtype=dtype)

def normalize(inputs, 
              epsilon = 1e-8,
              scope="ln",
              reuse=None):
    '''Applies layer normalization.
    
    Args:
      inputs: A tensor with 2 or more dimensions, where the first dimension has
        `batch_size`.
      epsilon: A floating number. A very small number for preventing ZeroDivision Error.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
      
    Returns:
      A tensor with the same shape and data dtype as `inputs`.
    '''
    with tf.variable_scope(scope, reuse=reuse):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]
    
        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta= tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))
        normalized = (inputs - mean) / ( (variance + epsilon) ** (.5) )
        outputs = gamma * normalized + beta
        
    return outputs

def embedding(inputs, 
              vocab_size, 
              num_units, 
              zero_pad=True, 
              scale=True,
              l2_reg=0.0,
              scope="embedding", 
              with_t=False,
              reuse=None):
    '''Embeds a given tensor.

    Args:
      inputs: A `Tensor` with type `int32` or `int64` containing the ids
         to be looked up in `lookup table`.
      vocab_size: An int. Vocabulary size.
      num_units: An int. Number of embedding hidden units.
      zero_pad: A boolean. If True, all the values of the fist row (id 0)
        should be constant zeros.
      scale: A boolean. If True. the outputs is multiplied by sqrt num_units.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A `Tensor` with one more rank than inputs's. The last dimensionality
        should be `num_units`.
        
    For example,
    
    ```
    import tensorflow as tf
    
    inputs = tf.to_int32(tf.reshape(tf.range(2*3), (2, 3)))
    outputs = embedding(inputs, 6, 2, zero_pad=True)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print sess.run(outputs)
    >>
    [[[ 0.          0.        ]
      [ 0.09754146  0.67385566]
      [ 0.37864095 -0.35689294]]

     [[-1.01329422 -1.09939694]
      [ 0.7521342   0.38203377]
      [-0.04973143 -0.06210355]]]
    ```
    
    ```
    import tensorflow as tf
    
    inputs = tf.to_int32(tf.reshape(tf.range(2*3), (2, 3)))
    outputs = embedding(inputs, 6, 2, zero_pad=False)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print sess.run(outputs)
    >>
    [[[-0.19172323 -0.39159766]
      [-0.43212751 -0.66207761]
      [ 1.03452027 -0.26704335]]

     [[-0.11634696 -0.35983452]
      [ 0.50208133  0.53509563]
      [ 1.22204471 -0.96587461]]]    
    ```    
    '''
    with tf.variable_scope(scope, reuse=reuse):
        lookup_table = tf.get_variable('lookup_table',
                                       dtype=tf.float32,
                                       shape=[vocab_size, num_units],
                                       #initializer=tf.contrib.layers.xavier_initializer(),
                                       regularizer=tf.contrib.layers.l2_regularizer(l2_reg))
        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1, num_units]),
                                      lookup_table[1:, :]), 0)
        outputs = tf.nn.embedding_lookup(lookup_table, inputs)
        
        if scale:
            outputs = outputs * (num_units ** 0.5) 
    if with_t: return outputs,lookup_table
    else: return outputs

def relative_position_embedding(query,
              num_units = None,
              zero_pad=True,
              scale=True,
              l2_reg=0.0,
              scope="relative_position_embedding",
              with_t=False,
              reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        maxlen = query.get_shape().as_list()[1]
        if num_units is None:
            num_units = query.get_shape().as_list()[-1]
        lookup_table = tf.get_variable('lookup_table',
                                       dtype=tf.float32,
                                       shape=[maxlen + maxlen, num_units],
                                       # initializer=tf.contrib.layers.xavier_initializer(),
                                       regularizer=tf.contrib.layers.l2_regularizer(
                                           l2_reg))
        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1, num_units]),
                                      lookup_table[1:, :]), 0)
        outputs = relative_position_embedding_lookup(lookup_table, query)

        if scale:
            outputs = outputs * (num_units ** 0.5)
    if with_t:
        return outputs, lookup_table
    else:
        return outputs

def relative_position_embedding_lookup(lookup_table, query):
    # query: [B, L, D]
    # lookup: [L, D]
    # output:[L, L, D]

    query_shape = query.get_shape().as_list()
    sequence_length = query_shape[1]

    sequence_length_range = tf.range(sequence_length)

    sequence_length_matrix = tf.expand_dims(sequence_length_range, -1) -\
                             tf.expand_dims(sequence_length_range, 0)

    sequence_length_matrix = sequence_length_matrix + sequence_length

    return tf.nn.embedding_lookup(lookup_table, sequence_length_matrix)

def user_embedding(inputs,
                   vocab_size,
                   num_units,
                   zero_pad=True,
                   scale=True,
                   l2_reg=0.0,
                   scope="embedding",
                   with_t=False,
                   reuse=None
                   ):
    with tf.variable_scope(scope, reuse=reuse):
        lookup_table = tf.get_variable('lookup_table',
                                       dtype=tf.float32,
                                       shape=[vocab_size, num_units],
                                       # initializer=tf.contrib.layers.xavier_initializer(),
                                       regularizer=tf.contrib.layers.l2_regularizer(
                                           l2_reg))
        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1, num_units]),
                                      lookup_table[1:, :]), 0)
        outputs = tf.nn.embedding_lookup(lookup_table, inputs)

        if scale:
            outputs = outputs * (num_units ** 0.5)
    if with_t:
        return outputs, lookup_table
    else:
        return outputs

def create_context_vector(context_mode, context_params):
    context_vec_output_dict = dict()
    if context_mode == 0: 
          context_vector = create_global_deep_context_vector(context_params)
    elif context_mode == 1: #T
        context_vector = create_deep_context_vector(context_params)
    elif context_mode == 2: 
        context_vector = create_combined_global_local_context_vector(context_params)
    elif context_mode == 3: #T
        context_vector = create_accumulative_global_deep_context_vector(context_params)
    elif context_mode == 4: #T
        context_vector, deep_context_vector, global_context_vector = create_combined_cumsum_deep_context_vector(context_params)
        context_vec_output_dict['deep_context_vector'] = deep_context_vector
        context_vec_output_dict['global_context_vector'] = global_context_vector
    elif context_mode == 5: #T
        context_vector, deep_context_vector, global_context_vector, local_context_vector = create_combined_deep_global_local_context_vector(context_params)
        context_vec_output_dict['deep_context_vector'] = deep_context_vector
        context_vec_output_dict['global_context_vector'] = global_context_vector
        context_vec_output_dict['local_context_vector'] = local_context_vector
    elif context_mode == 6: #T
        context_vector, deep_context_vector, global_context_vector = create_combined_deep_global_attentive_context_vector(context_params)
        context_vec_output_dict['deep_context_vector'] = deep_context_vector
        context_vec_output_dict['global_context_vector'] = global_context_vector
    elif context_mode == 7:#T
        context_vector, deep_context_vector, global_context_vector, max_context_vector = create_combined_deep_global_max_context_vector(context_params)
        context_vec_output_dict['deep_context_vector'] = deep_context_vector
        context_vec_output_dict['global_context_vector'] = global_context_vector
        context_vec_output_dict['max_context_vector'] = max_context_vector
    elif context_mode == 8: #T
        context_vector, deep_context_vector, global_context_vector, max_context_vector = create_combined_deep_global_most_attentive_context_vector(context_params)
        context_vec_output_dict['deep_context_vector'] = deep_context_vector
        context_vec_output_dict['global_context_vector'] = global_context_vector
        context_vec_output_dict['max_context_vector'] = max_context_vector
    elif context_mode == 9: #T
        context_vector, deep_context_vector, global_context_vector, max_context_vector = create_combined_deep_global_self_attentive_context_vector(context_params)
        context_vec_output_dict['deep_context_vector'] = deep_context_vector
        context_vec_output_dict['global_context_vector'] = global_context_vector
        context_vec_output_dict['max_context_vector'] = max_context_vector
    elif context_mode == 10: #T
        context_vector, deep_context_vector, global_context_vector = create_combined_deep_global_max_self_attentive_context_vector(context_params)
        context_vec_output_dict['deep_context_vector'] = deep_context_vector
        context_vec_output_dict['global_context_vector'] = global_context_vector
    elif context_mode == 11: #T
        context_vector, deep_context_vector, global_context_vector = create_combined_vdeep_global_max_self_attentive_context_vector(context_params)
        context_vec_output_dict['deep_context_vector'] = deep_context_vector
        context_vec_output_dict['global_context_vector'] = global_context_vector
    elif context_mode == 12: #T
        context_vector = create_accumulative_most_attentive_context_vector(context_params)
    elif context_mode == 13: #T
        context_vector = create_text_cnn_context_vector(context_params)
    elif context_mode == 14: #T
        context_vector, deep_context_vector, global_context_vector, text_cnn_context_vector = create_combined_deep_global_text_cnn_context_vector(context_params)
        context_vec_output_dict['deep_context_vector'] = deep_context_vector
        context_vec_output_dict['global_context_vector'] = global_context_vector
        context_vec_output_dict['text_cnn_context_vector'] = text_cnn_context_vector 
    print(context_vector)
    #context_vector = tf.Print(context_vector, [context_vector, tf.reduce_mean(context_vector)], message = 'context_vector')
    context_vector = tf.verify_tensor_all_finite(context_vector, 'context vector verify')
    return context_vector, context_vec_output_dict
 
def multihead_attention(queries, 
                        keys, 
                        num_units=None, 
                        num_heads=8, 
                        dropout_rate=0,
                        is_training=True,
                        causality=False,
                        apply_context=False,
                        context_combine_mode=0,
                        project_context_mode=0,
                        context_mode=0,
                        context_layer_mask=0,
                        context_dropout=0,
                        enable_rel_pos=False,
                        rel_pos_emb=None,
                        user_seq=None,
                        apply_local_modeling=False,
                        multi_head_attn_head_combine=0,
                        batch_size=128,
                        input_layers = [],
                        conv_layers = [],
                        local_attention_layers = [],
                        mask= None,
                        local_kernel_size = 0,
                        attention_weights_per_layer = [],
                        enable_value_context= False,
                        use_multihead_context= False,
                        multihead_disagreent_reg = 0.0,
                        include_current_item= False,
                        scope="multihead_attention", 
                        reuse=tf.AUTO_REUSE,
                        with_qk=False):
    '''Applies multihead attention.
    
    Args:
      queries: A 3d tensor with shape of [N, T_q, C_q].
      keys: A 3d tensor with shape of [N, T_k, C_k].
      num_units: A scalar. Attention size.
      dropout_rate: A floating point number.
      is_training: Boolean. Controller of mechanism for dropout.
      causality: Boolean. If true, units that reference the future are masked. 
      num_heads: An int. Number of heads.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
        
    Returns
      A 3d tensor with shape of (N, T_q, C)  
    '''
    print('multihead_attention scope:{}'.format(scope))
    with tf.variable_scope(scope, reuse=reuse):
        output_dict = dict()

        # Set the fall back option for num_units
        if num_units is None:
            num_units = queries.get_shape().as_list[-1]
        assert_ops = []
        assert_op = tf.Assert(queries.get_shape().as_list()[0] == keys.get_shape().as_list()[0], [queries, keys])
        assert_ops.append(assert_op)
        # Linear projections
        # Q = tf.layers.dense(queries, num_units, activation=tf.nn.relu) # (N, T_q, C)
        # K = tf.layers.dense(keys, num_units, activation=tf.nn.relu) # (N, T_k, C)
        # V = tf.layers.dense(keys, num_units, activation=tf.nn.relu) # (N, T_k, C)
        Q = tf.layers.dense(queries, num_units, activation=None) # (N, T_q, C)
        K = tf.layers.dense(keys, num_units, activation=None) # (N, T_k, C)
        V = tf.layers.dense(keys, num_units, activation=None) # (N, T_k, C)
        
        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0) # (h*N, T_q, C/h) 
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0) # (h*N, T_k, C/h) 
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0) # (h*N, T_k, C/h) 

        query_shape = queries.get_shape().as_list()
        sequence_length = query_shape[1] # T_q
        dim = num_units 

        print('apply_context flag with scope:' + str(scope))
        print(apply_context)

        normalized_input_layers = [normalize(layer) for layer in input_layers]

        if apply_context:
            print('Apply context with context mode:' + str(context_mode) + ", scope:" + str(scope))
            # Each layer has shape: [N, T_k, C]
            input_layers_reshape = [
                tf.concat(tf.split(layer, num_heads, axis=2), axis=0)
                for layer in input_layers
            ]
            normalized_input_layers_reshape = [
                tf.concat(tf.split(layer, num_heads, axis=2), axis=0)
                for layer in normalized_input_layers
            ]
            conv_layers_reshape = [
                tf.concat(tf.split(layer, num_heads, axis=2), axis=0)
                for layer in conv_layers
            ]
            local_attention_layers_reshape = [
                tf.concat(tf.split(layer, num_heads, axis=2), axis=0)
                for layer in local_attention_layers 
            ]
            print('input_layers_reshape')
            print(input_layers)
            print(input_layers_reshape)
            print('conv_layers_reshape')
            print(conv_layers)
            print(conv_layers_reshape)
            print('local_attention_layers_reshape')
            print(local_attention_layers)
            print(local_attention_layers_reshape)

            tile_mask = tf.tile(mask, [num_heads, 1, 1])

            context_emb_size = num_units 

            if use_multihead_context:
                context_input_layer = input_layers
            else:
                context_input_layer = input_layers_reshape 
                context_emb_size = num_units / num_heads

            # Generate query context
            context_params = {'input_layer': normalized_input_layers_reshape, 'conv_layer': conv_layers_reshape,'local_attention_layers':local_attention_layers_reshape, 'mask':tile_mask, 'context_combine_mode': context_combine_mode, 'exclude_top_layer': False, 'project_context_mode': project_context_mode, 'user_emb': user_seq, 'context_dropout': context_dropout, 'attention_weights_per_layer': attention_weights_per_layer, 'l2_reg': context_l2_reg, 'context_emb_size': context_emb_size, 'batch_size': batch_size, 'scope': scope, 'include_currrent_item': include_current_item, 'is_training':is_training}
            print('query context vector params')
            print(context_params)
            context_vector, context_vec_output_dict = create_context_vector(context_mode, context_params)
            params = dict()
            params['l2_reg'] = context_l2_reg
            params['context_combine_mode'] = context_combine_mode
            params['project_context_mode'] = project_context_mode 
            params['context_dropout'] = context_dropout 
            params['use_multihead_context'] = use_multihead_context 
            params['num_heads'] = num_heads 
            params['context_emb_size'] = context_emb_size 
            params['batch_size'] = batch_size 
            params['mask'] = mask 
            params['is_training'] = is_training
            if 'deep_context_vector' in context_vec_output_dict:
                params['deep_context_vector'] = context_vec_output_dict['deep_context_vector']
            if 'global_context_vector' in context_vec_output_dict:
                params['global_context_vector'] = context_vec_output_dict['global_context_vector'] 
            Qc, Qf = model_context_for_q(Q_, context_vector, params,
                scope=scope+"_context_query", reuse=reuse)

            # Generate key context
            context_params = {'input_layer': input_layers_reshape, 'conv_layer': conv_layers_reshape,'local_attention_layers':local_attention_layers_reshape, 'mask':tile_mask, 'context_combine_mode': context_combine_mode, 'exclude_top_layer': False, 'project_context_mode': project_context_mode, 'user_emb': user_seq, 'context_dropout': context_dropout, 'attention_weights_per_layer': attention_weights_per_layer, 'l2_reg': context_l2_reg, 'context_emb_size': context_emb_size, 'batch_size': batch_size, 'scope': scope, 'include_currrent_item': include_current_item, 'is_training':is_training}
            print('key context vector params')
            print(context_params)
            context_vector, context_vec_output_dict = create_context_vector(context_mode, context_params)
            params = dict()
            params['l2_reg'] = context_l2_reg
            params['context_combine_mode'] = context_combine_mode
            params['project_context_mode'] = project_context_mode 
            params['context_dropout'] = context_dropout 
            params['use_multihead_context'] = use_multihead_context 
            params['num_heads'] = num_heads 
            params['context_emb_size'] = context_emb_size 
            params['batch_size'] = batch_size 
            params['mask'] = mask 
            params['is_training'] = is_training
            if 'deep_context_vector' in context_vec_output_dict:
                params['deep_context_vector'] = context_vec_output_dict['deep_context_vector']
            if 'global_context_vector' in context_vec_output_dict:
                params['global_context_vector'] = context_vec_output_dict['global_context_vector'] 
            Kc, Kf = model_context_for_k(K_, context_vector, params,
                scope=scope + "_context_key", reuse=reuse)

            if enable_value_context:
                Vc, Vf = model_context_for_v(V_, context_vector, params,
                    scope=scope + "_context_value", reuse=reuse)
            print("context_layer_mask at scope:" + str(scope))
            print(context_layer_mask)

            Q_ = (1 - context_layer_mask) * Q_ + context_layer_mask * Qc
            K_ = (1 - context_layer_mask) * K_ + context_layer_mask * Kc
            print("Context Q")
            print(Q_)
            print("Context K")
            print(K_)
            if enable_value_context:
                V_ = (1 - context_layer_mask) * V_ + context_layer_mask * Vc
                print("Context V")
                print(V_)
                output_dict['Vf'] = Vf
            output_dict['Qf'] = Qf
            output_dict['Kf'] = Kf
            print("output_dict")
            print(output_dict)

        # Multiplication
        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1])) # (h*N, T_q, T_k)

        print("multi_atten outputs")
        print(outputs)
        
        # Scale
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

        if apply_local_modeling:
            print('Apply local modeling:')
            G, _ = model_localness(Q_, {
                'l2_reg': context_l2_reg,
                'num_heads': num_heads,
            }, scope=scope + "_apply_local_modeling", reuse=reuse)
            print("G:")
            print(G)
            output_dict['G'] = G
            outputs += G

        if enable_rel_pos and rel_pos_emb is not None:
            print('Apply relative position embedding:')
            rel_pos_impact = cal_relative_pos_impact(Q_, rel_pos_emb, num_heads)
            print(rel_pos_impact)
            outputs += rel_pos_impact

        # Key Masking
        key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1))) # (N, T_k)
        key_masks = tf.tile(key_masks, [num_heads, 1]) # (h*N, T_k)
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1]) # (h*N, T_q, T_k)
        
        paddings = tf.ones_like(outputs)*(-2**32+1)
        outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs) # (h*N, T_q, T_k)
  
        # Causality = Future blinding
        if causality:
            diag_vals = tf.ones_like(outputs[0, :, :]) # (T_q, T_k)
            tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense() # (T_q, T_k)
            masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1]) # (h*N, T_q, T_k)
            paddings = tf.ones_like(masks)*(-2**32+1)
            outputs = tf.where(tf.equal(masks, 0), paddings, outputs) # (h*N, T_q, T_k)
 
        if local_kernel_size > 0:
            # Local Attention weights
            query_length_r = tf.expand_dims(tf.range(sequence_length), -1)

            key_length_c = tf.expand_dims(tf.range(sequence_length), 0)

            local_attention_masks = query_length_r <= key_length_c + local_kernel_size
            local_attention_masks = tf.to_int32(local_attention_masks)
            local_attention_masks = tf.expand_dims(local_attention_masks, 0)

            print('local_attention_masks')
            print(local_attention_masks)

            local_paddings = tf.ones_like(local_attention_masks)*(-2**32+1)
            local_paddings = tf.to_float(local_paddings)
            print('local_paddings')
            print(local_paddings)

            local_attention_weights = tf.where(tf.equal(local_attention_masks, 0), local_paddings, outputs)
            print('local_attention_weights')
            print(local_attention_weights)
            local_attention_weights = tf.Print(local_attention_weights, [local_attention_weights, tf.shape(local_attention_weights), tf.reduce_mean(local_attention_weights)], name = 'local_attention_weights')
 
        # Activation
        outputs = tf.nn.softmax(outputs) # (h*N, T_q, T_k)

        attention_weights = tf.identity(outputs, name= 'attention_weights')
        variable_summaries(attention_weights, 'attention_weights_' + str(scope)) 
        output_dict['attention_weights'] = attention_weights

        if local_kernel_size > 0:
            local_attention_weights = tf.nn.softmax(local_attention_weights)  # (h*N, T_q, T_k)
 
        # Query Masking
        query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1))) # (N, T_q)
        query_masks = tf.tile(query_masks, [num_heads, 1]) # (h*N, T_q)
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]]) # (h*N, T_q, T_k)
        outputs *= query_masks # broadcasting. (N, T_q, C)

        if local_kernel_size > 0:
            local_attention_weights *= query_masks
          
        # Dropouts
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))
        if local_kernel_size > 0:
            local_attention_weights = tf.layers.dropout(local_attention_weights, rate=dropout_rate, training=tf.convert_to_tensor(is_training))

        # Attention weights
        attn_weights = outputs        
               
        # Weighted sum
        outputs = tf.matmul(outputs, V_) # ( h*N, T_q, C/h)

        print("outputs_v")
        print(outputs)

        variable_summaries(outputs, 'atten_outputs_' + str(scope)) 

        print("multihead_disagreent_reg")
        print(multihead_disagreent_reg)
        if multihead_disagreent_reg > 0.0 and num_heads > 1:
            print("Enable multihead_disagreement_reg:" + str(multihead_disagreent_reg))
            mut_head_dis_reg = mutual_head_disagreement_regularization(outputs, mask, num_heads)
            mut_head_dis_reg *= multihead_disagreent_reg
            tf.summary.scalar('mut_head_dis_reg_%sd' % scope, mut_head_dis_reg)
            mut_head_dis_reg = tf.Print(mut_head_dis_reg, [mut_head_dis_reg], message = 'mut_head_dis_reg_%s' % scope)
            output_dict['multihead_disagreent'] = mut_head_dis_reg 

        if local_kernel_size > 0:
            local_outputs = tf.matmul(local_attention_weights, V_)
            print('local_outputs')
            print(local_outputs)
            local_outputs = tf.Print(local_outputs, [local_outputs, tf.shape(local_outputs), tf.reduce_mean(local_outputs)], name = 'local_outputs') 
            output_dict['local_attention_weights'] = local_attention_weights
            output_dict['local_attention_outputs'] = local_outputs

        outputs_shape = outputs.get_shape().as_list()

        query_shape = queries.get_shape().as_list()
        batch_size = query_shape[0] # N
        sequence_length = outputs_shape[1] # T_q
        dim = outputs_shape[-1]  # C/h
        # Restore shape
        if multi_head_attn_head_combine & 1:
           print("Apply SE upon the outputs")
           # apply squeeze-excitation upon multiple heads
           # [n*t_q, 1, c/h, h]
           outputs_reshape = tf.reshape(outputs,
              [-1, 1, num_heads, dim])
           # [n*t_q, 1, c/h, h]
           se_outputs = squeeze_excitation_layer(outputs_reshape, 4, layer_name = scope)
           # [h*N, T_q, C/h]
           outputs = tf.reshape(se_outputs, [-1, sequence_length, dim]) 

        # Bi-interation
        if multi_head_attn_head_combine & 2:
           print('apply bi-interation')
           # [N*T_q, h, C/h]
           outputs_reshape = tf.reshape(outputs, [-1, num_heads, dim])
           print('outputs_reshape')
           print(outputs_reshape)
           # [N*T_q, C/h, h]
           outputs_tr = tf.transpose(outputs_reshape, [0, 2, 1])
           print('outputs_tr')
           print(outputs_tr)
           # [N*T_q, h, h]
           outputs_cross = tf.matmul(outputs_reshape, outputs_tr)
           outputs_cross = tf.nn.softmax(outputs_cross)
           print('outputs_cross')
           print(outputs_cross)
           # [N*T_q, h, C/h]
           outputs_comb = tf.matmul(outputs_cross, outputs_reshape)
           print('outputs_comb')
           print(outputs_comb)
           outputs = tf.reshape(outputs_comb, [-1, sequence_length, dim])

        if multi_head_attn_head_combine & 4:
           print("Apply SE upon the inputs for residual SE")
           project_query = tf.concat(tf.split(Q, num_heads, axis=2), axis=0) # (h*N, T_q, C/h) 
           # [N*T_q, 1, h, C/h]
           inputs_reshape = tf.reshape(project_query,
              [-1, 1, num_heads, dim])
           # [N*T_q, 1, h, C/h]
           se_outputs = squeeze_excitation_layer(inputs_reshape, 8, layer_name = scope + "_SE_PQ")
           # [h*N, T_q, C/h]
           project_query = tf.reshape(se_outputs, [-1, sequence_length, dim]) 
           queries = tf.concat(tf.split(project_query, num_heads, axis=0), axis=2 ) # (N, T_q, C)

        if multi_head_attn_head_combine & 32:
           print('Apply deep-cross upon multiple heads')
           # [N*T_q, h, C/h]
           outputs_reshape = tf.reshape(outputs, [-1, num_heads, dim])
           dcn_out = build_deep_cross_interaction_model(outputs_reshape, tf.contrib.training.HParams(num_cross_layers=1), scope + "_DCN_MA") 
           dcn_out = tf.reshape(dcn_out, [-1, sequence_length, dim])
           outputs = dcn_out
        
        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2 ) # (N, T_q, C)
        assert_op = tf.Assert(queries.get_shape().as_list()[0] == outputs.get_shape().as_list()[0], [queries, outputs])
        assert_ops.append(assert_op)
              
        # Residual connection
        outputs += queries

        # Normalize
        #outputs = normalize(outputs) # (N, T_q, C)
        #output_dict['attn_weights'] = attn_weights
    with tf.control_dependencies(assert_ops):  
       if with_qk: return Q,K,output_dict
       else: return outputs,output_dict

def feedforward(inputs, 
                num_units=[2048, 512],
                scope="multihead_attention", 
                dropout_rate=0.2,
                is_training=True,
                reuse=None):
    '''Point-wise feed forward net.
    
    Args:
      inputs: A 3d tensor with shape of [N, T, C].
      num_units: A list of two integers.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
        
    Returns:
      A 3d tensor with the same shape and dtype as inputs
    '''
    print('feedforward scope:{}, reuse:{}'.format(scope, reuse))
    with tf.variable_scope(scope, reuse=reuse):
        # Inner layer
        params = {"inputs": inputs, "filters": num_units[0], "kernel_size": 1,
                  "activation": tf.nn.relu, "use_bias": True}
        outputs = tf.layers.conv1d(**params)
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))
        # Readout layer
        params = {"inputs": outputs, "filters": num_units[1], "kernel_size": 1,
                  "activation": None, "use_bias": True}
        outputs = tf.layers.conv1d(**params)
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))
        
        # Residual connection
        outputs += inputs
        
        # Normalize
        #outputs = normalize(outputs)
    
    return outputs

def feedforward_with_routing(inputs, masks,
                             num_heads,
                             batch_size,
                             num_units=[2048, 512],
                             scope="multihead_attention",
                             dropout_rate=0.2,
                             is_training=True,
                             reuse=None
                             ):
    with tf.variable_scope(scope, reuse=reuse):
        # Inner layer non-linear transformations
        inputs_shape = inputs.get_shape().as_list()
        sequence_length = inputs_shape[1]

        params = {"inputs": inputs, "filters": num_units[0], "kernel_size": 1,
                  "activation": tf.nn.relu, "use_bias": True}
        outputs = tf.layers.conv1d(**params)
        # [B, L, D]
        outputs = tf.layers.dropout(outputs, rate=dropout_rate,
                                    training=tf.convert_to_tensor(is_training))

        # {[B, L, D/heads]}
        output_splits = tf.split(outputs, num_heads, axis=-1)

        tf.logging.info('feeddforward with routing output_splits')
        tf.logging.info(output_splits)

        output_splits = [
            tf.reshape(out, [-1, num_units[-1] / num_heads]) for out in output_splits]

        output_cap_info = combine_representations_by_dynamic_routing_aggrement(
            output_splits, batch_size * sequence_length, 2, num_heads, num_units[-1] / num_heads,
            apply_layer_transform=False)
        out_caps = output_cap_info[0]
        tf.logging.info('feedforward with routing output_caps')
        tf.logging.info(out_caps)
#        out_caps = tf.Print(out_caps, [tf.reduce_mean(out_caps), tf.shape(out_caps)], message = 'out_caps', summarize=100)
        tf.logging.info('sequence_length:' + str(sequence_length))
        tf.logging.info('output_dim:' + str(num_units[-1]))
        outputs = tf.reshape(out_caps, [-1, sequence_length, num_units[-1]])
        tf.logging.info(outputs)
#        outputs= tf.Print(outputs, [tf.reduce_mean(outputs), tf.shape(outputs)], message = 'outputs', summarize=100)
        variable_summaries(outputs, 'multihead_outputs_' + str(scope))

        # Residual connection
        outputs += inputs

        # Normalize
        # outputs = normalize(outputs)

    return outputs

def model_context_for_q(query, context_vector, params, scope,
                        reuse=None):
    atten_size = query.get_shape().as_list()[-1]
    l2_reg = params['l2_reg']
    context_combine_mode = params.get('context_combine_mode') or 0
    project_context_mode = params.get('project_context_mode') or 0
    context_dropout = params.get('context_dropout') or 0.0
    use_multihead_context = params.get('use_multihead_context') or False
    batch_size = params.get('batch_size')
    context_emb_size = params.get('context_emb_size') or atten_size 
    mask = params['mask']
    model_context_params = {
        'atten_size': atten_size,
        'l2_reg': l2_reg,
        'context_combine_mode': context_combine_mode,
        'context_vector': context_vector,
        'project_context_mode': project_context_mode,
        'context_dropout': context_dropout,
        'use_multihead_context': use_multihead_context,
        'batch_size': batch_size,
        'context_emb_size': context_emb_size,
        'is_training': params['is_training'],
        'mask': mask,
    }
    if 'deep_context_vector' in params:
        model_context_params['deep_context_vector'] = params['deep_context_vector']
    if 'global_context_vector' in params:
        model_context_params['global_context_vector'] = params['global_context_vector']
    print('model_context_params')
    print(model_context_params)
    return model_context(query, model_context_params, name='model_query_context',
                         scope = scope, reuse=reuse)

def model_context_for_k(key, context_vector, params, scope,
                        reuse=None):
    atten_size = key.get_shape().as_list()[-1]
    l2_reg = params['l2_reg']
    context_combine_mode = params.get('context_combine_mode') or 0
    project_context_mode = params.get('project_context_mode') or 0
    context_dropout = params.get('context_dropout') or 0.0
    use_multihead_context = params.get('use_multihead_context') or False
    batch_size = params.get('batch_size')
    context_emb_size = params.get('context_emb_size') or atten_size 
    mask = params['mask']
    model_context_params = {
        'atten_size': atten_size,
        'l2_reg': l2_reg,
        'context_combine_mode': context_combine_mode,
        'context_vector': context_vector,
        'project_context_mode': project_context_mode,
        'context_dropout': context_dropout,
        'use_multihead_context': use_multihead_context,
        'batch_size': batch_size,
        'context_emb_size': context_emb_size,
        'is_training': params['is_training'],
        'mask': mask,
    }
    if 'deep_context_vector' in params:
        model_context_params['deep_context_vector'] = params['deep_context_vector']
    if 'global_context_vector' in params:
        model_context_params['global_context_vector'] = params['global_context_vector']
    return model_context(key, model_context_params, name='model_key_context',
                         scope = scope, reuse=reuse)

def model_context_for_v(key, context_vector, params, scope,
                        reuse=None):
    atten_size = key.get_shape().as_list()[-1]
    l2_reg = params['l2_reg']
    context_combine_mode = params.get('context_combine_mode') or 0
    project_context_mode = params.get('project_context_mode') or 0
    context_dropout = params.get('context_dropout') or 0.0
    model_context_params = {
        'atten_size': atten_size,
        'l2_reg': l2_reg,
        'context_combine_mode': context_combine_mode,
        'context_vector': context_vector,
        'context_dropout': context_dropout,
        'is_training': params['is_training'],
        'project_context_mode': project_context_mode
    }
    if 'deep_context_vector' in params:
        model_context_params['deep_context_vector'] = params['deep_context_vector']
    if 'global_context_vector' in params:
        model_context_params['global_context_vector'] = params['global_context_vector']
    return model_context(key, model_context_params, name='model_value_context',
                         scope = scope, reuse=reuse)


    print('model_context_params')
    print(model_context_params)
    return model_context(query, model_context_params, name='model_query_context',
                         scope = scope, reuse=reuse)

def create_deep_context_vector(params):
    input_layers = params['input_layer']
    layers_below = input_layers[0:-1]
    # [batch size, seq length, 1]
    mask = params['mask']
    print("mask from params")
    print(mask)
    context_vectors = []
    for layer in layers_below:
        # layer has shape: [batch size, seq length, emb dim ]
        print("deep context layer")
        print(layer)
        context_vectors.append(layer)
    # [B, L, (l-1)*D]
    return tf.concat(context_vectors, -1)

def create_global_deep_context_vector(params):
    layers_below = params['input_layer']
    if 'exclude_top_layer' in params and params['exclude_top_layer']:
        layers_below = layers_below[0:-1]
    # [batch size, seq length, 1]
    mask = params['mask']
    print("mask from params")
    print(mask)
    context_vectors = []
    for layer in layers_below:
        # layer has shape: [batch size, seq length, emb dim ]
        print("global deep context layer")
        print(layer)
        layer_mean = tf.div(tf.reduce_sum(layer,1) , (tf.reduce_sum(mask, 1)))
        print(layer_mean)
        context_vectors.append(layer_mean)
    context_vector_exp = tf.expand_dims(tf.concat(context_vectors, -1), -2)
    layer = layers_below[0]
    layer_shape = layer.get_shape().as_list()
    context_vector_tile = tf.tile(context_vector_exp, [1, layer_shape[1], 1])
    return context_vector_tile

def create_accumulative_global_deep_context_vector(params):
    layers_below = params['input_layer']
    if 'exclude_top_layer' in params and params['exclude_top_layer']:
        layers_below = layers_below[0:-1]
    # [batch size, seq length, 1]
    mask = params['mask']
    print("mask from params")
    print(mask)
    mask_cumsum = tf.cumsum(mask, 1)
    print("mask_cumsum from params")
    print(mask_cumsum)
    print('create_accumulative_global_deep_context_vector params')
    print(params)

    context_vectors = []
    for layer in layers_below:
        # layer has shape: [batch size, seq length, emb dim ]
        print("create_accumulative_global_deep_context_vector layer")
        print(layer)
        # accumulative sum upon each position [B, L, D]
        mask_t = tf.cast(mask, layer.dtype)
        layer = layer * mask_t
        mask_cumsum_t = tf.where(tf.equal(mask_cumsum, 0),
                                 tf.ones_like(mask_cumsum),
                                 mask_cumsum)
        mask_cumsum_t = tf.cast(mask_cumsum_t, layer.dtype)
        #layer= tf.Print(layer, [tf.reduce_mean(layer), tf.reduce_min(layer), tf.reduce_max(layer), tf.shape(layer)], message = 'layer_', summarize=100)
        layer_cumsum = tf.cumsum(layer,1)
        layer_mean = tf.div(layer_cumsum , mask_cumsum_t)
        print(layer_mean)
        #layer_mean = tf.Print(layer_mean, [tf.reduce_mean(layer_mean), tf.reduce_min(layer_mean), tf.reduce_max(layer_mean), tf.shape(layer_mean)], message = 'layer_mean_', summarize=100)
        # [B,L,D]
        if 'include_currrent_item' in params and params['include_currrent_item']:
            print("create_accumulative_global_deep_context_vector include_current_item in layer_mean")
            context_vector = layer_mean
        else:
            context_vector = layer_mean - layer
        context_vectors.append(context_vector)
    return tf.concat(context_vectors, -1)

def create_accumulative_attentive_global_deep_context_vector(params):
    print('create_accumulative_attentive_global_deep_context_vector')
    layers_below = params['input_layer']
    last_layer = layers_below[-1]
    layers_below = layers_below[0:-1]
    # [batch size, seq length, 1]
    mask = params['mask']
    print("mask from params")
    print(mask)
    mask_cumsum = tf.cumsum(mask, 1)
    print("mask_cumsum from params")
    print(mask_cumsum)

    layer_shape = last_layer.get_shape().as_list()
    batch_size = layer_shape[0]
    sequence_length = layer_shape[1]

    context_vectors = []
    for layer in layers_below:
        # layer has shape: [batch size, seq length, emb dim ]
        print("create_accumulative_attentive_global_deep_context_vector layer")
        print(layer)
        # accumulative sum upon each position [B, L, D]
        mask_t = tf.cast(mask, layer.dtype)
        layer = layer * mask_t

        partial_layer_max_list = [
            tf.ones_like(layer[:, 0:1, :]) * 0.0]

        for idx in xrange(1, sequence_length):
            atten_query = last_layer[:, idx:idx+1, ]
            partial_layer = layer[:, 0:idx, :]
            partial_mask = mask[:, 0:idx, :]
            dot_products = atten_query * partial_layer
            dot_products_norm = tf.reduce_sum(dot_products, -1, keep_dims=True)
            dot_products_norm = tf.where(partial_mask > 0, dot_products_norm,
                                         tf.ones_like(
                                             dot_products_norm) * dot_products_norm.dtype.min)
            dot_products_norm = tf.nn.softmax(dot_products_norm)
            dot_products_sum = partial_layer * dot_products_norm
            dot_products_mean = tf.reduce_mean(dot_products_sum, 1, keep_dims=True)
            partial_layer_max_list.append(dot_products_mean)

        context_vector = tf.concat(partial_layer_max_list, 1)
        print('context_vector')
        print(context_vector)
        context_vectors.append(context_vector)

    mask_t = tf.cast(mask, layer.dtype)
    layer = last_layer * mask_t

    mask_cumsum_t = tf.where(tf.equal(mask_cumsum, 0),
                             tf.ones_like(mask_cumsum),
                             mask_cumsum)
    mask_cumsum_t = tf.cast(mask_cumsum_t, layer.dtype)
    # layer= tf.Print(layer, [tf.reduce_mean(layer), tf.reduce_min(layer),
    # tf.reduce_max(layer), tf.shape(layer)], message = 'layer_', summarize=100)
    layer_cumsum = tf.cumsum(layer, 1)
    layer_mean = tf.div(layer_cumsum, mask_cumsum_t)
    print(layer_mean)
    # layer_mean = tf.Print(layer_mean, [tf.reduce_mean(layer_mean),
    # tf.reduce_min(layer_mean), tf.reduce_max(layer_mean), tf.shape(
    # layer_mean)], message = 'layer_mean_', summarize=100)
    # [B,L,D]

    if 'include_current_item' in params and params['include_current_item']:
        print("create_accumulative_attentive_global_deep_context_vector include_current_item in layer_mean")
        context_vector = layer_mean
    else:
        context_vector = layer_mean - layer
    context_vectors.append(context_vector)

    return tf.concat(context_vectors, -1)

def create_accumulative_max_deep_context_vector(params):
    layers_below = params['input_layer']
    #if 'exclude_top_layer' in params and params['exclude_top_layer']:
    layers_below = layers_below[0:-1]
    # [batch size, seq length, 1]
    mask = params['mask']
    print("mask from params")
    print(mask)

    attention_weights_per_layer = params['attention_weights_per_layer']
    print("attention_weights_per_layer from params")
    print(attention_weights_per_layer) # [B, L, L]

    context_vectors = []
    for layer_idx, layer in enumerate(layers_below):
        # layer has shape: [batch size, seq length, emb dim ]
        print("create_accumulative_max_deep_context_vector layer")
        print(layer)

        attention_weights  = attention_weights_per_layer[layer_idx]

        print('attention_weights at layer:' + str(layer_idx))
        print(attention_weights)

        layer_shape = layer.get_shape().as_list()
        sequence_length = layer_shape[1]
        emb_dim = layer_shape[-1]

        partial_layer_max_list = [tf.ones_like(layer[:, 0:1, :]) * 0.0]

        print("initial partial max layer at idx:" + str(0))
        print(partial_layer_max_list)

        print("sequence_length")
        print(sequence_length)

        for idx in xrange(1, sequence_length):
            # [B, K, D]
            partial_layer = layer[:, 0:idx, :]
            # [B, K]
            partial_mask = mask[:, 0:idx]
            # [B, K]
            partial_attention_weights = attention_weights[:, idx, 0:idx]
            print("partial_layer at layer : {}, idx:{}".format(layer_idx, idx))
            print(partial_layer)
            print("partial_mask at layer : {}, idx:{}".format(layer_idx, idx))
            print(partial_mask)
            print("partial_attention_weights at layer : {}, idx:{}".format(layer_idx, idx))
            print(partial_attention_weights)
            partial_mask_tiled = tf.tile(partial_mask, [1,1,emb_dim])
            # [B, K , 1]
            partial_weights_tiled = tf.expand_dims(partial_attention_weights, -1)
            paddings = tf.ones_like(partial_layer) * partial_layer.dtype.min
            partial_layer_masked = tf.where(partial_mask_tiled > 0, partial_layer * partial_weights_tiled, paddings)
            partial_layer_max = tf.reduce_max(partial_layer_masked, -2, keep_dims=True)
            print("partial_layer_max idx:" + str(idx))
            print(partial_layer_max)
            partial_layer_max = tf.where(tf.equal(partial_layer_max , partial_layer.dtype.min), tf.zeros_like(partial_layer_max), partial_layer_max)
            partial_layer_max_list.append(partial_layer_max)

        partial_layer_max_context = tf.concat(partial_layer_max_list, -2)
        # [B,L,D]
        context_vectors.append(partial_layer_max_context)
    return tf.concat(context_vectors, -1)

def create_accumulative_most_attentive_context_vector(params):
    layers_below = params['input_layer']
    last_layer = layers_below[-1]
    layers_below = layers_below[0:-1]
    # [batch size, seq length, 1]
    mask = params['mask']
    print("mask from params")
    print(mask)

    attention_weights_per_layer = params['attention_weights_per_layer']
    print("attention_weights_per_layer from params")
    print(attention_weights_per_layer) # [B, L, L]

    context_vectors = []
    for layer_idx, layer in enumerate(layers_below):
        # layer has shape: [batch size, seq length, emb dim ]
        print("create_accumulative_most_attentive_context_vector layer")
        print(layer)

        layer_shape = layer.get_shape().as_list()
        sequence_length = layer_shape[1]
        emb_dim = layer_shape[-1]

        partial_layer_max_list = [tf.ones_like(layer[:, 0:1, :]) * 0.0]

        print("initial partial max layer at idx:" + str(layer_idx))
        print(partial_layer_max_list)

        print("sequence_length")
        print(sequence_length)

        attention_weights  = attention_weights_per_layer[layer_idx]

        print('attention_weights at layer:' + str(layer_idx))
        print(attention_weights)

        for idx in xrange(1, sequence_length):
            partial_attention_weights = attention_weights[:, idx, 0:idx]
            print("partial_attention_weights at layer:" + str(layer_idx) + "_" + str(idx))
            print(partial_attention_weights)
            # [B, L]
            top_k = 1

            # shape of indices:[B,]
            (values, indices) = tf.nn.top_k(partial_attention_weights, top_k)

            print("top indices at layer:" + str(layer_idx) + "_" + str(idx))
            print(indices)
            #indices = tf.Print(indices, [indices, tf.shape(indices)], message = 'indices_' + str(layer_idx) + "_" + str(idx), summarize=100)

            partial_layer = layer[:, 0:idx, :]
            partial_mask = mask[:, 0:idx, :]
            print("partial_layer at layer:" + str(layer_idx) + "_" + str(idx))
            print(partial_layer)
            print("partial_mask at layer:" + str(layer_idx) + "_" + str(idx))
            print(partial_mask)

            gathered_layer = gather_3d_along_seq_length_axis(
                partial_layer, indices)
            print("gathered_layer at layer:" + str(layer_idx) + "_" + str(idx))
            print(gathered_layer)
            gathered_mask = gather_3d_along_seq_length_axis(
                partial_mask, indices)
            print("gathered_mask at layer:" + str(layer_idx) + "_" + str(idx))
            print(gathered_mask)

            gathered_mask_tiled = tf.tile(gathered_mask, [1,emb_dim])
            paddings = tf.ones_like(gathered_layer) * 0.0
            gathered_layer_masked = tf.where(gathered_mask_tiled > 0,
                                             gathered_layer, paddings)
            print("gathered_layer_masked at layer:" + str(layer_idx) + "_" + str(idx))
            print(gathered_layer_masked)

            #gathered_layer_masked = gathered_layer_masked * values

            gathered_layer_masked = tf.expand_dims(gathered_layer_masked, 1)

            partial_layer_max_list.append(gathered_layer_masked)

        context_vector  = tf.concat(partial_layer_max_list, 1)

        print('context_vector at layer: '+ str(layer_idx))
        print(context_vector)

        context_vectors.append(context_vector)


    layer_shape = last_layer.get_shape().as_list()
    sequence_length = layer_shape[1]
    emb_dim = layer_shape[-1]

    partial_layer_max_list = [tf.ones_like(layer[:, 0:1, :]) * 0.0]

    for idx in xrange(1, sequence_length):
        partial_layer = last_layer[:, 0:idx, :]
        partial_mask = mask[:, 0:idx, :]
        print("last at layer:" + "_" + str(idx))
        print(partial_layer)
        print("last_mask at layer:" + "_" + str(idx))
        print(partial_mask)

        current_spot = last_layer[:, idx:idx+1, :]
        print("current spot at idx:" + str(idx))
        print(current_spot)
        dot_products = current_spot * partial_layer
        print("dot_products at idx:" + str(idx))
        print(dot_products)
        dot_products_norm = tf.reduce_sum(dot_products, -1, keep_dims=True)
        dot_products_norm = tf.where(partial_mask > 0, dot_products_norm,
                                     tf.ones_like(dot_products_norm) * dot_products_norm.dtype.min)
        dot_products_norm = tf.squeeze(dot_products_norm, -1)
        dot_products_norm = tf.nn.softmax(dot_products_norm)
        print("dot_products_norm at idx:" + str(idx))
        print(dot_products_norm)

        top_k = 1
        (values, indices) = tf.nn.top_k(dot_products_norm, top_k)

        gathered_layer = gather_3d_along_seq_length_axis(
            partial_layer, indices)
        print("gathered_layer at last layer:" + "_" + str(idx))
        print(gathered_layer)
        gathered_mask = gather_3d_along_seq_length_axis(
            partial_mask, indices)
        print("gathered_mask at last layer:"  + "_" + str(idx))
        print(gathered_mask)

        gathered_mask_tiled = tf.tile(gathered_mask, [1, emb_dim])
        paddings = tf.ones_like(gathered_layer) * 0.0
        gathered_layer_masked = tf.where(gathered_mask_tiled > 0,
                                         gathered_layer, paddings)
        gathered_layer_masked = tf.expand_dims(gathered_layer_masked, 1)
        partial_layer_max_list.append(gathered_layer_masked)

    context_vector = tf.concat(partial_layer_max_list, 1)

    print("context_vector")
    print(context_vector)

    context_vectors.append(context_vector)

    return tf.concat(context_vectors, -1)

def create_accumulative_max_self_attentional_global_deep_context_vector(params):
    layers_below = params['input_layer']
    if 'exclude_top_layer' in params and params['exclude_top_layer']:
        layers_below = layers_below[0:-1]
    first_layer = layers_below[0]
    # [batch size, seq length, 1]
    mask = params['mask']
    print("mask from params")
    print(mask)
    mask_cumsum = tf.cumsum(mask, 1)
    print("mask_cumsum from params")
    print(mask_cumsum)

    layer_shape = first_layer.get_shape().as_list()
    sequence_length = layer_shape[1]

    mask_t = tf.cast(mask, first_layer.dtype)

    first_layer = first_layer * mask_t

    context_vectors = []

    for layer in layers_below:
        # layer has shape: [batch size, seq length, emb dim ]
        print("create_accumulative_max_self_attentional_global_deep_context_vector layer")
        print(layer)
        # accumulative sum upon each position [B, L, D]
        layer = layer * mask_t

        partial_layer_list = [tf.ones_like(layer[:, 0:1, :]) * 0.0]

        for idx in xrange(1, sequence_length):
            partial_layer = layer[:, 0:idx, :]
            partial_mask = mask[:, 0:idx, :]
            partial_mask = tf.squeeze(partial_mask, -1)

            partial_first_layer = first_layer[:, 0:idx, :]
            partial_first_layer_max = tf.reduce_max(partial_first_layer, 1,
                                                    keep_dims=True)

            dot_product = partial_layer * partial_first_layer_max
            print("dot_product")
            print(dot_product)
            dot_product_sum = tf.reduce_sum(dot_product, -1)
            print("dot_product_sum")
            print(dot_product_sum)
            dot_product_mask = tf.where(partial_mask > 0, dot_product_sum,
                     tf.ones_like(dot_product_sum) * dot_product_sum.dtype.min)
            dot_product_norm = tf.nn.softmax(dot_product_mask)
            dot_product_norm = tf.expand_dims(dot_product_norm, -1)
            attentive_context = partial_layer * dot_product_norm
            attentive_context = tf.reduce_sum(attentive_context, 1, keep_dims=True)
            partial_layer_list.append(attentive_context)

        context_vector = tf.concat(partial_layer_list, 1)

        context_vectors.append(context_vector)

    return tf.concat(context_vectors, -1)

def create_accumulative_self_attentional_global_deep_context_vector(params):
    layers_below = params['input_layer']
    if 'exclude_top_layer' in params and params['exclude_top_layer']:
        layers_below = layers_below[0:-1]
    first_layer = layers_below[0]
    # [batch size, seq length, 1]
    mask = params['mask']
    print("mask from params")
    print(mask)
    mask_cumsum = tf.cumsum(mask, 1)
    print("mask_cumsum from params")
    print(mask_cumsum)

    layer_shape = first_layer.get_shape().as_list()
    batch_size = layer_shape[0]
    sequence_length = layer_shape[1]
    atten_size = layer_shape[-1]

    mask_t = tf.cast(mask, first_layer.dtype)

    first_layer = first_layer * mask_t

    context_vectors = []
    l2_reg = params['l2_reg']
    scope = params['scope']

    with tf.variable_scope(scope + '_acc_self_attentive_global_context', reuse=tf.AUTO_REUSE):
        vh = tf.get_variable("_acc_self_vh", dtype=first_layer.dtype,
                             shape=[8, atten_size],
                             regularizer=tf.contrib.layers.l2_regularizer(l2_reg))
        wh = tf.get_variable("_acc_self_wh", dtype=first_layer.dtype,
                             shape=[1, 8],
                             regularizer=tf.contrib.layers.l2_regularizer(l2_reg))

    for layer in layers_below:
        # layer has shape: [batch size, seq length, emb dim ]
        print("create_accumulative_self_attentional_global_deep_context_vector layer")
        print(layer)
        # accumulative sum upon each position [B, L, D]
        layer = layer * mask_t

        partial_layer_list = [tf.ones_like(layer[:, 0:1, :]) * 0.0]

        for idx in xrange(1, sequence_length):
            partial_layer = layer[:, 0:idx, :]
            partial_mask = mask[:, 0:idx, :]
            partial_mask = tf.squeeze(partial_mask, -1)

            partial_layer_reshape = tf.reshape(partial_layer, [-1, atten_size])

            partial_layer_reshape = tf.transpose(partial_layer_reshape, [1, 0])

            partial_layer_mul = tf.matmul(vh, partial_layer_reshape)

            partial_layer_mul = tf.nn.tanh(partial_layer_mul)

            partial_layer_mul = tf.matmul(wh, partial_layer_mul)

            print('partial_layer_mul')
            print(partial_layer_mul)          

            partial_layer_mul = tf.reshape(partial_layer_mul, [-1, idx])

            partial_layer_padding = tf.ones_like(partial_layer_mul) * partial_layer_mul.dtype.min

            partial_layer_logits = tf.where(partial_mask > 0, partial_layer_mul, partial_layer_padding)

            partial_layer_sm = tf.nn.softmax(partial_layer_logits)

            partial_layer_sm = tf.expand_dims(partial_layer_sm, -1)

            partial_layer_masked = partial_layer_sm * partial_layer

            partial_layer_sum = tf.reduce_sum(partial_layer_masked, 1, keep_dims=True)
 
            print("partial_layer_sum")
            print(partial_layer_sum)

            partial_layer_list.append(partial_layer_sum)

        context_vector = tf.concat(partial_layer_list, 1)

        context_vectors.append(context_vector)

    return tf.concat(context_vectors, -1)

def create_accumulative_vdeep_self_attentional_global_deep_context_vector(params):
    layers_below = params['input_layer']
    if 'exclude_top_layer' in params and params['exclude_top_layer']:
        layers_below = layers_below[0:-1]
    last_layer = layers_below[-1]

    # [batch size, seq length, 1]
    mask = params['mask']
    print("mask from params")
    print(mask)
    mask_cumsum = tf.cumsum(mask, 1)
    print("mask_cumsum from params")
    print(mask_cumsum)

    layer_shape = last_layer.get_shape().as_list()
    sequence_length = layer_shape[1]

    mask_t = tf.cast(mask, last_layer.dtype)

    last_layer = last_layer * mask_t

    context_vectors = []

    layers_below = layers_below[0:-1]

    for layer in layers_below:
        # layer has shape: [batch size, seq length, emb dim ]
        print("create_accumulative_vdeep_self_attentional_global_deep_context_vector layer")
        print(layer)
        # accumulative sum upon each position [B, L, D]
        layer = layer * mask_t

        partial_layer_list = [tf.concat([tf.ones_like(layer[:, 0:1, :]) * 0.0] * 2, -1)]

        for idx in xrange(1, sequence_length):
            partial_layer = layer[:, 0:idx, :]
            partial_mask = mask[:, 0:idx, :]
            partial_mask_sq = tf.squeeze(partial_mask, -1)

            partial_layer_shape = partial_layer.get_shape().as_list()
            emb_dim = partial_layer_shape[-1]

            partial_last_layer = last_layer[:, idx:idx+1, :]

            dot_product = partial_layer * partial_last_layer
            print("dot_product")
            print(dot_product)
            dot_product_sum = tf.reduce_sum(dot_product, -1)
            print("dot_product_sum")
            print(dot_product_sum)
            dot_product_mask = tf.where(partial_mask_sq > 0, dot_product_sum,
                     tf.ones_like(dot_product_sum) * dot_product_sum.dtype.min)
            dot_product_norm = tf.nn.softmax(dot_product_mask)
            dot_product_norm_exp = tf.expand_dims(dot_product_norm, -1)
            attentive_context = partial_layer * dot_product_norm_exp
            aggregated_attentive_context = tf.reduce_sum(attentive_context, 1,
                                                         keep_dims=True)

            top_k = 1
            (values, indices) = tf.nn.top_k(dot_product_norm, top_k)

            gathered_layer = gather_3d_along_seq_length_axis(
                partial_layer, indices)
            print("gathered_layer at layer:" + "_" + str(idx))
            print(gathered_layer)
            gathered_mask = gather_3d_along_seq_length_axis(
                partial_mask, indices)
            print("gathered_mask at  layer:" + "_" + str(idx))
            print(gathered_mask)

            gathered_mask_tiled = tf.tile(gathered_mask, [1, emb_dim])
            paddings = tf.ones_like(gathered_layer) * 0.0
            gathered_layer_masked = tf.where(gathered_mask_tiled > 0,
                                             gathered_layer, paddings)
            max_attentive_context = tf.expand_dims(gathered_layer_masked, 1)

            attentive_context = tf.concat([aggregated_attentive_context,
                                           max_attentive_context], -1)

            print("attentive_context at  layer:" + "_" + str(idx))
            print(attentive_context)

            partial_layer_list.append(attentive_context)

        context_vector = tf.concat(partial_layer_list, 1)

        context_vectors.append(context_vector)

    # The last layer is simple mean pooling
    mask_cumsum_t = tf.where(tf.equal(mask_cumsum, 0),
                             tf.ones_like(mask_cumsum),
                             mask_cumsum)
    mask_cumsum_t = tf.cast(mask_cumsum_t, last_layer.dtype)
    # layer= tf.Print(layer, [tf.reduce_mean(layer), tf.reduce_min(layer),
    # tf.reduce_max(layer), tf.shape(layer)], message = 'layer_', summarize=100)
    layer_cumsum = tf.cumsum(last_layer, 1)
    layer_mean = tf.div(layer_cumsum, mask_cumsum_t)
    print("last_layer_mean")
    print(layer_mean)

    mean_context_vector = layer_mean - last_layer

    layer_shape = last_layer.get_shape().as_list()
    sequence_length = layer_shape[1]
    emb_dim = layer_shape[-1]

    partial_layer_max_list = [tf.ones_like(layer[:, 0:1, :]) * 0.0]

    for idx in xrange(1, sequence_length):
        partial_layer = last_layer[:, 0:idx, :]
        partial_mask = mask[:, 0:idx, :]
        print("last at layer:" + "_" + str(idx))
        print(partial_layer)
        print("last_mask at layer:" + "_" + str(idx))
        print(partial_mask)

        current_spot = last_layer[:, idx:idx+1, :]
        print("current spot at idx:" + str(idx))
        print(current_spot)
        dot_products = current_spot * partial_layer
        print("dot_products at idx:" + str(idx))
        print(dot_products)
        dot_products_norm = tf.reduce_sum(dot_products, -1, keep_dims=True)
        dot_products_norm = tf.where(partial_mask > 0, dot_products_norm,
                                     tf.ones_like(dot_products_norm) * dot_products_norm.dtype.min)
        dot_products_norm = tf.squeeze(dot_products_norm, -1)
        dot_products_norm = tf.nn.softmax(dot_products_norm)
        print("dot_products_norm at idx:" + str(idx))
        print(dot_products_norm)

        top_k = 1
        (values, indices) = tf.nn.top_k(dot_products_norm, top_k)

        gathered_layer = gather_3d_along_seq_length_axis(
            partial_layer, indices)
        print("gathered_layer at last layer:" + "_" + str(idx))
        print(gathered_layer)
        gathered_mask = gather_3d_along_seq_length_axis(
            partial_mask, indices)
        print("gathered_mask at last layer:"  + "_" + str(idx))
        print(gathered_mask)

        gathered_mask_tiled = tf.tile(gathered_mask, [1, emb_dim])
        paddings = tf.ones_like(gathered_layer) * 0.0
        gathered_layer_masked = tf.where(gathered_mask_tiled > 0,
                                         gathered_layer, paddings)
        gathered_layer_masked = tf.expand_dims(gathered_layer_masked, 1)
        partial_layer_max_list.append(gathered_layer_masked)

    max_context_vector = tf.concat(partial_layer_max_list, 1)

    context_vector = tf.concat([mean_context_vector, max_context_vector], -1)

    context_vectors.append(context_vector)

    return tf.concat(context_vectors, -1)


def create_combined_global_local_context_vector(params):
#    deep_context_vector = create_deep_context_vector(params)
    local_context_vector = create_local_attention_context_vector(params)
    global_deep_context_vector = create_accumulative_global_deep_context_vector(params)
#    print('deep_context_vector ')
#    print(deep_context_vector)
    print('global_deep_context_vector ')
    print(global_deep_context_vector )
    print('local_context_vector ')
    print(local_context_vector)
    return tf.concat([local_context_vector, global_deep_context_vector], -1)

def create_combined_cumsum_deep_context_vector(params):
    deep_context_vector = create_deep_context_vector(params)
    #deep_context_vector_shape = deep_context_vector.get_shape().as_list()
    global_deep_context_vector = create_accumulative_global_deep_context_vector(params)
    print('global_deep_context_vector ')
    print(global_deep_context_vector )
    print('deep_context_vector ')
    print(deep_context_vector)

    combined_context_vector = [deep_context_vector, global_deep_context_vector]

    if 'user_emb' in params and params['user_emb'] is not None:
        # [B, D]
        user_emb = params['user_emb']
        sequence_length = deep_context_vector.get_shape().as_list()[1]
        user_emb_exp = tf.expand_dims(user_emb, 1)
        # [B, L, D]
        user_emb_tiled = tf.tile(user_emb_exp, [1, sequence_length, 1])
        print('user_emb_tiled')
        print(user_emb_tiled)
        combined_context_vector.append(user_emb_tiled)

    return tf.concat(combined_context_vector, -1), deep_context_vector, global_deep_context_vector

def create_combined_deep_global_local_context_vector(params):
    deep_context_vector = create_deep_context_vector(params)
    global_deep_context_vector = create_accumulative_global_deep_context_vector(params)
    #local_context_vector = create_local_context_vector(params)
    local_context_vector = create_local_attention_context_vector(params)
    print('global_deep_context_vector ')
    print(global_deep_context_vector )
    print('deep_context_vector ')
    print(deep_context_vector)
    print('local_context_vector')
    print(local_context_vector)

    combined_context_vector = [deep_context_vector, global_deep_context_vector, local_context_vector]

    if 'user_emb' in params and params['user_emb'] is not None:
        # [B, D]
        user_emb = params['user_emb']
        sequence_length = deep_context_vector.get_shape().as_list()[1]
        user_emb_exp = tf.expand_dims(user_emb, 1)
        # [B, L, D]
        user_emb_tiled = tf.tile(user_emb_exp, [1, sequence_length, 1])
        print('user_emb_tiled')
        print(user_emb_tiled)
        combined_context_vector.append(user_emb_tiled)

    return tf.concat(combined_context_vector, -1), deep_context_vector, global_deep_context_vector, local_context_vector

def create_combined_deep_global_attentive_context_vector(params):
    deep_context_vector = create_deep_context_vector(params)
    global_deep_context_vector = create_accumulative_attentive_global_deep_context_vector(params)
    print('global_deep_context_vector ')
    print(global_deep_context_vector )
    print('deep_context_vector ')
    print(deep_context_vector)

    combined_context_vector = [deep_context_vector, global_deep_context_vector]

    return tf.concat(combined_context_vector, -1), deep_context_vector, global_deep_context_vector

def create_combined_deep_global_max_context_vector(params):
    deep_context_vector = create_deep_context_vector(params)
    global_deep_context_vector = create_accumulative_global_deep_context_vector(params)
    max_deep_context_vector = create_accumulative_max_deep_context_vector(params)
    print('global_deep_context_vector ')
    print(global_deep_context_vector )
    print('deep_context_vector ')
    print(deep_context_vector)
    print('max_deep_context_vector')
    print(max_deep_context_vector)

    combined_context_vector = [deep_context_vector, global_deep_context_vector, max_deep_context_vector]

    return tf.concat(combined_context_vector, -1), deep_context_vector, global_deep_context_vector, max_deep_context_vector

def create_combined_deep_global_most_attentive_context_vector(params):
    deep_context_vector = create_deep_context_vector(params)
    global_deep_context_vector = create_accumulative_global_deep_context_vector(params)
    most_attentive_deep_context_vector = create_accumulative_most_attentive_context_vector(params)
    print('global_deep_context_vector ')
    print(global_deep_context_vector )
    print('deep_context_vector ')
    print(deep_context_vector)
    print('most_attentive_deep_context_vector')
    print(most_attentive_deep_context_vector)

    combined_context_vector = [deep_context_vector, global_deep_context_vector, most_attentive_deep_context_vector]

    return tf.concat(combined_context_vector, -1), deep_context_vector, global_deep_context_vector,most_attentive_deep_context_vector 

def create_combined_deep_global_self_attentive_context_vector(params):
    deep_context_vector = create_deep_context_vector(params)
    global_deep_context_vector = create_accumulative_self_attentional_global_deep_context_vector(params)
    most_attentive_deep_context_vector = create_accumulative_most_attentive_context_vector(params)
    print('global_deep_context_vector ')
    print(global_deep_context_vector )
    print('deep_context_vector ')
    print(deep_context_vector)
    print('max_deep_context_vector')
    print(most_attentive_deep_context_vector)

    combined_context_vector = [deep_context_vector, global_deep_context_vector, most_attentive_deep_context_vector]

    return tf.concat(combined_context_vector, -1), deep_context_vector, global_deep_context_vector, most_attentive_deep_context_vector

def create_combined_deep_global_max_self_attentive_context_vector(params):
    deep_context_vector = create_deep_context_vector(params)
    global_deep_context_vector = create_accumulative_max_self_attentional_global_deep_context_vector(params)
    print('global_deep_context_vector ')
    print(global_deep_context_vector )
    print('deep_context_vector ')
    print(deep_context_vector)

    combined_context_vector = [deep_context_vector, global_deep_context_vector]

    return tf.concat(combined_context_vector, -1), deep_context_vector, global_deep_context_vector

def create_combined_vdeep_global_max_self_attentive_context_vector(params):
    deep_context_vector = create_deep_context_vector(params)
    global_deep_context_vector = create_accumulative_vdeep_self_attentional_global_deep_context_vector(params)
    print('global_deep_context_vector ')
    print(global_deep_context_vector )
    print('deep_context_vector ')
    print(deep_context_vector)

    combined_context_vector = [deep_context_vector, global_deep_context_vector]

    return tf.concat(combined_context_vector, -1), deep_context_vector, global_deep_context_vector

def create_combined_deep_global_text_cnn_context_vector(params):
    deep_context_vector = create_deep_context_vector(params)
    global_deep_context_vector = create_accumulative_global_deep_context_vector(params)
    text_cnn_context_vector= create_text_cnn_context_vector(params)
    print('global_deep_context_vector ')
    print(global_deep_context_vector )
    print('deep_context_vector ')
    print(deep_context_vector)
    print('text_cnn_context_vector')
    print(text_cnn_context_vector)

    combined_context_vector = [deep_context_vector, global_deep_context_vector, text_cnn_context_vector]

    return tf.concat(combined_context_vector, -1), deep_context_vector, global_deep_context_vector, text_cnn_context_vector

def create_local_attention_context_vector(params):
    input_layers = params['input_layer']
    # [batch size, seq length, 1]
    mask = params['mask']
    print("mask from params")
    print(mask)
    scope = params['scope']
    is_training = params['is_training']
    context_vectors = []
    for layer in input_layers:
        # layer has shape: [batch size, seq length, emb dim ]

        # kernel_size = 2
        conv_layer = conv_block(layer, kernel_size=2, scope = "{}_conv_block_2".format(scope), is_training = is_training,
               reuse = tf.AUTO_REUSE, l2_reg = 0.1, dropout = 0.0)
        print('conv_layer_2')
        print(conv_layer)
        context_vectors.append(conv_layer)

        # kernel_size = 1
        conv_layer = conv_block(layer, kernel_size=1, scope = "{}_conv_block_1".format(scope), is_training = is_training,
               reuse = tf.AUTO_REUSE, l2_reg = 0.1, dropout = 0.0)
        print('conv_layer_1')
        print(conv_layer)
        context_vectors.append(conv_layer)

        # kernel_size = 3
        conv_layer = conv_block(layer, kernel_size=3, scope = "{}_conv_block_3".format(scope), is_training = is_training,
               reuse = tf.AUTO_REUSE, l2_reg = 0.1, dropout = 0.0)
        print('conv_layer_3')
        print(conv_layer)
        context_vectors.append(conv_layer)

    # [B, L, (l-1)*D]
    print('local_attention_context_vector')
    print(context_vectors)
    output = tf.concat(context_vectors, -1)
    print('local_attention_context_vector output')
    print(output)
    return output

def extract_text_cnn_block(layer, kernel_size, dilation_rate, is_training, scope, l2_reg, dropout):
    # layer has shape: [batch size, seq length, emb dim ]
    conv_layer = conv_block(layer, kernel_size=kernel_size, dilation_rate=dilation_rate, scope = scope, is_training = is_training,
           reuse = tf.AUTO_REUSE, l2_reg = l2_reg, dropout = dropout)
    print('text_cnn_block_{}'.format(kernel_size))
    print(conv_layer)
    conv_sub_layer_pooling_vec = []
    sequence_length = layer.get_shape().as_list()[1]
    for idx in range(sequence_length):
        conv_sub_layer = conv_layer[:, 0:idx+1, :]
        conv_sub_layer_max = tf.reduce_max(conv_sub_layer, -2, keep_dims=True)
        conv_sub_layer_pooling_vec.append(conv_sub_layer_max)
    conv_sub_layer_pooling = tf.concat(conv_sub_layer_pooling_vec, -2)
    print('text_cnn_pool_layer_{}'.format(kernel_size))
    print(conv_sub_layer_pooling)
    return conv_sub_layer_pooling
 
def create_text_cnn_context_vector(params):
    input_layers = params['input_layer']
    if 'exclude_top_layer' in params and params['exclude_top_layer']:
        input_layers = input_layers[0:-1]
    sequence_length= input_layers[0].get_shape().as_list()[1]
    # [batch size, seq length, 1]
    mask = params['mask']
    print("mask from params")
    print(mask)
    scope = params['scope']
    is_training = params['is_training']
    context_vectors = []
    for layer in input_layers:
       context_vector = extract_text_cnn_block(layer, 1, 1, is_training, "{}_text_cnn_conv_block_{}".format(scope, 1), 0.1, 0.0)
       print("text_cnn ctx vector k=1")
       print(context_vector)
       context_vectors.append(context_vector)

       context_vector = extract_text_cnn_block(layer, 2, 1, is_training, "{}_text_cnn_conv_block_{}".format(scope, 2), 0.1, 0.0)
       print("text_cnn ctx vector k=2")
       print(context_vector)
       context_vectors.append(context_vector)

       context_vector = extract_text_cnn_block(layer, 3, 1, is_training, "{}_text_cnn_conv_block_{}".format(scope, 3), 0.1, 0.0)
       print("text_cnn ctx vector k=3")
       print(context_vector)
       context_vectors.append(context_vector)

       context_vector = extract_text_cnn_block(layer, 2, 2, is_training, "{}_text_cnn_conv_block_{}_{}".format(scope, 2, 2), 0.1, 0.0)
       print("text_cnn ctx vector k=2 d=2")
       print(context_vector)
       context_vectors.append(context_vector)

       context_vector = extract_text_cnn_block(layer, 3, 2, is_training, "{}_text_cnn_conv_block_{}_{}".format(scope, 3, 2), 0.1, 0.0)
       print("text_cnn ctx vector k=3 d=2")
       print(context_vector)
       context_vectors.append(context_vector)

    # [B, L, (l-1)*D]
    print('text_cnn_context_vector')
    print(context_vectors)
    output = tf.concat(context_vectors, -1)
    print('text_cnn_context_output')
    print(output)
    return output


def create_local_context_vector(params):
    conv_layers = params['conv_layer']
    if 'exclude_top_layer' in params and params['exclude_top_layer']:
        conv_layers = conv_layers[0:-1]
    # [batch size, seq length, 1]
    mask = params['mask']
    print("mask from params")
    print(mask)

    context_vectors = []
    for layer_idx, layer in enumerate(conv_layers):
        # layer has shape: [batch size, seq length, emb dim ]
        print("create_local_context layer")
        print(layer)
        layer_slice = layer[:, 0:-1, ]
        layer_slice = tf.pad(layer_slice, [[0,0], [1, 0], [0,0]])
        mask_t = tf.cast(mask, layer.dtype)
        layer_slice = layer_slice * mask_t
        context_vectors.append(layer_slice)
    return tf.concat(context_vectors, -1)

def get_forgetting_factor(qk, atten_size, l2_reg, name, scope, reuse=None):
    qk_shape = qk.get_shape().as_list()
    seq_length = qk_shape[1]
    inner_dim = qk_shape[-1] 
    qk = tf.reshape(qk, [-1, inner_dim])
    print("qk reshape factor")
    print(qk)
    with tf.variable_scope(scope, reuse=reuse):
        vh = tf.get_variable(name + "_vh", dtype=qk.dtype,
                             shape=[atten_size, 1],
                             regularizer=tf.contrib.layers.l2_regularizer(l2_reg))
        print("vh")
        print(vh)
        fqk = tf.matmul(qk, vh)
        print("fqk")
        print(fqk)
        return tf.reshape(fqk, [-1, seq_length, 1])

def combine_context_with_query_v1(query, context, scope): #T
    # V1: combine query and projected context via concatenation and dense projection 
    query_shape = query.get_shape().as_list()
    batch_size = query_shape[0]
    sequence_length = query_shape[1]
    emb_dim = query_shape[-1]
    query_context_concat = tf.concat([query, context], -1)
    with tf.variable_scope(scope + "_qc_combine", reuse=tf.AUTO_REUSE):
        comb_out = tf.layers.dense(query_context_concat, emb_dim)
        print(comb_out)
        return comb_out

def combine_context_with_query_v2(query, projected_context, scope): #T
    # V2: bi_linear product between q and c and apply SE upon q and c and combine both
    #T
    # [B, L, 1, D]
    query_exp = tf.expand_dims(query, -2)
    # [B, L, 1, D]
    context_exp = tf.expand_dims(projected_context, -2)

    # [B, L, 1, D]
    bi_linear_qc = bi_linear(query, projected_context,
                                     scope = scope + "_project_query_context")
    print('combine_context_with_query_v2 bi_linear_qc')
    print(bi_linear_qc)

    # [B, L, 2, D]
    query_context_concat = tf.concat([query_exp, context_exp], -2)

    # [B, L, D, 2]
    query_context_concat = tf.transpose(query_context_concat, [0, 1, 3, 2])

    # [B, L, D, 2]
    sq_qc = squeeze_excitation_layer(query_context_concat, 2, layer_name=scope + "_orig_emb")

    # [B, L, 2, D]
    sq_qc = tf.transpose(sq_qc, [0,1,3,2])

    # {[B, L, 1, D]}
    sq_qc_splits = tf.split(sq_qc, 2, axis=-2)

    sq_query = sq_qc_splits[0]
    sq_context = sq_qc_splits[1]
    sq_query = tf.squeeze(sq_query, 2)
    sq_context = tf.squeeze(sq_context, 2)

    # [B, L, D]
    bi_linear_sq_qc = bi_linear(sq_query, sq_context, scope = scope + "_se_emb")
    print('combine_context_with_query_v2 bi_linear_sq_qc')
    print(bi_linear_sq_qc)

    bi_linear_concat = tf.concat([bi_linear_qc, bi_linear_sq_qc], -1)

    query_shape = query.get_shape().as_list()

    dim = query_shape[-1]

    with tf.variable_scope(scope + "_map_combined_context_query_v2"):
        return tf.layers.dense(bi_linear_concat, dim)

def combine_context_with_query_v3(query, projected_context, scope): #T
    # V3: combine q and c via co forget factor forget factor is a scalar 
    with tf.variable_scope(scope + "_combined_context_query_v3"):
        query_shape = query.get_shape().as_list()
        sequence_length = query_shape[1]
        atten_size = query_shape[-1]
        context_shape = projected_context.get_shape().as_list()
        context_size = context_shape[-1]
        query_plus_context = tf.concat([query, projected_context], -1)
        fzk = tf.layers.dense(query_plus_context, 1)
        print("fzk")
        print(fzk)
        fzk_sigmoid = tf.nn.sigmoid(fzk)
        return (1.0 - fzk_sigmoid) * query + fzk_sigmoid * projected_context 

def combine_context_with_query_v4(query, combined_context, l2_reg, name, scope, reuse):
    # V4: too complicated
    with tf.variable_scope(scope + "_combine_cq_v4", reuse=reuse):
        query_shape = query.get_shape().as_list()
        atten_size = query_shape[-1]
        context_shape = combined_context.get_shape().as_list()
        sequence_length = context_shape[1]
        context_size = context_shape[-1]
        qk = tf.get_variable(name + "_v4_qk", dtype=query.dtype,
                              shape=[atten_size, atten_size],
                              initializer = tf.contrib.layers.xavier_initializer(),
                              regularizer=tf.contrib.layers.l2_regularizer(l2_reg))
        query_reshape = tf.reshape(query, [-1, atten_size])
        query_transpose = tf.transpose(query_reshape, [1,0])
        query_w = tf.matmul(qk, query_transpose)
        query_w = tf.transpose(query_w, [1, 0])

        ck = tf.get_variable(name + "_v4_ck", dtype=combined_context.dtype,
                              shape=[atten_size, context_size],
                              initializer = tf.contrib.layers.xavier_initializer(),
                              regularizer=tf.contrib.layers.l2_regularizer(l2_reg))
        context_reshape = tf.reshape(combined_context, [-1, context_size])
        context_transpose = tf.transpose(context_reshape, [1,0])
        context_w = tf.matmul(ck, context_transpose)
        context_w = tf.transpose(context_w, [1, 0])
        context_w_reshape = tf.reshape(context_w, [-1, sequence_length, atten_size]) 

        query_context_w = query_w + context_w
        query_context_w = tf.reshape(query_context_w, [-1, sequence_length, atten_size])

        forget_factor = tf.nn.sigmoid(query_context_w)

        qkv = tf.get_variable(name + "_v4_qkv", dtype=query.dtype,
                              shape=[atten_size, atten_size],
                              initializer = tf.contrib.layers.xavier_initializer(),
                              regularizer=tf.contrib.layers.l2_regularizer(l2_reg))
        query_reshape = tf.reshape(query, [-1, atten_size])
        query_transpose = tf.transpose(query_reshape, [1,0])
        query_v = tf.matmul(qkv, query_transpose)
        query_v = tf.transpose(query_v, [1, 0])
        query_v = tf.reshape(query_v, [-1, sequence_length, atten_size])

        ckv = tf.get_variable(name + "_v4_ckv", dtype=combined_context.dtype,
                              shape=[atten_size, context_size],
                              initializer = tf.contrib.layers.xavier_initializer(),
                              regularizer=tf.contrib.layers.l2_regularizer(l2_reg))
        context_reshape = tf.reshape(combined_context, [-1, context_size])
        context_transpose = tf.transpose(context_reshape, [1,0])
        context_v = tf.matmul(ckv, context_transpose)
        context_v = tf.transpose(context_v, [1, 0])
        context_v = tf.reshape(context_v, [-1, sequence_length, atten_size])

        #output_qk = forget_factor * context_v 
        output_qk = forget_factor * query_v 
        output_ck = (1.0 - forget_factor) * context_v 
        #output_ck = query_v

        return output_qk + output_ck

def combine_context_with_query_v5(query, context_vector, l2_reg, name, scope, reuse): #T
    # V5: combine query and individual context vectors via SE followed by weighted sum
    with tf.variable_scope(scope + "_combine_cq_v5", reuse=reuse):
        query_shape = query.get_shape().as_list()
        emb_dim = query_shape[-1]

        context_vectors_shape = context_vector.get_shape().as_list()
        sequence_length = context_vectors_shape[-2]
        context_vector_dim = context_vectors_shape[-1]

        # context vector: [B, L, l, d]
        context_vector_reshape = tf.reshape(context_vector,
                                            [-1, sequence_length, context_vector_dim / emb_dim,
                                             emb_dim])

        # [B, L, 1, d]
        query_exp = tf.expand_dims(query, -2)


        # [B, L, l+1, D]
        query_context = tf.concat([query_exp, context_vector_reshape], -2)

        print('query_context')
        print(query_context)

        # [B, L, D, l+1]
        query_context = tf.transpose(query_context, [0,1,3,2])

        print('query_context')
        print(query_context)

        #query_context_proj = tf.layers.dense(query_context, 16)
        query_context_proj = query_context

        print('query_context_proj')
        print(query_context_proj)

        # query_context_weighted: [[B, L, D, l+1], weights: [B, 1, 1, l+1]]
        query_context_weighted, weights = squeeze_excitation_layer(query_context_proj, 3, l2_reg = l2_reg, layer_name=scope, ret_excitation=True)

        # [B, L, D]
        context_vec_pool = tf.reduce_sum(query_context_weighted, -1)
        # [B, 1, 1]
        weights_pool = tf.reduce_sum(weights, -1)

        # [B, L, D]
        query_context_weighted_sum = context_vec_pool / weights_pool

        return query_context_weighted_sum

def combine_context_with_query_v6(query, projected_context, l2_reg, name, scope, reuse):
    #V6: (1-f)*q + f * c
    with tf.variable_scope(scope + "_combine_cq_v6", reuse=reuse):
        query_shape = query.get_shape().as_list()
        sequence_length = query_shape[1]
        atten_size = query_shape[-1]
        context_shape = projected_context.get_shape().as_list()
        context_size = context_shape[-1]
        query_context_combo = tf.concat([query, projected_context], -1)
        uzk = tf.get_variable(name + "_v6_uzk", dtype=query.dtype,
                              shape=[1, atten_size + context_size],
                              initializer = tf.contrib.layers.xavier_initializer(),
                              regularizer=tf.contrib.layers.l2_regularizer(l2_reg))
        uzk_exp = tf.expand_dims(uzk, 0)
        uzk_exp = tf.expand_dims(uzk_exp, 0)
        query_context_combo_exp = tf.expand_dims(query_context_combo, -1)
        uzk_tiled = tf.tile(uzk_exp, [tf.shape(query_context_combo)[0], sequence_length, 1, 1])
        print('uzk_tiled')
        print(uzk_tiled)
        print('query_context')
        print(query_context_combo_exp)
        fzk = tf.matmul(uzk_tiled, query_context_combo_exp)
        fzk = tf.squeeze(fzk, -1)
        print("fzk")
        print(fzk)
        fzk_sigmoid = tf.nn.sigmoid(fzk)

        return (1.0 - fzk_sigmoid) * query + fzk_sigmoid * projected_context 

def combine_context_with_query_v7(query, projected_context, l2_reg, name, scope, reuse):
    #V7: (1-f) * q + f*c, f is element-wise forget factor
    with tf.variable_scope(scope + "_combine_cq_v7", reuse=reuse):
        query_shape = query.get_shape().as_list()
        sequence_length = query_shape[1]
        atten_size = query_shape[-1]
        context_shape = projected_context.get_shape().as_list()
        context_size = context_shape[-1]

#        query_plus_context = tf.concat([query, projected_context], -1)
#        fzk = tf.layers.dense(query_plus_context, atten_size)

        fzk = query * projected_context

        print("fzk")
        print(fzk)
        fzk_sigmoid = tf.nn.sigmoid(fzk)
        return (1.0 - fzk_sigmoid) * query + fzk_sigmoid * projected_context 

def combine_context_with_query_v8(query, context_vector, l2_reg, name, scope, reuse):
    # q + SUM([f * c]), f is element-wise forget factor
    with tf.variable_scope(scope + "_combine_cq_v8", reuse=reuse):
        context_vectors_shape = context_vector.get_shape().as_list()
        query_vector_shape = query.get_shape().as_list()
        sequence_length = query_vector_shape[-2]
        emb_dim = query_vector_shape[-1]

        context_vector_dim = context_vectors_shape[-1]

        # context vector: [B*L, l, d]
        context_vector_reshape = tf.reshape(context_vector,
                                            [-1, context_vector_dim / emb_dim,
                                             emb_dim])
        print('combine_context_with_query_v8 context_vector_reshape')
        print(context_vector_reshape)

        # query vector: [B*L, 1, d]
        query_vector_exp = tf.reshape(query, [-1, 1, emb_dim])
        # query vector: [B*L, l, d]
        query_vector_tiled = tf.tile(query_vector_exp,
                                     [1, context_vector_dim / emb_dim, 1])
        print('combine_context_with_query_v8 query_vector_tiled')
        print(query_vector_tiled)

        # [B*L, l, d+d]
        context_query_vector_comb =\
            tf.concat([context_vector_reshape, query_vector_tiled], -1)
        print('combine_context_with_query_v8 context_query_vector_comb')
        print(context_query_vector_comb)

        # [B*L, l, 1]
        context_query_vector_project =\
            tf.layers.dense(context_query_vector_comb, emb_dim,
                            kernel_regularizer =
                            tf.contrib.layers.l2_regularizer(l2_reg))

        context_query_vector_forget_factor =\
            (context_query_vector_project)

        context_query_vector_forget_factor = tf.nn.softmax(context_query_vector_forget_factor)

        print('combine_context_with_query_v8 context_query_vector_forget_factor')
        print(context_query_vector_forget_factor)

#        context_query_vector_forget_factor = tf.Print(context_query_vector_forget_factor, [tf.reduce_mean(context_query_vector_forget_factor), tf.shape(context_query_vector_forget_factor), tf.reduce_min(context_query_vector_forget_factor), tf.reduce_max(context_query_vector_forget_factor)])

        # [B*L, l, d]
        context_vector_with_forget_factor =\
            context_vector_reshape * context_query_vector_forget_factor

        # [B*L, d]
        context_vector_with_forget_factor_sum =\
            tf.reduce_sum(context_vector_with_forget_factor, -2)

        # [B, L, d]
        context_vector_output = tf.reshape(
            context_vector_with_forget_factor_sum,
            [-1, sequence_length, emb_dim]
        )

        return query + context_vector_output, context_query_vector_forget_factor

def combine_context_with_query_v9(query, projected_context, l2_reg, name, scope, reuse):
    bi_linear_qc = bi_linear(query, projected_context,
                                    scope = scope + "_combine_context_with_query_v9")
    print('combine_context_with_query_v9 bi_linear_qc')
    print(bi_linear_qc)
    emb_dim = query.get_shape().as_list()[-1]
    with tf.variable_scope(scope + "_combine_context_with_query_v9", reuse=reuse):
         return tf.layers.dense(bi_linear_qc, emb_dim, kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_reg)) 

def model_context(qk, params, name, scope = 'context_sa', reuse=None):
    atten_size = params['atten_size']
    l2_reg = params['l2_reg']
    context_vector = params['context_vector']
    context_combine_mode = params.get('context_combine_mode') or 0
    project_context_mode = params.get('project_context_mode') or 0
    context_size = context_vector.get_shape().as_list()[-1]
    context_dropout = params.get('context_dropout') or 0.0
    use_multihead_context = params.get('use_multihead_context') or False
    num_heads = params.get('num_heads') or 1 
    context_emb_size = params.get('context_emb_size') or atten_size

    print('model context params')
    print(params)
    print(atten_size)
    with tf.variable_scope(scope, reuse=reuse):
        uqk = tf.get_variable(name + "_uqk", dtype=qk.dtype,
                             shape=[context_size, atten_size],
                             regularizer=tf.contrib.layers.l2_regularizer(l2_reg))
        print("uqk")
        print(uqk)
        context_vector_shape = context_vector.get_shape().as_list()
        context_vector_reshaped = tf.reshape(context_vector, [-1, context_size])
        print("cqk")
        print(context_vector_reshaped)
        cqk = tf.matmul(context_vector_reshaped, uqk)
        print(cqk)
        cqk = tf.reshape(cqk, [-1, context_vector_shape[1], atten_size])
        if use_multihead_context:
           print("Use multihead context")
           cqk = tf.tile(cqk, [num_heads, 1, 1])
           print(cqk)

        if project_context_mode == 1:
           cqk = project_context_v1(qk, context_vector, params, scope) 
        if project_context_mode == 2:
           cqk = project_context_v2(qk, context_vector, params, scope) 
        if project_context_mode == 3:
           cqk = project_context_v3(qk, context_vector, params, scope) 
        if project_context_mode == 4:
           cqk = project_context_v4(qk, context_vector, params, scope) 
        if project_context_mode == 5:
           deep_context_vector = params['deep_context_vector']
           global_context_vector = params['global_context_vector']
           cqk = project_context_v5(qk, deep_context_vector, global_context_vector, params, scope) 
        if project_context_mode == 6:
           params['num_iterations'] = 2
           params['num_output_caps'] = 2
           cqk = project_context_v6(qk, context_vector, params, scope) 

#        cqk = tf.Print(cqk, [cqk, tf.shape(cqk)], message = 'cqk')

        if context_dropout > 0:
           print('Apply context dropout:' + str(context_dropout) + ", is_training:" + str(params['is_training']))
           cqk = tf.contrib.layers.dropout(cqk, keep_prob=1.0-context_dropout, is_training=params['is_training'])

        variable_summaries(cqk, 'context_vector_cqk_' + str(scope))

        print('cqk:')
        print(cqk)
        print('qk')
        print(qk)

        #cqk = tf.Print(cqk, [cqk, tf.shape(cqk)], message= 'cqk_' + str(scope))
        #qk = tf.Print(qk, [cqk, tf.shape(qk)], message= 'qk_' + str(scope))
        qk_forget_factor = get_forgetting_factor(qk, atten_size, l2_reg,
                                                name = name + "_qk_forget_factor", scope=scope + "_qk_forget_factor", reuse=reuse)
        cqk_forget_factor = get_forgetting_factor(cqk, atten_size, l2_reg,
                                                name = name + "_cqk_forget_factor", scope=scope + "_cqk_forget_factor", reuse=reuse)
        forget_factor = tf.nn.sigmoid(qk_forget_factor + cqk_forget_factor)
        variable_summaries(qk_forget_factor, 'context_vector_qk_forget_factor_' + str(scope))
        variable_summaries(cqk_forget_factor, 'context_vector_cqk_forget_factor_' + str(scope))
        variable_summaries(forget_factor, 'context_vector_forget_factor_' + str(scope))
        #forget_factor = tf.Print(forget_factor, [forget_factor, tf.reduce_mean(forget_factor), tf.reduce_max(forget_factor), tf.reduce_min(forget_factor), tf.shape(forget_factor)], message= 'forget_factor_' + str(scope))
        print("qk_forget_factor")
        print(qk_forget_factor)
        print("cqk_forget_factor")
        print(cqk_forget_factor)
        print("forget_factor")
        print(forget_factor)
        output_cqk = forget_factor * cqk
        #output_cqk = tf.Print(output_cqk, [output_cqk, tf.shape(output_cqk)], message= 'output_cqk_' + str(scope))
        output_qk = (1.0 - forget_factor) * qk
        #output_qk = qk
        #output_qk = tf.Print(output_qk, [output_qk, tf.shape(output_qk)], message= 'output_qk_' + str(scope))
        output = output_cqk + output_qk
        #output = tf.Print(output, [output, tf.shape(output)], message= 'output_' + str(scope))

        if context_combine_mode == 1:
           return combine_context_with_query_v1(qk, cqk, scope), forget_factor
        
        if context_combine_mode == 2:
           return combine_context_with_query_v2(qk, cqk, scope), forget_factor

        if context_combine_mode == 3:
           return combine_context_with_query_v3(qk, cqk, scope), forget_factor

        if context_combine_mode == 4:
            return combine_context_with_query_v4(qk, context_vector, l2_reg, name, scope, reuse), forget_factor 

        if context_combine_mode == 5:
            return combine_context_with_query_v5(qk , context_vector, l2_reg, name,scope, reuse), forget_factor 

        if context_combine_mode == 6:
            return combine_context_with_query_v6(qk , cqk, l2_reg, name,scope, reuse), forget_factor 

        if context_combine_mode == 7: #T
            return combine_context_with_query_v7(qk , cqk, l2_reg, name,scope, reuse), forget_factor 

        if context_combine_mode == 8:
            comb_output = combine_context_with_query_v8(qk , context_vector, l2_reg, name,scope, reuse)
            return comb_output 
        if context_combine_mode == 9:
            return combine_context_with_query_v9(qk , context_vector, l2_reg, name,scope, reuse), forget_factor

        #output_qk = tf.Print(output_qk, [output_qk, tf.shape(output_qk)], message= 'output_qk_' + str(scope))

        return output, forget_factor

def project_context_v1(query, context_vectors, params, scope):
    # context_vectors: [B, L, l * d]
    # query: [B, L, d]
    # Vanilla attention pooling: NOT WORKING
    print('project_context_v1')
    atten_vec = atten_on_deep_context_vectors(context_vectors, query, params, scope)
    return atten_vec

def project_context_v2(query, context_vectors, params, scope):
    # context_vectors: [B, L, l * d]
    # query: [B, L, d]
    # Self attention pooling: NOT WORKING
    print('project_context_v2')
    context_vectors_shape = context_vectors.get_shape().as_list()
    query_vector_shape = query.get_shape().as_list()
    sequence_length = query_vector_shape[1]
    emb_dim = query_vector_shape[2]

    context_vector_dim = context_vectors_shape[-1]

    # context vector: [B*L, l, d]
    context_vector_reshape = tf.reshape(context_vectors,
                                        [-1, context_vector_dim / emb_dim,
                                         emb_dim])

    # [B*L, d, l]
    context_vector_transpose = tf.transpose(context_vector_reshape, [0, 2, 1])

    # [B*L, l, l]
    context_vec_mul = tf.matmul(context_vector_reshape, context_vector_transpose)

    context_vector_softmax = tf.nn.softmax(context_vec_mul)

    # [B*L, l, d]
    context_vec_atten = tf.matmul(context_vector_softmax, context_vector_reshape)

    # [B*L, d]
    context_vec_pool = tf.reduce_mean(context_vec_atten, -2)

    return tf.reshape(context_vec_pool, [-1, sequence_length, emb_dim])

def project_context_v3(query, context_vectors, params, scope):
    # context_vectors: [B, L, l * d]
    # query: [B, L, d]
    # SE pooling: NOT WORKING
    print('project_context_v3')
    context_vectors_shape = context_vectors.get_shape().as_list()
    query_vector_shape = query.get_shape().as_list()
    sequence_length = query_vector_shape[1]
    emb_dim = query_vector_shape[2]

    context_vector_dim = context_vectors_shape[-1]

    # context vector: [B*L, 1, d, l]
    context_vector_reshape = tf.reshape(context_vectors,
                                        [-1, 1, emb_dim,
                                         context_vector_dim / emb_dim])

    # context_vec:[B*L, 1, d, l], weights:[B*L, 1, 1, l]
    context_vec_weighted, weights = squeeze_excitation_layer(context_vector_reshape, 3, layer_name=scope, ret_excitation=True)

    # [B*L, 1, d]
    context_vec_pool = tf.reduce_sum(context_vec_weighted, -1)
    # [B*L, 1, 1]
    #weights_pool = tf.reduce_sum(weights, -1)
    #context_vec_pool = context_vec_pool / weights_pool

    return tf.reshape(context_vec_pool, [-1, sequence_length, emb_dim])

def project_context_v4(query, context_vectors, params, scope):
    # context_vectors: [B, L, l * d]
    # query: [B, L, d]
    # summation: NOT WORKING
    print('project_context_v4')
    context_vectors_shape = context_vectors.get_shape().as_list()
    query_vector_shape = query.get_shape().as_list()
    sequence_length = query_vector_shape[1]
    emb_dim = query_vector_shape[2]

    context_vector_dim = context_vectors_shape[-1]

    # context vector: [B*L, d, l]
    context_vector_reshape = tf.reshape(context_vectors,
                                        [-1,  emb_dim,
                                         context_vector_dim / emb_dim])

    # [B*L, d]
    context_vec_pool = tf.reduce_mean(context_vector_reshape, -1)

    return tf.reshape(context_vec_pool, [-1, sequence_length, emb_dim])

def project_context_v5(query, deep_context_vectors, global_context_vectors, params, scope):
    print('project_context_v5')
    # query: [B, L, d]
    query_vector_shape = query.get_shape().as_list()
    sequence_length = query_vector_shape[1]
    emb_dim = query_vector_shape[2]
    l2_reg = params['l2_reg']

    with tf.variable_scope(scope + "_project_context_v5"):
        #context vector: [B, L, k*d]
        if (deep_context_vectors.get_shape().as_list()[-1]) == emb_dim:
            deep_context_projection = deep_context_vectors[0]
        else:
            print('deep_context_vector_combined')
            print(deep_context_vectors)
            deep_context_projection = tf.layers.dense(deep_context_vectors, emb_dim, kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_reg))
        if (global_context_vectors.get_shape().as_list()[-1]) == emb_dim:
            global_context_projection = global_context_vectors[0]
        else:
            print('global_context_vector_combined')
            print(global_context_vectors)
            global_context_projection = tf.layers.dense(global_context_vectors, emb_dim, kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_reg))
        return deep_context_projection + global_context_projection

def project_context_v6(query, context_vector_w, params, scope):
    print('project_context_v6')
    # query: [B, L, d]
    query_vector_shape = query.get_shape().as_list()
    sequence_length = query_vector_shape[1]
    emb_dim = query_vector_shape[2]
    l2_reg = params['l2_reg']
    batch_size = params['batch_size']
    context_emb_size = params['context_emb_size']
    num_iterations = params['num_iterations']
    num_output_caps= params['num_output_caps']
    masks = params['mask']

    context_vector_w_shape = context_vector_w.get_shape().as_list()

    num_contexts = context_vector_w_shape[-1] / context_emb_size

    context_vectors = tf.split(context_vector_w, num_contexts, axis=-1) 

    print("split context vectors")
    print(context_vectors)

    context_vectors_reshape = [tf.reshape(context_vector, [-1, context_vector.get_shape().as_list()[-1]]) for context_vector in context_vectors]     
    print("split context vectors reshape")
    print(context_vectors_reshape)

    batch_size = batch_size * sequence_length

    with tf.variable_scope(scope + "_project_context_v6"):
       print("_project_context_v6 input masks")
       print(masks)
       output_caps, _ = combine_representations_by_dynamic_routing_aggrement(context_vectors_reshape ,
                                                            batch_size, num_iterations, num_output_caps, context_emb_size / num_output_caps)  
       #output_caps = tf.Print(output_caps, [tf.reduce_mean(output_caps), tf.reduce_min(output_caps), tf.reduce_max(output_caps), tf.shape(output_caps)], message = 'output_caps')   
       output_caps = tf.reshape(output_caps, [-1, sequence_length, emb_dim])
       #output_caps = tf.Print(output_caps, [tf.reduce_mean(output_caps), tf.reduce_min(output_caps), tf.reduce_max(output_caps), tf.shape(output_caps)], message = 'output_caps_reshape')   
       output_caps *= masks
       #output_caps = tf.Print(output_caps, [tf.reduce_mean(output_caps), tf.reduce_min(output_caps), tf.reduce_max(output_caps), tf.shape(output_caps)], message = 'output_caps_reshape_mask')   
    return output_caps
    
def localness_position_projection(query, params, ffn_weights_scope,
                                  window_weights_scope, reuse):
    # [h*N, T_q, C/h]
    l2_reg = params['l2_reg']
    num_heads = params['num_heads']
    query_shape = query.get_shape().as_list()
    seq_len = query_shape[1]
    dim = query_shape[-1]

    query_reshape = tf.reshape(query, [num_heads, dim, -1])

    with tf.variable_scope(ffn_weights_scope, reuse=reuse):
        ffn_weights = tf.get_variable("ffn_weights", dtype=query.dtype,
                              shape=[num_heads, dim, dim],
                              regularizer=tf.contrib.layers.l2_regularizer(
                                  l2_reg))

        # [h, dim, N*T_q]
        ffn_projection = tf.nn.tanh(tf.matmul(ffn_weights, query_reshape))

        print("ffn_projection")
        print(ffn_projection)

    with tf.variable_scope(window_weights_scope, reuse=reuse):
        window_weights = tf.get_variable("window_weights",
                                     dtype=query.dtype,
                                     shape=[num_heads, 1, dim],
                                     regularizer=tf.contrib.layers.l2_regularizer(
                                         l2_reg))

        # [h, 1, N*T_q]
        ffn_projection = tf.matmul(window_weights, ffn_projection)
        print("ffn_projection")
        print(ffn_projection)

        # [N*h, T_q]
        ffn_projection_reshape = tf.reshape(ffn_projection, [-1, seq_len])

    return ffn_projection_reshape

# query and key are head_splitted
def model_localness(query, params, scope = 'localness_model', reuse=None):
    # query: (h*N, T_q, C/h)
    print("model localness with scope: " + str(scope) + ", reuse:" + str(reuse))
    with tf.variable_scope(scope + "_local_model", reuse=reuse):
        query_shape = query.get_shape().as_list()
        seq_len = query_shape[1]

        # [h*N, T_q]
        predicted_window = localness_position_projection(query, params,
                                      "ffn_projection",
                                      "window_prediction",
                                      reuse)

        print("predicted_window under scope:" + str(scope))
        print(predicted_window)

        normalized_window_weights = seq_len * tf.nn.sigmoid(predicted_window)

        # [L,]
        positions = tf.range(seq_len) + 1

        # [1, L]
        positions = tf.expand_dims(positions, 0)

        print("positions")
        print(positions)

        # [1, L, 1]
        positions_exp = tf.expand_dims(positions, -1)
        # [1, 1, L]
        positions_exp_second= tf.expand_dims(positions, -2)

        print("positions_exp under scope:" + str(scope))
        print(positions_exp)

        # [1, L, L]
        position_diff = positions_exp - positions_exp_second
        position_diff = tf.cast(position_diff, normalized_window_weights.dtype)
        position_diff_square = tf.square(position_diff)

        print("position_diff_square under scope:" + str(scope))
        print(position_diff_square)

        print("normalized_window_weights under scope:" + str(scope))
        print(normalized_window_weights)

        window_weights_square = tf.square(normalized_window_weights)

        window_weights_square = tf.expand_dims(window_weights_square, -1)

        window_weights_square = tf.tile(window_weights_square, [1, 1, seq_len])

        print("window_weights_square under scope:" + str(scope))
        print(window_weights_square)

        local_score = position_diff_square * -2 / window_weights_square

        print('local score under scope:' + str(scope))
        print(local_score)

        # [h*N, T_q, T_q]
        local_score = tf.reshape(local_score, [-1, seq_len, seq_len])

        return local_score, normalized_window_weights

def dot(x, y):
    """Modified from keras==2.0.6
    Multiplies 2 tensors (and/or variables) and returns a *tensor*.

    When attempting to multiply a nD tensor
    with a nD tensor, it reproduces the Theano behavior.
    (e.g. `(2, 3) * (4, 3, 5) -> (2, 4, 5)`)

    # Arguments
        x: Tensor or variable.
        y: Tensor or variable.

    # Returns
        A tensor, dot product of `x` and `y`.
    """
    if ndim(x) is not None and (ndim(x) > 2 or ndim(y) > 2):
        x_shape = []
        for i, s in zip(x.get_shape().as_list(), tf.unstack(tf.shape(x))):
            if i is not None:
                x_shape.append(i)
            else:
                x_shape.append(s)
        x_shape = tuple(x_shape)
        y_shape = []
        for i, s in zip(y.get_shape().as_list(), tf.unstack(tf.shape(y))):
            if i is not None:
                y_shape.append(i)
            else:
                y_shape.append(s)
        y_shape = tuple(y_shape)
        y_permute_dim = list(range(ndim(y)))
        y_permute_dim = [y_permute_dim.pop(-2)] + y_permute_dim
        xt = tf.reshape(x, [-1, x_shape[-1]])
        yt = tf.reshape(tf.transpose(y, perm=y_permute_dim), [y_shape[-2], -1])
        return tf.reshape(tf.matmul(xt, yt),
                          x_shape[:-1] + y_shape[:-2] + y_shape[-1:])
    if isinstance(x, tf.SparseTensor):
        out = tf.sparse_tensor_dense_matmul(x, y)
    else:
        out = tf.matmul(x, y)
    return out

def batch_dot(x, y, axes=None):
    """Copy from keras==2.0.6
    Batchwise dot product.

    `batch_dot` is used to compute dot product of `x` and `y` when
    `x` and `y` are data in batch, i.e. in a shape of
    `(batch_size, :)`.
    `batch_dot` results in a tensor or variable with less dimensions
    than the input. If the number of dimensions is reduced to 1,
    we use `expand_dims` to make sure that ndim is at least 2.

    # Arguments
        x: Keras tensor or variable with `ndim >= 2`.
        y: Keras tensor or variable with `ndim >= 2`.
        axes: list of (or single) int with target dimensions.
            The lengths of `axes[0]` and `axes[1]` should be the same.

    # Returns
        A tensor with shape equal to the concatenation of `x`'s shape
        (less the dimension that was summed over) and `y`'s shape
        (less the batch dimension and the dimension that was summed over).
        If the final rank is 1, we reshape it to `(batch_size, 1)`.
    """
    if isinstance(axes, int):
        axes = (axes, axes)
    x_ndim = ndim(x)
    y_ndim = ndim(y)
    if x_ndim > y_ndim:
        diff = x_ndim - y_ndim
        y = tf.reshape(y, tf.concat([tf.shape(y), [1] * (diff)], axis=0))
    elif y_ndim > x_ndim:
        diff = y_ndim - x_ndim
        x = tf.reshape(x, tf.concat([tf.shape(x), [1] * (diff)], axis=0))
    else:
        diff = 0
    if ndim(x) == 2 and ndim(y) == 2:
        if axes[0] == axes[1]:
            out = tf.reduce_sum(tf.multiply(x, y), axes[0])
        else:
            out = tf.reduce_sum(tf.multiply(tf.transpose(x, [1, 0]), y), axes[1])
    else:
        if axes is not None:
            adj_x = None if axes[0] == ndim(x) - 1 else True
            adj_y = True if axes[1] == ndim(y) - 1 else None
        else:
            adj_x = None
            adj_y = None
        out = tf.matmul(x, y, adjoint_a=adj_x, adjoint_b=adj_y)
    if diff:
        if x_ndim > y_ndim:
            idx = x_ndim + y_ndim - 3
        else:
            idx = x_ndim - 1
        out = tf.squeeze(out, list(range(idx, idx + diff)))
    if ndim(out) == 1:
        out = tf.expand_dims(out, 1)
    return out

initializer = lambda: tf.contrib.layers.variance_scaling_initializer(factor=1.0,
                                                             mode='FAN_AVG',
                                                             uniform=True,
                                                             dtype=tf.float32)

def fuse_layer_attention_v3(all_layer_inputs, params=dict()):
    batch_size = params['batch_size']
    num_iterations = params['num_iterations']
    num_out_caps = params['num_out_caps']
    #batch_size = tf.Print(batch_size, [batch_size], message = 'batch_size')
    print("Fuse layer attention")
    print(all_layer_inputs)
    layer = all_layer_inputs[0]
    sequence_length = layer.get_shape().as_list()[1]
    emb_dim = layer.get_shape().as_list()[-1]
    layer_inputs_reshape = [tf.reshape(x, [-1, emb_dim]) for x in all_layer_inputs]
    print("layer_inputs_reshape")
    print(layer_inputs_reshape)
    output_caps, _ = combine_representations_by_dynamic_routing_aggrement(layer_inputs_reshape, batch_size * sequence_length, num_iterations, num_out_caps, emb_dim / num_out_caps) 
    #output_caps = tf.Print(output_caps, [output_caps, tf.shape(output_caps)], message = 'out_cpas')
    output_caps = tf.reshape(output_caps, [-1, sequence_length, emb_dim])
    print('output_layer_attention_caps')
    print(output_caps)
    return output_caps

def compute_forget_factor(query, key, project_dim = None, name=None):
    key_shape = key.get_shape().as_list()
    emb_dim = key_shape[-1]
    if project_dim is None:
        project_dim = emb_dim
    query_key = tf.concat([query, key], -1)
    query_key_projection = tf.layers.dense(query_key, project_dim, name=name)
    return tf.nn.sigmoid(query_key_projection)

def fuse_layer_attention_v1(all_layer_inputs, params=dict()):
    #batch_size = tf.Print(batch_size, [batch_size], message = 'batch_size')
    print("Fuse layer attention v1")
    print(all_layer_inputs)
    layer = all_layer_inputs[-1]
    sequence_length = layer.get_shape().as_list()[1]
    emb_dim = layer.get_shape().as_list()[-1]

    other_layers = all_layer_inputs[0:-1]
    output_layer = layer
    for idx, xother_layer in enumerate(other_layers):
        # compute forget factor
        with tf.variable_scope('fuse_layer_atten_{}'.format(idx)):
            forget_factor = compute_forget_factor(layer, xother_layer, project_dim=1,
                                                  name = 'layer_forget_factor_{}'.format(idx))
            #forget_factor = tf.Print(forget_factor, [tf.reduce_min(forget_factor), tf.reduce_max(forget_factor), tf.reduce_mean(forget_factor)], name = 'forget_factor_fuse_layer_atten_{}'.format(idx))
            variable_summaries(forget_factor, 'fuse_layer_atten_forget_factor_{}'.format(idx))
        forget_other_layer = xother_layer * forget_factor
        output_layer += forget_other_layer
    return output_layer

def fuse_layer_attention_v2(all_layer_inputs, params=dict()):
    print("Fuse layer attention v2")
    print(all_layer_inputs)
    num_layers = len(all_layer_inputs)
    layer = all_layer_inputs[-1]
    sequence_length = layer.get_shape().as_list()[1]
    emb_dim = layer.get_shape().as_list()[-1]

    all_layer_combs = [tf.expand_dims(x, -2) for x in all_layer_inputs]
    # [B, L, N, D]
    all_layer_concats = tf.concat(all_layer_combs, -2)
    # [B, L, N*D]
    all_layer_combs = tf.reshape(all_layer_concats, [-1, sequence_length, num_layers * emb_dim])

    print("all_layer_concats")
    print(all_layer_concats)
    print("all_layer_combs")
    print(all_layer_combs)  
 
    with tf.variable_scope('fuse_layer_atten_v2_weights'):
       projection_weights = tf.layers.dense(all_layer_combs, num_layers * emb_dim) 
       # [B, L, N, D]
       projection_weights = tf.reshape(projection_weights, [-1, sequence_length, num_layers, emb_dim]) 
       # [B, L, N, D]
       projection_weights = tf.nn.sigmoid(projection_weights)
       print("projection_weights")
       print(projection_weights)
#       projection_weights = tf.Print(projection_weights, [tf.reduce_min(projection_weights), tf.reduce_max(projection_weights), tf.reduce_mean(projection_weights)], name = 'projection_weights_fuse_layer_atten_v2')
       variable_summaries(projection_weights, name = 'projection_weights_fuse_layer_atten_v2')


    # [B, L, N, D]
    weighted_all_layer_combs = all_layer_concats * projection_weights

    return tf.reduce_sum(weighted_all_layer_combs, -2) 

def fuse_layer_attention_v4(all_layer_inputs, params=dict()):
    print("Fuse layer attention v4")
    print(all_layer_inputs)
    layer = all_layer_inputs[-1]
    sequence_length = layer.get_shape().as_list()[1]
    emb_dim = layer.get_shape().as_list()[-1]
    num_layers = len(all_layer_inputs)

    all_layer_combs = [tf.reshape(x, [-1, emb_dim]) for x in all_layer_inputs]

    print("all_layer_combs")
    print(all_layer_combs)

    gru_cell = tf.contrib.rnn.GRUCell(num_units=emb_dim)

    hiddens, state = tf.contrib.rnn.static_rnn(cell=gru_cell, inputs=all_layer_combs,
                                       dtype=all_layer_combs[0].dtype)

    #states: [B*L, D]

    last_state = tf.reshape(state, [-1, sequence_length, emb_dim])

    print('last_state')
    print(last_state)

    return last_state
    
def layer_wise_attention(all_layer_inputs, scope,
                         l2_reg = context_l2_reg, layer_dropout_prob=0.1):
    # {[B, L, D]}
    weighted_layer_embs = []
    dropout_layer_weights = []
    print("all_layer_inputs")
    print(all_layer_inputs)
    with tf.variable_scope("layer_wise_attention_"+ str(scope)):
        for layer in xrange(len(all_layer_inputs)):
            layer_weight = tf.get_variable(
                "layer_weight_%d" % layer, [1],
                initializer=initializer())
            print('layer_weight: %d' % layer)
            print(layer_weight)
            if layer + 1 == len(all_layer_inputs):
                dropout_layer_weight = layer_weight
            else:
                dropout_layer_weight = tf.cond(
                    tf.greater(tf.reduce_max(tf.random_uniform([1])),
                            tf.constant(layer_dropout_prob)),
                    lambda :  layer_weight,
                    lambda : tf.constant(-1e10, shape=layer_weight.shape))
            print('dropout_layer_weight')
            print(dropout_layer_weight)
            dropout_layer_weights.append(dropout_layer_weight)
        dropout_layer_weights = tf.concat(dropout_layer_weights, -1)
        dropout_layer_weights = tf.nn.softmax(dropout_layer_weights)
        print("droput_layer_weights")
        print(dropout_layer_weights)

        for layer, layer_emb in enumerate(all_layer_inputs):
            # [B,L,D]
            weighted_layer_emb = layer_emb * dropout_layer_weights[layer]
            print('weighted_layer_emb')
            print(weighted_layer_emb)
            print(dropout_layer_weights[layer])
            weighted_layer_embs.append(weighted_layer_emb)
        weighted_layer_emb_agg = tf.add_n(weighted_layer_embs)
        global_layer_weight = tf.get_variable(
            "layer_weight_global", [1],
            regularizer=tf.contrib.layers.l2_regularizer(l2_reg),
            initializer=initializer())
        # [B, L, D]
        weighted_layer_emb_agg *= global_layer_weight
        print('weighted_layer_emb_agg')
        print(weighted_layer_emb_agg)
        return weighted_layer_emb_agg        

def optimized_trilinear_for_attention(args, c_maxlen, q_maxlen, input_keep_prob=1.0,
    scope='efficient_trilinear',
    bias_initializer=tf.zeros_initializer(),
    kernel_initializer=initializer()):
    assert len(args) == 2, "just use for computing attention with two input"
    arg0_shape = args[0].get_shape().as_list()
    arg1_shape = args[1].get_shape().as_list()
    print(arg0_shape)
    print(arg1_shape)
    if len(arg0_shape) != 3 or len(arg1_shape) != 3:
        raise ValueError("`args` must be 3 dims (batch_size, len, dimension)")
    if arg0_shape[2] != arg1_shape[2]:
        raise ValueError("the last dimension of `args` must equal")
    arg_size = arg0_shape[2]
    dtype = args[0].dtype
    droped_args = [tf.nn.dropout(arg, input_keep_prob) for arg in args]
    with tf.variable_scope(scope):
        weights4arg0 = tf.get_variable(
            "linear_kernel4arg0", [arg_size, 1],
            dtype=dtype,
            regularizer=regularizer,
            initializer=kernel_initializer)
        weights4arg1 = tf.get_variable(
            "linear_kernel4arg1", [arg_size, 1],
            dtype=dtype,
            regularizer=regularizer,
            initializer=kernel_initializer)
        weights4mlu = tf.get_variable(
            "linear_kernel4mul", [1, 1, arg_size],
            dtype=dtype,
            regularizer=regularizer,
            initializer=kernel_initializer)
        biases = tf.get_variable(
            "linear_bias", [1],
            dtype=dtype,
            regularizer=regularizer,
            initializer=bias_initializer)
        subres0 = tf.tile(dot(droped_args[0], weights4arg0), [1, 1, q_maxlen])
        print("subres0")
        print(subres0)
        subres1 = tf.tile(tf.transpose(dot(droped_args[1], weights4arg1), perm=(0, 2, 1)), [1, c_maxlen, 1])
        print("subres1")
        print(subres1)
        subres2 = batch_dot(droped_args[0] * weights4mlu, tf.transpose(droped_args[1], perm=(0, 2, 1)))
        print("subres2")
        print(subres2)
        res = subres0 + subres1 + subres2
        res = res + biases
        return res

def mask_logits(inputs, mask, mask_value = -1e30):
    shapes = inputs.shape.as_list()
    mask = tf.cast(mask, tf.float32)
    return inputs + mask_value * (1 - mask)

def context_to_query_attention(context, query, c_mask, q_mask, c_maxlen, q_maxlen, dropout=0.0, scope = 'Context_to_Query_Attention_Layer'):
        with tf.variable_scope(scope):
            # C = tf.tile(tf.expand_dims(c,2),[1,1,self.q_maxlen,1])
            # Q = tf.tile(tf.expand_dims(q,1),[1,self.c_maxlen,1,1])
            # S = trilinear([C, Q, C*Q], input_keep_prob = 1.0 - self.dropout)
            # shape of S: [batch_size, c_maxlen, q_maxlen]
            c = context
            q = query
            S = optimized_trilinear_for_attention([c, q], c_maxlen, q_maxlen, input_keep_prob = 1.0 -dropout)
            mask_q = tf.expand_dims(q_mask, 1)
            S_ = tf.nn.softmax(mask_logits(S, mask = mask_q))

            mask_c = tf.expand_dims(c_mask, 2)
            S_T = tf.transpose(tf.nn.softmax(mask_logits(S, mask = mask_c), dim = 1),(0,2,1))

            c2q = tf.matmul(S_, q)
            q2c = tf.matmul(tf.matmul(S_, S_T), c)
            attention_outputs = [c, c2q, c * c2q, c * q2c]
            return tf.concat(attention_outputs, -1)


def flatten(tensor, keep):
    fixed_shape = tensor.get_shape().as_list()
    start = len(fixed_shape) - keep
    left = reduce(mul, [fixed_shape[i] or tf.shape(tensor)[i] for i in range(start)])
    out_shape = [left] + [fixed_shape[i] or tf.shape(tensor)[i] for i in range(start, len(fixed_shape))]
    flat = tf.reshape(tensor, out_shape)
    return flat

def reconstruct(tensor, ref, keep, dim_reduced_keep=None):
    dim_reduced_keep = dim_reduced_keep or keep

    ref_shape = ref.get_shape().as_list() # original shape
    tensor_shape = tensor.get_shape().as_list() # current shape
    ref_stop = len(ref_shape) - keep # flatten dims list
    tensor_start = len(tensor_shape) - dim_reduced_keep  # start
    pre_shape = [ref_shape[i] or tf.shape(ref)[i] for i in range(ref_stop)] #
    keep_shape = [tensor_shape[i] or tf.shape(tensor)[i] for i in range(tensor_start, len(tensor_shape))] #
    # pre_shape = [tf.shape(ref)[i] for i in range(len(ref.get_shape().as_list()[:-keep]))]
    # keep_shape = tensor.get_shape().as_list()[-keep:]
    target_shape = pre_shape + keep_shape
    out = tf.reshape(tensor, target_shape)
    return out

def _linear(xs, output_size, bias, bias_start=0., scope=None):
    with tf.variable_scope(scope or 'linear_layer'):
        x = tf.concat(xs, -1)
        input_size = x.get_shape()[-1]
        W = tf.get_variable('W', shape=[input_size, output_size],
                            dtype=tf.float32,
                            )
        if bias:
            bias = tf.get_variable('bias', shape=[output_size],
                                   dtype=tf.float32,
                                   initializer=tf.constant_initializer(
                                       bias_start))
            out = tf.matmul(x, W) + bias
        else:
            out = tf.matmul(x, W)
        return out

def add_reg_without_bias(scope=None):
    scope = scope or tf.get_variable_scope().name
    variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
    counter = 0
    for var in variables:
        if len(var.get_shape().as_list()) <= 1: continue
        tf.add_to_collection('reg_vars', var)
        counter += 1
    return counter

def linear(args, output_size, bias, bias_start=0.0, scope=None,
           squeeze=False, wd=0.0, input_keep_prob=1.0,
           is_train=None):
    if args is None or (isinstance(args, (tuple, list)) and not args):
        raise ValueError("`args` must be specified")
    if not isinstance(args, (tuple, list)):
        args = [args]

    flat_args = [flatten(arg, 1) for arg in
                 args]  # for dense layer [(-1, d)]
    if input_keep_prob < 1.0:
        assert is_train is not None
        flat_args = [
            tf.cond(is_train, lambda: tf.nn.dropout(arg, input_keep_prob),
                    lambda: arg)  # for dense layer [(-1, d)]
            for arg in flat_args]
    flat_out = _linear(flat_args, output_size, bias, bias_start=bias_start,
                       scope=scope)  # dense
    out = reconstruct(flat_out, args[0], 1)  # ()
    if squeeze:
        out = tf.squeeze(out, [len(args[0].get_shape().as_list()) - 1])

    if wd:
        add_reg_without_bias()

    return out

def mask_for_high_rank(val, val_mask, name=None):
    val_mask = tf.expand_dims(val_mask, -1)
    return tf.multiply(val, tf.cast(val_mask, tf.float32),
                       name=name or 'mask_for_high_rank')


# # ----------- with normalization ------------
def bn_dense_layer(input_tensor, hn, bias, bias_start=0.0, scope=None,
                   activation='relu', enable_bn=True,
                   wd=0., keep_prob=1.0, is_train=None):
    if is_train is None:
        is_train = False

    # activation
    if activation == 'linear':
        activation_func = tf.identity
    elif activation == 'relu':
        activation_func = tf.nn.relu
    elif activation == 'elu':
        activation_func = tf.nn.elu
    elif activation == 'sigmoid':
        activation_func = tf.nn.sigmoid
    elif activation == 'tanh':
        activation_func = tf.nn.tanh
    else:
        raise AttributeError('no activation function named as %s' % activation)

    with tf.variable_scope(scope or 'bn_dense_layer'):
        linear_map = linear(input_tensor, hn, bias, bias_start, 'linear_map',
                            False, wd, keep_prob, is_train)
        if enable_bn:
            # with tf.variable_scope('bn_module'):
            #     linear_map = tf.cond(
            #         is_train,
            #         lambda: tf.contrib.layers.batch_norm(
            #             linear_map, center=True, scale=True, is_training=True,
            #             scope='bn'),
            #         lambda: tf.contrib.layers.batch_norm(
            #             linear_map, center=True, scale=True, is_training=False,
            #             scope='bn', reuse=True),
            #     )
            linear_map = tf.contrib.layers.batch_norm(
                linear_map, center=True, scale=True, is_training=is_train,
                updates_collections=None,  decay=0.9,
                scope='bn')

        return activation_func(linear_map)

VERY_NEGATIVE_NUMBER=-1e30

def exp_mask_for_high_rank(val, val_mask, name=None):
    val_mask = tf.expand_dims(val_mask, -1)
    return tf.add(val, (1 - tf.cast(val_mask, tf.float32)) * VERY_NEGATIVE_NUMBER,
                  name=name or 'exp_mask_for_high_rank')

def simple_block_attention(
        rep_tensor, rep_mask, block_len=5, scope=None, direction='forward',
        keep_prob=1., is_train=None, activation = 'elu'):
    def scaled_tanh(x, scale=5.):
        return scale * tf.nn.tanh(1. / scale * x)

    # rep_tensor: [B, L, D]
    # rep_mask: [B, L]
    bs, sl, vec = tf.shape(rep_tensor)[0],\
                  tf.shape(rep_tensor)[1], tf.shape(rep_tensor)[2]
    org_ivec = rep_tensor.get_shape().as_list()[2]
    ivec = org_ivec
    print('rep_tensor')
    print(rep_tensor)
    print('rep_mask')
    print(rep_mask)
    with tf.variable_scope(scope or 'block_simple'):
        # @1. split sequence
        with tf.variable_scope('split_seq'):
            block_num = tf.cast(tf.ceil(tf.divide(tf.cast(sl, tf.float32),
                                                  tf.cast(block_len, tf.float32))),
                                tf.int32) # L / block_len
            comp_len = block_num * block_len - sl

            rep_tensor_comp = tf.concat([rep_tensor, tf.zeros([bs, comp_len, org_ivec], tf.float32)], 1)
            rep_mask_comp = tf.concat([rep_mask, tf.cast(tf.zeros([bs, comp_len], tf.int32), rep_mask.dtype)], 1)

            rep_tensor_split_4dim = tf.reshape(rep_tensor_comp, [bs, block_num, block_len, org_ivec])  # bs*bn,bl,d

            rep_tensor_split = tf.reshape(rep_tensor_comp, [bs, block_num, block_len, org_ivec])  # bs*bn,bl,d
            rep_mask_split = tf.reshape(rep_mask_comp, [bs, block_num, block_len])  # bs, bn,bl

            print('rep_tensor_split')
            print(rep_tensor_split)
            print('rep_mask_split')
            print(rep_mask_split)

            # non-linear
            rep_map = bn_dense_layer(rep_tensor_split, ivec, True, 0., 'bn_dense_map', activation,
                                     False, 0, keep_prob, is_train)  # bs,bn,bl,vec
            print("rep_map_after_dense_layer")
            print(rep_map)
            rep_map_tile = tf.tile(tf.expand_dims(rep_map, 2), [1, 1, block_len, 1, 1])  # bs,bn,bl,bl,vec
            print("rep_map_tile")
            print(rep_map_tile)

            bn = block_num
            bl = block_len


        with tf.variable_scope('self_attention'):
            # @2.self-attention in block
            # mask generation
            sl_indices = tf.range(block_len, dtype=tf.int32)
            sl_col, sl_row = tf.meshgrid(sl_indices, sl_indices)
            if direction == 'forward':
                direct_mask = tf.greater(sl_row, sl_col)  # bl,bl
                print('forward direction mask')
                print(direct_mask)
            else:
                direct_mask = tf.greater(sl_col, sl_row)  # bl,bl
            direct_mask_tile = tf.tile(
                tf.expand_dims(tf.expand_dims(direct_mask, 0), 0), [bs, bn, 1, 1])  # bs,bn,bl,bl
            rep_mask_tile_1 = tf.tile(tf.expand_dims(rep_mask_split, 2), [1, 1, bl, 1])  # bs,bn,bl,bl
            rep_mask_tile_2 = tf.tile(tf.expand_dims(rep_mask_split, 3), [1, 1, 1, bl])  # bs,bn,bl,bl
            rep_mask_tile = tf.logical_and(rep_mask_tile_1, rep_mask_tile_2)
            print('rep_mask_tile')
            print(rep_mask_tile)
            print('direct_mask_tile')
            print(direct_mask_tile)
            attn_mask = tf.logical_and(direct_mask_tile, rep_mask_tile, name='attn_mask')  # bs,bn,bl,bl
            print('attn_mask')
            print(attn_mask)

            # attention
            f_bias = tf.get_variable('f_bias', [ivec], tf.float32, tf.constant_initializer(0.))
            dependent_head = linear(
                rep_map, 2 * ivec, False, 0., 'linear_dependent_head', False, 0.0, keep_prob, is_train)  # bs,bn,bl,2vec
            dependent, head = tf.split(dependent_head, 2, 3)
            dependent_etd = tf.expand_dims(dependent, 2)  # bs,bn,1,bl,vec
            head_etd = tf.expand_dims(head, 3)  # bs,bn,bl,1,vec
            print('dependent_etd')
            print(dependent_etd)
            print('head_etd')
            print(head_etd)
            logits = scaled_tanh(dependent_etd + head_etd + f_bias, 5.0)  # bs,bn,bl,bl,vec
            logits_masked = exp_mask_for_high_rank(logits, attn_mask)
            attn_score = tf.nn.softmax(logits_masked, 3)  # bs,bn,bl,bl,vec
            attn_score = mask_for_high_rank(attn_score, attn_mask)  # bs,bn,bl,bl,vec
            print('attn_score')
            print(attn_score)
            self_attn_result = tf.reduce_sum(attn_score * rep_map_tile, 3)  # bs,bn,bl,vec
            print('self_attn_result')
            print(self_attn_result)

        with tf.variable_scope('source2token_self_attn'):
            inter_block_logits = bn_dense_layer(self_attn_result, ivec, True, 0., 'bn_dense_map', 'linear',
                                                False, 0.0, keep_prob, is_train)  # bs,bn,bl,vec
            inter_block_logits_masked = exp_mask_for_high_rank(
                inter_block_logits, rep_mask_split)  # bs,bn,bl,vec
            inter_block_soft = tf.nn.softmax(inter_block_logits_masked,
                                             2)  # bs,bn,bl,vec
            inter_block_attn_output = tf.reduce_sum(
                self_attn_result * inter_block_soft, 2)  # bs,bn,vec
            print('inter_block_soft')
            print(inter_block_soft)
            print(inter_block_attn_output)

        with tf.variable_scope('self_attn_inter_block'):
            inter_block_attn_output_mask = tf.cast(tf.ones([bs, bn], tf.int32), tf.bool)
            block_ct_res = directional_attention_with_dense(
                inter_block_attn_output, inter_block_attn_output_mask,
                scope = 'directional_attention_with_dense',
                direction='forward',
                keep_prob=keep_prob, is_train=is_train
            )  # [bs,bn,vec]

            block_ct_res_tile = tf.tile(tf.expand_dims(block_ct_res, 2), [1, 1, bl, 1])#[bs,bn,vec]->[bs,bn,bl,vec]
            print('block_ct_res_tile')
            print(block_ct_res_tile)

        with tf.variable_scope('combination'):
            # input:1.rep_map[bs,bn,bl,vec]; 2.self_attn_result[bs,bn,bl,vec]; 3.rnn_res_tile[bs,bn,bl,vec]
            rep_tensor_with_ct = tf.concat([rep_tensor_split_4dim, self_attn_result, block_ct_res_tile], -1)  # [bs,bn,bl,3vec]
            print('rep_tensor_with_ct')
            print(rep_tensor_with_ct)
            new_context_and_gate = linear(rep_tensor_with_ct, 2 * ivec, True, 0., 'linear_new_context_and_gate',
                                          False, wd=0.0, input_keep_prob=keep_prob,
                                          is_train=is_train)  # [bs,bn,bl,2vec]
            print('new_context_and_gate')
            print(new_context_and_gate)
            new_context, gate = tf.split(new_context_and_gate, 2, 3)  # bs,bn,bl,vec
            if activation == "relu":
                new_context_act = tf.nn.relu(new_context)
            elif activation == "elu":
                new_context_act = tf.nn.elu(new_context)
            elif activation == "linear":
                new_context_act = tf.identity(new_context)
            else:
                raise RuntimeError
            gate_sig = tf.nn.sigmoid(gate)
            combination_res = gate_sig * new_context_act + (1 - gate_sig) * rep_tensor_split_4dim  # bs,bn,bl,vec

        with tf.variable_scope('restore_original_length'):
            combination_res_reshape = tf.reshape(combination_res, [bs, bn * bl, ivec])  # bs,bn*bl,vec
            output = combination_res_reshape[:, :sl, :]
            return output

def directional_attention_with_dense(
        rep_tensor, rep_mask, direction=None, scope=None,
        keep_prob=1., is_train=None, wd=0., activation='elu',
        tensor_dict=None, name=None, hn=None):

    def scaled_tanh(x, scale=5.):
        return scale * tf.nn.tanh(1./scale * x)

    bs, sl, vec = tf.shape(rep_tensor)[0], tf.shape(rep_tensor)[1], tf.shape(rep_tensor)[2]
    ivec = rep_tensor.get_shape().as_list()[2]
    ivec = hn or ivec
    with tf.variable_scope(scope or 'directional_attention_%s' % direction or 'diag'):
        # mask generation
        sl_indices = tf.range(sl, dtype=tf.int32)
        sl_col, sl_row = tf.meshgrid(sl_indices, sl_indices)
        if direction is None:
            direct_mask = tf.cast(tf.diag(- tf.ones([sl], tf.int32)) + 1, tf.bool)
        else:
            if direction == 'forward':
                direct_mask = tf.greater(sl_row, sl_col)
            else:
                direct_mask = tf.greater(sl_col, sl_row)
        direct_mask_tile = tf.tile(tf.expand_dims(direct_mask, 0), [bs, 1, 1])  # bs,sl,sl
        rep_mask_tile = tf.tile(tf.expand_dims(rep_mask, 1), [1, sl, 1])  # bs,sl,sl
        attn_mask = tf.logical_and(direct_mask_tile, rep_mask_tile)  # bs,sl,sl

        # non-linear
        rep_map = bn_dense_layer(rep_tensor, ivec, True, 0., 'bn_dense_map', activation,
                                 False, wd, keep_prob, is_train)
        rep_map_tile = tf.tile(tf.expand_dims(rep_map, 1), [1, sl, 1, 1])  # bs,sl,sl,vec
        rep_map_dp = tf.contrib.layers.dropout(rep_map, keep_prob=keep_prob, is_training=is_train)

        # attention
        with tf.variable_scope('attention'):  # bs,sl,sl,vec
            f_bias = tf.get_variable('f_bias',[ivec], tf.float32, tf.constant_initializer(0.))
            dependent = linear(rep_map_dp, ivec, False, scope='linear_dependent')  # bs,sl,vec
            dependent_etd = tf.expand_dims(dependent, 1)  # bs,1,sl,vec
            head = linear(rep_map_dp, ivec, False, scope='linear_head') # bs,sl,vec
            head_etd = tf.expand_dims(head, 2)  # bs,sl,1,vec

            logits = scaled_tanh(dependent_etd + head_etd + f_bias, 5.0)  # bs,sl,sl,vec

            logits_masked = exp_mask_for_high_rank(logits, attn_mask)
            attn_score = tf.nn.softmax(logits_masked, 2)  # bs,sl,sl,vec
            attn_score = mask_for_high_rank(attn_score, attn_mask)

            attn_result = tf.reduce_sum(attn_score * rep_map_tile, 2)  # bs,sl,vec

        with tf.variable_scope('output'):
            o_bias = tf.get_variable('o_bias',[ivec], tf.float32, tf.constant_initializer(0.))
            # input gate
            fusion_gate = tf.nn.sigmoid(
                linear(rep_map, ivec, True, 0., 'linear_fusion_i', False, wd, keep_prob, is_train) +
                linear(attn_result, ivec, True, 0., 'linear_fusion_a', False, wd, keep_prob, is_train) +
                o_bias)
            output = fusion_gate * rep_map + (1-fusion_gate) * attn_result
            output = mask_for_high_rank(output, rep_mask)

        # save attn
        if tensor_dict is not None and name is not None:
            tensor_dict[name + '_dependent'] = dependent
            tensor_dict[name + '_head'] = head
            tensor_dict[name] = attn_score
            tensor_dict[name + '_gate'] = fusion_gate
        return output


def depthwise_separable_convolution(inputs, kernel_size, num_filters,
                                    scope = "depthwise_separable_convolution",
                                    bias = True, is_training = True, reuse = None):
    with tf.variable_scope(scope, reuse = reuse):
        shapes = inputs.shape.as_list()
        depthwise_filter = tf.get_variable("depthwise_filter",
                                        (kernel_size[0], kernel_size[1], shapes[-1], 1),
                                        dtype = tf.float32,
                                        regularizer=regularizer,
                                        initializer = initializer_relu())
        pointwise_filter = tf.get_variable("pointwise_filter",
                                        (1,1,shapes[-1],num_filters),
                                        dtype = tf.float32,
                                        regularizer=regularizer,
                                        initializer = initializer_relu())
        outputs = tf.nn.separable_conv2d(inputs,
                                        depthwise_filter,
                                        pointwise_filter,
                                        strides = (1,1,1,1),
                                        padding = "SAME")
        if bias:
            b = tf.get_variable("bias",
                    outputs.shape[-1],
                    regularizer=regularizer,
                    initializer = tf.zeros_initializer())
            outputs += b
        outputs = tf.nn.relu(outputs)
        return outputs

def layer_dropout(inputs, residual, dropout):
    pred = tf.random_uniform([]) < dropout
    return tf.cond(pred, lambda: residual, lambda: tf.nn.dropout(inputs, 1.0 - dropout) + residual)

def slice(input_seq_emb, left_context):
    # [B, L, D]
    emb_dim = input_seq_emb.get_shape().as_list()[-1]
    sequence_length = input_seq_emb.get_shape().as_list()[1]
    print('slice context:{}, sequence_length:{}'.format(left_context, sequence_length))
    # [B, 1, D]
    first_column = input_seq_emb[:, 0:1, :]
    first_column_tiled = tf.tile(first_column, [1, left_context, 1])
    print('first_column_tiled')
    print(first_column_tiled)

    # [B, L+left_context, D]
    pad_input_seq_emb = tf.concat([first_column_tiled, input_seq_emb], -2)
#    pad_input_seq_emb = tf.Print(pad_input_seq_emb, [pad_input_seq_emb], message = 'pad_input_seq_emb')
    sliced_contexts = []
    for idx in range(left_context):
       # [B, L, D]
       sliced_context = pad_input_seq_emb[:, idx:idx+sequence_length, :] 
       #sliced_context = tf.expand_dims(sliced_context, -2)
       sliced_contexts.append(sliced_context)
    return tf.concat(sliced_contexts, -1) 
     

def conv_block(inputs, kernel_size, dilation_rate = 1, scope = "conv_block", is_training = True,
               reuse = None, l2_reg = 0.0, dropout = 0.0):
    with tf.variable_scope(scope, reuse = reuse):
        emb_dim = inputs.get_shape().as_list()[-1]
        sequence_length = inputs.get_shape().as_list()[-2]
        first_column = inputs[:, 0:1, :]
        # [B, K, D]
        first_column_tiled = tf.tile(first_column, [1, kernel_size + dilation_rate - 1, 1])
        # [B, L+K, D]
        padded_inputs = tf.concat([first_column_tiled, inputs], -2)
        conv_output = tf.layers.conv1d(padded_inputs, emb_dim, kernel_size, dilation_rate = dilation_rate, padding='VALID', kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_reg))
        conv_output = conv_output[:, 0:sequence_length, :]
        return conv_output

def variable_summaries(var, name=""):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries_' + name):
    print('variable summary of summaries_' + str(name))
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)


def global_average_pooling_2d(input_x):
    pooling_inst = tf.keras.layers.GlobalAveragePooling2D()
    pooled_x = pooling_inst(input_x)
    pooled_x = tf.expand_dims(pooled_x, 1)
    pooled_x = tf.expand_dims(pooled_x, 1)
    return pooled_x

def squeeze_excitation_layer(input_x, ratio, l2_reg = 0.1, out_dim=None, layer_name='', ret_excitation=False):
    if out_dim is None:
        out_dim = input_x.get_shape().as_list()[-1]
    
    with tf.name_scope(layer_name + "squeeze_excitation"),\
         tf.variable_scope(layer_name + "squeeze_excitation") :
        squeeze = global_average_pooling_2d(input_x)

        print('squeeze pooled output')
        print(squeeze)
        #squeeze = tf.Print(squeeze, [squeeze, tf.shape(squeeze), tf.reduce_mean(squeeze)], message = 'squeeze_' + str(layer_name))

        excitation = tf.layers.dense(squeeze, out_dim / ratio,
                                     activation=tf.nn.relu,
                                     use_bias = False,
                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(
                                         l2_reg),
                                     name=layer_name+'fully_connected1')

        excitation = tf.layers.dense(excitation, units=out_dim,
                                     activation=tf.nn.sigmoid,
                                     use_bias = False,
                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(
                                         l2_reg),
                                     name=layer_name+'fully_connected2')


        excitation = tf.reshape(excitation, [-1,1,1,out_dim])

        excitation = tf.nn.softmax(excitation)

        print('squeeze excitation output')
        print(excitation)

        #excitation = tf.Print(excitation, [excitation, tf.shape(excitation), tf.reduce_mean(excitation)], message = 'excitation_' + str(layer_name), summarize=50)

        scale = input_x * excitation

        #scale = tf.Print(scale, [scale, tf.shape(scale), tf.reduce_mean(scale), tf.shape(input_x)], message = 'scale_' + str(layer_name), summarize=50)

        if ret_excitation:
           return scale, excitation
        else:
           return scale

def squeeze_and_excite_seq_emb(seq_emb, ratio):
    if ratio <= 0:
        return seq_emb

    # [B, L, D]
    seq_emb_shape = seq_emb.get_shape().as_list()
    batch_size = seq_emb_shape[0]
    seq_length = seq_emb_shape[1]
    emb_dim = seq_emb_shape[2]
    seq_emb_reshape = tf.reshape(seq_emb,
                                 [batch_size * seq_length, 1, 1, emb_dim])
    squeeze_excitation_output = squeeze_excitation_layer(seq_emb_reshape, ratio)
    return tf.reshape(squeeze_excitation_output, seq_emb_shape)

def squeeze_and_excite_context_emb_layers(context_emb_layers, ratio):
    if ratio <= 0:
        return context_emb_layers 

    # [B, L,l, D]
    seq_emb = context_emb_layers
    seq_emb_shape = seq_emb.get_shape().as_list()
    batch_size = seq_emb_shape[0]
    seq_length = seq_emb_shape[1]
    num_layers = seq_emb_shape[-2]
    emb_dim = seq_emb_shape[-1]
    seq_emb_reshape = tf.reshape(seq_emb,
                                 [batch_size * seq_length, 1, num_layers, emb_dim])
    squeeze_excitation_output = squeeze_excitation_layer(seq_emb_reshape, ratio)
    return tf.reshape(squeeze_excitation_output, seq_emb_shape)

def outer_product(x, y):
    # x: [B, L, D]
    # y: [B, L, D]
    xp = tf.expand_dims(x, -1)
    yp = tf.expand_dims(y, -2)
    xyp = tf.matmul(xp, yp)
    return tf.reduce_mean(xyp, -1)

def bi_linear(x, y, scope):
    y_shape = y.get_shape().as_list()
    n_dims = y_shape[-1]
    with tf.variable_scope(scope + "_bi_linear", reuse=tf.AUTO_REUSE):
        xw = tf.layers.dense(x, n_dims)
        return xw * y

def bi_linear_product(x, scope=None):
    x_shape = x.get_shape().as_list()
    batch_size = x_shape[0]
    feat_num = x_shape[1]
    dim = x_shape[-1]

    with tf.variable_scope(scope + "_bi_linear", reuse=tf.AUTO_REUSE):
        # [B, L, d]
        xr = tf.reshape(x, [-1, dim])
        W = tf.get_variable("bi_linear_weights",
                            [dim, dim],
                            regularizer=regularizer,
                            initializer=initializer_relu())
        xw  = tf.matmul(xr, W)
        xw = tf.reshape(xw, [-1, feat_num, dim])

    xw_splits = tf.split(xw, feat_num, 1)

    y_splits = tf.split(x, feat_num, 1)

    total_product_vec = y_splits

    for idx in xrange(len(xw_splits)):
        xw_split = xw_splits[idx]
        for jdx in xrange(len(y_splits)):
            if idx == jdx:
                continue
            y_split = y_splits[idx]
            print('xsplit at idx:' + str(idx))
            print(xw_split)
            print('ysplit at jdx:' + str(jdx))
            print(y_split)
            total_product_vec.append(xw_split * y_split)

    # [B, L', D]
    return tf.concat(total_product_vec, 1)

def bi_hadamard_product(x, scope=None):
    x_shape = x.get_shape().as_list()
    feat_num = x_shape[1]

    x_splits = tf.split(x, feat_num, 1)

    total_product_vec = []

    for idx in xrange(len(x_splits)):
        x_split = x_splits[idx]
        total_product_vec.append(x_split)
        for jdx in xrange(len(x_splits)):
            if idx >= jdx:
                continue
            y_split = x_splits[idx]
            print('xsplit at idx:' + str(idx))
            print(x_split)
            print('ysplit at jdx:' + str(jdx))
            print(y_split)
            total_product_vec.append(x_split * y_split)

    # [B, L', D]
    return tf.concat(total_product_vec, 1)

def prediction_logits_with_atten(input_emb, target_emb, num_heads, input_masks):
    # [N, T_q, C]
    query = target_emb
    # [N, T_k, C]
    key = input_emb
    value = input_emb

    Q_ = tf.concat(tf.split(query, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
    K_ = tf.cncat(tf.split(key, num_heads, axis=2), axis=0) # [h*N, T_k, C/h]

    outputs = tf.matmul(Q_, K_, transpose_b=True) # [h*N, T_q, T_k]

    outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

    diag_vals = tf.ones_like(outputs)  # [h*N, T_q, T_k]
    tril = tf.matrix_band_part(diag_vals, -1, 0)
    tril_diag = tf.matrix_band_part(tril, 0, 0)
    masks = tril - tril_diag

    paddings = tf.ones_like(masks) * (-2 ** 32 + 1)
    outputs = tf.where(tf.equal(masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

    key_masks = tf.squeeze(input_masks, -1)
    key_masks = tf.tile(key_masks, [num_heads, 1])  # (h*N, T_k)
    key_masks = tf.tile(tf.expand_dims(key_masks, 1),
                        [1, tf.shape(query)[1], 1])  # (h*N, T_q, T_k)

    paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
    outputs = tf.where(tf.equal(key_masks, 0), paddings,
                       outputs)  # (h*N, T_q, T_k)

    outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)

    outputs = tf.matmul(outputs, value)  # ( h*N, T_q, C/h)

    outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2) # [N, T_q, C]

    outputs_input_comb = tf.concat([input_emb, outputs], -1)

    # [N, T_q, 1]
    return tf.squeeze(tf.layers.dense(outputs_input_comb, 1))


def prediction_logits(input_emb, target_emb, l2_reg,
                      num_heads, input_masks,
                      scope='predict_logits', output_logits_mode=0):
    """

    :param input_seq_emb_list: {[B, L, D]}
    :param target_emb:  [B, L, D]
    :param args:
    :return:
    """
    # Self-attention
    atten_emb_list = []
    assert_ops = []
    print('input_emb')
    print(input_emb)
    print('target_emb')
    print(target_emb)
    if output_logits_mode == 1:
        output_emb = outer_product(input_emb, target_emb)
        output = tf.reduce_sum(output_emb, -1)
    elif output_logits_mode == 2:
        output_emb = input_emb * target_emb
        print('output_emb_from_dot_prodct_atten')
        print(output_emb)
        output = tf.reduce_sum(output_emb, -1)
        output= tf.Print(output, [output, tf.shape(output)], message = 'output_emb_from_dot_prodct_atten')
    elif output_logits_mode == 3:
        print("bilinear input")
        print(input_emb)
        print(target_emb)
        output_emb = bi_linear(input_emb, target_emb, scope)
        print('output_emb')
        print(output_emb)
        output = tf.reduce_sum(output_emb, -1)
    elif output_logits_mode == 4:
        output =\
            prediction_logits_with_atten(input_emb,
                                         target_emb, num_heads, input_masks)
    with tf.control_dependencies(assert_ops):
       return tf.identity(output)

def aux_loss(pos_logits, neg_logits, istarget):
    return tf.reduce_sum(
        - tf.log(tf.sigmoid(pos_logits) + 1e-24) * istarget -
        tf.log(1 - tf.sigmoid(neg_logits) + 1e-24) * istarget
    ) / tf.reduce_sum(istarget)

def bpr_loss(pos_logits, neg_logits, istarget):
    return -tf.reduce_sum(tf.log(
        tf.nn.sigmoid(pos_logits - neg_logits) + 1e-24) * istarget) \
           / tf.reduce_sum(istarget)

def atten_on_deep_context_vectors(lower_layers, current_layer, params, scope):
    # deep_context_vectors: [B, L, (l-1)*D]
    # current_layer: [B, L, D]

    deep_context_vectors_shape = lower_layers.get_shape().as_list()
    print("deep_context_vectors_shape")
    print(deep_context_vectors_shape)

    current_layer_shape = current_layer.get_shape().as_list()
    print("current_layer_shape")
    print(current_layer_shape)
#    current_layer = tf.Print(current_layer, [current_layer, tf.shape(current_layer), tf.reduce_mean(current_layer)], message = 'current_layer')

    # [B*L, D]
    current_layer_reshape = tf.reshape(current_layer,
                                       [-1, current_layer_shape[-1]])

#    current_layer_reshape = tf.Print(current_layer_reshape, [current_layer_reshape, tf.shape(current_layer_reshape), tf.reduce_mean(current_layer_reshape)], message = 'current_layer_reshape')
    dim = current_layer_shape[-1]
    context_dim = deep_context_vectors_shape[-1]

    # [B*L, l-1, D]
    deep_context_vectors_reshape = tf.reshape(lower_layers,
                                              [-1,  context_dim / dim, dim])

    print('deep_context_vector_reshape')
    print(deep_context_vectors_reshape)
#    deep_context_vectors_reshape = tf.Print(deep_context_vectors_reshape, [deep_context_vectors_reshape, tf.shape(deep_context_vectors_reshape), tf.reduce_mean(deep_context_vectors_reshape)], message = 'deep_context_vectors_reshape_next')

    # [B*L, 1, D]
    current_layer_reshape_exp = tf.expand_dims(current_layer_reshape, -2)

    print('current_layer_reshape_exp')
    print(current_layer_reshape_exp)

    # [B*L, l-1, D]
    mul_deep_cur = deep_context_vectors_reshape * current_layer_reshape_exp

    print('mul_deep_cur')
    print(mul_deep_cur)

#    mul_deep_cur = tf.Print(mul_deep_cur, [mul_deep_cur, tf.shape(mul_deep_cur), tf.reduce_mean(mul_deep_cur)], message = 'mul_deep_cur')
    # [B*L, l - 1]
    mul_deep_cur_sum = tf.reduce_sum(mul_deep_cur, -1)

    mul_deep_cur_softmax = tf.nn.softmax(mul_deep_cur_sum)

    print('mul_deep_cur_softmax')
    print(mul_deep_cur_softmax)
#    mul_deep_cur_softmax = tf.Print(mul_deep_cur_softmax, [mul_deep_cur_softmax, tf.shape(mul_deep_cur_softmax), tf.reduce_mean(mul_deep_cur_softmax)], message = 'mul_deep_cur_softmax')

    # [B*L, l - 1, D]
    deep_context_vectors_attn = deep_context_vectors_reshape * tf.expand_dims(mul_deep_cur_softmax, -1)
    print("deep_context_vectors_attn")
    print(deep_context_vectors_attn)
#    deep_context_vectors_attn= tf.Print(deep_context_vectors_attn, [deep_context_vectors_attn, tf.shape(deep_context_vectors_attn), tf.reduce_mean(deep_context_vectors_attn)], message = 'deep_context_vectors_attn')

    # [B*L, D]
    if hasattr(params, 'output_projection') and params.output_projection == 0:
        deep_context_vector_proj = tf.reshape(
        deep_context_vectors_attn, [
            tf.shape(deep_context_vectors_attn)[0], -1
        ]
        )

        with tf.variable_scope(scope + "_atten_deep_ctx", reuse=tf.AUTO_REUSE):
            deep_context_vector_pool = tf.layers.dense(deep_context_vector_proj,
                                                   dim)
    else:
        deep_context_vector_pool = tf.reduce_sum(deep_context_vectors_attn, -2)

    deep_context_vector_pool_shape = deep_context_vector_pool.get_shape().as_list()

    # [B, L, D]
    deep_context_vector_pool = tf.reshape(deep_context_vector_pool,
                                          [-1,
                                           deep_context_vectors_shape[1],
                                           deep_context_vector_pool_shape[-1]])

#    deep_context_vector_pool = tf.Print(deep_context_vector_pool, [deep_context_vector_pool, tf.shape(deep_context_vector_pool), tf.reduce_mean(deep_context_vector_pool)], message = 'deep_context_vector_pool_reshape')

    return deep_context_vector_pool

def build_interaction_model(nn_input, params, scope):
    # nn_input: [N, T_q, C]
    sequence_length = tf.shape(nn_input)[1]
    nn_input_splits = tf.concat(tf.split(nn_input, params.num_heads, axis=2),
                          axis=0)  # (h*N, T_q, C/h)

    # Explicit multi-head interaction
    print('apply cross upon heads')
    # [N*T_q, h, C/h]
    nn_input_reshape = tf.reshape(nn_input_splits, [-1, params.num_heads,
                                           params.hidden_units / params.num_heads])

    if params.multi_head_feat_interaction == 1:
        explicit_interaction =  build_simple_interaction_model(nn_input_reshape, params, scope)
        return tf.reshape(explicit_interaction, [-1, sequence_length, params.hidden_units])
    if params.multi_head_feat_interaction == 2:
        explicit_interaction = build_fibnet_interaction_model(nn_input_reshape, params, scope)
        explicit_interaction_shape = explicit_interaction.get_shape().as_list()
        return tf.reshape(explicit_interaction, [-1, sequence_length, explicit_interaction_shape[-1] * explicit_interaction_shape[-2]])
    if params.multi_head_feat_interaction == 3:
        explicit_interaction = build_deep_cross_interaction_model(nn_input_reshape, params, scope)
        return tf.reshape(explicit_interaction, [-1, sequence_length, params.hidden_units])
    if params.multi_head_feat_interaction == 4:
        explicit_interaction = build_xdeepfm_interaction_model(nn_input_reshape, params, scope)
        return tf.reshape(explicit_interaction, [-1, sequence_length, params.hidden_units])

    return None

def build_simple_interaction_model(nn_input, params, scope):
    print("Apply simple interaction model")
    # nn_input: [B, h, d]

    # [B, d, h]
    nn_input_transpose = tf.transpose(nn_input, [0, 2, 1])

    # [B, h, h]
    nn_input_cr = tf.matmul(nn_input, nn_input_transpose)
    nn_input_cr = tf.nn.softmax(nn_input_cr)

    print("nn_input")
    print(nn_input)
    print("nn_input_transpose")
    print(nn_input_transpose)
    print("nn_input_cr")
    print(nn_input_cr)

    # [B, h, d]
    return tf.matmul(nn_input_cr, nn_input)


def build_fibnet_interaction_model(nn_input, hparams, scope):
    print("Apply fibnet interaction model")
    # [B, h, d] h is the number of features
    nn_input_shape = nn_input.get_shape().as_list()
    # [B, d, h]
    nn_input_reshape = tf.transpose(nn_input, [0, 2, 1])
    # [B, 1, d, h]
    nn_input_reshape = tf.expand_dims(nn_input_reshape, 1)

    sq_reduction_ratio = hparams.sq_reduction_ratio
    nn_input_sq = squeeze_excitation_layer(nn_input_reshape,
                                           sq_reduction_ratio)
    nn_input_sq = tf.squeeze(nn_input_sq, 1)
    # [B, h, d]
    nn_input_sq = tf.transpose(nn_input_sq, [0, 2, 1])

    nn_input_interacted = bi_hadamard_product(nn_input, scope=scope+"_nn_input_interacted")
    nn_input_sq_interacted = bi_hadamard_product(nn_input_sq, scope=scope+"_nn_input_sq_interacted")

    nn_output = tf.concat([nn_input_interacted, nn_input_sq_interacted], 1)

    # [B, h, D]
    return nn_output

def build_deep_cross_interaction_model(nn_input, hparams, scope):
    print("Apply deep cross interaction model")
    # [b, h, d]
    layer_output = nn_input
    nn_input_shape = nn_input.get_shape().as_list()
    emb_dim = nn_input_shape[-1]
    for layer_idx in xrange(hparams.num_cross_layers):
        # [b, h, 1, d]
        prev_layer_exp = tf.expand_dims(layer_output, -2)
        # [b, h, d, 1]
        nn_input_exp = tf.expand_dims(nn_input, -1)
        print('prev_layer_exp at layer:' + str(layer_idx))
        print(prev_layer_exp)
        print('nn_input_exp at layer:' + str(layer_idx))
        print(nn_input_exp)

        prev_layer_exp_shape = prev_layer_exp.get_shape().as_list()
        batch_size = prev_layer_exp_shape[0]
        length = prev_layer_exp_shape[1]

        with tf.variable_scope(scope + "_dc_w_%d" % layer_idx):
            # [b, h, 1, 1]
            prev_ws = tf.layers.dense(prev_layer_exp, 1, use_bias=False)
            print("prev_ws at layer: " + str(layer_idx))
            print(prev_ws)
            # [B, h, d, 1]
            prev_cur_interact = tf.matmul(nn_input_exp, prev_ws)
            print("prev_cur_interact at layer: " + str(layer_idx))
            print(prev_cur_interact)
            # [B, h, d]
            prev_cur_interact = tf.squeeze(prev_cur_interact, -1)
            bs = tf.get_variable('dc_biases_%d' % layer_idx, [emb_dim],
                                 initializer = tf.zeros_initializer())
            prev_cur_interact_bias = prev_cur_interact + bs
            print("prev_cur_interact_bias at layer: " + str(layer_idx))
            print(prev_cur_interact_bias)
            layer_output = prev_cur_interact_bias + nn_input
    return layer_output

def build_xdeepfm_interaction_model(nn_input, hparams, scope):
    print("Apply deep fm interaction model")
    # shape of nn_input: [B, h, D]
    nn_input_shape = nn_input.get_shape().as_list()
    hidden_nn_layers = []
    field_nums = []
    final_len = 0
    field_num = nn_input_shape[-2]
    emb_dim = nn_input_shape[-1]
    field_nums.append(int(field_num))
    hidden_nn_layers.append(nn_input)
    final_result = []
    # {[B, h]}
    split_tensor0 = tf.split(hidden_nn_layers[0], emb_dim * [1], -1)
    with tf.variable_scope(scope + "_interactoin_exfm"):
        for idx, layer_size in enumerate(hparams.cross_layer_sizes):
            # {[B, h]}
            split_tensor = tf.split(hidden_nn_layers[-1], emb_dim * [1], -1)
            print('split_tensor')
            print(split_tensor)
            print(split_tensor0)
            dot_result_m = tf.matmul(split_tensor0, split_tensor,
                                     transpose_b=True)
            print('dot_result_m')
            print(dot_result_m)
            dot_result_o = tf.reshape(dot_result_m, shape=[emb_dim, -1,
                                                           field_nums[0] *
                                                           field_nums[-1]])
            dot_result = tf.transpose(dot_result_o, perm=[1, 0, 2])

            print('dot_result')
            print(dot_result)

            filters = tf.get_variable(name="f_" + str(idx),
                                      shape=[1, field_nums[-1] * field_nums[0],
                                             layer_size],
                                      dtype=tf.float32)
            # dot_result = tf.transpose(dot_result, perm=[0, 2, 1])
            curr_out = tf.nn.conv1d(dot_result, filters=filters, stride=1,
                                    padding='VALID')
            print('curr_out')
            print(curr_out)
            curr_out = tf.transpose(curr_out, perm=[0, 2, 1])

            print("residual at xdeepfm layer " + str(idx))
            print(curr_out)
            final_result.append(curr_out)

        result = tf.concat(final_result, axis=1)
        result = tf.reshape(result, [nn_input_shape[0], -1])
        result = tf.layers.dense(result, field_num * emb_dim)
        return tf.reshape(result, [-1, field_num, emb_dim])

def cal_relative_pos_impact(query, relative_pos_emb_table, num_heads):
    # Q: (h*N, T_q, C/h), rel_pos_emb_table: [T_q, T_q, h, C/h]
    query_shape = query.get_shape().as_list()
    sequence_length = query_shape[1]
    emb_dim = query_shape[-1]
    # split Q upon the sequence length dimension
    q_splits = tf.split(query, sequence_length, 1)
    # rel_pos_emb_table: [T_q, T_k, C]
    rel_pos_emb_splits = tf.split(relative_pos_emb_table, num_heads, axis=2) # [T_q, T_q, C/h]
    rel_pos_emb_splits = [tf.expand_dims(emb, -2) for emb in rel_pos_emb_splits]
    # [T_q, T_k, h, C/h]
    rel_pos_emb_splits = tf.concat(rel_pos_emb_splits, -2)
    print('rel_pos_emb_splits')
    print(rel_pos_emb_splits)

    # [h*T_q, N, C/h]
    query_reshape = tf.reshape(query, [num_heads * sequence_length, -1, emb_dim])
    # [h*T_q, C/h, T_k]
    rel_pos_emb_splits_reshape = tf.reshape(rel_pos_emb_splits, [-1, emb_dim,sequence_length])

    # [h*T_q, N, T_k]
    query_rel_pos_mul = tf.matmul(query_reshape, rel_pos_emb_splits_reshape)

    # [h*N, T_q, T_k]
    return tf.reshape(query_rel_pos_mul, [-1, sequence_length, sequence_length])

def gather_3d_along_seq_length_axis(sequence_emb, seq_indices):
    sequence_emb_shape = sequence_emb.get_shape().as_list()
    sequence_length = sequence_emb_shape[1]
    seq_indices_squeeze = tf.squeeze(seq_indices, -1)
    seq_indices_onehot = tf.one_hot(seq_indices_squeeze, sequence_length)
    seq_indices_onehot = tf.expand_dims(seq_indices_onehot, -1)
    sequence_emb_masked = seq_indices_onehot * sequence_emb
    sequence_emb_gathered = tf.reduce_sum(sequence_emb_masked, 1)
    return sequence_emb_gathered

def eu_distance(a, b):
    return tf.norm(a - b, ord=2, axis=-1) / a.get_shape().as_list()[-1]

def mutual_head_disagreement_regularization(outputs, mask, num_heads):
    #( h*N, T_q, C/h)
    split_by_heads = tf.split(outputs, num_heads, axis=0)

    cosine_sims = []
    cosine_sim_masks = []
    for i in xrange(len(split_by_heads)):
        for j in xrange(i + 1, len(split_by_heads)):
            split_by_heads[i] = tf.Print(split_by_heads[i], [split_by_heads[i], tf.reduce_max(split_by_heads[i]), tf.reduce_min(split_by_heads[i])], message='split_by_heads_%d' % i)
            split_by_heads[j] = tf.Print(split_by_heads[j], [split_by_heads[j], tf.reduce_max(split_by_heads[j]), tf.reduce_min(split_by_heads[j])], message='split_by_heads_%d' % j)
            cosine_sim = eu_distance(split_by_heads[i], split_by_heads[j])
            cosine_sim_exp = tf.expand_dims(cosine_sim, -1)
            masked_cosine_sim = tf.where(mask > 0, cosine_sim_exp, tf.zeros_like(cosine_sim_exp))
            masked_cosine_sim = tf.Print(masked_cosine_sim, [masked_cosine_sim, tf.shape(masked_cosine_sim)], message = 'cosine_sim_%d_%d' % (i, j), summarize=100)
            mask = tf.Print(mask, [mask, tf.shape(mask)], message = 'mask_%d_%d' % (i, j), summarize=100)
            cosine_sims.append(masked_cosine_sim)
            cosine_sim_masks.append(tf.cast(mask, masked_cosine_sim.dtype))

    cosine_sims_sum = tf.concat(cosine_sims, -1)
    cosine_sim_masks_sum = tf.concat(cosine_sim_masks, -1)

    cosine_sims_sum = tf.reduce_sum(cosine_sims_sum)
    cosine_sims_sum = tf.Print(cosine_sims_sum, [cosine_sims_sum], message = 'cosine_sims_sum')
    cosine_sim_masks_sum = tf.reduce_sum(cosine_sim_masks_sum)
    cosine_sim_masks_sum = tf.Print(cosine_sim_masks_sum, [cosine_sim_masks_sum], message = 'cosine_sim_masks_sum')
    
    return 1.0 - tf.nn.sigmoid(cosine_sims_sum / cosine_sim_masks_sum) 

def voting_vector_input_layer_transformation(input_layers):
    # [B, D*L]
    input_layers_concat = tf.concat(input_layers, -1)
#    input_layers_concat = tf.Print(input_layers_concat, [tf.shape(input_layers_concat)], message = 'input_layers_concat')
    layer_shape = input_layers_concat.get_shape().as_list()
    combined_emb_dim = layer_shape[-1]
    with tf.variable_scope("calcualte_voting_vector"):
        input_layers_fn_output = tf.layers.dense(input_layers_concat,
                                                 combined_emb_dim)
#    input_layers_fn_output = tf.Print(input_layers_fn_output, [tf.shape(input_layers_fn_output)], message = 'input_layers_fn_output')

    return tf.split(input_layers_fn_output, len(input_layers), axis=-1)

def squash_func(vector, axis):
    vec_norm = tf.sqrt(tf.reduce_sum(tf.square(vector), axis=axis, keep_dims=True) + 1e-8)
    scale = (vec_norm) / (1 + vec_norm)
    vec_1_norm = (tf.reduce_sum(tf.abs(vector), axis=axis, keep_dims=True) + 1e-8)
    squashed_vector = scale * (vector / vec_1_norm)
    return squashed_vector

def calculate_voting_vectors(input_layers, out_dim, num_out_caps, batch_size):
#    batch_size = tf.Print(batch_size, [batch_size], message = 'batch_size')
    # H(l) = F(H1, H2, .. HL)
    input_layer_shape = input_layers[0].get_shape().as_list()

    input_caps = len(input_layers)

    emb_dim = input_layer_shape[-1]

    input_layers_transformed =\
        [tf.expand_dims(trans_layer, -2)
         for trans_layer in input_layers]

    # [B, IC, D]
    input_layers_transformed = tf.concat(input_layers_transformed, -2)
#    input_layers_transformed = tf.Print(input_layers_transformed, [tf.shape(input_layers_transformed)], message='nput_layers_transformed', summarize=200)

    print("input layer transformed")
    print(input_layers_transformed)


    # [B, IC, D, 1]
    input_layers_transformed = tf.expand_dims(input_layers_transformed, -1)

    # [B, IC, 1, D, 1]
    input_layers_transformed = tf.expand_dims(input_layers_transformed, -3)

    # [B, IC, OC, D, 1]
    input_layers_transformed_tiled = tf.tile(input_layers_transformed,
                                             [1, 1, num_out_caps, 1, 1])

    print('input_layers_transformed_tiled')
    print(input_layers_transformed_tiled)

#    input_layers_transformed_tiled = tf.Print(input_layers_transformed_tiled, [tf.shape(input_layers_transformed_tiled)], message='nput_layers_transformed_tiled', summarize=200)
    with tf.variable_scope('voting_vector_projection_weights'):
        # [OD, D]
        weights = tf.get_variable(shape = [1, input_caps, num_out_caps, out_dim, emb_dim],
                                  name = 'weight_matrix',
                                  dtype = input_layers_transformed_tiled.dtype)
        tf.logging.info('voting tensor weights')
        tf.logging.info(weights)

#    weights = tf.Print(weights, [tf.shape(weights)], message = 'weights', summarize=200)

    # [B, IC, OC, OD, D]
    weights_tiled = tf.tile(weights,
                            [batch_size, 1, 1, 1, 1])
    print('weights_tiled')
    print(weights_tiled)
 #   weights_tiled = tf.Print(weights_tiled, [tf.shape(weights_tiled)], message='weights_tiled', summarize=200)

    # [B, IC, OC, OD, 1]
    voting_vector = tf.matmul(weights_tiled, input_layers_transformed_tiled)
#    voting_vector = tf.Print(voting_vector, [voting_vector, tf.shape(voting_vector)], message='voting_vector')

    # [B, IC, OC, OD]
    return voting_vector


def combine_representations_by_dynamic_routing_aggrement(layers, batch_size,
                                                         num_iterations,
                                                         num_output_capsules,
                                                         output_dim,
                                                         apply_layer_transform=False):
    # layers: {[batch_size, D]}
    layer_shape = layers[0].get_shape().as_list()
    emb_dim = layer_shape[-1]
    number_of_layers = len(layers)
    tf.logging.info('emb_dim: {}, number of layers: {}'.format(emb_dim , number_of_layers))
    # output: [batch_size, D/n * n], n is the number of output capsules

    print("number of input caps:" + str(number_of_layers))
    print("number of output caps:" + str(num_output_capsules))
    print("batch_size in routing:" + str(batch_size))
    print("output_dim in routing:" + str(output_dim))
    print("layer_transform_flag in routing:" + str(apply_layer_transform))
    print('input layers to dynamic routing')
    print(layers) 

    # [B, IC, OC, 1, 1]
    B = tf.zeros([batch_size, number_of_layers, num_output_capsules, 1, 1],
                 dtype=layers[0].dtype)

    print("B:")
    print(B)

    if apply_layer_transform:
        input_layers_transformed = voting_vector_input_layer_transformation(layers)
    else:
        input_layers_transformed = layers
    # [B, IC, OC, OD, 1]
    voting_tensor = calculate_voting_vectors(input_layers_transformed,
                             output_dim,
                             num_output_capsules, batch_size)

    print("voting_tensor")
    print(voting_tensor)

#    voting_tensor = tf.Print(voting_tensor, [voting_tensor, tf.shape(voting_tensor), tf.reduce_mean(voting_tensor)], message = 'voting_tensor', summarize=100)

    for iter in xrange(num_iterations):
        # [B, IC, OC, 1, 1]
        B_t = tf.squeeze(B, -1)
        B_t = tf.squeeze(B_t, -1)
        # [B, IC, OC]
        C = tf.nn.softmax(B_t)

        # [B, IC, OC, 1, 1]
        C = tf.expand_dims(C, -1)
        C = tf.expand_dims(C, -1)

#        C = tf.Print(C, [C, tf.shape(C)], message = 'C_%d' % iter, summarize = 100)

        with tf.variable_scope('dynamic_routing_aggrement_%d' % iter):
            # [B, IC, OC, OD, 1]
            weighted_voting_tensor = voting_tensor * C
            print("weighted_voting_tensor")
            print(weighted_voting_tensor)
            # [B, 1, OC, OD, 1]
            sj = tf.reduce_sum(weighted_voting_tensor, 1, keep_dims=True)
            print("sj")
            print(sj)
            # [B, 1, OC, OD, 1]
#            sj = tf.Print(sj, [sj], message = 'sj_%d' % iter)
            vj = squash_func(sj, axis=-2)
#            vj = tf.Print(vj, [vj], message = 'vj_%d' % iter)
            print("vj")
            print(vj)
            # [B, IC, OC, OD, 1]
            vj_tiled = tf.tile(vj, [1, number_of_layers, 1, 1, 1])
#            vj_tiled = tf.Print(vj_tiled, [vj_tiled], message = 'vj_tiled_%d' % iter)

            # [B, IC, OC, OD, 1]
            voting_vj = vj_tiled * voting_tensor
#            voting_vj = tf.Print(voting_vj, [voting_vj], message = 'voting_vj_%d' % iter)

            # [B, IC, OC, 1, 1]
            update_b = tf.reduce_sum(voting_vj, -2, keep_dims=True)

            # voting_tensor: [B, IC, OC, OD, 1]
            # vj_tiled: [B, IC, OC, OD, 1]
            #update_b = tf.matmul(voting_tensor, vj_tiled, transpose_a=True,
             #             name='agreement')
#            update_b = tf.Print(update_b, [update_b], message = 'update_b_%d' % iter)

            B = B + update_b

            output_capsules = vj

    #B = tf.Print(B, [B, tf.reduce_mean(B), tf.reduce_max(B), tf.shape(B)], message = 'B', summarize=100)

    print('output_capsules')
    print(output_capsules)

    output_capsules_shape = output_capsules.get_shape().as_list()

    final_output_dim = output_capsules_shape[-2] * output_capsules_shape[-3]

    output_capsules = tf.reshape(output_capsules, [-1, final_output_dim])

    print('output_capsules_reshape')
    print(output_capsules)

    return output_capsules, B

def combine_representations_by_em_routing_aggrement(layers, batch_size,
                                                    num_iterations,
                                                    num_output_capsules,
                                                    output_dim,
                                                    apply_layer_transform=True
                                                    ):
    # layers: {[batch_size, D]}
    layer_shape = layers[0].get_shape().as_list()
    emb_dim = layer_shape[-1]
    number_of_layers = len(layers)
    # output: [batch_size, D/n * n], n is the number of output capsules

    print("number of input caps:" + str(number_of_layers))
    print("number of output caps:" + str(num_output_capsules))


    # [B, IC, OC, 1, 1]
    C = tf.ones([batch_size, number_of_layers, num_output_capsules, 1, 1],
                 dtype=layers[0].dtype) * 1.0 / num_output_capsules

    print("C:")
    print(C)

    # [B, IC, OC, OD, 1]
    if apply_layer_transform:
        input_layers_transformed = voting_vector_input_layer_transformation(
            layers)
    else:
        input_layers_transformed = layers
    voting_tensor = calculate_voting_vectors(input_layers_transformed,
                                             output_dim,
                                             num_output_capsules, batch_size)

    print("voting_tensor")
    print(voting_tensor)

    input_layers_transformed =\
        [tf.expand_dims(trans_layer, -2)
         for trans_layer in input_layers_transformed]

    # [B, IC, D]
    input_layers_transformed = tf.concat(input_layers_transformed, -2)

    # [B, IC, 1]
    input_layers_act_prob = tf.layers.dense(input_layers_transformed, 1)

    input_layers_act_prob = tf.nn.sigmoid(input_layers_act_prob)

    params = dict()

    params['num_output_capsules'] = num_output_capsules
    params['batch_size'] = batch_size

    beta_a = tf.get_variable(
        name='beta_a', shape=[1, 1, 1, 1, 1], dtype=tf.float32,
        initializer=tf.contrib.layers.xavier_initializer()
    )

    beta_u = tf.get_variable(
        name='beta_u', shape=[1, 1, num_output_capsules, 1, 1], dtype=tf.float32,
        initializer=tf.contrib.layers.xavier_initializer()
    )

    params['beta_a'] = beta_a
    params['beta_u'] = beta_u


    for iter in xrange(num_iterations):
        C = tf.Print(C, [tf.reduce_min(C), tf.reduce_max(C), tf.reduce_mean(C)], message = 'C')

        inverse_temperature = 1 + (num_iterations - 1) * iter / max(1.0,
                                                                    iter - 1.0)

        params['lambda'] = inverse_temperature

   
        (out_cap_mean, out_cap_var, out_cap_act_prob) =\
            em_routing_m_step(C, input_layers_act_prob, voting_tensor, params)
        C = em_routing_e_step(out_cap_mean, out_cap_var, out_cap_act_prob, voting_tensor)
        print('C')
        print(C)

    return out_cap_mean * out_cap_act_prob, {'out_cap_mean': out_cap_mean, 'out_cap_act_prob':out_cap_act_prob}

def em_routing_m_step(C, input_layers_act_prob, voting_tensor, params):
    # C: [B, IC, OC, 1, 1]
    # input_layers_act_prob: [B, IC, 1]
    # voting_tensor: [B, IC, OC, OD, 1]

    def calculate_output_cap_mean_var(C, V):
        # C: [B, IC, OC, 1, 1]
        # V: [B, IC, OC, OD, 1]

        # [B, IC, OC, OD, 1]
        cv = C * V

        print("cv")
        print(cv)

        cv = tf.Print(cv , [tf.reduce_min(cv), tf.reduce_max(cv), tf.reduce_mean(cv), tf.shape(cv)], message = 'cv', summarize=100)

        # [B, 1, OC, OD, 1]
        cv_sum = tf.reduce_sum(cv, 1, keep_dims=True)

        print('cv_sum')
        print(cv_sum)

        # [B, 1, OC, 1, 1]
        c_sum = tf.reduce_sum(C, 1, keep_dims=True) + 1e-8

        print('c_sum')
        print(c_sum)

        # [B, 1, OC, OD, 1]
        mean = cv_sum / c_sum

        mean = tf.Print(mean , [tf.reduce_min(mean), tf.reduce_max(mean), tf.reduce_mean(mean), tf.shape(mean)], message = 'mean', summarize=100)

        v_shape = V.get_shape().as_list()

        # [B, IC, OC, OD, 1]
        mean_tiled = tf.tile(mean, [1, v_shape[1], 1, 1, 1])

        # [B, IC, OC, OD, 1]
        cv_minus_mean = C * tf.square(V - mean_tiled)

        print("cv_minus_mean")
        print(cv_minus_mean)

        # [B, 1, OC, OD, 1]
        cv_minus_mean_sum = tf.reduce_sum(cv_minus_mean, 1, keep_dims=True)

        cv_minus_mean_sum = tf.Print(cv_minus_mean_sum, [tf.reduce_min(cv_minus_mean_sum), tf.reduce_max(cv_minus_mean_sum), tf.reduce_mean(cv_minus_mean_sum)], message = 'cv_minus_mean_sum')

        c_sum = tf.Print(c_sum, [tf.reduce_min(c_sum), tf.reduce_max(c_sum), tf.reduce_mean(c_sum)], message = 'c_sum')


        # [B, 1, OC, OD, 1]
        var = cv_minus_mean_sum / c_sum

        print('var')
        print(var)

        var = tf.Print(var , [tf.reduce_min(var), tf.reduce_max(var), tf.reduce_mean(var), tf.shape(var)], message = 'var', summarize=100)

        # mean: [B, 1, OC, OD, 1]
        # var: [B, 1, OC, OD, 1]
        return mean, var

    def calculate_out_cap_cost(var, C):
        # C: [B, IC, OC, 1, 1]
        # var: [B, 1, OC, OD, 1]
        # reduce_sum(C, 1): [B, 1, OC, 1, 1]
        import math
        # [B, 1, OC, OD, 1]
        return (tf.log(var+1e-5) * 0.5 + (1 + tf.log(2 * tf.constant(math.pi))) / 2) * tf.reduce_sum(C, 1, keep_dims=True)

    def calculate_out_cap_act_prob(C, cost, lamb, beta_a, beta_u):
        # C: [B, IC, OC, 1, 1]
        # cost: [B, 1, OC, OD, 1]

        # [B, 1, OC, 1, 1]
        c_sum = tf.reduce_sum(C, 1, keep_dims=True)
        c_sum = tf.Print(c_sum, [tf.reduce_min(c_sum), tf.reduce_max(c_sum), tf.reduce_mean(c_sum), tf.shape(c_sum)], message = 'c_sum', summarize=100)
        # [B, 1, OC, 1, 1]
        cost_sum = tf.reduce_sum(cost, -2, keep_dims=True)
        cost_sum = tf.Print(cost_sum, [tf.reduce_min(cost_sum), tf.reduce_max(cost_sum), tf.reduce_mean(cost_sum), tf.shape(cost_sum)], message = 'cost_sum', summarize=100)

        # [B, 1, OC, 1, 1]
        return tf.nn.sigmoid(lamb * (beta_a - beta_u * c_sum - cost_sum))

    input_layers_act_prob_exp = tf.expand_dims(input_layers_act_prob, -1)
    # [B, IC, 1, 1, 1]
    input_layers_act_prob_exp = tf.expand_dims(input_layers_act_prob_exp, -1)

    mean, variance = calculate_output_cap_mean_var(
        C * input_layers_act_prob_exp, voting_tensor)
    cost = calculate_out_cap_cost(variance, C)

    print('cost')
    print(cost)

    cost = tf.Print(cost, [tf.reduce_mean(cost), tf.reduce_min(cost), tf.reduce_max(cost)], message = 'cost')

    print('params')
    print(params)

    lamb = params['lambda']
    beta_a = params['beta_a']
    beta_u = params['beta_u']

    out_cap_act_prob = calculate_out_cap_act_prob(C, cost, lamb, beta_a, beta_u)
    out_cap_act_prob= tf.Print(out_cap_act_prob, [tf.reduce_mean(out_cap_act_prob), tf.reduce_min(out_cap_act_prob), tf.reduce_max(out_cap_act_prob)], message = 'out_cap_act_prob')

    print('out_cap_act_prob')
    print(out_cap_act_prob)

    return mean, variance, out_cap_act_prob


def em_routing_e_step(out_cap_mean, out_cap_var, out_cap_act_prob, voting_tensor):
    def calculate_output_cap_gaussian(voting_tensor, mean, variance):
        # voting_tensor: [B, IC, OC, OD, 1]
        # mean: [B, 1, OC, OD, 1]
        # variance: [B, 1, OC, OD, 1]
        import math

        norm_denom = tf.exp((-1.0 * tf.square(voting_tensor - mean)) / (2 * variance + 1e-4)) 
        norm_nom = tf.sqrt(2 * tf.constant(math.pi) * variance)+1e-4
        norm_denom = tf.Print(norm_denom, [tf.reduce_mean(norm_denom), tf.reduce_min(norm_denom), tf.reduce_max(norm_denom)], message = 'norm_denom')
        norm_nom = tf.Print(norm_nom, [tf.reduce_mean(norm_nom), tf.reduce_min(norm_nom), tf.reduce_max(norm_nom)], message = 'norm_nom')
        # [B, IC, OC, 1, 1]
        return tf.reduce_sum(-1.0 * norm_denom / norm_nom, -2, keep_dims=True) 

    def update_assignment_prob(out_cap_act_prob, output_cap_gaussian):
        # out_cap_act_prob: [B, 1, OC, 1, 1]
        # output_cap_gaussian: [B, IC, OC, 1, 1]

        # [B, IC, OC, 1, 1]
        output_cap_assignments = out_cap_act_prob * output_cap_gaussian
        output_cap_assignments= tf.Print(output_cap_assignments, [tf.reduce_mean(output_cap_assignments), tf.reduce_min(output_cap_assignments), tf.reduce_max(output_cap_assignments)], message = 'output_cap_assignments')

        # [B, IC, OC, 1, 1]
        output_cap_assignments_sum = tf.reduce_sum(output_cap_assignments,
                                                   -2, keep_dims=True) + 1e-5
        output_cap_assignments_sum = tf.Print(output_cap_assignments_sum, [tf.reduce_mean(output_cap_assignments_sum), tf.reduce_min(output_cap_assignments_sum), tf.reduce_max(output_cap_assignments_sum)], message = 'output_cap_assignments_sum')

        return output_cap_assignments / output_cap_assignments_sum

    output_cap_gaussian = calculate_output_cap_gaussian(voting_tensor,
                                                        out_cap_mean,
                                                        out_cap_var)

    print('output_cap_gaussian')
    print(output_cap_gaussian)
    output_cap_gaussian = tf.Print(output_cap_gaussian, [tf.reduce_mean(output_cap_gaussian), tf.reduce_min(output_cap_gaussian), tf.reduce_max(output_cap_gaussian)], message = 'output_cap_gaussian')

    return update_assignment_prob(out_cap_act_prob, output_cap_gaussian)
