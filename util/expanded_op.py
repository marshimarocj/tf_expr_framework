import tensorflow as tf

# Note: the concat operation leads to batch_size as the inner for loop and 
# max_words_in_caption - 1 as outer for loop
# <BOS> is not in the output, so need to delete the 1st word in _captionids and _caption_masks when calculating loss
# tricky, refer to figure.3 in paper "Show and Tell: A Neural Image Caption Generator" for details
def cross_entropy_loss_on_rnn_outputs(_captionids, _caption_masks, output_ops,
    softmax_W, softmax_B,
    basegraph, name_scope):
  with basegraph.as_default():
    with tf.variable_scope(name_scope):
      # ((max_words_in_caption-1)*batch_size, dim_hidden)
      output_ops = tf.concat(0, output_ops)
      # ((max_words_in_caption-1)*batch_size, num_words)
      logits = tf.nn.xw_plus_b(output_ops, softmax_W, softmax_B)
      
      return cross_entropy_loss_on_rnn_logits(
        _captionids, _caption_masks, logits, basegraph, name_scope)


# Note: the concat operation leads to batch_size as the inner for loop and 
# max_words_in_caption - 1 as outer for loop
# <BOS> is not in the output, so need to delete the 1st word in _captionids and _caption_masks when calculating loss
# tricky, refer to figure.3 in paper "Show and Tell: A Neural Image Caption Generator" for details
def cross_entropy_loss_on_rnn_logits(_captionids, _caption_masks, logits,
    basegraph, name_scope):
  with basegraph.as_default():
    with tf.variable_scope(name_scope):
      # align shape to logits: ((max_words_in_caption-1)*batch_size,)
      labels = tf.reshape(tf.transpose(_captionids[:, 1:]), (-1,))
      # (max_words_in_caption-1)*batch_size, )
      label_masks = tf.reshape(tf.transpose(_caption_masks[:, 1:]), (-1, ))
      cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels)

      loss_op = tf.reduce_sum(cross_entropy * label_masks) / tf.reduce_sum(label_masks)

  return loss_op


# attention: (batch_size, row, cols)
# calculate completeness on column and sparsity on row
# e.g. row is the time step and cols is the attention number in each step
def lplq_norm_on_attention(attentions, mask_sum, p, q, 
    basegraph, name_scope, eps=1e-5):
  with basegraph.as_default():
    with tf.variable_scope(name_scope) as variable_scope:
      attention_completeness = tf.pow(attentions + eps, p)
      attention_completeness = tf.reduce_sum(attention_completeness, 1) # (batch_size, cols)
      attention_completeness = tf.pow(attention_completeness, 1.0/p)
      attention_completeness = tf.reduce_sum(attention_completeness, 1) #(batch_size,)
      attention_completeness = tf.reduce_sum(attention_completeness) / mask_sum

      attention_sparsity = tf.pow(attentions + eps, q)
      attention_sparsity = tf.reduce_sum(attention_sparsity, 2) # (batch_size, rows)
      attention_sparsity = tf.pow(attention_sparsity, 1.0/q)
      attention_sparsity = tf.reduce_sum(attention_sparsity, 1) #(batch_size,)
      attention_sparsity = tf.reduce_sum(attention_sparsity) / mask_sum

  return attention_completeness, attention_sparsity


# # alpha_ops: list of (None, different attention)
# def lplq_norm_on_attention(alpha_ops, mask, p, q, 
#     basegraph, name_scope, eps=1e-5):
#   with basegraph.as_default():
#     with tf.variable_scope(name_scope) as variable_scope:
#       mask = mask[:, 1:]
#       mask = tf.expand_dims(mask, 2) # (None, max_words_in_caption-1, 1)
#       mask = tf.transpose(mask, perm=[1, 0, 2])

#       alphas = tf.pack(alpha_ops) #(max_words_in_caption, None, len(dim_attention_fts))
#       alphas = alphas * mask

#       alpha_completeness = tf.pow(alphas + eps, p)
#       alpha_completeness = tf.reduce_sum(alpha_completeness, 0) # (None, attr_num)
#       alpha_completeness = tf.pow(alpha_completeness, 1.0/p)
#       alpha_completeness = tf.reduce_sum(alpha_completeness, 1) #(None,)
#       alpha_completeness = tf.reduce_sum(alpha_completeness) / tf.reduce_sum(mask)

#       alpha_sparsity = tf.pow(alphas + eps, q)
#       alpha_sparsity = tf.reduce_sum(alpha_sparsity, 2) # (max_words_in_caption, None)
#       alpha_sparsity = tf.pow(alpha_sparsity, 1.0/q)
#       alpha_sparsity = tf.reduce_sum(alpha_sparsity, 0) #(None,)
#       alpha_sparsity = tf.reduce_sum(alpha_sparsity) / tf.reduce_sum(mask)

#   return alpha_completeness, alpha_sparsity


def robust_softmax(x, basegraph, name_scope, eps=1e-5):
  x = tf.nn.softmax(x)
  x = tf.minimum(tf.maximum(x, eps), 1.0-eps)
  return x


def one_layer_LSTM(dim_input, dim_hidden):
  return tf.nn.rnn_cell.LSTMCell(dim_hidden, dim_input, use_peepholes=False)


def multi_layer_LSTM(num_layer, dim_input, dim_hidden):
  cells = [tf.nn.rnn_cell.LSTMCell(dim_input, dim_hidden, use_peepholes=False)
    for l in range(num_layer)]
  return tf.nn.rnn_cell.MultiRNNCell(cells)


def one_layer_GRU(num_unit):
  return tf.nn.rnn_cell.GRUCell(num_unit)


def multi_layer_GRU(num_layer, num_unit):
  cells = [tf.nn.rnn_cell.GRUCell(num_unit) for l in range(num_layer)]
  return tf.nn.rnn_cell.MultiRNNCell(cells)
