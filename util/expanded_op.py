import tensorflow as tf


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
