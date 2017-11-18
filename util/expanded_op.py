import tensorflow as tf
from tensorflow.python.util import nest


# Note: the concat operation leads to batch_size as the inner for loop and 
# max_words_in_caption - 1 as outer for loop
# <BOS> is not in the output, so need to delete the 1st word in _captionids and _caption_masks when calculating loss
# tricky, refer to figure.3 in paper "Show and Tell: A Neural Image Caption Generator" for details
def cross_entropy_loss_on_rnn_logits(_captionids, _caption_masks, logits):
  # align shape to logits: ((max_words_in_caption-1)*batch_size,)
  labels = tf.reshape(tf.transpose(_captionids[:, 1:]), (-1,))
  # (max_words_in_caption-1)*batch_size, )
  label_masks = tf.reshape(tf.transpose(_caption_masks[:, 1:]), (-1, ))
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)

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


def flatten(tensor):
  return tf.reshape(tensor, (-1,))


# unified interface with LSTMCell's state_is_tuple mode
class GRUCell(tf.contrib.rnn.RNNCell):
  """Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078)."""

  def __init__(self, num_units, activation=tf.nn.tanh, reuse=None):
    super(GRUCell, self).__init__(_reuse=reuse)
    self._num_units = num_units
    self._activation = activation

  @property
  def state_size(self):
    return (self._num_units,)

  @property
  def output_size(self):
    return self._num_units

  def call(self, inputs, state):
    """Gated recurrent unit (GRU) with nunits cells."""
    with tf.variable_scope("gates"):  # Reset gate and update gate.
      # We start with bias of 1.0 to not reset and not update.
      value = tf.nn.sigmoid(
        tf.nn.rnn_cell._linear([inputs, state[0]], 2 * self._num_units, True, 1.0))
      r, u = tf.split(value=value, num_or_size_splits=2, axis=1)
    with tf.variable_scope("candidate"):
      c = self._activation(
        tf.nn.rnn_cell._linear([inputs, r * state[0]], self._num_units, True))
    new_h = u * state[0] + (1 - u) * c
    return new_h, (new_h,)


# def beam_decode(next_step_func, 
#     init_input, init_state, scope,
#     state_size, beam_width, num_step, 
#     reuse_only_after_first_step=True, init_output=None):
#   if not reuse_only_after_first_step:
#     scope.reuse_variables()

#   state_struct = state_size
#   state_sizes = nest.flatten(state_struct)

#   k = beam_width
#   m = num_step
#   batch_size = tf.shape(init_input)[0]

#   # auxiliary idx variable for topk selection operations
#   row_idx = tf.tile(tf.expand_dims(tf.range(0, batch_size), 1), (1, k)) # (batch_size, k) 
#   row_idx = tf.reshape(row_idx, (-1, 1)) # (batch_size*k, 1)
#   # [0...0, ..., batch_size-1...batch_size-1]

#   output_ops = []
#   beam_pre_ops = []
#   beam_cum_log_prob_ops = []
#   beam_end_ops = []

#   wordids = init_input # (batch_size,)
#   states = init_state
#   outputs = init_output
#   for i in xrange(m):
#     if i > 0:
#       scope.reuse_variables()

#     # (batch_size,) in step 0 and (batch_size*k,) in other steps
#     log_prob, states, outputs = next_step_func(wordids, states, outputs, i)

#     if i == 0:
#       log_prob_topk, word_topk = tf.nn.top_k(log_prob, k) # (batch_size, k)
#       output_ops.append(word_topk)
#       pre = -tf.ones((batch_size, k), dtype=tf.int32)
#       beam_pre_ops.append(pre)

#       # set cumulated probability of completed sentences to -inf
#       is_end = tf.equal(word_topk, tf.ones_like(word_topk, dtype=tf.int32))
#       end_idx = tf.where(is_end)
#       beam_cum_log_prob_ops.append(log_prob_topk)
#       beam_end_ops.append(end_idx)
#       log_prob_topk = tf.where(is_end, -100000000*tf.ones_like(log_prob_topk), log_prob_topk) 

#       wordids = flatten(word_topk) # (batch_size*k,)

#       # expand state and outputs
#       states = nest.flatten(states) # (batch_size, hidden_size)
#       states = [
#         tf.reshape(tf.tile(state, [1, k]), (-1, state_size)) # (batch_size*k, hidden_size)
#         for state, state_size in zip(states, state_sizes)
#       ]
#       states = nest.pack_sequence_as(state_struct, states)
#       if outputs is not None:
#         outputs = tf.reshape(tf.tile(outputs, [1, k]), (-1, tf.shape(outputs)[1]))
#     else:
#       # first select top k*k; then select top k
#       # log_prob += tf.reshape(beam_cum_log_prob_ops[-1], (-1, 1))
#       log_prob += tf.reshape(log_prob_topk, (-1, 1))
#       log_prob_topk2, word_topk2 = tf.nn.top_k(log_prob, k) # (batch_size*k, k)
#       log_prob_topk2 = tf.reshape(log_prob_topk2, (-1, k*k)) # (batch_size, k*k)
#       word_topk2 = tf.reshape(word_topk2, (-1, k*k)) # (batch_size, k*k)
#       log_prob_topk, idx_topk = tf.nn.top_k(log_prob_topk2, k) # (batch_size, k)

#       pre = idx_topk//k # (batch_size, k)
#       beam_pre_ops.append(pre)
#       col_idx_topk = tf.reshape(idx_topk, (-1, 1)) # (batch_size*k, 1)
#       row_idx_topk = row_idx
#       idx = tf.concat([row_idx_topk, col_idx_topk], 1) # (batch_size*k, 2)
#       word_topk = tf.gather_nd(word_topk2, idx) # (batch_size*k, )
#       word_topk = tf.reshape(word_topk, (-1, k)) # (batch_size, k)
#       output_ops.append(word_topk)

#       # set cumulated probability of completed sentences to -inf
#       is_end = tf.equal(word_topk, tf.ones_like(word_topk, dtype=tf.int32))
#       end_idx = tf.where(is_end)
#       beam_cum_log_prob_ops.append(log_prob_topk)
#       beam_end_ops.append(end_idx)
#       log_prob_topk = tf.where(is_end, -100000000*tf.ones_like(log_prob_topk), log_prob_topk) 

#       wordids = flatten(word_topk) # (batch_size*k,)

#       # rearrange state and outputs indexs based on selection
#       states = nest.flatten(states)
#       _states = []
#       for state, state_size in zip(states, state_sizes):
#         state = tf.reshape(state, (-1, k, state_size))
#         col_pre = tf.reshape(pre, (-1, 1)) # (batch_size*k, 1)
#         row_pre = row_idx # (batch_size*k, 1)
#         idx = tf.concat([row_pre, col_pre], 1) # (batch_size*k, 2)
#         state = tf.gather_nd(state, idx)
#         _states.append(state)
#       states = nest.pack_sequence_as(state_struct, _states)

#       if outputs is not None:
#         outputs = _states[-1]
#         # col_pre = tf.reshape(pre, (-1, 1))
#         # row_pre = row_idx
#         # idx = tf.concat([row_pre, col_pre], 1)
#         # outputs = tf.gather_nd(outputs, idx)

#   return output_ops, beam_pre_ops, beam_cum_log_prob_ops, beam_end_ops


def beam_decode(next_step_func, 
    init_input, init_state,
    state_size, beam_width, num_step, 
    init_output=None):
  state_struct = state_size
  state_sizes = nest.flatten(state_struct)

  k = beam_width
  m = num_step
  batch_size = tf.shape(init_input)[0]

  # auxiliary idx variable for topk selection operations
  row_idx = tf.tile(tf.expand_dims(tf.range(0, batch_size), 1), (1, k)) # (batch_size, k) 
  row_idx = tf.reshape(row_idx, (-1, 1)) # (batch_size*k, 1)
  # [0...0, ..., batch_size-1...batch_size-1]

  output_ops = []
  beam_pre_ops = []
  beam_cum_log_prob_ops = []
  beam_end_ops = []

  wordids = init_input # (batch_size,)
  states = init_state
  outputs = init_output
  for i in xrange(m):
    # (batch_size,) in step 0 and (batch_size*k,) in other steps
    log_prob, states, outputs = next_step_func(wordids, states, outputs, i)

    if i == 0:
      log_prob_topk, word_topk = tf.nn.top_k(log_prob, k) # (batch_size, k)
      output_ops.append(word_topk)
      pre = -tf.ones((batch_size, k), dtype=tf.int32)
      beam_pre_ops.append(pre)

      # set cumulated probability of completed sentences to -inf
      is_end = tf.equal(word_topk, tf.ones_like(word_topk, dtype=tf.int32))
      end_idx = tf.where(is_end)
      beam_cum_log_prob_ops.append(log_prob_topk)
      beam_end_ops.append(end_idx)
      log_prob_topk = tf.where(is_end, -100000000*tf.ones_like(log_prob_topk), log_prob_topk) 

      wordids = flatten(word_topk) # (batch_size*k,)

      # expand state and outputs
      states = nest.flatten(states) # (batch_size, hidden_size)
      states = [
        tf.reshape(tf.tile(state, [1, k]), (-1, state_size)) # (batch_size*k, hidden_size)
        for state, state_size in zip(states, state_sizes)
      ]
      states = nest.pack_sequence_as(state_struct, states)
      if outputs is not None:
        outputs = tf.reshape(tf.tile(outputs, [1, k]), (-1, tf.shape(outputs)[1]))
    else:
      # first select top k*k; then select top k
      # log_prob += tf.reshape(beam_cum_log_prob_ops[-1], (-1, 1))
      log_prob += tf.reshape(log_prob_topk, (-1, 1))
      log_prob_topk2, word_topk2 = tf.nn.top_k(log_prob, k) # (batch_size*k, k)
      log_prob_topk2 = tf.reshape(log_prob_topk2, (-1, k*k)) # (batch_size, k*k)
      word_topk2 = tf.reshape(word_topk2, (-1, k*k)) # (batch_size, k*k)
      log_prob_topk, idx_topk = tf.nn.top_k(log_prob_topk2, k) # (batch_size, k)

      pre = idx_topk//k # (batch_size, k)
      beam_pre_ops.append(pre)
      col_idx_topk = tf.reshape(idx_topk, (-1, 1)) # (batch_size*k, 1)
      row_idx_topk = row_idx
      idx = tf.concat([row_idx_topk, col_idx_topk], 1) # (batch_size*k, 2)
      word_topk = tf.gather_nd(word_topk2, idx) # (batch_size*k, )
      word_topk = tf.reshape(word_topk, (-1, k)) # (batch_size, k)
      output_ops.append(word_topk)

      # set cumulated probability of completed sentences to -inf
      is_end = tf.equal(word_topk, tf.ones_like(word_topk, dtype=tf.int32))
      end_idx = tf.where(is_end)
      beam_cum_log_prob_ops.append(log_prob_topk)
      beam_end_ops.append(end_idx)
      log_prob_topk = tf.where(is_end, -100000000*tf.ones_like(log_prob_topk), log_prob_topk) 

      wordids = flatten(word_topk) # (batch_size*k,)

      # rearrange state and outputs indexs based on selection
      states = nest.flatten(states)
      _states = []
      for state, state_size in zip(states, state_sizes):
        state = tf.reshape(state, (-1, k, state_size))
        col_pre = tf.reshape(pre, (-1, 1)) # (batch_size*k, 1)
        row_pre = row_idx # (batch_size*k, 1)
        idx = tf.concat([row_pre, col_pre], 1) # (batch_size*k, 2)
        state = tf.gather_nd(state, idx)
        _states.append(state)
      states = nest.pack_sequence_as(state_struct, _states)

      if outputs is not None:
        outputs = _states[-1]

  return output_ops, beam_pre_ops, beam_cum_log_prob_ops, beam_end_ops
