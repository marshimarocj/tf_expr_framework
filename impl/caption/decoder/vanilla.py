import tensorflow as tf
from tensorflow.python.util import nest

import framework.util.expanded_op
import base


class Config(base.ConfigBase):
  pass


# exactly same as figure 3 in paper "Show and Tell: A Neural Image Caption Generator"
class Decoder(base.DecoderBase):
  # name_scope = 'vanilla.Decoder'
  name_scope = 'vanilla.Decoder/'

  def __init__(self, config):
    base.DecoderBase.__init__(self, config)

    # input
    self._init_state = () # bypass unknown batch size

    # output
    # trn only
    self._trn_ft_state = ()
    # tst only
    self._tst_ft_state = ()

  @property
  def init_state(self):
    if len(self._init_state) == 0:
      raise NotImplementedError("""please implement vanilla._init_state""")
    return self._init_state

  @init_state.setter
  def init_state(self, val):
    self._init_state = val

  @property
  def trn_ft_state(self):
    return self._trn_ft_state

  @property
  def tst_ft_state(self):
    return self._tst_ft_state

  @property
  def logit_ops(self):
    return 

  def build_inference_graph_in_tst(self, basegraph):
    with basegraph.as_default():
      with tf.variable_scope(self.name_scope) as scope:
        cell = tf.contrib.rnn.MultiRNNCell(self._cells, state_is_tuple=True)
        self._tst_ft_state = self._ft_step(cell, scope, False)
        if self.config.greedy_or_beam:
          self._greedy_word_steps(cell, scope)
        else:
          self._beam_search_word_steps(cell, scope)

  def build_inference_graph_in_trn_tst(self, basegraph):
    with basegraph.as_default():
      with tf.variable_scope(self.name_scope) as scope:
        cells = []
        for cell in self._cells[:-1]:
          cells.append(tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=0.5))
        cells.append(tf.contrib.rnn.DropoutWrapper(self._cells[-1], input_keep_prob=0.5, output_keep_prob=0.5))
        cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)

        self._trn_ft_state = self._ft_step(cell, scope, False)
        self._word_steps(cell, scope)

        cell = tf.contrib.rnn.MultiRNNCell(self._cells, state_is_tuple=True)
        self._tst_ft_state = self._ft_step(cell, scope, True)
        self._greedy_word_steps(cell, scope)

  def _ft_step(self, cell, scope, reuse):
    if reuse:
      scope.reuse_variables()

    _, state = cell(self._ft_embeds, self._init_state) # ft step output is ignored

    return state # (batch_size, state_size)

  def _word_steps(self, cell, scope):
    scope.reuse_variables()

    state = self._trn_ft_state
    outputs = []
    # <EOS> won't be feed back into LSTM, so loop for max_words_in_caption-1 outputs
    for i in xrange(self._config.max_words_in_caption-1):
      input = tf.nn.embedding_lookup(self.word_embedding_W, self._captionids[:, i])
      output, state = cell(input, state)
      outputs.append(output)

    outputs = tf.concat(outputs, 0)
    self._logit_ops = tf.nn.xw_plus_b(outputs, self.softmax_W, self.softmax_B)

  def _greedy_word_steps(self, cell, scope):
    scope.reuse_variables()

    wordids = self._init_wordids # (batch_size,)
    states = self._tst_ft_state
    for i in xrange(self._config.max_words_in_caption):
      input = tf.nn.embedding_lookup(self.word_embedding_W, wordids)
      outputs, states = cell(input, states)
      logits = tf.nn.xw_plus_b(outputs, self.softmax_W, self.softmax_B) # (batch_size, num_words)
      wordids = tf.argmax(logits, axis=1)

      self._output_ops.append(wordids)

  def _beam_search_word_steps(self, cell, scope):
    scope.reuse_variables()

    state_struct = self.state_size
    state_sizes = nest.flatten(state_struct)

    k = self.config.beam_width
    m = self.config.max_words_in_caption
    n = self.config.num_words
    batch_size = tf.shape(self.init_wordids)[0]

    # auxiliary idx variable for topk selection operations
    row_idx = tf.tile(tf.expand_dims(tf.range(0, batch_size), 1), (1, k)) # (batch_size, k) 
    row_idx = tf.reshape(row_idx, (-1, 1)) # (batch_size*k, 1)
    # [0...0, ..., batch_size-1...batch_size-1]

    wordids = self._init_wordids # (batch_size,)
    states = self._tst_ft_state
    for i in xrange(m):
      # (batch_size,) in step 0 and (batch_size*k,) in other steps
      input = tf.nn.embedding_lookup(self.word_embedding_W, wordids) 
      outputs, states = cell(input, states)
      logit = tf.nn.xw_plus_b(outputs, self.softmax_W, self.softmax_B)

      if i == 0:
        logit_topk, word_topk = tf.nn.top_k(logit, k) # (batch_size, k)
        self._output_ops.append(word_topk)
        pre = -tf.ones((batch_size, k), dtype=tf.int32)
        self._beam_pre_ops.append(pre)

        # set cumulated probability of completed sentences to -inf
        is_end = tf.equal(word_topk, tf.ones_like(word_topk, dtype=tf.int32))
        logit_topk = tf.where(is_end, -100000000*tf.ones_like(logit_topk), logit_topk) 
        end_idx = tf.where(is_end)
        self._beam_cum_logit_ops.append(logit_topk)
        self._beam_end_ops.append(end_idx)

        wordids = framework.util.expanded_op.flatten(word_topk) # (batch_size*k,)

        # expand state
        states = nest.flatten(states) # (batch_size, hidden_size)
        states = [
          tf.reshape(tf.tile(state, [1, k]), (-1, state_size)) # (batch_size*k, hidden_size)
          for state, state_size in zip(states, state_sizes)
        ]
        states = nest.pack_sequence_as(state_struct, states)
      else:
        # first select top k*k; then select top k
        logit += tf.reshape(self._beam_cum_logit_ops[-1], (-1, 1))
        logit_topk2, word_topk2 = tf.nn.top_k(logit, k) # (batch_size*k, k)
        logit_topk2 = tf.reshape(logit_topk2, (-1, k*k)) # (batch_size, k*k)
        word_topk2 = tf.reshape(word_topk2, (-1, k*k)) # (batch_size, k*k)
        logit_topk, idx_topk = tf.nn.top_k(logit_topk2, k) # (batch_size, k)

        pre = idx_topk//k # (batch_size, k)
        self._beam_pre_ops.append(pre)
        col_idx_topk = tf.reshape(idx_topk, (-1, 1)) # (batch_size*k, 1)
        row_idx_topk = row_idx
        idx = tf.concat([row_idx_topk, col_idx_topk], 1) # (batch_size*k, 2)
        word_topk = tf.gather_nd(word_topk2, idx) # (batch_size*k, )
        word_topk = tf.reshape(word_topk, (-1, k)) # (batch_size, k)
        self._output_ops.append(word_topk)

        # set cumulated probability of completed sentences to -inf
        is_end = tf.equal(word_topk, tf.ones_like(word_topk, dtype=tf.int32))
        logit_topk = tf.where(is_end, -100000000*tf.ones_like(logit_topk), logit_topk) 
        end_idx = tf.where(is_end)
        self._beam_cum_logit_ops.append(logit_topk)
        self._beam_end_ops.append(end_idx)

        wordids = framework.util.expanded_op.flatten(word_topk) # (batch_size*k,)

        # rearrange state indexs based on selection
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


# exactly same as figure 3 in MSR-VTT: A Large Video Description Dataset for Bridging Video and Language
# remove init state from Decoder
class DecoderHiddenSet(base.DecoderBase):
  # name_scope = 'vanilla.DecoderWithHiddenSet'
  name_scope = 'vanilla.DecoderWithHiddenSet/'

  def __init__(self, config):
    base.DecoderBase.__init__(self, config)

    # output
    self._init_state = ()
    # trn only
    self._trn_ft_state = ()
    # tst only
    self._tst_ft_state = ()

  @property
  def trn_ft_state(self):
    return self._trn_ft_state

  @property
  def tst_ft_state(self):
    return self._tst_ft_state

  def build_inference_graph_in_tst(self, basegraph):
    with basegraph.as_default():
      with tf.variable_scope(self.name_scope) as scope:
        cell = tf.contrib.rnn.MultiRNNCell(self._cells, state_is_tuple=True)
        self._tst_ft_state = self._ft_step(cell, scope, None)
        if self.config.greedy_or_beam:
          self._greedy_word_steps(cell, scope)
        else:
          self._beam_search_word_steps(cell, scope)

  def build_inference_graph_in_trn_tst(self, basegraph):
    with basegraph.as_default():
      with tf.variable_scope(self.name_scope) as scope:
        cells = []
        for cell in self._cells[:-1]:
          cells.append(tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=0.5))
        cells.append(tf.contrib.rnn.DropoutWrapper(self._cells[-1], input_keep_prob=0.5, output_keep_prob=0.5))
        cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)

        self._trn_ft_state = self._ft_step(cell, scope, None)
        self._word_steps(cell, scope)

        cell = tf.contrib.rnn.MultiRNNCell(self._cells, state_is_tuple=True)
        self._tst_ft_state = self._ft_step(cell, scope, None)
        self._greedy_word_steps(cell, scope)

  def _ft_step(self, cell, scope, reuse):
    state_size = cell.state_size
    states = [self._ft_embeds for _ in nest.flatten(state_size)]
    states = nest.pack_sequence_as(state_size, states)

    return states

  def _word_steps(self, cell, scope):
    state = self._trn_ft_state
    outputs = []
    # <EOS> won't be feed back into LSTM, so loop for max_words_in_caption-1 outputs
    for i in xrange(self._config.max_words_in_caption-1):
      if i > 0: 
        scope.reuse_variables()
      input = tf.nn.embedding_lookup(self.word_embedding_W, self._captionids[:, i])
      output, state = cell(input, state) # (None, dim_hidden)
      outputs.append(output)

    outputs = tf.concat(outputs, 0)
    self._logit_ops = tf.nn.xw_plus_b(outputs, self.softmax_W, self.softmax_B)

  def _greedy_word_steps(self, cell, scope):
    wordids = self._init_wordids # (batch_size,)
    states = self._tst_ft_state
    for i in xrange(self._config.max_words_in_caption):
      if i > 0:
        scope.reuse_variables()

      input = tf.nn.embedding_lookup(self.word_embedding_W, wordids)
      outputs, states = cell(input, states)
      logits = tf.nn.xw_plus_b(outputs, self.softmax_W, self.softmax_B) # (batch_size, num_words)
      wordids = tf.argmax(logits, axis=1)
      predict_prob = tf.nn.softmax(logits) # (batch_size, num_words)

      self._output_ops.append(wordids)
      self._predict_prob_ops.append(predict_prob)

  def _beam_search_word_steps(self, cell, scope):
    state_struct = self.state_size
    state_sizes = nest.flatten(state_struct)

    k = self.config.beam_width
    m = self.config.max_words_in_caption
    n = self.config.num_words
    batch_size = tf.shape(self.init_wordids)[0]

    # auxiliary idx variable for topk selection operations
    row_idx = tf.tile(tf.expand_dims(tf.range(0, batch_size), 1), (1, k)) # (batch_size, k) 
    row_idx = tf.reshape(row_idx, (-1, 1)) # (batch_size*k, 1)
    # [0...0, ..., batch_size-1...batch_size-1]

    wordids = self._init_wordids # (batch_size,)
    states = self._tst_ft_state
    for i in xrange(m):
      if i > 0:
        scope.reuse_variables()

      # (batch_size,) in step 0 and (batch_size*k,) in other steps
      input = tf.nn.embedding_lookup(self.word_embedding_W, wordids) 
      outputs, states = cell(input, states)
      logit = tf.nn.xw_plus_b(outputs, self.softmax_W, self.softmax_B)

      if i == 0:
        logit_topk, word_topk = tf.nn.top_k(logit, k) # (batch_size, k)
        self._output_ops.append(word_topk)
        pre = -tf.ones((batch_size, k), dtype=tf.int32)
        self._beam_pre_ops.append(pre)

        # set cumulated probability of completed sentences to -inf
        is_end = tf.equal(word_topk, tf.ones_like(word_topk, dtype=tf.int32))
        logit_topk = tf.where(is_end, -100000000*tf.ones_like(logit_topk), logit_topk) 
        end_idx = tf.where(is_end)
        self._beam_cum_logit_ops.append(logit_topk)
        self._beam_end_ops.append(end_idx)

        wordids = framework.util.expanded_op.flatten(word_topk) # (batch_size*k,)

        # expand state
        states = nest.flatten(states) # (batch_size, hidden_size)
        states = [
          tf.reshape(tf.tile(state, [1, k]), (-1, state_size)) # (batch_size*k, hidden_size)
          for state, state_size in zip(states, state_sizes)
        ]
        states = nest.pack_sequence_as(state_struct, states)
      else:
        # first select top k*k; then select top k
        logit += tf.reshape(self._beam_cum_logit_ops[-1], (-1, 1))
        logit_topk2, word_topk2 = tf.nn.top_k(logit, k) # (batch_size*k, k)
        logit_topk2 = tf.reshape(logit_topk2, (-1, k*k)) # (batch_size, k*k)
        word_topk2 = tf.reshape(word_topk2, (-1, k*k)) # (batch_size, k*k)
        logit_topk, idx_topk = tf.nn.top_k(logit_topk2, k) # (batch_size, k)

        pre = idx_topk//k # (batch_size, k)
        self._beam_pre_ops.append(pre)
        col_idx_topk = tf.reshape(idx_topk, (-1, 1)) # (batch_size*k, 1)
        row_idx_topk = row_idx
        idx = tf.concat([row_idx_topk, col_idx_topk], 1) # (batch_size*k, 2)
        word_topk = tf.gather_nd(word_topk2, idx) # (batch_size*k, )
        word_topk = tf.reshape(word_topk, (-1, k)) # (batch_size, k)
        self._output_ops.append(word_topk)

        # set cumulated probability of completed sentences to -inf
        is_end = tf.equal(word_topk, tf.ones_like(word_topk, dtype=tf.int32))
        logit_topk = tf.where(is_end, -100000000*tf.ones_like(logit_topk), logit_topk) 
        end_idx = tf.where(is_end)
        self._beam_cum_logit_ops.append(logit_topk)
        self._beam_end_ops.append(end_idx)

        wordids = framework.util.expanded_op.flatten(word_topk) # (batch_size*k,)

        # rearrange state indexs based on selection
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
