import tensorflow as tf
from tensorflow.python.util import nest

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
        self._greedy_word_steps(cell, scope)

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
    logits = tf.nn.xw_plus_b(outputs, self.softmax_W, self.softmax_B)
    self._logit_ops = tf.split(logits, self._config.max_words_in_caption-1, axis=0)
    self._logit_ops = list(self._logit_ops)

  def _greedy_word_steps(self, cell, scope):
    scope.reuse_variables()

    wordids = self._init_wordids # (batch_size,)
    states = self._tst_ft_state
    for i in xrange(self._config.max_words_in_caption):
      outputs, states = cell(wordids, states)
      logits = tf.nn.xw_plus_b(outputs, self.softmax_W, self.softmax_B) # (batch_size, num_words)
      wordids = tf.argmax(logits, axis=1)
      predict_prob = tf.nn.softmax(logits) # (batch_size, num_words)

      self._output_ops.append(wordids)
      self._predict_prob_ops.append(predict_prob)


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
        self._greedy_word_steps(cell, scope)

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
    logits = tf.nn.xw_plus_b(outputs, self.softmax_W, self.softmax_B)
    self._logit_ops = tf.split(logits, self._config.max_words_in_caption-1, axis=0)
    self._logit_ops = list(self._logit_ops)

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
