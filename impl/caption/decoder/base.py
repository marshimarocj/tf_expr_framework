import sys

import tensorflow as tf

import framework.model.proto
import framework.util.expanded_op


class ConfigBase(framework.model.proto.ProtoConfig):
  def __init__(self):
    self.dim_hidden = 512
    self.dim_input = 512
    self.max_words_in_caption = 30
    self.num_words = 10870

    self.greedy_or_beam = False
    self.beam_width = 5
    self.sent_pool_size = 5

    self.tied_embed = False
    self.variational_recurrent = False

    self.reg_type = None
    self.reg_lambda = 0.

    self.cell_type = '' # lstm|gru
    self.num_layer = 1


class DecoderBase(framework.model.proto.ModelProto):
  name_scope = 'base.DecoderBase'

  def __init__(self, config):
    framework.model.proto.ModelProto.__init__(self, config)

    self._cells = []

    if config.cell_type == 'lstm':
      self._cells = [
        tf.contrib.rnn.LSTMCell(config.dim_hidden, state_is_tuple=True) \
        for _ in range(config.num_layer)
      ]
    elif config.cell_type == 'gru':
      assert config.dim_input == config.dim_hidden, \
        "require config.dim_input == config.dim_hidden in vanilla.DecoderBase.__init__"
      self._cells = [
        framework.util.expanded_op.GRUCell(config.dim_hidden) for _ in range(config.num_layer)
      ]

    # input
    self._trn_ft_embeds = tf.no_op()
    self._tst_ft_embeds = tf.no_op()
    # trn only
    self._captionids = tf.no_op()
    self._caption_masks = tf.no_op()
    # tst only
    self._init_wordids = tf.no_op()

    # output
    # trn only
    self._logit_ops = tf.no_op()
    self._regularize_op = tf.no_op()
    # tst only
    self._output_ops = []
    self._beam_cum_log_prob_ops = []
    self._beam_pre_ops = []
    self._beam_end_ops = []

  ###############input###############
  @property
  def trn_ft_embeds(self):
    if self._trn_ft_embeds is tf.Operation:
      raise NotImplementedError("""please implement decoder.base.DecoderBase._trn_ft_embeds""")
    return self._trn_ft_embeds

  @trn_ft_embeds.setter
  def trn_ft_embeds(self, val):
    self._trn_ft_embeds = val

  @property
  def tst_ft_embeds(self):
    if self._tst_ft_embeds is tf.Operation:
      raise NotImplementedError("""please implement decoder.base.DecoderBase._tst_ft_embeds""")
    return self._tst_ft_embeds

  @tst_ft_embeds.setter
  def tst_ft_embeds(self, val):
    self._tst_ft_embeds = val

  # trn
  @property
  def captionids(self):
    if self._captionids is tf.Operation:
      raise NotImplementedError("""please implement decoder.base.DecoderBase._captionids""")
    return self._captionids
  
  @captionids.setter
  def captionids(self, val):
    self._captionids = val

  @property
  def caption_masks(self):
    if self._caption_masks is tf.Operation:
      raise NotImplementedError("""please implement decoder.DecoderBase._caption_masks""")
    return self._caption_masks

  @caption_masks.setter
  def caption_masks(self, val):
    self._caption_masks = val

  # tst
  @property
  def init_wordids(self):
    if self._init_wordids is tf.Operation:
      raise NotImplementedError("""please implement decoder.DecoderBase._wordids""")
    return self._init_wordids
 
  @init_wordids.setter
  def init_wordids(self, val):
    self._init_wordids = val

  ###############output###############
  # trn
  @property
  def logit_ops(self):
    if self._logit_ops is tf.Operation:
      raise NotImplementedError("""please implement decoder.DecoderBase._logit_ops""")
    return self._logit_ops

  @property
  def regularize_op(self):
    if self._regularize_op is tf.Operation:
      raise NotImplementedError("""please implement decoder.DecoderBase._regularize_op""")
    return self._regularize_op

  # tst
  @property
  def output_ops(self):
    if len(self._output_ops) == 0:
      raise NotImplementedError("""please implement decoder.DecoderBase._output_ops""")
    return self._output_ops

  @property
  def beam_cum_log_prob_ops(self):
    if len(self._beam_cum_log_prob_ops) == 0:
      raise NotImplementedError("""please implement decoder.DecoderBase._beam_cum_log_prob_ops""")
    return self._beam_cum_log_prob_ops

  @property
  def beam_pre_ops(self):
    if len(self._beam_pre_ops) == 0:
      raise NotImplementedError("""please implement decoder.DecoderBase._beam_pre_ops""")
    return self._beam_pre_ops

  @property
  def beam_end_ops(self):
    if len(self._beam_end_ops) == 0:
      raise NotImplementedError("""please implement decoder.DecoderBase._beam_end_ops""")
    return self._beam_end_ops

  @property
  def state_size(self):
    return tuple([cell.state_size for cell in self._cells])

  def build_parameter_graph(self, basegraph):
    with basegraph.as_default():
      with tf.variable_scope(self.name_scope):
        scale = 1.0 / (self._config.num_words**0.5)
        self.word_embedding_W = tf.get_variable('word_embedding_W',
          shape=(self._config.num_words, self._config.dim_input), dtype=tf.float32,
          initializer=tf.random_uniform_initializer(-scale, scale))

        if self._config.tied_embed:
          self.softmax_W = tf.transpose(self.word_embedding_W)
        else:
          scale = 1.0/ (self._config.dim_hidden**0.5)
          self.softmax_W = tf.get_variable('softmax_W',
            shape=(self._config.dim_hidden, self._config.num_words), dtype=tf.float32,
            initializer=tf.random_uniform_initializer(-scale, scale))

        self.softmax_B = tf.get_variable('softmax_B',
            shape=(self._config.num_words,), dtype=tf.float32,
            initializer=tf.random_uniform_initializer(-0.1, 0.1))

  # it's not a good idea on add regularizer on rnn weight according to :
  # https://www.reddit.com/r/MachineLearning/comments/3nwukm/can_you_use_l1l2_regularization_on_rnn_or_lstms/
  def add_reg(self, basegraph):
    pass
