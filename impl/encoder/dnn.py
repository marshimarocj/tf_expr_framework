import enum

import tensorflow as tf

import framework.model.module


class Config(framework.model.module.ModuleConfig):
  def __init__(self):
    framework.model.module.ModuleConfig.__init__(self)

    self.dim_fts = [0,]
    self.dim_hiddens = []
    # self.dim_output = 512 # dim of feature layer output
    self.keepin_prob = 1.

  def _assert(self):
    pass


class Encoder(framework.model.module.AbstractModule):
  name_scope = 'vanilla.Encoder'

  class InKey(enum.Enum):
    FT = 'ft' # (None, dim_ft)
    IS_TRN = 'is_training'

  class OutKey(enum.Enum):
    EMBED = 'ft_embed' # (None, dim_output)

  def _set_submods(self):
    return {}

  def _build_parameter_graph(self):
    with tf.variable_scope(self.name_scope):
      dim_inputs = [sum(self._config.dim_fts)] + self._config.dim_hiddens[:-1]
      dim_outputs = self._config.dim_hiddens
      layer = 0
      self.fc_Ws = []
      self.fc_Bs = []
      for dim_input, dim_output in zip(dim_inputs, dim_outputs):
        fc_W = tf.contrib.framework.model_variable('fc_W_%d'%layer,
          shape=(dim_input, dim_output), dtype=tf.float32,
          initializer=tf.contrib.layers.xavier_initializer())
        self._weights.append(fc_W)
        self.fc_Ws.append(fc_W)
        fc_B = tf.contrib.framework.model_variable('fc_B_%d'%layer,
          shape=(dim_output,), dtype=tf.float32,
          initializer=tf.random_uniform_initializer(-0.1, 0.1))
        self._weights.append(fc_B)
        self.fc_Bs.append(fc_B)
        layer += 1

  def _embed(self, in_ops):
    ft = in_ops[self.InKey.FT]
    is_training = in_ops[self.InKey.IS_TRN]

    for fc_W, fc_B in zip(self.fc_Ws[:-1], self.fc_Bs[:-1]):
      ft = tf.nn.xw_plus_b(ft, fc_W, fc_B)
      ft = tf.contrib.layers.dropout(ft, keep_prob=self._config.keepin_prob, is_training=is_training)
      ft = tf.nn.relu(ft)
    ft = tf.nn.xw_plus_b(ft, self.fc_Ws[-1], self.fc_Bs[-1])

    return ft

  def get_out_ops_in_mode(self, in_ops, mode, reuse=True, **kwargs):
    with tf.variable_scope(self.name_scope):
      embed_op = self._embed(in_ops)
    return {
      self.OutKey.EMBED: embed_op,
    }
