import enum

import tensorflow as tf

import framework.model.module


class Config(framework.model.module.ModuleConfig):
  def __init__(self):
    framework.model.module.ModuleConfig.__init__(self)

    self.dim_fts = [0,]
    self.dim_output = 512 # dim of feature layer output
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
      dim_input = sum(self._config.dim_fts)
      self.fc_W = tf.contrib.framework.model_variable('fc_W',
        shape=(dim_input, self._config.dim_output), dtype=tf.float32,
        initializer=tf.contrib.layers.xavier_initializer())
      self._weights.append(self.fc_W)
      self.fc_B = tf.contrib.framework.model_variable('fc_B',
        shape=(self._config.dim_output,), dtype=tf.float32,
        initializer=tf.random_uniform_initializer(-0.1, 0.1))
      self._weights.append(self.fc_B)

  def _embed(self, in_ops):
    ft = in_ops[self.InKey.FT]
    is_training = in_ops[self.InKey.IS_TRN]

    ft = tf.contrib.layers.dropout(ft, keep_prob=self._config.keepin_prob, is_training=is_training)
    embed = tf.nn.xw_plus_b(ft, self.fc_W, self.fc_B)

    return embed

  def get_out_ops_in_mode(self, in_ops, mode, reuse=True, **kwargs):
    with tf.variable_scope(self.name_scope):
      embed_op = self._embed(in_ops)
    return {
      self.OutKey.EMBED: embed_op,
    }
