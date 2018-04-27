import enum
import math

import tensorflow as tf
from tensorflow.python.util import nest

import framework.model.module
import framework.util.expanded_op


class CellConfig(framework.model.module.ModuleConfig):
  def __init__(self):
    framework.model.module.ModuleConfig.__init__(self)

    self.dim_hidden = 512
    self.dim_input = 512
    self.keepout_prob = 1.
    self.keepin_prob = 1.

  def _assert(self):
    pass


class SCCellConfig(CellConfig):
  def __init__(self):
    CellConfig.__init__(self)

    self.dim_latent = 512
    self.nnz_s = 10
    self.dim_s = 100


class LSTMCell(framework.model.module.AbstractModule):
  name_scope = 'cell.LSTMCell'

  class InKey(enum.Enum):
    INPUT = 'input' # (None, dim_input)
    STATE = 'state' # ((None, dim_hidden), (None, dim_hidden))
    IS_TRN = 'is_training'

  class OutKey(enum.Enum):
    OUTPUT = 'output'
    STATE = 'state'

  @property
  def state_size(self):
    return (self._config.dim_hidden, self._config.dim_hidden)

  def _set_submods(self):
    return {}

  def _build_parameter_graph(self):
    with tf.variable_scope(self.name_scope):
      stddev = 1 / math.sqrt(self._config.dim_hidden + self._config.dim_input)
      self.W = tf.contrib.framework.model_variable('W', 
        shape=(self._config.dim_hidden + self._config.dim_input, 4*self._config.dim_hidden),
        dtype=tf.float32,
        initializer=tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32))
      self._weights.append(self.W)
      self.b = tf.contrib.framework.model_variable('b',
        shape=(4*self._config.dim_hidden,),
        dtype=tf.float32,
        initializer=tf.constant_initializer(0.0, dtype=tf.float32))
      self._weights.append(self.b)

  def _step(self, input, state, is_training):
    input = tf.contrib.layers.dropout(input, keep_prob=self._config.keepin_prob, is_training=is_training)
    c, h = state
    ijfo = tf.nn.xw_plus_b(tf.concat([input, h], 1), self.W, self.b) # (None, 4*dim_hidden)
    i, j, f, o = tf.split(ijfo, 4, 1)
    new_c = c * tf.sigmoid(f + 1.0) + tf.sigmoid(i) * tf.tanh(j)
    new_h = tf.tanh(new_c) * tf.sigmoid(o)
    new_h = tf.contrib.layers.dropout(new_h, keep_prob=self._config.keepout_prob, is_training=is_training)
    return new_h, (new_c, new_h)

  def get_out_ops_in_mode(self, in_ops, mode):
    with tf.variable_scope(self.name_scope):
      output, state = self._step(in_ops[self.InKey.INPUT], in_ops[self.InKey.STATE], in_ops[self.InKey.IS_TRN])
    return {
      self.OutKey.OUTPUT: output,
      self.OutKey.STATE: state,
    }


class SCLSTMCell(framework.model.module.AbstractModule):
  name_scope = 'cell.SCLSTMCell'

  class InKey(enum.Enum):
    INPUT = 'input' # (None, dim_input)
    SIDX = 's_idx' # ()
    SWEIGHT = 's_weight' # ()
    STATE = 'state' # ((None, dim_hidden), (None, dim_hidden))
    IS_TRN = 'is_training'

  class OutKey(enum.Enum):
    OUTPUT = 'output'
    STATE = 'state'

  @property
  def state_size(self):
    return (self._config.dim_hidden, self._config.dim_hidden)

  def _set_submods(self):
    return {}

  def _build_parameter_graph(self):
    with tf.variable_scope(self.name_scope):
      stddev = 1 / math.sqrt(self._config.dim_latent)
      self.W_b = tf.get_variable('W_b', 
        shape=(self._config.dim_s, 4 * self._config.dim_latent),
        dtype=tf.float32,
        initializer=tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32))
      self.U_b = tf.get_variable('U_b', 
        shape=(self._config.dim_s, 4 * self._config.dim_latent),
        dtype=tf.float32,
        initializer=tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32))

      stddev = 1 / math.sqrt(self._config.dim_input)
      self.W_c = tf.get_variable('W_c',
        shape=(self._config.dim_input, 4 * self._config.dim_latent),
        dtype=tf.float32,
        initializer=tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32))
      stddev = 1 / math.sqrt(self._config.dim_hidden)
      self.U_c = tf.get_variable('U_c',
        shape=(self._config.dim_hidden, 4 * self._config.dim_latent),
        dtype=tf.float32,
        initializer=tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32))

      stddev = 1 / math.sqrt(self._config.dim_latent)
      self.W_a = tf.get_variable('W_a',
        shape=(4, self._config.dim_latent, self._config.dim_hidden),
        dtype=tf.float32,
        initializer=tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32))
      self.U_a = tf.get_variable('U_a',
        shape=(4, self._config.dim_latent, self._config.dim_hidden),
        dtype=tf.float32,
        initializer=tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32))

      self.b = tf.get_variable('b',
        shape=(4, 1, self._config.dim_hidden),
        dtype=tf.float32,
        initializer=tf.constant_initializer(0.0, dtype=tf.float32))

  def _step(self, input, s_idx, s_weight, state, is_training):
    input = tf.contrib.layers.dropout(input, keep_prob=self._config.keepin_prob, is_training=is_training)
    c, h = state

    latent_x = tf.matmul(input, self.W_c)
    latent_h = tf.matmul(h, self.U_c)

    # sparse lookup
    s_weight = tf.expand_dims(s_weight, 2) # (None, nnz_s, 1)
    Ws = tf.nn.embedding_lookup(self.W_b, tf.reshape(s_idx, (-1,))) # (None*nnz_s, 4*dim_latent)
    Ws = tf.reshape(Ws, (-1, self._config.nnz_s, 4*self._config.dim_latent)) # (None, nnz_s, 4*dim_latent)
    Ws = tf.reduce_sum(Ws * s_weight, 1) # (None, 4*dim_latent)
    Us = tf.nn.embedding_lookup(self.U_b, tf.reshape(s_idx, (-1,)))
    Us = tf.reshape(Us, (-1, self._config.nnz_s, 4*self._config.dim_latent))
    Us = tf.reduce_sum(Us * s_weight, 1) # (None, 4*dim_latent)
    latent_x *= Ws # (None, 4*dim_latent)
    latent_h *= Us # (None, 4*dim_latent)
    latent_x = tf.transpose(tf.reshape(latent_x, (-1, 4, self._config.dim_latent)), (1, 0, 2)) # (4, None, dim_latent)
    latent_h = tf.transpose(tf.reshape(latent_h, (-1, 4, self._config.dim_latent)), (1, 0, 2)) # (4, None, dim_latent)

    ijfo = tf.matmul(latent_x, self.W_a) + tf.matmul(latent_h, self.U_a) + self.b # (4, None, dim_hidden)
    i, j, f, o = ijfo[0], ijfo[1], ijfo[2], ijfo[3]

    new_c = (
        c * tf.sigmoid(f + 1.0) + tf.sigmoid(i) * tf.tanh(j))
    new_h = tf.tanh(new_c) * tf.sigmoid(o)
    new_h = tf.contrib.layers.dropout(new_h, keep_prob=self._config.keepout_prob, is_training=is_training)
    return new_h, (new_c, new_h)

  def get_out_ops_in_mode(self, in_ops, mode):
    with tf.variable_scope(self.name_scope):
      output, state = self._step(in_ops[self.InKey.INPUT], in_ops[self.InKey.SIDX], in_ops[self.InKey.SWEIGHT],
        in_ops[self.InKey.STATE], in_ops[self.InKey.IS_TRN])
    return {
      self.OutKey.OUTPUT: output,
      self.OutKey.STATE: state,
    }


class GRUCell(framework.model.module.AbstractModule):
  name_scope = 'rnn.GRUCell'

  class InKey(enum.Enum):
    INPUT = 'input' # (None, dim_input)
    STATE = 'state' # ((None, dim_hidden), (None, dim_hidden))
    IS_TRN = 'is_training'

  class OutKey(enum.Enum):
    OUTPUT = 'output'
    STATE = 'state'

  @property
  def state_size(self):
    return self._config.dim_hidden

  def _set_submods(self):
    return {}

  def _build_parameter_graph(self):
    with tf.variable_scope(self.name_scope):
      stddev = 1 / math.sqrt(self._config.dim_hidden + self._config.dim_input)
      self.gate_W = tf.contrib.framework.model_variable('gate_W', 
        shape=(self._config.dim_hidden + self._config.dim_input, 2*self._config.dim_hidden),
        dtype=tf.float32,
        initializer=tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32))
      self.gate_b = tf.contrib.framework.model_variable('gate_b',
        shape=(2*self._config.dim_hidden,),
        dtype=tf.float32,
        initializer=tf.constant_initializer(1.0, dtype=tf.float32))
      self.candidate_W = tf.contrib.framework.model_variable('candidate_W',
        shape=(self._config.dim_hidden + self._config.dim_input, self._config.dim_hidden),
        dtype=tf.float32,
        initializer=tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32))
      self.candidate_b = tf.contrib.framework.model_variable('candidate_b',
        shape=(self._config.dim_hidden,),
        dtype=tf.float32,
        initializer=tf.constant_initializer(0.0, dtype=tf.float32))
      self._weights.append(self.gate_W)
      self._weights.append(self.gate_b)
      self._weights.append(self.candidate_W)
      self._weights.append(self.candidate_b)

  def _step(self, input, state, is_training):
    input = tf.contrib.layers.dropout(input, keep_prob=self._config.keepin_prob, is_training=is_training)
    gate_inputs = tf.nn.xw_plus_b(tf.concat([input, state], 1), self.gate_W, self.gate_b)
    
    value = tf.sigmoid(gate_inputs)
    r, u = tf.split(value, num_or_size_splits=2, axis=1)

    r_state = r * state

    candidate = tf.nn.xw_plus_b(tf.concat([input, r_state], 1), self.candidate_W, self.candidate_B)

    c = tf.tanh(candidate)
    new_h = u * state + (1 - u) * c
    return new_h

  def get_out_ops_in_mode(self, in_ops, mode):
    with tf.variable_scope(self.name_scope):
      state = self._step(in_ops[self.InKey.INPUT], in_ops[self.InKey.STATE], in_ops[self.InKey.IS_TRN])
    return {
      self.OutKey.OUTPUT: state,
      self.OutKey.STATE: state,
    }
