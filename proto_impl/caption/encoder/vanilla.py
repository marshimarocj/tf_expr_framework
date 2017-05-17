import sys
sys.path.append('../../../')

import tensorflow as tf

import model.proto


class Config(model.proto.ProtoConfig):
  def __init__(self):
    self.dim_fts = [0, 0, 0, 0] # c3d, mfccbox, mfccfv, category
    self.dim_output = 512 # dim of feature layer output

    self.reg_type = 'l1l2' # l1l2|l2|None
    self.reg_lambda = 0.

    self.dummy = False

  # @override
  def load(self, cfg_dict):
    model.proto.ProtoConfig.load(self, cfg_dict)

    if sum(self.dim_fts) == self.dim_output:
      assert self.reg_type == None


class Encoder(model.proto.ModelProto):
  name_scope = 'vanilla.Encoder'

  def __init__(self, config):
    model.proto.ModelProto.__init__(self, config)

    # input
    self._fts = tf.no_op()
    # output
    self._feature_op = tf.no_op()
    # trn only
    self._regularize_op = tf.no_op()

  @property
  def fts(self):
    return self._fts
 
  @fts.setter
  def fts(self, val):
    self._fts = val 

  @property
  def feature_op(self):
    return self._feature_op

  @property
  def regularize_op(self):
    return self._regularize_op

  def build_parameter_graph(self, basegraph):
    if sum(self._config.dim_fts) != self._config.dim_output or self._config.dummy:
      with basegraph.as_default():
        with tf.variable_scope(self.name_scope):
          scale = 1.0 / (sum(self._config.dim_fts) ** 0.5)
          self.fc_W = tf.get_variable('fc_W', 
            shape=(sum(self._config.dim_fts), self._config.dim_output), dtype=tf.float32,
            initializer=tf.random_uniform_initializer(-scale, scale))
          self.fc_B = tf.get_variable('fc_B',
            shape=(self._config.dim_output,), dtype=tf.float32,
            initializer=tf.random_uniform_initializer(-0.1, 0.1))

  def build_inference_graph_in_tst(self, basegraph):
    with basegraph.as_default():
      with tf.variable_scope(self.name_scope):
        if sum(self._config.dim_fts) == self._config.dim_output and not self._config.dummy:
          self._feature_op = tf.identity(self._fts) # not an efficient implementation, just to make the interface consistent
        else:
          self._feature_op = tf.nn.xw_plus_b(self._fts, self.fc_W, self.fc_B)

  def build_inference_graph_in_trn_tst(self, basegraph):
    self.build_inference_graph_in_tst(basegraph)

  def add_reg(self, basegraph):
    if self._config.reg_type == 'l1l2':
      self._l1l2_reg(basegraph)
    elif self._config.reg_type == 'l2':
      self._l2_reg(basegraph)

  def _l1l2_reg(self, basegraph):
    with basegraph.as_default():
      with tf.variable_scope(self.name_scope):
        norms = []
        cum_dim_ft = 0
        for dim_ft in self._config.dim_fts:
          one_modal_W = self.fc_W[cum_dim_ft:cum_dim_ft+dim_ft, :]
          norm = tf.reduce_sum(one_modal_W**2)**0.5

          norms.append(norm)
          cum_dim_ft += dim_ft

        self._regularize_op = tf.add_n(norms)

  def _l2_reg(self, basegraph):
    with basegraph.as_default():
      with tf.variable_scope(self.name_scope):
        self._regularize_op = tf.reduce_sum(self.fc_W**2)**0.5


class AttentionFtConfig(model.proto.ProtoConfig):
  def __init__(self):
    self.dim_fts = []
    self.dim_time = 0
    self.dim_output = 512

    self.reg_type = None # l1l2|l2|None
    self.reg_lambda = 0.

    self.dummy = False

  # @override
  def load(self, cfg_dict):
    model.proto.ProtoConfig.load(self, cfg_dict)

    if sum(self.dim_fts) == self.dim_output:
      assert self.reg_type == None


class AttentionFtEncoder(Encoder):
  name_scope='vanilla.AttentionFtEncoder'

  def __init__(self, config):
    model.proto.ModelProto.__init__(self, config)
    
    # input
    self._fts = tf.no_op()
    # output
    self._feature_op = tf.no_op()
    # trn only
    self._regularize_op = tf.no_op()

  @property
  def fts(self):
    return self._fts
 
  @fts.setter
  def fts(self, val):
    self._fts = val 

  @property
  def feature_op(self):
    return self._feature_op

  def build_parameter_graph(self, basegraph):
    if sum(self._config.dim_fts) != self._config.dim_output or self._config.dummy:
      with basegraph.as_default():
        with tf.variable_scope(self.name_scope):
          scale = 1.0 / (sum(self._config.dim_fts) ** 0.5)
          self.fc_W = tf.get_variable('fc_W', 
            shape=(1, 1, sum(self._config.dim_fts), self._config.dim_output), dtype=tf.float32,
            initializer=tf.random_uniform_initializer(-scale, scale))
          self.fc_B = tf.get_variable('fc_B',
            shape=(self._config.dim_output,), dtype=tf.float32,
            initializer=tf.random_uniform_initializer(-0.1, 0.1))

  def build_inference_graph_in_tst(self, basegraph):
    with basegraph.as_default():
      with tf.variable_scope(self.name_scope):
        if sum(self._config.dim_fts) == self._config.dim_output and not self._config.dummy:
          self._feature_op = tf.identity(self._fts)
        else:
          fts = tf.expand_dims(self._fts, 2) # (None, dim_time, 1, sum(dim_fts))
          ft_embeddings = tf.nn.conv2d(fts, self.fc_W, [1, 1, 1, 1], 'VALID') # (None, dim_time, 1, dim_output)
          ft_embeddings = tf.reshape(ft_embeddings, 
            [-1, self._config.dim_time, self._config.dim_output])
          self._feature_op = ft_embeddings + self.fc_B

  def build_inference_graph_in_trn_tst(self, basegraph):
    self.build_inference_graph_in_tst(basegraph)

  def add_reg(self):
    pass
