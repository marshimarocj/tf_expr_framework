import json

import tensorflow as tf


class ProtoConfig(object):
  def load(self, json_str):
    data = json.loads(json_str)
    for key in data:
      if key in self.__dict__:
        setattr(self, key, data[key])


class FullModelConfig(object):
  def __init__(self):
    self.trn_batch_size = 256
    self.tst_batch_size = 128
    self.num_epoch = 100
    self.val_iter = 100
    self.summary_iter = 10
    self.learning_rate = 1e-4
    self.optimizer_alg = 'Adam' # Adam, SGD

  def load(self, file):
    data = json.load(open(file))
    for key in data:
      if key in self.__dict__:
        setattr(self, key, data[key])

    return data


class ModelCombinerConfig(FullModelConfig):
  def __init__(self):
    FullModelConfig.__init__(self)
    self.learning_rates = []
    self.stop_gradients = []


# ModelProto: graph prototype of the model
# should implement:
#   build_parameter_graph,
#   build_inference_graph_in_trn_tst,
#   build_inference_graph_in_tst,
#   regularization
# these three functions will be called in FullModel
class ModelProto(object):
  name_scope = 'ModelProto'


  ######################################
  # functions to customize
  ######################################
  # return basegraph 
  def build_parameter_graph(self, basegraph):
    raise NotImplementedError("""please customize ModelProto.build_parameter_graph""")

  # return basegraph 
  def build_inference_graph_in_trn_tst(self, basegraph):
    raise NotImplementedError("""please customize ModelProto.build_inference_graph""")

  # return basegraph 
  def build_inference_graph_in_tst(self, basegraph):
    raise NotImplementedError("""please customize ModelProto.build_inference_graph""")

  # return basegraph 
  def add_reg(self, basegraph):
    raise NotImplementedError("""please cutomize ModelProto.add_reg""")


# FullModel: include trn and tst boilerpipe ops
# functions to implement:
#   get_model_proto
#   add_tst_input
#   add_trn_tst_input
#   add_loss
#   op_in_val
#   op_in_tst
# utility functions:
#   append_op2monitor
# implemented boilerpipe fuctions:
#   op_in_trn
#   _add_init
#   _add_saver
#   _add_summary
#   _build_parameter_graph
#   build_tst_graph, 
#   build_trn_tst_graph
# build_tst_graph and build_trn_tst_graph will be called in TrnTst
class FullModel(object):
  name_scope = 'FullModel' 

  def __init__(self, config):
    self._config = config
    self._model_proto = self.get_model_proto()

    self._init_op = tf.no_op()
    self._saver = tf.no_op()
    self._summary_op = tf.no_op()
    self._loss_op = tf.no_op()
    self._gradient_op = tf.no_op()
    self._train_op = tf.no_op()
    self._op2monitor = []

  @property
  def config(self):
    return self._config

  @property
  def model_proto(self):
    return self._model_proto

  @property
  def summary_op(self):
    return self._summary_op

  @property
  def loss_op(self):
    return self._loss_op

  @property
  def gradient_op(self):
    return self._gradient_op

  @property
  def train_op(self):
    return self._train_op

  @property
  def op2monitor(self):
      return self._op2monitor

  ######################################
  # functions to customize
  ######################################
  # return model proto
  def get_model_proto(self):
    raise NotImplementedError("""please customize FullModel.get_model_proto""")

  def add_tst_input(self, basegraph):
    raise NotImplementedError("""please customize FullModel.add_tst_input""")

  def add_trn_tst_input(self, basegraph):
    raise NotImplementedError("""please customize FullModel.add_trn_tst_input""")

  # return loss op
  def add_loss(self, basegraph):
    raise NotImplementedError("""please customize FullModel.add_loss""")

  def op_in_val(self):
    raise NotImplementedError("""please customize FullModel.op_in_val""")

  def op_in_tst(self):
    raise NotImplementedError("""please customize FullModel.op_in_tst""")

  ######################################
  # utility functions
  ######################################
  def append_op2monitor(name, op):
    self._op2monitor.append((name, op))

  ######################################
  # boilerpipe functions
  ######################################
  def op_in_trn(self):
    return self.loss_op, self.train_op, self.summary_op

  def _add_init(self, basegraph):
    with basegraph.as_default():
      with tf.variable_scope(self.name_scope):
        self._init_op = tf.initialize_all_variables()

  def _add_summary(self, basegraph):
    with basegraph.as_default():
      with tf.variable_scope(self.name_scope):
        tf.scalar_summary('loss', self._loss_op)
        for var in tf.trainable_variables():
          tf.histogram_summary(var.name + '/activations', var)
        for grad, var in self.gradient_op:
          if grad is not None:
            tf.histogram_summary(var.name + '/gradients', grad)
        self._summary_op = tf.merge_all_summaries()

  def _build_parameter_graph(self, basegraph):
    basegraph = tf.Graph()
    self.model_proto.build_parameter_graph(basegraph)

  def _add_saver(self, basegraph):
    with basegraph.as_default():
      self._saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=1000)

  def _build_inference_graph_in_trn_tst(self, basegraph):
    with basegraph.as_default():
      self.model_proto.build_inference_graph_in_trn_tst(basegraph)

  def _build_inference_graph_in_tst(self, basegraph):
    with basegraph.as_default():
      self.model_proto.build_inference_graph_in_tst(basegraph)

  def build_tst_graph(self):
    basegraph = tf.Graph()

    basegraph = self._build_parameter_graph(basegraph)
    self.add_tst_input(basegraph)

    self._build_inference_graph_in_tst(basegraph)

    self._add_saver(basegraph)
    self._add_init(basegraph)

    return basegraph 

  def build_trn_tst_graph(self):
    basegraph = tf.Graph()

    basegraph = self._build_parameter_graph(basegraph)
    self.add_trn_tst_input(basegraph)

    self._build_inference_graph_in_trn_tst(basegraph)

    self._loss_op = self.add_loss(basegraph)

    with basegraph.as_default():
      with tf.variable_scope(self.name_scope):
        with tf.device('/gpu:0'):
          if self.config.optimizer_alg == 'Adam':
            optimizer = tf.train.AdamOptimizer(self.config.learning_rate)
          elif self.config.optimizer_alg == 'SGD':
            optimizer = tf.train.GradientDescentOptimizer(self.config.learning_rate)
          self._gradient_op = optimizer.compute_gradients(self._loss_op)
          self._train_op = optimizer.apply_gradients(self._gradient_op)

    self._add_saver(basegraph)
    self._add_summary(basegraph)
    self._add_init(basegraph)

    return basegraph


# should implement:
#   combine: combine computation graph from model_protos, 
#             will be called in _build_inference_graph
#   all the functions to implement in FullModel
class ModelCombiner(FullModel):
  name_scope = 'ModelCombiner'

  def __init__(self, config):
    FullModel.__init__(self, config)
    self._model_protos = self.get_model_proto()
    self._train_ops = []

  @property
  def model_protos(self):
    return self._model_protos

  @property
  def train_ops(self):
    return self._train_ops

  ######################################
  # functions to customize
  ######################################
  def combine(self, basegraph):
    raise NotImplementedError("""please customize ModelCombiner.combine""")

  ######################################
  # boilerpipe functions
  ######################################

  def _build_parameter_graph(self):
    basegraph = tf.Graph()
    for model_proto in self.model_protos:
      model_proto.build_parameter_graph(basegraph)

  def _build_inference_graph_in_tst(self, basegraph):
    for model_proto in self.model_protos:
      model_proto.build_inference_graph_in_tst(basegraph)
    self.combine(basegraph)

  def _build_inference_graph_in_trn_tst(self, basegraph):
    for model_proto in self.model_protos:
      model_proto.build_inference_graph_in_trn_tst(basegraph)
    self.combine(basegraph)

  def build_trn_tst_graph(self):
    basegraph = tf.Graph()

    basegraph = self._build_parameter_graph(basegraph)
    self.add_trn_tst_input(basegraph)

    self._build_inference_graph_in_trn_tst(basegraph)

    self.add_loss(basegraph)

    with basegraph.as_default():
      with tf.variable_scope(self.name_scope):
        self.gradient_op = []
        for m, model_proto in enumerate(self.model_protos):
          if self.config.optimizer_alg == 'Adam':
            optimizer = tf.train.AdamOptimizer(self.config.learning_rates[m])
          elif self.config.optimizer_alg == 'SGD':
            optimizer = tf.train.GradientDescentOptimizer(self.config.learning_rate)

          if not self.config.stop_gradients[m]:
            weight = tf.get_collection(
              tf.GraphKeys.TRAINABLE_VARIABLES, model_proto.name_scope)
            grads_and_weights = optimizer.compute_gradients(self.loss_op, weight)
            self.gradient_op += grads_and_weights[0]
            self.train_ops.append(optimizer.apply_gradients(grads_and_weights))

    self._add_saver(basegraph)
    self._add_summary(basegraph)
    self._add_init(basegraph)

    return basegraph
