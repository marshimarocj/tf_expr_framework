"""
Complex modules could be composed from several simple basic modules. 
As you can see from the above description, this is a recursive process. 
Thus, both ModuleConfig and AbstractModule are designed to be recursive. 
"""
import json

import tensorflow as tf


class ModuleConfig(object):
  """
  config of a module
  in addition to the customized parameters in the config, it contains tree special attributes:
  [subconfigs] a dictionary of configs belong to the submodules in this module
  [freeze] boolean, whether to freeze the weights in this module in training. N.B. it doesn't affect the training of submodules
  [lr_mult] float, the multiplier to the base learning rate for weights in this modules. N.B. it doesn't affect the traninig of submodules
  [opt_alg] string, 'Adam|SGD|RMSProp', optimizer
  """
  def __init__(self):
    self.subconfigs = {}
    self.freeze = False
    self.lr_mult = 1.0
    self.opt_alg = 'Adam'

  def load(self, cfg_dict):
    for key in cfg_dict:
      if key == 'subconfigs': # recursion
        data = cfg_dict[key]
        for key in data:
          self.subconfigs[key].load(data[key])
      elif key in self.__dict__:
        setattr(self, key, cfg_dict[key])


class AbstractModule(object):
  """
  a module only contains the weight to share
  and provides the function to construct the inference graph
  N.B. it doesn't construct the inference graph, it only provides the function to do so
  it doesn't contain loss and the gradient
  in addition to the customized members, it contains three special members:
  [config] the config of this module
  [op2monitor] a dictionary of ops to monitor for debug
  [submodules] a dictionary of submodules in this module
  """
  name_scope = 'AbstractModule'

  def __init__(self, config):
    self._config = config
    self._submodules = {}
    self._op2monitor = {}
    self._input_keys = []
    self._output_keys = []

  @property
  def config(self):
    return self._config

  @property
  def op2monitor(self):
    return self._op2monitor

  @property
  def submodules(self):
    return self._submodules

  def build_parameter_graph(self, basegraph):
    """
    this would be called before get_out_ops_in_trn and get_out_ops_in_tst. 
    shared parts between trn and tst are encouraged to be placed in this function
    """
    raise NotImplementedError("""please customize AbstractModule.build_parameter_graph""")

  def get_out_ops_in_trn(self, basegraph, in_ops):
    """return out_ops (a dictionary) in trn given in_ops (a dictionary)"""
    raise NotImplementedError("""please customize AbstractModule.get_out_ops_in_trn""")

  def get_out_ops_in_val(self, basegraph, in_ops):
    """return out_ops (a dictionary) in trn given in_ops (a dictionary)"""
    raise NotImplementedError("""please customize AbstractModule.get_out_ops_in_trn""")

  def get_out_ops_in_tst(self, basegraph, in_ops):
    """return out_ops (a dictionary) in tst given in_ops (a dictionary)"""
    raise NotImplementedError("""please customize AbstractModule.get_out_ops_in_trn""")


class ModelConfig(object):
  def __init__(self):
    self.module_config = ModuleConfig()

    self.trn_batch_size = 256
    self.tst_batch_size = 128
    self.num_epoch = 100
    self.val_iter = 100
    self.monitor_iter = 1
    self.base_lr = 1e-4

  def load(self, file):
    with open(file) as f:
      data = json.load(f)
      for key in data:
      if key == 'module': # recursion
        self.module_config.load(data[key])
      elif key in self.__dict__:
        setattr(self, key, data[key])


class AbstractModel(object):
  """
  model contains the full computation graph, including loss, graident, save, summary in addition to inference
  a model has the following special members:
  [_module] the module object, which is actually the root node of the module recursive tree
  [trn_inputs] dictionary of trn input placeholder
  [val_inputs] dictionary of val input placeholder
  [tst_inputs] dictionary of tst input placeholder
  [val_outputs] dictionary of val output ops
  [tst_outputs] dictionary of tst output ops
  """
  name_scope = 'AbstractModel'

  def __init__(self, config):
    self.config = config
    self._module = self._set_module()

    self._trn_inputs = {}
    self._val_inputs = {}
    self._tst_inputs = {}
    self._val_outputs = {}
    self._tst_outputs = {}

    self._loss_op = tf.no_op()
    self._train_ops = []
    self._saver = tf.no_op()
    self._summary_op = tf.no_op()
    self._init_op = tf.no_op()
    self._op2monitor = {}

  def _set_module(self):
    """
    return the module for the model
    """
    raise NotImplementedError("""please customize AbstractModel.set_module""")

  def _add_trn_input(self):
    """
    return dictionary of input placeholders
    """
    raise NotImplementedError("""please customize AbstractModel.set_trn_input""")

  def _add_val_input(self):
    """
    return dictionary of input placeholders
    """
    raise NotImplementedError("""please customize AbstractModel.set_val_input""")

  def _add_tst_input(self):
    """
    return dictionary of input placeholders
    """
    raise NotImplementedError("""please customize AbstractModel.set_tst_input""")

  def _add_loss(self, trn_inputs, trn_outputs):
    """
    return loss op
    """
    raise NotImplementedError("""please customize AbstractModel.set_loss""")

  @property
  def trn_inputs(self):
    return self._trn_inputs

  @property
  def val_inputs(self):
    return self._val_inputs

  @property
  def tst_inputs(self):
    return self._tst_inputs

  @property
  def val_outputs(self):
    return self._val_outputs

  @property
  def tst_outputs(self):
    return self._tst_outputs

  @property
  def init_op(self):
    return self._init_op

  @property
  def saver(self):
    return self._saver

  @property
  def summary_op(self):
    return self._summary_op

  @property
  def op2monitor(self):
      return self._op2monitor

  def build_trn_val_graph(self):
    basegraph = tf.Graph()
    self._trn_inputs = self._add_trn_input()
    self._val_inputs = self._add_val_input()
    self._module.build_parameter_graph(basegraph)
    trn_outputs = self._module.get_out_ops_in_trn(basegraph, self.trn_inputs)
    self._val_outputs = self._module.get_out_ops_in_val(basegraph, self.val_inputs)
    self._loss_op = self._add_loss(self.trn_inputs, trn_outputs)
    self._calculate_gradient(basegraph)

    _recursive_gather_op2monitor_helper(self._module, self._op2monitor)
    self._add_saver(basegraph)
    self._add_summary(basegraph)
    self._add_init(basegraph)

    return basegraph

  def build_trn_tst_graph(self):
    """
    provided to make the inteface coherent with old version
    this is actually building trn and val graph
    """
    self.build_trn_val_graph()

  def build_tst_graph(self):
    basegraph = tf.Graph()
    self._tst_inputs = self._add_tst_input()
    self._module.build_parameter_graph(basegraph)
    self._tst_outputs = self._module.get_out_ops_in_tst(basegraph, self.tst_inputs)

    self._add_saver(basegraph)
    self._add_init(basegraph)

    return basegraph

  def op_in_trn(self):
    return {
      'loss_op': self._loss_op, 
      'train_ops': self._train_ops,
    }

  def _add_summary(self, basegraph):
    with basegraph.as_default():
      with tf.variable_scope(self.name_scope):
        tf.summary.scalar('loss', self._loss_op)
        for var in tf.trainable_variables():
          tf.summary.histogram(var.name + '/activations', var)
        self._summary_op = tf.summary.merge_all()

  def _add_saver(self, basegraph):
    with basegraph.as_default():
      self._saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=1000)

  def _add_init(self, basegraph):
    with basegraph.as_default():
      with tf.variable_scope(self.name_scope):
        self._init_op = tf.global_variables_initializer()

  def _calculate_gradient(self, basegraph):
    with basegraph.as_default():
      with tf.variable_scope(self.name_scope):
        _recursive_gradient_helper(self.module, self._loss_op, self.config.base_lr,
          self._train_ops)


def _recursive_gradient_helper(module, loss_op, base_lr,
    train_ops):
  weight = tf.get_collection(
    tf.GraphKeys.TRAINABLE_VARIABLES, module.name_scope)
  if len(weight) > 0 and not module.config.freeze:
    learning_rate = base_lr * self.config.lr_mult

    if module.config.opt_alg == 'Adam':
      optimizer = tf.train.AdamOptimizer(learning_rate)
    elif self.config.opt_alg == 'SGD':
      optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    elif self.config.opt_alg == 'RMSProp':
      optimizer = tf.train.RMSPropOptimizer(learning_rate)
    grads_and_weights = optimizer.compute_gradients(loss_op, weight)
    train_ops.append(optimizer.apply_gradients(grads_and_weights))
  # recursive
  for key in module.submodules:
    submodule = module.submodules[key]
    _recursive_gradient_helper(submodule, loss_op, base_lr, 
      train_ops)


def _recursive_gather_op2monitor_helper(module, op2monitor):
  op2monitor.update(module.op2monitor)
  for key in module.submodules:
    submodule = module.submodules[key]
    _recursive_gather_op2monitor_helper(submodule, op2monitor)
