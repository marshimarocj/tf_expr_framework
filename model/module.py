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
  """
  def __init__(self):
    self.subconfigs = {}
    self.freeze = False
    self.lr_mult = 1.0

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
  a module only contains the inference graph
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
    self.base_lr = 1e-4
    self.module_config = ModuleConfig()

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
  [module] the module object, which is actually the root node of the module recursive tree
  """
  def __init__(self, config):
    self.config = config
    self.module = self._set_module()

    self._loss_op = tf.no_op()
    self._train_op = tf.no_op()
    self._saver = tf.no_op()
    self._summary_op = tf.no_op()

  def _set_module(self):
    """
    return the module for the model
    """
    raise NotImplementedError("""please customize AbstractModel.set_module""")

  def _set_trn_input(self):
    """
    return dictionary of input placeholders
    """
    raise NotImplementedError("""please customize AbstractModel.set_trn_input""")

  def _set_val_input(self):
    """
    return dictionary of input placeholders
    """
    raise NotImplementedError("""please customize AbstractModel.set_val_input""")

  def _set_tst_input(self):
    """
    return dictionary of input placeholders
    """
    raise NotImplementedError("""please customize AbstractModel.set_tst_input""")

  def _set_loss(self):
    raise NotImplementedError("""please customize AbstractModel.set_loss""")

  @property
  def saver(self):
    return self._saver

  @property
  def summary_op(self):
    return self._summary_op

  def build_trn_val_graph(self):
    basegraph = tf.Graph()
    self.trn_inputs = self.set_trn_input()
    self.val_inputs = self.set_val_input()
    self.module.build_parameter_graph(basegraph)
    self.trn_outputs = self.module.get_out_ops_in_trn(basegraph, trn_inputs)
    self.val_outputs = self.module.get_out_ops_in_val(basegraph, val_inputs)

  def build_trn_tst_graph(self):
    """
    provided to make the inteface coherent with old version
    this is actually building trn and val graph
    """
    self.build_trn_val_graph()

  def op_in_trn(self):
    return {
      'loss_op': self.loss_op, 
      'train_op': self.train_op,
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

  def _calculate_gradient(self):
    pass

