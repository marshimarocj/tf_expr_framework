"""
Complex modules could be composed from several simple basic modules. 
As you can see from the above description, this is a recursive process. 
Thus, both ModuleConfig and AbstractModule are designed to be recursive. 
"""
import json
import enum

import tensorflow as tf
import numpy as np
import horovod.tensorflow as hvd

import module


class AbstractModel(AbstractModule):
  """
  model is the root node in the tree of module composition and is a special type of module. 
  therefore, it is an inheritance of AbstractModule. 
  it contains the full computation graph, including loss, graident, save, summary in addition to inference
  a model has the following special members:
  """
  name_scope = 'AbstractModel'

  class DefaultKey(enum.Enum):
    INIT = 'init'
    LOSS = 'loss'
    TRAIN = 'train'
    SAVER = 'saver'
    SUMMARY = 'summary'
    BCAST = 'broadcast'

  def __init__(self, config):
    AbstractModule.__init__(self, config)

    self._inputs = {}
    self._outputs = {}

  def _add_input_in_mode(self, mode):
    """
    return dictionary of inputs
    """
    raise NotImplementedError("""please customize AbstractModel._add_input_in_mode""")

  def _add_loss(self):
    """
    return loss op
    """
    raise NotImplementedError("""please customize AbstractModel._add_loss""")

  def op_in_tst(self, **kwargs):
    """
    return dictionary of op in tst
    """
    raise NotImplementedError("""please customize AbstractModel.op_in_tst""")

  @property
  def inputs(self):
    return self._inputs

  @property
  def outputs(self):
    return self._outputs

  @property
  def init_op(self):
    return self._outputs[self.DefaultKey.INIT]

  @property
  def bcast(self):
    return self._outputs[self.DefaultKey.BCAST]

  @property
  def saver(self):
    return self._outputs[self.DefaultKey.SAVER]

  @property
  def summary_op(self):
    return self._outputs[self.DefaultKey.SUMMARY]

  def build_trn_tst_graph(self, decay_boundarys=[], step=0):
    basegraph = tf.Graph()
    with basegraph.as_default():
      self._inputs = self._add_input_in_mode(Mode.TRN_VAL)
      self.build_parameter_graph()
      self._outputs = self.get_out_ops_in_mode(self._inputs, Mode.TRN_VAL)

      if len(decay_boundarys) > 0:
        global_step = tf.Variable(step, trainable=False)
        base_lr = tf.train.piecewise_constant(global_step, decay_boundarys, self._config.decay_values)
      else:
        base_lr = self._config.base_lr

      self._outputs[self.DefaultKey.LOSS] = self._add_loss()
      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      with tf.control_dependencies(update_ops):
        self._outputs[self.DefaultKey.TRAIN] = self._calculate_gradient(base_lr)

      _recursive_gather_op2monitor_helper(self, self._op2monitor)
      self._outputs[self.DefaultKey.SAVER] = self._add_saver()
      self._outputs[self.DefaultKey.SUMMARY] = self._add_summary()
      self._outputs[self.DefaultKey.INIT] = self._add_init()
      self._outputs[self.DefaultKey.BCAST] = self._add_bcast()

  def build_tst_graph(self):
    basegraph = tf.Graph()
    with basegraph.as_default():
      self._inputs = self._add_input_in_mode(Mode.TST)
      self.build_parameter_graph()
      self._outputs = self.get_out_ops_in_mode(self._inputs, Mode.TST)

      self._outputs[self.DefaultKey.SAVER] = self._add_saver()
      self._outputs[self.DefaultKey.INIT] = self._add_init()
      self._outputs[self.DefaultKey.BCAST] = self._add_bcast()

    return basegraph

  def op_in_trn(self, **kwargs):
    return {
      self.DefaultKey.LOSS: self._outputs[self.DefaultKey.LOSS],
      self.DefaultKey.TRAIN: self._outputs[self.DefaultKey.TRAIN],
    }

  def op_in_val(self, **kwargs):
    return {
      self.DefaultKey.LOSS: self._outputs[self.DefaultKey.LOSS],
    }

  def _add_summary(self):
    with tf.variable_scope(self.name_scope):
      tf.summary.scalar('loss', self._outputs[self.DefaultKey.LOSS])
      for var in tf.trainable_variables():
        tf.summary.histogram(var.name + '/activations', var)
      summary_op = tf.summary.merge_all()
    return summary_op

  def _add_saver(self):
    # model_vars = tf.trainable_variables() + self._get_batchnorm_stat_vars()
    # print [d.op.name for d in self._get_batchnorm_stat_vars()]
    model_vars = tf.get_collection(tf.GraphKeys.MODEL_VARIABLES)
    # model_vars = tf.global_variables()
    saver = tf.train.Saver(model_vars, max_to_keep=1000)
    return saver

  def _add_init(self):
    with tf.variable_scope(self.name_scope):
      init = tf.global_variables_initializer()
    return init

  def _add_bcast(self):
    with tf.variable_scope(self.name_scope):
      bcast = hvd.broadcast_global_variables(0)
    return bcast

  def _calculate_gradient(self, base_lr):
    loss_op = self._outputs[self.DefaultKey.LOSS]

    train_ops = _recursive_train_ops(self, base_lr, loss_op)
    return train_ops


def _recursive_train_ops(module, base_lr, loss_op):
  weights = module.weights

  all_train_ops = []
  if len(weights) > 0 and not module.config.freeze:
    learning_rate = base_lr * module.config.lr_mult
    if self.config.opt_alg == 'Adam':
      optimizer = tf.train.AdamOptimizer(learning_rate)
    elif self.config.opt_alg == 'SGD':
      optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    elif self.config.opt_alg == 'RMSProp':
      optimizer = tf.train.RMSPropOptimizer(learning_rate)
    optimizer = hvd.DistributedOptimizer(optimizer)

    grad_vars = optimizer.compute_gradients(loss_op, var_list=weights)
    train_op = optimizer.apply_gradients(grad_vars)
    all_train_ops.append(train_op)

  # recursive
  for key in module.submods:
    submod = module.submods[key]
    train_ops = _recursive_train_ops(submod, loss_op)
    all_train_ops += train_ops

  return all_train_ops


def _recursive_gather_op2monitor_helper(module, op2monitor):
  op2monitor.update(module.op2monitor)
  for key in module.submods:
    submod = module.submods[key]
    _recursive_gather_op2monitor_helper(submod, op2monitor)
