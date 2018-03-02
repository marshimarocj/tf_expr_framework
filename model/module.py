"""
Complex modules could be composed from several simple basic modules. 
As you can see from the above description, this is a recursive process. 
Thus, both ModuleConfig and AbstractModule are designed to be recursive. 
"""
import json
import enum

import tensorflow as tf
import numpy as np

import memory_saving_gradients


class Mode(enum.Enum):
  TRN_VAL = 0
  TST = 1
  ROLLOUT = 2
  SCORE = 3


class ModuleConfig(object):
  """
  config of a module
  in addition to the customized parameters in the config, it contains three special attributes:
  [subcfgs] a dictionary of configs belong to the submodules in this module
  [freeze] boolean, whether to freeze the weights in this module in training. N.B. it doesn't affect the training of submodules
  [lr_mult] float, the multiplier to the base learning rate for weights in this modules. N.B. it doesn't affect the traninig of submodules
  [opt_alg] string, 'Adam|SGD|RMSProp', optimizer
  """
  def __init__(self):
    self.subcfgs = {}
    self.freeze = False
    self.clip = False
    self.clip_interval = [-1., 1.]
    self.lr_mult = 1.0
    self.opt_alg = 'Adam'
    self.device = '/device:GPU:0'

  def load(self, cfg_dict):
    for key in cfg_dict:
      if key == 'subcfgs': # recursion
        data = cfg_dict[key]
        for key in data:
          self.subcfgs[key].load(data[key])
      elif key in self.__dict__:
        setattr(self, key, cfg_dict[key])

    self._assert()

  def save(self):
    out = {}
    for attr in self.__dict__:
      if attr == 'subcfgs':
        cfgs = self.__dict__[attr]
        out['subcfgs'] = {}
        for key in cfgs:
          out['subcfgs'][key] = cfgs[key].save()
      else:
        val = self.__dict__[attr]
        if type(val) is not np.ndarray: # ignore nparray fields, which are used to initialize weights
          out[attr] = self.__dict__[attr]
    return out

  def _assert(self):
    """
    check compatibility between configs
    """
    pass
    # raise NotImplementedError("""please customize ModuleConfig._assert""")


class AbstractModule(object):
  """
  a module only contains the weight to share
  and provides the function to construct the inference graph
  N.B. it doesn't construct the inference graph, it only provides the function to do so
  it doesn't contain loss and the gradient
  in addition to the customized members, it contains three special members:
  [config] the config of this module
  [op2monitor] a dictionary of ops to monitor for debug
  [submods] a dictionary of submodules in this module
  """
  name_scope = 'AbstractModule'

  class InKey(enum.Enum):
    pass

  class OutKey(enum.Enum):
    pass

  def __init__(self, config):
    self._config = config
    self._op2monitor = {}
    self._submods = self._set_submods()
    self._weights = []

  @property
  def config(self):
    return self._config

  @property
  def op2monitor(self):
    return self._op2monitor

  @property
  def submods(self):
    return self._submods

  @property
  def weights(self):
    return self._weights

  def _set_submods(self):
    """
    return a dictionary of submods
    """
    raise NotImplementedError("""please customize AbstractModule._set_submods""")

  def _build_parameter_graph(self):
    """
    this would be called before get_out_ops_in_mode. 
    shared parts between trn and tst are encouraged to be placed in this function
    """
    raise NotImplementedError("""please customize AbstractModule.build_parameter_graph""")

  def get_out_ops_in_mode(self, in_ops, mode, reuse=False):
    """
    return out_ops (a dictionary) given in_ops (a dictionary), 
    """
    raise NotImplementedError("""please customize AbstractModule.get_out_ops_in_mode""")

  def build_parameter_graph(self):
    with tf.device(self._config.device):
      self._build_parameter_graph()
    for key in self._submods:
      submod = self._submods[key]
      with tf.device(self._config.subcfgs[key].device):
        submod.build_parameter_graph()


class ModelConfig(ModuleConfig):
  def __init__(self):
    ModuleConfig.__init__(self)

    self.trn_batch_size = 256
    self.tst_batch_size = 128
    self.num_epoch = 100
    self.val_iter = 100
    self.val_loss = True
    self.monitor_iter = 1
    self.base_lr = 1e-4
    self.decay_schema = '' # , piecewise_constant
    self.decay_boundarys = []
    self.decay_values = []
    self.save_memory = False

  def load(self, file):
    with open(file) as f:
      data = json.load(f)
      ModuleConfig.load(self, data)

  def save(self, out_file):
    out = ModuleConfig.save(self)
    with open(out_file, 'w') as fout:
      json.dump(out, fout, indent=2)


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

  def op_in_tst(self):
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
  def saver(self):
    return self._outputs[self.DefaultKey.SAVER]

  @property
  def summary_op(self):
    return self._outputs[self.DefaultKey.SUMMARY]

  def build_trn_tst_graph(self, decay_boundarys=[]):
    basegraph = tf.Graph()
    with basegraph.as_default():
      self._inputs = self._add_input_in_mode(Mode.TRN_VAL)
      self.build_parameter_graph()
      self._outputs = self.get_out_ops_in_mode(self._inputs, Mode.TRN_VAL)
      self._outputs[self.DefaultKey.LOSS] = self._add_loss()
      self._outputs[self.DefaultKey.TRAIN] = self._calculate_gradient()
      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      self._outputs[self.DefaultKey.TRAIN].extend(update_ops)

      _recursive_gather_op2monitor_helper(self, self._op2monitor)
      self._outputs[self.DefaultKey.SAVER] = self._add_saver()
      self._outputs[self.DefaultKey.SUMMARY] = self._add_summary()
      self._outputs[self.DefaultKey.INIT] = self._add_init()

    return basegraph

  def build_tst_graph(self):
    basegraph = tf.Graph()
    with basegraph.as_default():
      self._inputs = self._add_input_in_mode(Mode.TST)
      self.build_parameter_graph()
      self._outputs = self.get_out_ops_in_mode(self._inputs, Mode.TST)

      self._outputs[self.DefaultKey.SAVER] = self._add_saver()
      self._outputs[self.DefaultKey.INIT] = self._add_init()

    return basegraph

  def op_in_trn(self):
    return {
      self.DefaultKey.LOSS: self._outputs[self.DefaultKey.LOSS],
      self.DefaultKey.TRAIN: self._outputs[self.DefaultKey.TRAIN],
    }

  def op_in_val(self):
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
    # saver = tf.train.Saver(tf.trainable_variables() + tf.get_collection(tf.GraphKeys.MOVING_AVERAGE_VARIABLES), max_to_keep=1000)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=1000)
    # saver = tf.train.Saver(max_to_keep=1000)
    return saver

  def _add_init(self):
    with tf.variable_scope(self.name_scope):
      init = tf.global_variables_initializer()
    return init

  # def _calculate_gradient(self):
  #   train_ops = []
  #   loss_op = self._outputs[self.DefaultKey.LOSS]
  #   with tf.variable_scope(self.name_scope):
  #     _recursive_gradient_helper(self, loss_op, self.config.base_lr,
  #       train_ops)
  #   return train_ops

  def _calculate_gradient(self):
    loss_op = self._outputs[self.DefaultKey.LOSS]
    optimizer2ws = {}
    _recursive_collect_weight_and_optimizers(self, self.config.base_lr, 
      optimizer2ws)
    ws = []
    optimizer2idxs = {}
    for optimizer in optimizer2ws:
      start_idx = len(ws)
      ws += optimizer2ws[optimizer]
      end_idx = len(ws)
      optimizer2idxs[optimizer] = (start_idx, end_idx)

    train_ops = []
    if self._config.save_memory:
      grads = memory_saving_gradients.gradients(loss_op, ws, gate_gradients=True)
    else:
      grads = tf.gradients(loss_op, ws, gate_gradients=True)
    grads_and_weights = zip(grads, ws)
    # for grad, w in grads_and_weights:
    #   print grad.op.name, w.op.name
    for optimizer in optimizer2idxs:
      start_idx, end_idx = optimizer2idxs[optimizer]
      train_ops.append(optimizer.apply_gradients(grads_and_weights[start_idx:end_idx]))

    return train_ops


# def _recursive_gradient_helper(module, loss_op, base_lr,
#     train_ops):
#   weight = tf.get_collection(
#     tf.GraphKeys.TRAINABLE_VARIABLES, module.name_scope)
#   if len(weight) > 0 and not module.config.freeze:
#     learning_rate = base_lr * module.config.lr_mult

#     if module.config.opt_alg == 'Adam':
#       optimizer = tf.train.AdamOptimizer(learning_rate)
#     elif self.config.opt_alg == 'SGD':
#       optimizer = tf.train.GradientDescentOptimizer(learning_rate)
#     elif self.config.opt_alg == 'RMSProp':
#       optimizer = tf.train.RMSPropOptimizer(learning_rate)
#     grads_and_weights = optimizer.compute_gradients(loss_op, weight)
#     if module._config.clip:
#       grads_and_weights = [
#         (tf.clip_by_value(grad, module._config.clip_interval[0], module._config.clip_interval[1]), var) 
#         for grad, var in grads_and_weights
#       ]
#     train_ops.append(optimizer.apply_gradients(grads_and_weights))
#   # recursive
#   for key in module.submods:
#     submod = module.submods[key]
#     _recursive_gradient_helper(submod, loss_op, base_lr, 
#       train_ops)


def _recursive_collect_weight_and_optimizers(module, base_lr, optimizer2ws):
  # weight = tf.get_collection(
  #   tf.GraphKeys.TRAINABLE_VARIABLES, module.name_scope)
  weight = module.weights
  # for w in weight:
  #   print module.name_scope, w.op.name

  if len(weight) > 0 and not module.config.freeze:
    learning_rate = base_lr * module.config.lr_mult

    if module.config.opt_alg == 'Adam':
      optimizer = tf.train.AdamOptimizer(learning_rate)
    elif self.config.opt_alg == 'SGD':
      optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    elif self.config.opt_alg == 'RMSProp':
      optimizer = tf.train.RMSPropOptimizer(learning_rate)

    optimizer2ws[optimizer] = weight
  # recursive
  for key in module.submods:
    submod = module.submods[key]
    _recursive_collect_weight_and_optimizers(submod, base_lr, optimizer2ws)

def _recursive_gather_op2monitor_helper(module, op2monitor):
  op2monitor.update(module.op2monitor)
  for key in module.submods:
    submod = module.submods[key]
    _recursive_gather_op2monitor_helper(submod, op2monitor)


class AbstractPGModel(AbstractModel):
  """
  model is the root node in the tree of module composition and is a special type of module. 
  therefore, it is an inheritance of AbstractModule. 
  it contains the full computation graph, including loss, graident, save, summary in addition to inference
  a model has the following special members:
  """
  def __init__(self, config):
    AbstractModule.__init__(self, config)

    self._inputs = {}
    self._outputs = {}
    self._rollout_inputs = {}
    self._rollout_outputs = {}

  def op_in_rollout(self):
    """
    return dictionary of op in rollout
    """
    raise NotImplementedError("""please customize AbstractPGModel.op_in_rollout""")

  @property
  def rollout_inputs(self):
    return self._rollout_inputs

  @property
  def rollout_outputs(self):
    return self._rollout_outputs

  def build_trn_tst_graph(self, decay_boundarys=[]):
    basegraph = tf.Graph()
    with basegraph.as_default():
      self._inputs = self._add_input_in_mode(Mode.TRN_VAL)
      self._rollout_inputs = self._add_input_in_mode(Mode.ROLLOUT)
      self.build_parameter_graph()
      self._outputs = self.get_out_ops_in_mode(self._inputs, Mode.TRN_VAL)

      self._outputs[self.DefaultKey.LOSS] = self._add_loss()
      self._outputs[self.DefaultKey.TRAIN] = self._calculate_gradient()
      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      self._outputs[self.DefaultKey.TRAIN].extend(update_ops)
      self._rollout_outputs = self.get_out_ops_in_mode(self._rollout_inputs, Mode.ROLLOUT)

      _recursive_gather_op2monitor_helper(self, self._op2monitor)
      self._outputs[self.DefaultKey.SAVER] = self._add_saver()
      self._outputs[self.DefaultKey.SUMMARY] = self._add_summary()
      self._outputs[self.DefaultKey.INIT] = self._add_init()

    return basegraph


class AbstractStructModel(AbstractPGModel):
  """
  model is the root node in the tree of module composition and is a special type of module. 
  therefore, it is an inheritance of AbstractModule. 
  it contains the full computation graph, including loss, graident, save, summary in addition to inference
  a model has the following special members:
  """
  def __init__(self, config):
    AbstractModule.__init__(self, config)

    self._inputs = {}
    self._outputs = {}
    self._rollout_inputs = {}
    self._rollout_outputs = {}
    self._score_inputs = {}
    self._score_outputs = {}

  def op_in_score(self):
    """
    return dictionary of op in rollout
    """
    raise NotImplementedError("""please customize AbstractStructModel.op_in_score""")

  @property
  def score_inputs(self):
    return self._score_inputs

  @property
  def score_outputs(self):
    return self._score_outputs

  def build_trn_tst_graph(self, decay_boundarys=[]):
    basegraph = tf.Graph()
    with basegraph.as_default():
      self._inputs = self._add_input_in_mode(Mode.TRN_VAL)
      self._rollout_inputs = self._add_input_in_mode(Mode.ROLLOUT)
      self._score_inputs = self._add_input_in_mode(Mode.SCORE)
      self.build_parameter_graph()
      self._outputs = self.get_out_ops_in_mode(self._inputs, Mode.TRN_VAL)
      self._outputs[self.DefaultKey.LOSS] = self._add_loss()
      self._outputs[self.DefaultKey.TRAIN] = self._calculate_gradient()
      self._rollout_outputs = self.get_out_ops_in_mode(self._rollout_inputs, Mode.ROLLOUT)
      self._score_outputs = self.get_out_ops_in_mode(self._score_inputs, Mode.SCORE)

      _recursive_gather_op2monitor_helper(self, self._op2monitor)
      self._outputs[self.DefaultKey.SAVER] = self._add_saver()
      self._outputs[self.DefaultKey.SUMMARY] = self._add_summary()
      self._outputs[self.DefaultKey.INIT] = self._add_init()

    return basegraph
