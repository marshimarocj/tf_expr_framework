import enum

import tensorflow as tf

import framework.model.module
import framework.util.expanded_op


class Config(framework.model.module.ModuleConfig):
  def __init__(self):
    framework.model.module.ModuleConfig.__init__(self)

    self.opt_alg = 'SGD'


class Module(framework.model.module.AbstractModule):
  name_scope = 'poincare.Module'

  class InKey(enum.Enum):
    INPUT = 'input'

  class OutKey(enum.Enum):
    OUTPUT = 'output'

  def _set_submods(self):
    return {}

  def _build_parameter_graph(self):
    pass

  def get_out_ops_in_mode(self, in_ops, mode, reuse=False, **kwargs):
    with tf.variable_scope(self.name_scope):
      input = in_ops[self.InKey.INPUT]
      output = framework.util.expanded_op.lorentz_gradient(input)
    return {self.OutKey.OUTPUT: output}
