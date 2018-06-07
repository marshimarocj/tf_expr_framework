import os
import sys
import random
import enum
import math

import tensorflow as tf
import numpy as np

import framework.model.module


class Config(framework.model.module.ModuleConfig):
  def __init__(self):
    framework.model.module.ModuleConfig.__init__(self)

    self.dim_ft = 0
    self.dim_pca_ft = 0
    self.num_ft = 0
    self.num_center = 0

    self.l2_norm_input = False

    self.centers = np.empty((0,)) # (dim_ft, num_center)

  def _assert(self):
    pass


class Encoder(framework.model.module.AbstractModel):
  name_scope = 'vlad.Encoder'

  class InKey(enum.Enum):
    FT = 'fts' # (None, num_ft, dim_ft)
    MASK = 'masks' # (None, num_ft)

  class OutKey(enum.Enum):
    VLAD = 'vlad'

  def _set_submods(self):
    return {}

  def _build_parameter_graph(self):
    with tf.variable_scope(self.name_scope):
      if self._config.centers.shape[0] > 0:
        self.centers = tf.contrib.framework.model_variable('centers',
          shape=(self._config.dim_pca_ft, self._config.num_center), dtype=tf.float32,
          initializer=tf.constant_initializer(self._config.centers))
      else:
        self.centers = tf.contrib.framework.model_variable('centers',
          shape=(self._config.dim_pca_ft, self._config.num_center), dtype=tf.float32,
          initializer=tf.contrib.layers.xavier_initializer())
      self._weights.append(self.centers)
      if self._config.dim_pca_ft != self._config.dim_ft:
        stddev = math.sqrt(3.0 / (self._config.dim_ft + self._config.dim_pca_ft))
        self.conv_W = tf.contrib.framework.model_variable('conv_W',
          shape=(1, self._config.dim_ft, self._config.dim_pca_ft), dtype=tf.float32,
          initializer=tf.random_normal_initializer(mean=0., stddev=stddev))
        self.conv_B = tf.contrib.framework.model_variable('conv_B',
          shape=(self._config.dim_pca_ft,), dtype=tf.float32,
          initializer=tf.constant_initializer(0.))
        self._weights.append(self.conv_W)
        self._weights.append(self.conv_B)
      scale = 1.0 / (self._config.dim_pca_ft ** 0.5)
      self.w = tf.contrib.framework.model_variable('w',
        shape=(self._config.dim_pca_ft, self._config.num_center), dtype=tf.float32,
        initializer=tf.random_uniform_initializer(-scale, scale))
      self.b = tf.contrib.framework.model_variable('b',
        shape=(self._config.num_center,), dtype=tf.float32,
        initializer=tf.random_uniform_initializer(-0.1, 0.1))
      self._weights.append(self.w)
      self._weights.append(self.b)

  def _netvlad(self, fts, masks):
    if self._config.l2_norm_input:
      fts = tf.nn.l2_normalize(fts, dim=1)
    if self._config.dim_ft != self._config.dim_pca_ft:
      fts = tf.nn.conv1d(fts, self.conv_W, 1, 'VALID')
      fts = tf.nn.bias_add(fts, self.conv_B)

    fts = tf.reshape(fts, (-1, self._config.dim_pca_ft)) # (None*num_ft, dim_pca_ft)
    logits = tf.nn.xw_plus_b(fts, self.w, self.b) # (None*num_ft, num_center)
    a = tf.nn.softmax(logits)

    a = tf.expand_dims(a, 1) # (None*num_ft, 1, num_center)
    fts = tf.expand_dims(fts, 2) # (None*num_ft, dim_ft, 1)
    centers = tf.expand_dims(self.centers, 0) # (1, dim_pca_ft, num_center)
    diff = fts - centers # (None*num_ft, dim_pca_ft, num_center)
    V_ijk = a * diff # (None*num_ft, dim_pca_ft, num_center)
    masks = tf.reshape(masks, (-1, 1, 1))
    V_ijk *= masks
    dim_vlad = self._config.dim_pca_ft* self._config.num_center
    V_ijk = tf.reshape(V_ijk, (-1, self._config.num_ft, dim_vlad))
    V_jk = tf.reduce_sum(V_ijk, 1) # (None, dim_vlad)

    return V_jk

  def get_out_ops_in_mode(self, in_ops, mode):
    with tf.variable_scope(self.name_scope):
      V_jk = self._netvlad(in_ops[self.InKey.FT], in_ops[self.InKey.MASK])

    return {
      self.OutKey.VLAD: V_jk,
    }
