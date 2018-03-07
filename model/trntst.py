import os
import sys
import time
import cPickle
import collections
import json
sys.path.append('../')

import tensorflow as tf
from tensorflow.python import debug as tf_debug
import framework.util.graph_ckpt

import toolkit


class PathCfg(object):
  def __init__(self):
    self.log_dir = ''
    self.model_dir = ''

    self.log_file = ''
    # self.val_metric_file = ''
    self.model_file = ''
    self.predict_file = ''

  def load(self, file):
    data = json.load(open(file))
    for key in data:
      setattr(self, key, data[key])


# note: 
#   please look at examples to get an idea of what to implement in each function
# functions to implement:
#   feed_data_and_trn,
#   feed_data_and_monitor_in_trn,
#   feed_data_and_summary,
#   feed_data_and_run_loss_op_in_val,
#   predict_and_eval_in_val
#   predict_in_tst
# implemented boilerpipe functions:
#   _iterate_epoch
#   _validation
#   train
#   test
class TrnTst(object):
  def __init__(self, model_cfg, path_cfg, model, debug=False):
    self._model_cfg = None
    self._path_cfg = None
    self._model = None
    self._logger = None
    self._debug = debug

    self._model_cfg = model_cfg
    self._path_cfg = path_cfg
    self._model = model

    self._logger = toolkit.set_logger('TrnTst', path_cfg.log_file)

  @property
  def model_cfg(self):
    return self._model_cfg

  @property
  def path_cfg(self):
    return self._path_cfg

  @property
  def model(self):
      return self._model

  ######################################
  # functions to customize 
  ######################################
  def feed_data_and_trn(self, data, sess):
    raise NotImplementedError("""please customize feed_data_and_trn""")

  def feed_data_and_monitor_in_trn(self, data, sess, step):
    raise NotImplementedError("""please customize feed_data_and_monitor""")

  def feed_data_and_summary(self, data, sess):
    """
    return summarystr
    """
    raise NotImplementedError("""please customize feed_data_and_summary""")

  def feed_data_and_run_loss_op_in_val(self, data, sess):
    """
    return loss value
    """
    raise NotImplementedError("""please customize feed_data_and_run_loss_op_in_val""")

  def predict_and_eval_in_val(self, sess, tst_reader, metrics):
    """
    add eval result to metrics dictionary, key is metric name, val is metric value
    """
    raise NotImplementedError("""please customize predict_and_eval_in_val""")

  def predict_in_tst(self, sess, tst_reader, predict_file):
    """
    write predict result to predict_file
    """
    raise NotImplementedError("""please customize predict_in_tst""")

  def customize_func_after_each_epoch(self, epoch):
    pass

  def manual_init_weight(self, sess):
    """
    manual initialize weight
    """
    raise NotImplementedError("""please customize manual_init_weight""")

  ######################################
  # boilerpipe functions
  ######################################
  def _iterate_epoch(self,
      sess, trn_reader, tst_reader, summarywriter, step, total_step, epoch):
    trn_batch_size = self.model_cfg.trn_batch_size
    trn_time = 0.
    trn_reader.reset()
    for data in trn_reader.yield_trn_batch(trn_batch_size):
      tic = time.time()
      self.feed_data_and_trn(data, sess)
      toc = time.time()
      trn_time += toc - tic

      step += 1

      if self.model_cfg.monitor_iter > 0 and step % self.model_cfg.monitor_iter == 0:
        self.feed_data_and_monitor_in_trn(data, sess, step)

      if self.model_cfg.val_iter > 0 and step % self.model_cfg.val_iter == 0:
        tic = time.time()
        metrics = self._validation(sess, tst_reader)
        toc = time.time()
        val_time = toc - tic

        self._logger.info('step (%d/%d)', step, total_step)
        self._logger.info('%f s for trn', trn_time)
        self._logger.info('%f s for val', val_time)
        trn_time = 0.
        rollout_time = 0.
        for key in metrics:
          self._logger.info('%s:%.4f', key, metrics[key])

    summarystr = self.feed_data_and_summary(data, sess)
    summarywriter.add_summary(summarystr, step)
    self.model.saver.save(
      sess, os.path.join(self.path_cfg.model_dir, 'epoch'), global_step=epoch)

    return step

  def _validation(self, sess, tst_reader):
    metrics = collections.OrderedDict()
    batch_size = self.model_cfg.tst_batch_size

    # loss on validation
    if self.model_cfg.val_loss:
      iter_num = 0
      avg_loss = 0.
      for data in tst_reader.yield_val_batch(batch_size):
        loss = self.feed_data_and_run_loss_op_in_val(data, sess)
        avg_loss += loss
        iter_num += 1
        # print loss

      avg_loss /= iter_num
      metrics['loss'] = avg_loss

    self.predict_and_eval_in_val(sess, tst_reader, metrics)

    return metrics

  def train(self, trn_reader, tst_reader, memory_fraction=1.0, resume=False, manual_init=False):
    batch_size = self.model_cfg.trn_batch_size
    batches_per_epoch = (trn_reader.num_record() + batch_size - 1) / batch_size
    total_step = batches_per_epoch * self.model_cfg.num_epoch

    decay_boundarys = []
    if self.model_cfg.decay_schema == 'piecewise_constant':
      decay_boundarys = self.model_cfg.decay_boundarys
      decay_boundarys = [int(d*total_step) for d in decay_boundarys]
    trn_tst_graph = self.model.build_trn_tst_graph(decay_boundarys=decay_boundarys)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=memory_fraction)
    configProto = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
    with tf.Session(graph=trn_tst_graph, config=configProto) as sess:
      if self._debug:
        sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
      sess.run(self.model.init_op)
      if resume:
        # self.model.saver.restore(sess, self.path_cfg.model_file)
        self._restore(sess, trn_tst_graph, self.path_cfg.model_file)
        name = os.path.basename(self.path_cfg.model_file)
        data = name.split('-')
        try:
          base_epoch = int(data[-1]) + 1
        except:
          base_epoch = 0
      else:
        base_epoch = 0
        if manual_init:
          self.manual_init_weight(sess)
      summarywriter = tf.summary.FileWriter(self.path_cfg.log_dir, graph=sess.graph)

      # round 0, just for quick checking
      metrics = self._validation(sess, tst_reader)
      self._logger.info('step (%d/%d)', 0, total_step)
      for key in metrics:
        self._logger.info('%s:%.4f', key, metrics[key])

      step = 0
      for epoch in xrange(base_epoch, self.model_cfg.num_epoch):
        step = self._iterate_epoch(
          sess, trn_reader, tst_reader, summarywriter, step, total_step, epoch)

        metrics = self._validation(sess, tst_reader)
        metrics['epoch'] = epoch

        self._logger.info('epoch (%d/%d)', epoch, self.model_cfg.num_epoch)
        for key in metrics:
          self._logger.info('%s:%.4f', key, metrics[key])
        val_log_file = os.path.join(self.path_cfg.log_dir, 'val_metrics.%d.json'%epoch)
        with open(val_log_file, 'w') as fout:
          json.dump(metrics, fout, indent=2)

        self.customize_func_after_each_epoch(epoch)

  def test(self, tst_reader, memory_fraction=1.0):
    tst_graph = self.model.build_tst_graph()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=memory_fraction)
    config_proto = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
    with tf.Session(graph=tst_graph, config=config_proto) as sess:
      if self._debug:
        sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
      sess.run(self.model.init_op)
      if self.path_cfg.model_file is not None:
        # self.model.saver.restore(sess, self.path_cfg.model_file)
        self._restore(sess, trn_tst_graph, self.path_cfg.model_file)

      self.predict_in_tst(sess, tst_reader, self.path_cfg.predict_file)

  def _restore(self, sess, graph, ckpt_file):
    with graph.as_default():
      all_var_names = set([v.op.name for v in  tf.global_variables()] + \
        [v.op.name for v in tf.get_collection(tf.GraphKeys.MOVING_AVERAGE_VARIABLES)])

    key2val = framework.util.graph_ckpt.load_variable_in_ckpt(ckpt_file)

    out_key2val = {}
    for key in all_var_names:
      if key in key2val:
        out_key2val[key] = key2val[key]

    with graph.as_default():
      assign_op, feed_dict = tf.contrib.framework.assign_from_values(out_key2val)

    sess.run(assign_op, feed_dict=feed_dict)


class PGTrnTst(TrnTst):
  ######################################
  # functions to customize 
  ######################################
  def feed_data_and_rollout(self, data, sess):
    """
    return data
    """
    raise NotImplementedError("""please customize feed_data_and_rollout""")

  ######################################
  # boilerpipe functions
  ######################################
  def _iterate_epoch(self,
      sess, trn_reader, tst_reader, summarywriter, step, total_step, epoch):
    trn_batch_size = self.model_cfg.trn_batch_size
    trn_time = 0.
    rollout_time = 0.
    trn_reader.reset()
    for data in trn_reader.yield_trn_batch(trn_batch_size):
      tic = time.time()
      data = self.feed_data_and_rollout(data, sess)
      toc = time.time()
      rollout_time += toc - tic

      tic = toc
      self.feed_data_and_trn(data, sess)
      toc = time.time()
      trn_time += toc - tic

      step += 1

      if self.model_cfg.monitor_iter > 0 and step % self.model_cfg.monitor_iter == 0:
        self.feed_data_and_monitor_in_trn(data, sess, step)

      if self.model_cfg.val_iter > 0 and step % self.model_cfg.val_iter == 0:
        tic = time.time()
        metrics = self._validation(sess, tst_reader)
        toc = time.time()
        val_time = toc - tic

        self._logger.info('step (%d/%d)', step, total_step)
        self._logger.info('%f s for trn', trn_time)
        self._logger.info('%f s for val', val_time)
        self._logger.info('%f s for rollout', rollout_time)
        trn_time = 0.
        rollout_time = 0.
        for key in metrics:
          self._logger.info('%s:%.4f', key, metrics[key])

    summarystr = self.feed_data_and_summary(data, sess)
    summarywriter.add_summary(summarystr, step)
    self.model.saver.save(
      sess, os.path.join(self.path_cfg.model_dir, 'epoch'), global_step=epoch)

    return step


class StructTrnTst(PGTrnTst):
  ######################################
  # functions to customize 
  ######################################
  def init_pool(self):
    raise NotImplementedError("""please customzie init_pool""")

  def feed_data_and_score(self, data, sess):
    """
    return data
    """
    raise NotImplementedError("""please customize feed_data_and_score""")

  def update_pool(self, data):
    """
    return data
    """
    raise NotImplementedError("""please customize update_pool""")

  ######################################
  # boilerpipe functions
  ######################################
  def _iterate_epoch(self,
      sess, trn_reader, tst_reader, summarywriter, step, total_step, epoch):
    trn_batch_size = self.model_cfg.trn_batch_size
    trn_time = 0.
    rollout_time = 0.
    trn_reader.reset()
    for data in trn_reader.yield_trn_batch(trn_batch_size):
      tic = time.time()
      data = self.feed_data_and_rollout(data, sess)
      data = self.feed_data_and_score(data, sess)
      data = self.update_pool(data)
      toc = time.time()
      rollout_time += toc - tic

      tic = toc
      self.feed_data_and_trn(data, sess)
      toc = time.time()
      trn_time += toc - tic

      step += 1

      if self.model_cfg.monitor_iter > 0 and step % self.model_cfg.monitor_iter == 0:
        self.feed_data_and_monitor_in_trn(data, sess, step)

      if self.model_cfg.val_iter > 0 and step % self.model_cfg.val_iter == 0:
        tic = time.time()
        metrics = self._validation(sess, tst_reader)
        toc = time.time()
        val_time = toc - tic

        self._logger.info('step (%d/%d)', step, total_step)
        self._logger.info('%f s for trn', trn_time)
        self._logger.info('%f s for val', val_time)
        self._logger.info('%f s for finding most violated constraints', rollout_time)
        trn_time = 0.
        rollout_time = 0.
        for key in metrics:
          self._logger.info('%s:%.4f', key, metrics[key])

    summarystr = self.feed_data_and_summary(data, sess)
    summarywriter.add_summary(summarystr, step)
    self.model.saver.save(
      sess, os.path.join(self.path_cfg.model_dir, 'epoch'), global_step=epoch)

    return step

  def train(self, trn_reader, tst_reader, memory_fraction=1.0, resume=False):
    self.init_pool()

    PGTrnTst.train(self, trn_reader, tst_reader, memory_fraction=memory_fraction, resume=resume)
