import os
import sys
import time
import cPickle
import collections
import time
import json
sys.path.append('../')

import tensorflow as tf

import toolkit


class PathCfg(object):
  def __init__(self):
    self.log_dir = ''
    self.model_dir = ''

    self.log_file = ''
    self.val_metric_file = ''
    self.model_file = ''
    self.predict_file = ''

  def load(self, file):
    data = json.load(open(file))
    for key in data:
      setattr(self, key, data[key])

# note: 
#   please look at examples to get an idea of what to implement in each function
# functions to implement:
#   feed_data_and_run_train_summary_op_in_trn,
#   feed_data_and_run_loss_op_in_val,
#   predict_and_eval_in_val
#   predict_in_tst
# implemented boilerpipe functions:
#   _iterate_epoch
#   _validation
#   train
#   test
class TrnTst(object):
  def __init__(self, model_cfg, path_cfg, model):
    self._model_cfg = None
    self._path_cfg = None
    self._model = None
    self._logger = None

    self._model_cfg = model_cfg
    self._path_cfg = path_cfg
    self._model = model

    self._logger = toolkit.set_logger('TrnTst%f'%time.time(), path_cfg.log_file)

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
  # return summarystr
  def feed_data_and_run_train_summary_op_in_trn(self, data, sess):
    raise NotImplementedError("""please customize feed_data_and_run_train_summary_op_in_trn""")

  # return loss value  
  def feed_data_and_run_loss_op_in_val(self, data, sess):
    raise NotImplementedError("""please customize feed_data_and_run_loss_op_in_val""")

  # add eval result to metrics dictionary, key is metric name, val is metric value
  def predict_and_eval_in_val(self, sess, tst_reader, metrics):
    raise NotImplementedError("""please customize predict_and_eval_in_val""")

  # write predict result to predict_file
  def predict_in_tst(self, sess, tst_reader, predict_file):
    raise NotImplementedError("""please customize predict_in_tst""")

  ######################################
  # boilerpipe functions
  ######################################
  def _iterate_epoch(self,
      sess, trn_reader, tst_reader, summarywriter, step, total_step, epoch):
    trn_batch_size = self.model_cfg.trn_batch_size
    for data in trn_reader.yield_trn_batch(trn_batch_size):
      summarystr = self.feed_data_and_run_train_summary_op_in_trn(data, self.model, sess)

      step += 1

      if step % self.model_cfg.summary_iter == 0:
        summarywriter.add_summary(summarystr, step)

      if step % self.model_cfg.val_iter == 0:
        metrics = self._validation(sess, tst_reader)

        self._logger.info('step (%d/%d)', step, total_step)
        for key in metrics:
          self._logger.info('%s:%.4f', key, metrics[key])

    self.model.saver.save(
      sess, os.path.join(self.path_cfg.model_dir, 'epoch'), global_step=epoch)

    return step

  def _validation(self, sess, tst_reader):
    metrics = collections.OrderedDict()
    batch_size = self.model_cfg.tst_batch_size

    # loss on validation
    iter_num = 0
    avg_loss = 0.
    for data in tst_reader.yield_val_batch(batch_size):
      loss = self.feed_data_and_run_loss_op_in_val(data, self.model, sess)
      avg_loss += loss
      iter_num += 1

    avg_loss /= iter_num
    metrics['loss'] = avg_loss

    metrics = self.predict_and_eval_in_val(sess, tst_reader, metrics)

    return metrics

  def train(self, trn_reader, tst_reader, memory_fraction=1.0, resume=False):
    batch_size = self.model_cfg.trn_batch_size
    batches_per_epoch = (trn_reader.num_caption + batch_size - 1) / batch_size
    total_step = batches_per_epoch * self.model_cfg.num_epoch

    trn_tst_graph = self.model.build_trn_tst_graph()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=memory_fraction)
    configProto = tf.ConfigProto(gpu_options=gpu_options)
    with tf.Session(graph=trn_tst_graph, config=configProto) as sess:
      sess.run(self.model.init_op)
      if resume:
        self.model.saver.restore(sess, self.path_cfg.model_file)
        name = os.path.basename(self.path_cfg.model_file)
        data = name.split('-')
        base_epoch = int(data[-1]) + 1
      else:
        base_epoch = 0
      summarywriter = tf.train.SummaryWriter(self.path_cfg.log_dir, graph=sess.graph)

      # round 0, just for quick checking
      metrics = self._validation(sess, tst_reader)
      self._logger.info('step (%d/%d)', 0, total_step)
      for key in metrics:
        self._logger.info('%s:%.4f', key, metrics[key])

      metric_history = []
      step = 0
      for epoch in xrange(self.model_cfg.num_epoch):
        step = self._iterate_epoch(
          sess, trn_reader, tst_reader, summarywriter, step, total_step, base_epoch + epoch)

        metrics = self._validation(sess, tst_reader)
        metric_history.append(metrics)

        self._logger.info('epoch (%d/%d)', epoch, self.model_cfg.num_epoch)
        for key in metrics:
          self._logger.info('%s:%.4f', key, metrics[key])

      val_metric_file = self.path_cfg.val_metric_file
      cPickle.dump(metric_history, open(val_metric_file, 'w'))

  def test(self, tst_reader, memory_fraction=1.0):
    batch_size = self.model_cfg.tst_batch_size

    tst_graph = self.model.build_tst_graph()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=memory_fraction)
    config_proto = tf.ConfigProto(gpu_options=gpu_options)
    with tf.Session(graph=tst_graph, config=config_proto) as sess:
      sess.run(self.model.init_op)
      self.model.saver.restore(sess, self.path_cfg.model_file)

    self.predict_in_tst(sess, tst_reader, self.path_cfg.predict_file)
