import os
import sys
import json
import cPickle
import random
sys.path.append('../../')

import tensorflow as tf
from tensorflow.python.util import nest
import numpy as np
from bleu import bleu
from cider import cider

import model.proto
import model.trntst
import model.data
import util.utility
import util.expanded_op


class EncoderDecoderBase(model.proto.ModelCombiner):
  name_scope = 'model.vevd.EncoderDecoderBase/'

  def add_tst_input(self, basegraph):
    encoder = self.model_protos[0]
    decoder = self.model_protos[1]

    with basegraph.as_default():
      with tf.variable_scope(self.name_scope):
        self._fts = tf.placeholder(
          tf.float32, 
          shape=(None, sum(self.config.encoder_cfg.dim_fts)), 
          name='fts')
        self._init_wordids = tf.placeholder(
          tf.int32, shape=(None, ), name='wordIds')
        # self._states = [
        #   tf.placeholder(tf.float32, shape=(None, state_size)) 
        #   for state_size in nest.flatten(decoder.state_size)
        # ]
        # self._states = nest.pack_sequence_as(decoder.state_size, self._states)

    encoder.fts = self._fts
    decoder.init_wordids = self._wordids

  def add_trn_tst_input(self, basegraph):
    encoder = self.model_protos[0]
    decoder = self.model_protos[1]

    with basegraph.as_default():
      with tf.variable_scope(self.name_scope):
        self._fts = tf.placeholder(
          tf.float32, shape=(None, sum(self.config.encoder_cfg.dim_fts)), name='fts')
        # trn only
        self._captionids = tf.placeholder(
          tf.int32, shape=(None, self.config.decoder_cfg.max_words_in_caption), name='captionids')
        self._caption_masks = tf.placeholder(
          tf.float32, shape=(None, self.config.decoder_cfg.max_words_in_caption), name='caption_masks')
        # tst only
        self._init_wordids = tf.placeholder(
          tf.int32, shape=(None, ), name='wordIds')
        # self._states = [
        #   tf.placeholder(tf.float32, shape=(None, state_size)) 
        #   for state_size in nest.flatten(decoder.state_size)
        # ]
        # self._states = nest.pack_sequence_as(decoder.state_size, self._states)

    encoder.fts = self._fts
    decoder.captionids = self._captionids
    decoder.caption_masks = self._caption_masks
    decoder.wordids = self._wordids

  def add_loss(self, basegraph):
    encoder = self.model_protos[0]
    decoder = self.model_protos[1]

    with basegraph.as_default():
      with tf.variable_scope(self.name_scope):
        logits = tf.concat(0, decoder.logit_ops)
        loss_op = util.expanded_op.cross_entropy_loss_on_rnn_logits(
          self._captionids, self._caption_masks, logits)
        self.append_op2monitor('cross_entropy_loss_on_rnn_logits', loss_op)

        if self.config.encoder_cfg.reg_type != None:
          encoder.add_reg()
          loss_op += self.config.encoder_cfg.reg_lambda * encoder.regularize_op
          self.append_op2monitor(
            'encoder_reg_%s'%self.config.encoder_cfg.reg_type, encoder.regularize_op)

        if self.config.decoder_cfg.reg_type != None:
          decoder.add_reg()
          loss_op += self.config.decoder_cfg.reg_lambda * decoder.regularize_op
          self.append_op2monitor(
            'decoder_reg_%s'%self.config.decoder_cfg.ret_type, decoder.regularize_op)

    return loss_op

  def build_inference_graph_in_tst(self, basegraph):
    encoder = self.model_protos[0]
    decoder = self.model_protos[1]

    encoder.build_inference_graph_in_tst(basegraph)
    decoder.ft_embeds = encoder.feature_op
    decoder.build_inference_graph_in_tst(basegraph)

  def build_inference_graph_in_trn_tst(self, basegraph):
    encoder = self.model_protos[0]
    decoder = self.model_protos[1]

    encoder.build_inference_graph_in_trn_tst(basegraph)
    decoder.ft_embeds = encoder.feature_op
    decoder.build_inference_graph_in_trn_tst(basegraph)

  def op_in_val(self):
    encoder = self.model_protos[0]
    decoder = self.model_protos[1]

    return {
      'loss_op': self.loss_op,
      'decoder.tst_ft_state_op': decoder.tst_ft_state_op,
      'decoder.predict_prob_ops': decoder.predict_prob_ops,
      'decoder.output_ops': decoder.output_ops,
    }

  def op_in_tst(self):
    encoder = self.model_protos[0]
    decoder = self.model_protos[1]

    return {
      'decoder.tst_ft_state_op': decoder.tst_ft_state_op,
      'decoder.predict_prob_ops': decoder.predict_prob_ops,
      'decoder.output_ops': decoder.output_ops,
    }


class PathCfgBase(model.trntst.PathCfg):
  def __init__(self):
    model.trntst.PathCfg.__init__(self)
    # manually provided in the cfg file
    self.split_dir = ''
    self.annotation_dir = ''
    self.output_dir = ''
    self.trn_ftfiles = []
    self.val_ftfiles = []
    self.tst_ftfiles = []

    # automatically generated paths
    self.trn_videoid_file = ''
    self.val_videoid_file = ''
    self.tst_videoid_file = ''
    self.trn_annotation_file = ''
    self.val_annotation_file = ''
    self.groundtruth_file = ''
    self.word_file = ''


# functions to implement:
#   predict_and_eval_in_val
#   predict_in_tst
# implemented boilerpipe functions:
#   feed_data_and_trn,
#   feed_data_and_monitor_in_trn,
#   feed_data_and_summary,
#   feed_data_and_run_loss_op_in_val,
# provide helper functions:
#   output_by_sent_mode
class TrnTstBase(model.trntst.TrnTst):
  def __init__(self, model_cfg, path_cfg, model, gen_sent_mode=1):
    model.trntst.TrnTst.__init__(self, model_cfg, path_cfg, model)

    # caption int to string
    self.int2str = util.utility.CaptionInt2str(path_cfg.word_file)

    self.gen_sent_mode = gen_sent_mode

  def feed_data_and_trn(self, data, sess):
    op_dict = self.model.op_in_trn()

    feed_dict = self._construct_feed_dict_in_trn(data)
    out = sess.run(
      [op_dict['loss_op']] + op_dict['train_ops'],
      feed_dict=feed_dict)

  def feed_data_and_monitor_in_trn(self, data, sess, step):
    op2monitor = self.model.op2monitor
    names = op2monitor.keys()
    ops = op2monitor.values()

    feed_dict = self._construct_feed_dict_in_trn(data)
    out = sess.run(ops, feed_dict=feed_dict)
    for name, val in zip(names, out):
      print '(step %d) monitor "%s":'%(step, name)
      print val

  def feed_data_and_summary(self, data, sess):
    summary_op = self.model.summary_op

    feed_dict = self._construct_feed_dict_in_trn(data)
    out = sess.run(summary_op, feed_dict=feed_dict)

    return out

  def feed_data_and_run_loss_op_in_val(self, data, sess):
    op_dict = self.model.op_in_val()

    feed_dict = self._construct_feed_dict_in_trn(data)
    loss = sess.run(op_dict['loss_op'], feed_dict=feed_dict)

    return loss

  def output_by_sent_mode(self, sent_pool, videoid, videoid2caption):
    sent_pool_size = self.model_cfg.decoder_cfg.sent_pool_size
    if self.gen_sent_mode == 1:
      captionid = np.expand_dims(sent_pool[0][1], 0)
      videoid2caption[videoid] = self.int2str(captionid)[0]
    elif self.gen_sent_mode == 2:
      videoid2caption[videoid] = []
      for k in xrange(sent_pool_size):
        captionid = np.expand_dims(sent_pool[k][1], 0)
        out = (
          float(sent_pool[k][0]),
          self.int2str(captionid)[0]
        )
        videoid2caption[videoid].append(out)
    elif self.gen_sent_mode == 3:
      videoid2caption[videoid] = []
      for k in xrange(sent_pool_size):
        captionid = np.expand_dims(sent_pool[k][1], 0)
        out = (
          float(sent_pool[k][0]),
          self.int2str(captionid)[0],
          [float(x) for x in sent_pool[k][2]]
        )
        videoid2caption[videoid].append(out)

  def _construct_feed_dict_in_trn(self, data):
    raise NotImplementedError("""please customize TrnTstBase._construct_encoder_feed_dict_in_trn""")

  def _construct_encoder_feed_dict_in_tst(self, data):
    raise NotImplementedError("""please customize TrnTstBase._construct_encoder_feed_dict_in_tst""")

  def _construct_decoder_feed_dict_in_tst(self, data, **kwargs):
    raise NotImplementedError("""please customize TrnTstBase._construct_decoder_feed_dict_in_tst""")


# def predict_eval_in_val(trntst, sess, tst_reader, metrics):
#   # greedy generator on validation
#   videoid2caption = {}
#   base = 0
#   op_dict = trntst.model.op_in_val()

#   batch_size = trntst.model_cfg.tst_batch_size
#   for data in tst_reader.yield_tst_batch(batch_size):
#     encoder_feed_dict = trntst._construct_encoder_feed_dict_in_tst(data)
#     states = sess.run(
#       op_dict['decoder.tst_ft_state_op'], feed_dict=encoder_feed_dict)

#     decoder_feed_dict = trntst._construct_decoder_feed_dict_in_tst(data)
#     sent_pool = util.utility.greedy_word_decode(
#       sess, states, 
#       trntst.model._states, trntst.model._wordids, 
#       op_dict['decoder.update_state_op'], 
#       op_dict['decoder.predict_prob_op'], 
#       op_dict['decoder.output_op'],
#       trntst.model_cfg.decoder_cfg.max_words_in_caption, 
#       addition_input_placeholders=decoder_feed_dict.keys(), addition_inputs=decoder_feed_dict.values())

#     for k, sent in enumerate(sent_pool):
#       videoid = tst_reader.videoids[base + k]
#       videoid2caption[videoid] = trntst.int2str(np.expand_dims(sent, 0))
#     base += batch_size

#   bleu_scorer = bleu.Bleu(4)
#   bleu_score, _ = bleu_scorer.compute_score(tst_reader.videoid2captions, videoid2caption)
#   for i in range(4):
#     metrics['bleu%d'%(i+1)] = bleu_score[i]

#   cider_scorer = cider.Cider()
#   cider_score, _ = cider_scorer.compute_score(tst_reader.videoid2captions, videoid2caption)
#   metrics['cider'] = cider_score


# def predict_eval_in_val_general_attention(trntst, sess, tst_reader, metrics):
#   # greedy generator on validation
#   videoid2caption = {}
#   base = 0
#   op_dict = trntst.model.op_in_val()

#   batch_size = trntst.model_cfg.tst_batch_size
#   for data in tst_reader.yield_tst_batch(batch_size):
#     encoder_feed_dict = trntst._construct_encoder_feed_dict_in_tst(data)
#     states, ft_embeds, attention_ft_embeds  = sess.run(
#       [
#         op_dict['decoder.init_state_op'], 
#         op_dict['mean_encoder.feature_op'], 
#         op_dict['attention_encoder.feature_op']
#       ], 
#       feed_dict=encoder_feed_dict)

#     # share computation of Uav_plus_ba in decoder steps
#     decoder_feed_dict = trntst._construct_decoder_feed_dict_in_tst(
#       data, attention_ft_embeds=attention_ft_embeds)
#     Uav_plus_ba = sess.run(
#       op_dict['decoder.Uav_plus_ba_op'], feed_dict=decoder_feed_dict)
#     decoder_feed_dict[trntst.model._Uav_plus_ba] = Uav_plus_ba

#     sent_pool = util.utility.greedy_word_decode(
#       sess, states, 
#       trntst.model._states, trntst.model._wordids, 
#       op_dict['decoder.update_state_op'], 
#       op_dict['decoder.predict_prob_op'], 
#       op_dict['decoder.output_op'],
#       trntst.model_cfg.decoder_cfg.max_words_in_caption, 
#       _last_output=trntst.model._last_output, init_output=ft_embeds,
#       addition_input_placeholders=decoder_feed_dict.keys(), addition_inputs=decoder_feed_dict.values())

#     for k, sent in enumerate(sent_pool):
#       videoid = tst_reader.videoids[base + k]
#       videoid2caption[videoid] = trntst.int2str(np.expand_dims(sent, 0))
#     base += batch_size

#   bleu_scorer = bleu.Bleu(4)
#   bleu_score, _ = bleu_scorer.compute_score(tst_reader.videoid2captions, videoid2caption)
#   for i in range(4):
#     metrics['bleu%d'%(i+1)] = bleu_score[i]

#   cider_scorer = cider.Cider()
#   cider_score, _ = cider_scorer.compute_score(tst_reader.videoid2captions, videoid2caption)
#   metrics['cider'] = cider_score


# def predict_in_tst(trntst, sess, tst_reader, predict_file):
#   videoid2caption = {}
#   base = 0
#   op_dict = trntst.model.op_in_tst()

#   max_words_in_caption = trntst.model_cfg.decoder_cfg.max_words_in_caption
#   width = trntst.model_cfg.decoder_cfg.beam_width
#   sent_pool_size = trntst.model_cfg.decoder_cfg.sent_pool_size

#   for data in tst_reader.yield_tst_batch(trntst.model_cfg.tst_batch_size):
#     encoder_feed_dict = trntst._construct_encoder_feed_dict_in_tst(data)
#     states = sess.run(
#       op_dict['decoder.tst_ft_state_op'], feed_dict=encoder_feed_dict)

#     decoder_feed_dict = trntst._construct_decoder_feed_dict_in_tst(data)
#     sent_pool = util.utility.beamsearch_word_decode(
#       sess, states,
#       trntst.model._states, trntst.model._wordids, 
#       op_dict['decoder.update_state_op'], 
#       op_dict['decoder.predict_prob_op'], 
#       op_dict['decoder.output_op'],
#       max_words_in_caption, width, sent_pool_size,
#       addition_input_placeholders=decoder_feed_dict.keys(), addition_inputs=decoder_feed_dict.values())

#     for b in xrange(len(sent_pool)):
#       videoid = str(tst_reader.videoids[b+base])

#       trntst.output_by_sent_mode(sent_pool[b], videoid, videoid2caption)

#     base += len(sent_pool)

#   json.dump(videoid2caption, open(predict_file, 'w'))


# def predict_in_tst_general_attention(trntst, sess, tst_reader, predict_file):
#   videoid2caption = {}
#   base = 0
#   op_dict = trntst.model.op_in_tst()

#   max_words_in_caption = trntst.model_cfg.decoder_cfg.max_words_in_caption
#   width = trntst.model_cfg.decoder_cfg.beam_width
#   sent_pool_size = trntst.model_cfg.decoder_cfg.sent_pool_size

#   for data in tst_reader.yield_tst_batch(trntst.model_cfg.tst_batch_size):
#     encoder_feed_dict = trntst._construct_encoder_feed_dict_in_tst(data)
#     states, ft_embeds, attention_ft_embeds = sess.run(
#       [
#         op_dict['decoder.init_state_op'], 
#         op_dict['mean_encoder.feature_op'], 
#         op_dict['attention_encoder.feature_op']
#       ],
#       feed_dict=encoder_feed_dict)

#     # share computation of Uav_plus_ba among decoder steps
#     decoder_feed_dict = trntst._construct_decoder_feed_dict_in_tst(
#       data, attention_ft_embeds=attention_ft_embeds)
#     Uav_plus_ba = sess.run(
#       op_dict['decoder.Uav_plus_ba_op'], feed_dict=decoder_feed_dict)
#     decoder_feed_dict[trntst.model._Uav_plus_ba] = Uav_plus_ba

#     sent_pool = util.utility.beamsearch_word_decode(
#       sess, states,
#       trntst.model._states, trntst.model._wordids, 
#       op_dict['decoder.update_state_op'], 
#       op_dict['decoder.predict_prob_op'], 
#       op_dict['decoder.output_op'],
#       max_words_in_caption, width, sent_pool_size, 
#       _last_outputs=trntst.model._last_output, init_output=ft_embeds,
#       addition_input_placeholders=decoder_feed_dict.keys(), addition_inputs=decoder_feed_dict.values())

#     for b in xrange(len(sent_pool)):
#       videoid = str(tst_reader.videoids[b+base])

#       trntst.output_by_sent_mode(sent_pool[b], videoid, videoid2caption)

#     base += len(sent_pool)

#   json.dump(videoid2caption, open(predict_file, 'w'))


class ReaderBase(model.data.Reader):
  def __init__(self, ft_files, videoid_file, 
      shuffle=True, annotation_file=None, captionstr_file=None):
    self.fts = np.empty(0) # (numVideo, dimVideo)
    self.ft_idxs = np.empty(0) # (num_caption,)
    self.captionids = np.empty(0) # (num_caption, maxWordsInCaption)
    self.caption_masks = np.empty(0) # (num_caption, maxWordsInCaption)
    self.videoids = []
    self.videoid2captions = {} # (numVideo, numGroundtruth)

    self.shuffled_idxs = [] # (num_caption,)
    self.num_caption = 0 # used in trn and val
    self.num_ft = 0
    self.caption_batch_pos = 0
    self.ft_batch_pos = 0

    fts = []
    for ft_file in ft_files:
      ft = np.load(ft_file)
      fts.append(ft)
    self.fts = np.concatenate(tuple(fts), axis=1)
    self.num_ft = self.fts.shape[0]

    self.videoids = np.load(open(videoid_file))

    if annotation_file is not None:
      self.ft_idxs, self.captionids, self.caption_masks = cPickle.load(file(annotation_file))
      self.num_caption = self.ft_idxs.shape[0]
    if captionstr_file is not None:
      videoid2captions = cPickle.load(open(captionstr_file))
      for videoid in self.videoids:
        self.videoid2captions[videoid] = videoid2captions[videoid]

    self.caption_batch_pos = 0
    self.ft_batch_pos = 0

    self.shuffled_idxs = range(self.num_caption)
    if shuffle:
      random.shuffle(self.shuffled_idxs)

  def num_record(self):
    return self.num_caption


class ReaderTrnSubsetBase(model.data.Reader):
  def __init__(self, ft_files, videoid_file, 
      shuffle=True, annotation_file=None, captionstr_file=None):
    self.fts = np.empty(0) # (numVideo, dimVideo)
    self.ft_idxs = [] # (num_caption,)
    self.captionids = [] # (num_caption, maxWordsInCaption)
    self.caption_masks = [] # (num_caption, maxWordsInCaption)
    self.videoids = []
    self.videoid2captions = {} # (numVideo, numGroundtruth)

    self.shuffled_idxs = [] # (num_caption,)
    self.num_caption = 0 # used in trn and val
    self.num_ft = 0
    self.caption_batch_pos = 0
    self.ft_batch_pos = 0

    self.videoids = np.load(open(videoid_file))
    videoid_set = set(self.videoids)

    fts = []
    for ft_file in ft_files:
      ft = np.load(ft_file)
      fts.append(ft)
    self.fts = np.concatenate(tuple(fts), axis=1)
    # self.fts = self.fts[self.videoids]
    self.num_ft = self.fts.shape[0]

    if annotation_file is not None:
      ft_idxs, captionids, caption_masks = cPickle.load(file(annotation_file))
      for i in range(ft_idxs.shape[0]):
        if ft_idxs[i] in videoid_set:
          self.ft_idxs.append(ft_idxs[i])
          self.captionids.append(captionids[i])
          self.caption_masks.append(caption_masks[i])
      self.ft_idxs = np.array(self.ft_idxs)
      self.captionids = np.array(self.captionids)
      self.caption_masks = np.array(self.caption_masks)
      self.num_caption = self.ft_idxs.shape[0]
    if captionstr_file is not None:
      videoid2captions = cPickle.load(open(captionstr_file))
      for videoid in self.videoids:
        self.videoid2captions[videoid] = videoid2captions[videoid]

    self.caption_batch_pos = 0
    self.ft_batch_pos = 0

    self.shuffled_idxs = range(self.num_caption)
    if shuffle:
      random.shuffle(self.shuffled_idxs)

  def num_record(self):
    return self.num_caption
