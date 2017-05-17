import os
import sys
import json
import cPickle

import tensorflow as tf
from tensorflow.python.util import nest
import numpy as np
from bleu import bleu
from cider import cider

import framework.model.proto
import framework.model.trntst
import framework.model.data
import framework.impl.caption.encoder.vanilla
import framework.impl.caption.decoder.vanilla
import base


class ModelConfigBase(framework.model.proto.ModelCombinerConfig):
  def __init__(self):
    framework.model.proto.ModelCombinerConfig.__init__(self)

    self.encoder_cfg = framework.impl.caption.encoder.vanilla.Config()
    self.decoder_cfg = framework.impl.caption.decoder.vanilla.Config()

  def load(self, file):
    data = framework.model.proto.ModelCombinerConfig.load(self, file)

    self.encoder_cfg.load(data['encoder'])
    self.decoder_cfg.load(data['decoder'])


class ModelConfig(ModelConfigBase):
  def load(self, file):
    ModelConfigBase.load(self, file)
    assert self.encoder_cfg.dim_output == self.decoder_cfg.dim_input


class ModelHiddentSetConfig(ModelConfigBase):
  def load(self, file):
    ModelConfigBase.load(self, file)
    assert self.encoder_cfg.dim_output == self.decoder_cfg.dim_hidden


class EncoderDecoder(base.EncoderDecoderBase):
  name_scope = 'framework.model.vevd.EncoderDecoder/'

  def get_model_proto(self):
    _encoder = framework.impl.caption.encoder.vanilla.Encoder(self.config.encoder_cfg)
    _decoder = framework.impl.caption.decoder.vanilla.Decoder(self.config.decoder_cfg)

    return [_encoder, _decoder]

  def add_tst_input(self, basegraph):
    base.EncoderDecoderBase.add_tst_input(self, basegraph)

    with basegraph.as_default():
      with tf.variable_scope(self.name_scope):
        self._init_state = [tf.placeholder(tf.float32, shape=(none, state_size))
          for state_size in nest.flatten(framework.impl.caption.decoder.state_size)]
        self._init_state = nest.pack_sequence_as(framework.impl.caption.decoder.state_size, self._init_state)

    decoder = self.model_protos[1]
    framework.impl.caption.decoder.init_state = self._init_state

  def add_trn_tst_input(self, basegraph):
    base.EncoderDecoderBase.add_trn_tst_input(self, basegraph)

    with basegraph.as_default():
      with tf.variable_scope(self.name_scope):
        self._init_state = tf.placeholder(
          tf.float32, shape=(None, framework.impl.caption.decoder.state_size), name='_init_state')

    decoder = self.model_protos[1]
    framework.impl.caption.decoder.init_state = self._init_state


class EncoderDecoderHiddenSet(base.EncoderDecoderBase):
  # name_scope = 'framework.model.vevd.EncoderDecoderHiddenSet'
  name_scope = 'framework.model.vevd.EncoderDecoderHiddenSet/'

  def get_model_proto(self):
    _encoder = framework.impl.caption.encoder.vanilla.Encoder(self.config.encoder_cfg)
    _decoder = framework.impl.caption.decoder.vanilla.DecoderHiddenSet(self.config.decoder_cfg)

    return [_encoder, _decoder]


class PathCfg(base.PathCfgBase):
  pass


class TrnTst(base.TrnTstBase):
  def _construct_feed_dict_in_trn(self, data):
    fts = data[0]
    captionids = data[1]
    caption_masks = data[2]

    batch_size = fts.shape[0]
    decoder = self.framework.model.model_protos[1]
    state_init = [
      np.zeros((batch_size, state_size), dtype=np.float)
        for state_size in nest.flatten(framework.impl.caption.decoder.state_size)
    ]
    init_wordids = np.zeros((batch_size,), dtype=np.int32)

    keys = [
        self.framework.model._fts, 
        self.framework.model._captionids, 
        self.framework.model._caption_masks, 
        self.framework.model._init_wordids
      ]
    keys += nest.flatten(self.framework.model._init_state)
    values = [fts, captionids, caption_masks, init_wordids] + state_init

    return dict(zip(keys, values))

  def _construct_encoder_feed_dict_in_tst(self, data):
    fts = data

    batch_size = fts.shape[0]
    decoder = self.framework.model.model_protos[1]
    state_init = [
      np.zeros((batch_size, state_size), dtype=np.float)
      for state_size in nest.flatten(framework.impl.caption.decoder.state_size)
    ]
    init_wordids = np.zeros((batch_size,), dtype=np.int32)

    keys = [self.framework.model._fts, self.framework.model._init_wordids]
    keys += nest.flatten(self.framework.model._init_state)
    values = [fts, init_wordids] + state_init

    return dict(zip(keys, values))

  def _construct_decoder_feed_dict_in_tst(self, data):
    return {}

  def predict_and_eval_in_val(self, sess, tst_reader, metrics):
    base.predict_eval_in_val(self, sess, tst_reader, metrics)

  def predict_in_tst(self, sess, tst_reader, predict_file):
    base.predict_in_tst(self, sess, tst_reader, predict_file) 
  

class TrnTstHiddenSet(TrnTst):
  # @override
  def _construct_feed_dict_in_trn(self, data):
    fts = data[0]
    captionids = data[1]
    caption_masks = data[2]
    batch_size = fts.shape[0]
    init_wordids = np.zeros((batch_size,), dtype=np.int32)

    return {
      self.framework.model._fts: fts,
      self.framework.model._captionids: captionids,
      self.framework.model._caption_masks: caption_masks,
      self.framework.model._init_wordids: init_wordids,
    }

  # @override
  def _construct_encoder_feed_dict_in_tst(self, data):
    fts = data[0]
    batch_size = fts.shape[0]
    init_wordids = np.zeros((batch_size,), dtype=np.int32)

    return {
      self.framework.model._fts: fts,
      self.framework.model._init_wordids: init_wordids,
    }


class Reader(base.ReaderBase):
  def yield_trn_batch(self, batch_size):
    for i in range(0, self.num_caption, batch_size):
      start = i
      end = i + batch_size
      idxs = self.shuffled_idxs[start:end]

      yield (self.fts[self.ft_idxs[idxs]],
        self.captionids[idxs],
        self.caption_masks[idxs])

  def yield_val_batch(self, batch_size):
    for i in range(0, self.num_caption, batch_size):
      start = i
      end = i + batch_size
      idxs = self.shuffled_idxs[start:end]

      yield (self.fts[self.ft_idxs[idxs]],
        self.captionids[idxs],
        self.caption_masks[idxs])

  # when we generate tst batch, we never shuffle as we are not doing training
  def yield_tst_batch(self, batch_size):
    for i in range(0, self.num_ft, batch_size):
      start = i
      end = i + batch_size

      yield (self.fts[start:end],)
