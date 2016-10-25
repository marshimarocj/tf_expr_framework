import os
import sys
import time
import json
import cPickle
import collections
import time
sys.path.append('../')

import numpy as np
from bleu import bleu
from cider import cider

import util
import toolkit


class TrnTst(trn_tst.TrnTst):
  """
  gen_sent_mode:
    1: top_1 sentence, {vid: caption}
    2: top_k loss+sentence {vid: [(loss, sent), ...]}
    3: top_k loss+sentence+word_loss {vid: [(loss, sent, (w_loss, ...)), ...]}
  """
  def __init__(self, model_cfg, path_cfg, model, gen_sent_mode=1):
    TrnTst.__init__(self, model_cfg, path_cfg, model)

    # caption int to string
    self.int2str = util.utility.CaptionInt2str(path_cfg.word_file)

    self.gen_sent_mode = gen_sent_mode

  def _predict_and_eval(self, sess, tst_reader, metrics):
    # greedy generator on validation
    videoid2caption = {}
    base = 0
    for fts in tst_reader.yield_tst_batch(batch_size):
      sent_pool = self.model.simple_decoder.greedy_generator(
        sess, {self.model._fts:fts})
      for k, sent in enumerate(sent_pool):
        videoid = tst_reader.videoids[base + k]
        videoid2caption[videoid] = self.int2str(np.expand_dims(sent, 0))
      base += batch_size

    bleu_scorer = bleu.Bleu(4)
    bleu_score, _ = bleu_scorer.compute_score(tst_reader.videoid2captions, videoid2caption)
    for i in range(4):
      metrics['bleu%d'%(i+1)] = bleu_score[i]

    cider_scorer = cider.Cider()
    cider_score, _ = cider_scorer.compute_score(tst_reader.videoid2captions, videoid2caption)
    metrics['cider'] = cider_score

    return metrics

  def _predict(self, sess, tst_reader, predict_file):
    videoid2caption = {}
    base = 0
    for fts in tst_reader.yield_tst_batch(batch_size):
      sent_pool = self.model.simple_decoder.beamsearch_generator(
        sess, {self.model._fts: fts})

      for b in xrange(len(sent_pool)):
        videoid = str(tst_reader.videoids[b+base])

        if self.gen_sent_mode == 1:
          captionid = np.expand_dims(sent_pool[b][0][1], 0)
          videoid2caption[videoid] = self.int2str(captionid)[0]
        elif self.gen_sent_mode == 2:
          videoid2caption[videoid] = []
          for k in xrange(self.model_cfg.sent_pool_size):
            captionid = np.expand_dims(sent_pool[b][k][1], 0)
            out = (
              float(sent_pool[b][k][0]),
              self.int2str(captionid)[0]
            )
            videoid2caption[videoid].append(out)
        elif self.gen_sent_mode == 3:
          videoid2caption[videoid] = []
          for k in xrange(self.model_cfg.sent_pool_size):
            captionid = np.expand_dims(sent_pool[b][k][1], 0)
            out = (
              float(sent_pool[b][k][0]),
              self.int2str(captionid)[0],
              [float(x) for x in sent_pool[b][k][2]]
            )
            videoid2caption[videoid].append(out)

      base += len(sent_pool)

    json.dump(videoid2caption, open(predict_file, 'w'))
