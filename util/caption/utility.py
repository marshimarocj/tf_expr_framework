import cPickle
import numpy as np

import tensorflow as tf

BOS = 0
EOS = 1
UNK = 2


# load int2wordFile and 
# transfrom each captionid to a string sentence without punctuation
class CaptionInt2str(object):
  def __init__(self, int2word_file):
    self.int2word = []

    self.int2word = cPickle.load(file(int2word_file))
    # print len(self.int2word)

  # captionInt should be a batch of captionInts
  def __call__(self, captionid):
    batch_size = captionid.shape[0]

    captionStr = []
    for i in xrange(batch_size):
      sent = []
      for t in captionid[i]:
        if t == BOS:
          continue
        elif t == EOS:
          break
        else:
          # if t >= len(self.int2word):
          #   print t
          sent.append(self.int2word[t])
      captionStr.append(' '.join(sent))

    return captionStr


# pool_size <= beam_width
def beamsearch_recover_captions(wordids, cum_log_probs, pres, ends, pool_size):
  batch_size = wordids[0].shape[0]
  beam_width = wordids[0].shape[1]
  num_step = len(wordids)
  sent_pool = [[] for _ in range(batch_size)]

  for n in range(num_step):
    _ends = ends[n]
    for end in _ends:
      b = end[0]
      k = end[1]
      pre = pres[n][b, k]
      caption = []
      log_prob = cum_log_probs[n][b, k] / (n+1)
      for t in xrange(n-1, -1, -1):
        caption.append(wordids[t][b, pre])
        pre = pres[t][b, pre]
      caption = np.array(caption, np.int32)[::-1]
      sent_pool[b].append((log_prob, caption))

  out = []
  for b, sents in enumerate(sent_pool):
    sents = sorted(sents, key=lambda x:x[0], reverse=True)
    out.append(sents)
  return out
