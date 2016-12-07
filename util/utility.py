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


def sentence_encode(sess, states, sentenceids, _states, _wordids, update_state_op, prob_op):
  num_step = sentenceids.shape[1]
  for i in xrange(num_step):
    states, word_probs = sess.run([update_state_op, prob_op],
      feed_dict={
        _states: states,
        _wordids: sentenceids[:, i]
      })
  
  return states


def sample_word_decode(sess, states, wordids,
    _states, _wordids, update_state_op, prob_op, max_step):
  batch_size = states.shape[0]

  caption = np.zeros((batch_size, max_step), dtype=np.int32)
  for i in xrange(max_step):
    states, word_probs = sess.run([update_state_op, prob_op],
      feed_dict={
        _states: states,
        _wordids: wordids
      })

    num_word = word_probs.shape[1]
    cum_probs = np.cumsum(word_probs, axis=1)
    rand = np.random.rand(batch_size)
    wordids = np.empty(batch_size)
    for j in range(batch_size):
      wordid = np.searchsorted(cum_probs[j], rand[j])
      if wordid == num_word: wordid -= 1
      wordids[j] = wordid

    caption[:, i] = wordids

  return caption


def sample_top_word_decode(sess, states, wordids,
    _states, _wordids, update_state_op, prob_op, max_step, topk):
  batch_size = states.shape[0]

  caption = np.zeros((batch_size, max_step), dtype=np.int32)
  for i in xrange(max_step):
    states, word_probs = sess.run([update_state_op, prob_op],
      feed_dict={
        _states: states,
        _wordids: wordids
      })

    num_word = word_probs.shape[1]
    wids = np.argsort(-word_probs, axis=1)
    wordids = np.empty(batch_size)
    for j in range(batch_size):
      topk_probs = word_probs[j][wids[j][:topk]]
      cum_probs = np.cumsum(topk_probs)
      rand = np.random.random(1)*np.max(cum_probs)
      idx = np.searchsorted(cum_probs, rand)
      if idx == topk: idx -= 1
      wordids[j] = wids[j, idx]

    caption[:, i] = wordids

  return caption


# note: the addition_inputs remain same across different time steps
def greedy_word_decode(
    sess, states, _states, _wordids, update_state_op, prob_op, output_op,
    max_step, 
    _last_output=None, init_output=None, # general attention mechanism 
    addition_input_placeholders=[], addition_inputs=[]): # input placeholders and input values 
  batch_size = states.shape[0]

  # initialize start words <BOS>
  wordids = np.zeros((batch_size, ), dtype=np.int32)

  caption = np.zeros((batch_size, max_step), dtype=np.int32)
  outputs = init_output
  for i in xrange(max_step): # assume longest sentence <= max_step
    feed_dict = {
      _states: states,
      _wordids: wordids,
    }
    if _last_output is not None:
      feed_dict[_last_output] = outputs
    num = len(addition_input_placeholders)
    for j in range(num):
      feed_dict[addition_input_placeholders[j]] = addition_inputs[j]

    states, word_probs, outputs = sess.run(
      [update_state_op, prob_op, output_op],
      feed_dict=feed_dict)
    
    wordids = np.argmax(word_probs, 1)
    caption[:, i] = wordids

  return caption


# return sent_pool in shape (batch_size, sent_pool_size, 3)
# last dimension: (loss, captionId, word_loss)
def beamsearch_word_decode(
    sess, states, _states, _wordids, update_state_op, prob_op, output_op,
    max_step, width, sent_pool_size, 
    _last_outputs=None, init_output=None, # general attention mechanism
    addition_input_placeholders=[], addition_inputs=[]): # input placeholders and input values 
  batch_size = states.shape[0]

  # initialize start words <BOS>
  word_topk = np.zeros((batch_size, ), dtype=np.int32)

  sent_pool = [[] for _ in xrange(batch_size)]
  wordids = []
  pres = []
  word_losses = []
  # sum log probability for each beams
  log_probs = np.zeros((batch_size * width, ), dtype=np.float32)
  # exist beams for each input feature
  batch_sent_pool_remain_cnt = np.zeros((batch_size,), dtype=np.float32) + sent_pool_size

  # expand addition_input for beam search from step 1, excluding step 0
  expand_addition_inputs = []
  num = len(addition_input_placeholders)
  for j in range(num):
    addition_input = addition_inputs[j]
    shape = list(addition_input.shape)
    shape[0] *= width
    expand_addition_input = np.zeros(tuple(shape), dtype=np.float32)
    for ib in xrange(batch_size):
      expand_addition_input[ib*width: (ib+1)*width] = addition_input[ib]
    expand_addition_inputs.append(expand_addition_input)
 
  outputs = init_output
  for i in xrange(0, max_step): # assume longest sentence <= max_step
    # state: (batch_size, state_units) if i == 0 else (batch_size*width, state_units)
    # word_topk: (batch_size,) if i == 0 else (batch_size*width,)
    feed_dict = {
      _states: states,
      _wordids: word_topk,
    }

    if _last_outputs is not None:
      feed_dict[_last_outputs] = outputs
    for j in range(num):
      if i == 0:
        feed_dict[addition_input_placeholders[j]] = addition_inputs[j]
      else:
        # addition_input = addition_inputs[j]
        # shape = list(addition_input.shape)
        # shape[0] *= width
        # expand_addition_input = np.zeros(tuple(shape), dtype=np.float32)
        # for ib in xrange(batch_size):
        #   expand_addition_input[ib*width: (ib+1)*width] = addition_input[ib]
        # feed_dict[addition_input_placeholders[j]] = expand_addition_input
        feed_dict[addition_input_placeholders[j]] = expand_addition_inputs[j]
        
    states, prob, outputs = sess.run([update_state_op, prob_op, output_op],
      feed_dict=feed_dict)

    if i == 0:
      # select top width for each video feature
      # (batch_size, width)
      word_topk = np.argsort(-prob)[:, :width]
      prob_topk = np.zeros_like(word_topk, dtype=np.float32)
      for l in xrange(prob_topk.shape[0]):
        prob_topk[l] = prob[l, word_topk[l]]

      # expand state
      expand_state = np.zeros((batch_size * width, states.shape[1]), dtype=np.float32)
      for ib in xrange(batch_size):
        expand_state[ib*width: (ib+1)*width, :] = states[ib]
      states = expand_state

      # expand outputs
      expand_output = np.zeros((batch_size * width, outputs.shape[1]), dtype=np.float32)
      for ib in xrange(batch_size):
        expand_output[ib*width: (ib+1*width), :] = outputs[ib]
      outputs = expand_output

      word_topk = word_topk.flatten()
      log_probs = np.log(prob_topk.flatten())

      wordids.append(word_topk)
      pres.append([])
      word_losses.append(np.log(prob_topk).flatten())
    else:
      # select top width**2 for each video feature
      # (batch_size*width, width)
      word_topk2 = np.argsort(-prob)[:, :width]
      prob_topk2 = np.zeros_like(word_topk2, dtype=np.float32)
      for l in xrange(prob_topk2.shape[0]):
        prob_topk2[l] = prob[l, word_topk2[l]]

      # shape of word_topk2, prob_topk2: (batch_size*width, width)
      log_probs = np.log(prob_topk2) + np.repeat(np.expand_dims(log_probs, 1), width, 1)
      log_probs = log_probs.reshape((batch_size, width * width))
      topk_indices = np.argsort(-log_probs)[:, :width] # shape=(batch_size, width)

      # get pre indexes (batch_size*width,)
      # one dimension array makes selection operation easy
      for b in xrange(batch_size):
        topk_indices[b, :] += b * width * width
      topk_indices = topk_indices.flatten() # shape=(batch_size*width,)
      topk_pre_indices = topk_indices // width # shape=(batch_size*width,)
      states = states[topk_pre_indices]

      # reshape to (batch_size*width*width, ) for topK selection
      word_topk2 = word_topk2.flatten()
      prob_topk2 = prob_topk2.flatten()
      log_probs = log_probs.flatten()

      # select topK from topK**2 candidates to shape=(batch_size*width,)
      word_topk = word_topk2[topk_indices]
      log_probs = log_probs[topk_indices]
      word_loss = np.log(prob_topk2)[topk_indices]

      # save
      pres.append(topk_pre_indices)
      wordids.append(word_topk)
      word_losses.append(word_loss)

      # in case never ending, manually set the last word to EOS
      if i == max_step - 1:
        word_topk = np.ones((batch_size * width, ), dtype=np.int32)

      # set complete sentences probs -inf
      for iw, w in enumerate(word_topk):
        if w == 1:
          complete_sentence, complete_loss = beamsearch_recover_one_caption(
            wordids, pres, iw, word_loss=word_losses)

          # find batch
          complete_batch = iw // width
          if np.sum(batch_sent_pool_remain_cnt[complete_batch]) > 0:
            sent_pool[complete_batch].append(
              (log_probs[iw], complete_sentence, complete_loss))
            batch_sent_pool_remain_cnt[complete_batch] -= 1  # decrease beam width budget

          log_probs[iw] = -np.inf
          if batch_sent_pool_remain_cnt[complete_batch] == 0:
            log_probs[complete_batch*width: complete_batch*width+width] = -np.inf

      # early break if sent_pool budget in the batch is exhausted
      if np.sum(batch_sent_pool_remain_cnt) == 0:
        break

  # sort for each batch
  for i, sents in enumerate(sent_pool):
    for k in xrange(len(sents)):
      sents[k] = (sents[k][0] / len(sents[k][1]), sents[k][1], sents[k][2])
    sents.sort(key=lambda x: -x[0])

  return sent_pool


def beamsearch_recover_one_caption(wordids, pre, ith, word_loss=None):
  """
  wordids: list, the index corresponds to one time_step
  pre: list, the index corresponds to one timestep
  ith: the last idx of wordids
  word_loss: if given, will output the array of each word's loss
  """

  time_step = len(wordids)
  caption = []
  caption_loss = []

  for t in xrange(time_step-1, 0, -1):
    caption.append(wordids[t][ith])
    if word_loss is not None:
      caption_loss.append(word_loss[t][ith])
    ith = pre[t][ith]

  caption.append(wordids[0][ith])
  if word_loss is not None:
    caption_loss.append(word_loss[0][ith])

  caption = np.array(caption, np.int32)[::-1]
  if word_loss is not None:
    caption_loss = np.array(caption_loss, np.float32)[::-1]
  else:
    caption_loss = np.empty(0)

  return caption, caption_loss

