import os
import random
from collections import deque

import tensorflow as tf


def bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def meta_record(num_record):
  meta = tf.train.Example(features=tf.train.Features(feature={
    'num_record': int64_feature([num_record]),
  }))
  return meta


class RandomShuffleQueue(object):
  def __init__(self, capacity):
    self.capacity = capacity
    self.q = deque([])

  def is_full(self):
    return len(self.q) == self.capacity

  def is_empty(self):
    return len(self.q) == 0

  def enqueue(self, ele):
    self.q.append(ele)

  def dequeue(self):
    r = random.randint(0, len(self.q)-1)
    tmp = self.q[r]
    self.q[r] = self.q[0]
    self.q.popleft()
    return tmp


class ShuffleBatchJoin(object):
  def __init__(self, files, capacity, shuffle_files, **kwargs):
    self.capacity = capacity
    self.files = files
    if shuffle_files:
      random.shuffle(self.files)
    self.random_shuffle_queue = RandomShuffleQueue(capacity)

  def generate_data_from_record(self, example):
    raise NotImplementedError("""please customize generate_data_from_record""")

  def num_record(self):
    total = 0
    for file in self.files:
      options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
      record_iterator = tf.python_io.tf_record_iterator(path=file, options=options)
      string_record = record_iterator.next()
      example = tf.train.Example()
      example.ParseFromString(string_record)
      total += int(example.features.feature['num_record'].int64_list.value[0])
    return total

  def iterator(self, batch_size):
    assert batch_size <= self.capacity

    for file in self.files:
      options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
      record_iterator = tf.python_io.tf_record_iterator(path=file, options=options)
      record_iterator.next() # skip meta data
      for string_record in record_iterator:
        example = tf.train.Example()
        example.ParseFromString(string_record)
        data = self.generate_data_from_record(example)
        if self.random_shuffle_queue.is_full():
          batch_data = []
          for i in range(batch_size):
            batch_data.append(self.random_shuffle_queue.dequeue())
          yield batch_data
        self.random_shuffle_queue.enqueue(data)
    batch_data = []
    while not self.random_shuffle_queue.is_empty():
      batch_data.append(self.random_shuffle_queue.dequeue())
      if len(batch_data) == batch_size:
        yield batch_data
        batch_data = []
    if len(batch_data) > 0:
      yield batch_data


# Note: never ending circular queue
# don't call by for loop
# call next() instead
class CircularShuffleBatchJoin(ShuffleBatchJoin):
  def iterator(self, batch_size):
    assert batch_size <= self.capacity

    while True:
      for file in self.files:
        options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
        record_iterator = tf.python_io.tf_record_iterator(path=file, options=options)
        record_iterator.next() # skip meta data
        for string_record in record_iterator:
          example = tf.train.Example()
          example.ParseFromString(string_record)
          data = self.generate_data_from_record(example)
          if self.random_shuffle_queue.is_full():
            batch_data = []
            for i in range(batch_size):
              batch_data.append(self.random_shuffle_queue.dequeue())
            yield batch_data
          self.random_shuffle_queue.enqueue(data)
      batch_data = []
      while not self.random_shuffle_queue.is_empty():
        batch_data.append(self.random_shuffle_queue.dequeue())
        if len(batch_data) == batch_size:
          yield batch_data
          batch_data = []
      if len(batch_data) > 0:
        yield batch_data
