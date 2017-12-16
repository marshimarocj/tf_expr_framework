import tensorflow as tf
from tensorflow.python.pywrap_tensorflow import NewCheckpointReader


'''func
'''
def load_variable_in_ckpt(ckpt_file):

  reader = NewCheckpointReader(ckpt_file)
  # get_variable_to_shape_map() returns a dict of the restored Variables with their shape
  var_to_shape_map = reader.get_variable_to_shape_map()
  out = {}
  for key in var_to_shape_map:
    # get_tensor() returns the narray value of the restored Variables
    val = reader.get_tensor(key)
    out[key] = val

  return out


def init_weight_from_singlemodel(ckpt_file, map_var_from_single_to_ensemble):
  single_key2val = load_variable_in_ckpt(ckpt_file)
  ensemble_key2val = {}
  for key in map_var_from_single_to_ensemble.iterkeys():
    ensemble_key = map_var_from_single_to_ensemble[key]
    ensemble_key2val[key] = single_key2val[key]
  assign_op, feed_dict = tf.contrib.framework.assign_from_values(ensemble_key2val)
  return assign_op, feed_dict


def load_from_single_ckpts_and_save_ensemble_ckpt(graph, init_op, init_feed, saver, out_file):
  with tf.Session(graph=graph) as sess:
    sess.run(init_op, init_feed)
    saver.save(sess, out_file, global_step=0)


