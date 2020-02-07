import tensorflow as tf
from libs.preprocess import *


def x_to_dict(x, a):
    return {'x': x, 'a': a}


def get_dataset(smi,
                shuffle_buffer_size,
                batch_size):

    smi = tf.data.Dataset.from_tensor_slices(smi)
    smi = smi.shuffle(shuffle_buffer_size)
    ds = smi.map(
        lambda x: tf.py_function(func=convert_smiles_to_graph,
                                 inp=[x],
                                 Tout=[tf.float32, tf.float32]),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.padded_batch(batch_size, padded_shapes=([None, 58], [None, None]))
    ds = ds.map(x_to_dict)

    y = smi.map(
        lambda x: tf.py_function(func=calc_properties,
                                 inp=[x], Tout=[tf.float32, tf.float32, tf.float32, tf.float32]),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    y = y.padded_batch(batch_size, padded_shapes=([], [], [], []))
    ds = tf.data.Dataset.zip((ds, y))
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    return ds


def get_benchmark_dataset(smi,
                          property_func,
                          shuffle_buffer_size,
                          batch_size):

    smi = tf.data.Dataset.from_tensor_slices(smi)
    smi = smi.shuffle(shuffle_buffer_size)
    ds = smi.map(
        lambda x: tf.py_function(func=convert_smiles_to_graph,
                                 inp=[x],
                                 Tout=[tf.float32, tf.float32]),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.padded_batch(batch_size, padded_shapes=([None, 58], [None, None]))
    ds = ds.map(x_to_dict)

    y = smi.map(
        lambda x: tf.py_function(func=property_func,
                                 inp=[x], Tout=tf.float32),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    y = y.padded_batch(batch_size, padded_shapes=([]))
    ds = tf.data.Dataset.zip((ds, y))
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    return ds
