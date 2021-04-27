import os

import tensorflow.compat.v1 as tf

from magenta.models.onsets_frames_transcription.data import provide_batch

# tf.disable_v2_behavior()

from magenta.models.onsets_frames_transcription import train_util
from magenta.models.onsets_frames_transcription.configs import CONFIG_MAP

CHECKPOINT_PATH = "models/piano_transcriber_checkpoint"
CONFIG_KEY = "onsets_frames"  # "drums" for drums

config = CONFIG_MAP[CONFIG_KEY]
hparams = config.hparams
hparams.parse("")
hparams.batch_size = 1
hparams.truncated_length_secs = 0
hparams.use_tpu = False


# def _serving_input_receiver_fn():
#     serialized_tf_example = tf.placeholder(dtype=tf.string, shape=[None])
#     receiver_tensors = {'examples': serialized_tf_example}
#
#     dataset = provide_batch(
#         examples=serialized_tf_example,
#         preprocess_examples=True,
#         params=hparams,
#         is_training=False,
#         shuffle_examples=False,
#         skip_n_initial_records=0)
#
#     iterator = tf.data.make_initializable_iterator(dataset)
#     feature_tensors, _ = iterator.get_next()
#
#     return tf.estimator.export.ServingInputReceiver(feature_tensors.spec, receiver_tensors)

def _serving_input_receiver_fn():
    tf_example = tf.placeholder(dtype=tf.float32, shape=[1, None, 229, 1])
    # receiver_tensors = {'feat': tf_example}
    return tf.estimator.export.ServingInputReceiver(tf_example, tf_example)


# def transcription_data(params):
#     del params
#     return _serving_input_receiver_fn()


# new_serving_fn = labels_to_features_wrapper(transcription_data)

#
# def actual_fn():
#     return new_serving_fn(hparams)


if __name__ == "__main__":
    estimator = train_util.create_estimator(config.model_fn, os.path.expanduser(CHECKPOINT_PATH), hparams)
    estimator.export_saved_model("export", _serving_input_receiver_fn)
