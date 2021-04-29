import os

import tensorflow.compat.v1 as tf

from magenta.models.onsets_frames_transcription.data import provide_batch

# tf.disable_v2_behavior()

from magenta.models.onsets_frames_transcription import train_util
from magenta.models.onsets_frames_transcription.configs import CONFIG_MAP

CHECKPOINT_PATH = "models/piano_transcriber_checkpoint"
CONFIG_KEY = "onsets_frames"
# CHECKPOINT_PATH = "models/drum_transcriber_checkpoint"
# CONFIG_KEY = "drums"  # "drums" for drums

config = CONFIG_MAP[CONFIG_KEY]
hparams = config.hparams
hparams.parse("")
hparams.batch_size = 1
hparams.truncated_length_secs = 0
hparams.use_tpu = False


def _serving_input_receiver_fn():
    # 229 for piano, 250 for drums I think
    tf_example = tf.placeholder(dtype=tf.float32, shape=[1, None, hparams.spec_n_bins, 1])
    return tf.estimator.export.ServingInputReceiver(tf_example, tf_example)


if __name__ == "__main__":
    estimator = train_util.create_estimator(config.model_fn, os.path.expanduser(CHECKPOINT_PATH), hparams)
    estimator.export_saved_model("export", _serving_input_receiver_fn)
