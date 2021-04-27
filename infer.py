from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

from magenta.models.onsets_frames_transcription import train_util
from magenta.models.onsets_frames_transcription.configs import CONFIG_MAP

CHECKPOINT_PATH = "models/piano_transcriber_checkpoint"
CONFIG_KEY = "onsets_frames"  # "drums" for drums

config = CONFIG_MAP[CONFIG_KEY]
hparams = config.hparams
hparams.parse("")
hparams.batch_size = 1
hparams.truncated_length_secs = 0

estimator = train_util.create_estimator(config.model_fn, os.path.expanduser(CHECKPOINT_PATH), hparams)

import os
import six
from magenta.models.onsets_frames_transcription import audio_label_data_utils
from magenta.models.onsets_frames_transcription.data import provide_batch
from magenta.models.onsets_frames_transcription import infer_util
from magenta.music.protobuf import music_pb2
import tensorflow.compat.v1 as tf


def create_example(filename, sample_rate, load_audio_with_librosa):
    """Processes an audio file into an Example proto."""
    wav_data = tf.gfile.Open(filename, 'rb').read()
    example_list = list(
        audio_label_data_utils.process_record(
            wav_data=wav_data,
            sample_rate=sample_rate,
            ns=music_pb2.NoteSequence(),
            # decode to handle filenames with extended characters.
            example_id=six.ensure_text(filename, 'utf-8'),
            min_length=0,
            max_length=-1,
            allow_empty_notesequence=True,
            load_audio_with_librosa=load_audio_with_librosa))
    assert len(example_list) == 1
    return example_list[0].SerializeToString()


with tf.Graph().as_default():
    examples = tf.placeholder(tf.string, [None])

    dataset = provide_batch(
        examples=examples,
        preprocess_examples=True,
        params=hparams,
        is_training=False,
        shuffle_examples=False,
        skip_n_initial_records=0)

    iterator = tf.data.make_initializable_iterator(dataset)
    next_record = iterator.get_next()

    output_filenames = []
    with tf.Session() as sess:
        sess.run([
            tf.initializers.global_variables(),
            tf.initializers.local_variables()
        ])

        for filename in ["alkan-short.wav"]:
            tf.logging.info('Starting transcription for %s...', filename)
            # The reason we bounce between two Dataset objects is so we can use
            # the data processing functionality in data.py without having to
            # construct all the Example protos in memory ahead of time or create
            # a temporary tfrecord file.
            tf.logging.info('Processing file...')
            # restorer = tf.train.Saver(None, write_version=tf.train.SaverDef.V2)
            # restorer.restore(sess, model_path)
            example = create_example(filename, hparams.sample_rate, False)
            # print('this is the input what is it', type(example))
            sess.run(iterator.initializer, {examples: [example]})


            def transcription_data(params):
                del params
                return tf.data.Dataset.from_tensors(sess.run(next_record))
            #
            # def transcription_data(params):
            #     # del params
            #     return tf.data.Dataset.from_tensors(sess.run(next_record))


            input_fn = infer_util.labels_to_features_wrapper(transcription_data)
            tf.logging.info('Running inference...')

            prediction_list = list(
                estimator.predict(
                    input_fn,
                    checkpoint_path=None,
                    yield_single_examples=False))
            assert len(prediction_list) == 1
            sequence_prediction = music_pb2.NoteSequence.FromString(
                prediction_list[0]['sequence_predictions'][0])
            # midi_filename = filename + data.transcribed_file_suffix + '.midi'
            # magenta_utils.mm.sequence_proto_to_midi_file(sequence_prediction, midi_filename)
            #
            # output_filenames.append(midi_filename)
            #
            # tf.logging.info('Transcription written to %s.', midi_filename)
