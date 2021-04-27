import functools
import tensorflow.compat.v1 as tf
# import tensorflow as tf
from note_seq import sequence_proto_to_midi_file
from note_seq.protobuf import music_pb2

from magenta.models.onsets_frames_transcription import infer_util, constants
from magenta.models.onsets_frames_transcription.data import read_examples, preprocess_example, \
    input_tensors_to_model_input, splice_examples, create_batch
from magenta.models.onsets_frames_transcription.onsets_frames_transcription_transcribe import create_example

from export import hparams


def prepare_input(file_path):
    example = create_example(file_path, hparams.sample_rate, False)
    input_dataset = read_examples([example], False, False, 0, hparams)
    input_map_fn = functools.partial(preprocess_example, hparams=hparams, is_training=False)
    input_tensors = input_dataset.map(input_map_fn)
    model_input = input_tensors.map(
        functools.partial(input_tensors_to_model_input, hparams=hparams, is_training=False))
    model_input = splice_examples(model_input, hparams, False)
    dataset = create_batch(model_input, hparams=hparams, is_training=False)
    t, _ = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE).make_one_shot_iterator().get_next()
    return t.spec


def _predict(frame_probs, onset_probs, frame_predictions, onset_predictions, offset_predictions, velocity_values):
    sequence = infer_util.predict_sequence(
        frame_probs=frame_probs,
        onset_probs=onset_probs,
        frame_predictions=frame_predictions,
        onset_predictions=onset_predictions,
        offset_predictions=offset_predictions,
        velocity_values=velocity_values,
        hparams=hparams,
        min_pitch=constants.MIN_MIDI_PITCH)
    return sequence.SerializeToString()


# def prepare_output(model_result):
#     frame_probs = model_result["frame_probs"][0].numpy()
#     frame_predictions = model_result["frame_predictions"][0].numpy()
#     onset_probs = model_result["onset_probs"][0].numpy()
#     onset_predictions = model_result["onset_predictions"][0].numpy()
#     offset_predictions = model_result["offset_predictions"][0].numpy()
#     velocity_values = model_result["velocity_values"][0].numpy()
#
#     sequence = _predict(frame_probs, onset_probs, frame_predictions, onset_predictions, offset_predictions,
#                         velocity_values)
#     sequence = music_pb2.NoteSequence.FromString(sequence)
#     midi_filename = 'output.midi'
#     sequence_proto_to_midi_file(sequence, midi_filename)

def _predict_sequence(model_result):
    """Convert frame predictions into a sequence (TF)."""

    def _predict(frame_probs, onset_probs, frame_predictions, onset_predictions,
                 offset_predictions, velocity_values):
        print('lengths', len(frame_probs), len(onset_probs), len(frame_predictions), len(onset_predictions), len(offset_predictions), len(velocity_values))
        print('len frame_predictions', len(frame_predictions))
        """Convert frame predictions into a sequence (Python)."""
        sequence = infer_util.predict_sequence(
            frame_probs=frame_probs,
            onset_probs=onset_probs,
            frame_predictions=frame_predictions,
            onset_predictions=onset_predictions,
            offset_predictions=offset_predictions,
            velocity_values=velocity_values,
            hparams=hparams,
            min_pitch=constants.MIN_MIDI_PITCH)
        print('sequence', sequence)
        return sequence.SerializeToString()

    sequence = tf.py_func(
        _predict,
        inp=[
            model_result["frame_probs"][0],
            model_result["onset_probs"][0],
            model_result["frame_predictions"][0],
            model_result["onset_predictions"][0],
            model_result["offset_predictions"][0],
            model_result["velocity_values"][0]
        ],
        Tout=tf.string,
        stateful=False)
    print('sequence', sequence)
    sequence.set_shape([])
    print('after set shape', sequence)
    print('to return', tf.expand_dims(sequence, axis=0))
    return tf.expand_dims(sequence, axis=0)


def prepare_output(model_result):
    sequence = _predict_sequence(model_result)
    sequence = music_pb2.NoteSequence.FromString(sequence)
    midi_filename = 'output.midi'
    sequence_proto_to_midi_file(sequence, midi_filename)
    print('yo')
