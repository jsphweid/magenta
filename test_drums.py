import functools

import tensorflow as tf
import tensorflow.compat.v1 as tf_compat
import grpc
from note_seq import sequence_proto_to_midi_file
from note_seq.protobuf import music_pb2

from export import hparams
from magenta.models.onsets_frames_transcription import infer_util, constants

tf.compat.v1.enable_eager_execution()

from preprocess import prepare_input, prepare_output
from tensorflow_serving.apis import prediction_service_pb2_grpc, predict_pb2

channel = grpc.insecure_channel("192.168.1.18:8500")
# channel = grpc.insecure_channel("127.0.0.1:8500")
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

f = tf.saved_model.load("export/1619576949", tags=["serve"]).signatures["serving_default"]


def get_grpc_prediction_drums(t):
    request = predict_pb2.PredictRequest()
    request.model_spec.name = "onsets-and-frames-drum-transcriber"
    request.model_spec.signature_name = 'serving_default'
    request.inputs['input'].CopyFrom(tf.make_tensor_proto(t.numpy()))
    result = stub.Predict(request)
    keys = ["frame_probs", "onset_probs", "frame_predictions", "onset_predictions", "offset_predictions",
            "velocity_values"]
    return {k: tf.make_ndarray(result.outputs[k]) for k in keys}


def _predict_sequences_drums(model_result):
    def predict_sequence(frame_probs, onset_probs, frame_predictions,
                         onset_predictions, offset_predictions, velocity_values,
                         hparams):
        sequence_prediction = infer_util.predict_sequence(
            frame_probs=frame_probs,
            onset_probs=onset_probs,
            frame_predictions=onset_predictions,
            onset_predictions=onset_predictions,
            offset_predictions=onset_predictions,
            velocity_values=velocity_values,
            min_pitch=constants.MIN_MIDI_PITCH,
            hparams=hparams,
            onsets_only=True)
        for note in sequence_prediction.notes:
            note.is_drum = True
        return sequence_prediction.SerializeToString()

    sequences = []
    for i in range(model_result["frame_predictions"].shape[0]):
        sequence = tf_compat.py_func(
            functools.partial(predict_sequence, hparams=hparams),
            inp=[
                model_result["frame_probs"][i],
                model_result["onset_probs"][i],
                model_result["frame_predictions"][i],
                model_result["onset_predictions"][i],
                model_result["offset_predictions"][i],
                model_result["velocity_values"][i],
            ],
            Tout=tf.string,
            stateful=False)
        sequence.set_shape([])
        sequences.append(sequence)
    return tf.stack(sequences)


def prepare_output_drums(model_result):
    sequence = _predict_sequences_drums(model_result)[0].numpy()
    sequence = music_pb2.NoteSequence.FromString(sequence)
    midi_filename = 'output_drums.midi'
    sequence_proto_to_midi_file(sequence, midi_filename)


input_data = prepare_input("nirvana.wav")
result = get_grpc_prediction_drums(input_data)
# result = f(input_data)
final_result = prepare_output_drums(result)
# print("okay", result)
