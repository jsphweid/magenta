import tensorflow.compat.v1 as tf
from tensor2tensor.utils import decoding
from tensor2tensor.utils import trainer_lib
from magenta.models.score2perf import score2perf
import numpy as np

tf.disable_v2_behavior()

model_name = 'transformer'
hparams_set = 'transformer_tpu'
ckpt_path = './models/piano_transformer_checkpoint/unconditional_model_16.ckpt'


class PianoPerformanceLanguageModelProblem(score2perf.Score2PerfProblem):
    @property
    def add_eos_symbol(self):
        return True


problem = PianoPerformanceLanguageModelProblem()

# Set up HParams.
hparams = trainer_lib.create_hparams(hparams_set=hparams_set)
trainer_lib.add_problem_hparams(hparams, problem)
hparams.num_hidden_layers = 16
hparams.sampling_method = 'random'

# Set up decoding HParams.
decode_hparams = decoding.decode_hparams()
decode_hparams.alpha = 0.0
decode_hparams.beam_size = 1


def _serving_fn():
    features = {
        'targets': tf.placeholder(dtype=tf.int32, shape=[1, None], name="targets_lol"),
        'decode_length': tf.placeholder(dtype=tf.int32, shape=[], name="decode_length_lol"),
    }
    return tf.estimator.export.ServingInputReceiver(features, features)


if __name__ == "__main__":
    run_config = trainer_lib.create_run_config(hparams)
    estimator = trainer_lib.create_estimator(model_name, hparams, run_config, decode_hparams=decode_hparams)
    estimator.export_saved_model("export", _serving_fn, checkpoint_path=ckpt_path)


#
# # Create input generator (so we can adjust priming and
# # decode length on the fly).
def input_generator():
    global targets
    global decode_length
    while True:
        yield {
            'targets': np.array([targets], dtype=np.int32),
            'decode_length': np.array(decode_length, dtype=np.int32)
        }
#
#
# # These values will be changed by subsequent cells.
# targets = []
# decode_length = 0
#
# # Start the Estimator, loading from the specified checkpoint.
# input_fn = decoding.make_input_fn_from_generator(input_generator())
# print('apseu')
# unconditional_samples = estimator.predict(input_fn, checkpoint_path=ckpt_path)
#
# # "Burn" one.
# _ = next(unconditional_samples)
#
# targets = []
# decode_length = 1024
#
# # Generate sample events.
# sample_ids = next(unconditional_samples)['outputs']
#
# # Decode to NoteSequence.
# midi_filename = decode(sample_ids, encoder=unconditional_encoders['targets'])
# print('midi_filename', midi_filename)
# unconditional_ns = note_seq.midi_file_to_note_sequence(midi_filename)
#
# note_seq.note_sequence_to_midi_file(unconditional_ns, 'lmao.midi')

# Play and plot.
# note_seq.play_sequence(
#     unconditional_ns,
#     synth=note_seq.fluidsynth, sample_rate=SAMPLE_RATE, sf2_path=SF2_PATH)
# note_seq.plot_sequence(unconditional_ns)
#
# note_seq.sequence_proto_to_midi_file(
#     unconditional_ns, '/tmp/unconditional.mid')
# files.download('/tmp/unconditional.mid')
