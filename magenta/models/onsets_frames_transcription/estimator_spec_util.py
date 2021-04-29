# Copyright 2021 The Magenta Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Utilities for creating EstimatorSpecs for Onsets and Frames models."""

import functools

from magenta.models.onsets_frames_transcription import constants
from magenta.models.onsets_frames_transcription import drum_mappings
from magenta.models.onsets_frames_transcription import infer_util
from magenta.models.onsets_frames_transcription import metrics

import tensorflow.compat.v1 as tf
import tf_slim


def _drums_only_metric_ops(features, labels, frame_probs, onset_probs,
                           frame_predictions, onset_predictions,
                           offset_predictions, velocity_values, hparams):
  """Generate drum metrics: offsets/frames are ignored."""
  del frame_predictions, offset_predictions  # unused
  metric_ops = metrics.define_metrics(
      frame_probs=frame_probs,
      onset_probs=onset_probs,
      frame_predictions=onset_predictions,
      onset_predictions=onset_predictions,
      offset_predictions=onset_predictions,
      velocity_values=velocity_values,
      length=features.length,
      sequence_label=labels.note_sequence,
      frame_labels=labels.labels,
      sequence_id=features.sequence_id,
      hparams=hparams,
      min_pitch=constants.MIN_MIDI_PITCH,
      max_pitch=constants.MAX_MIDI_PITCH,
      prefix='drums/',
      onsets_only=True,
      pitch_map=drum_mappings.GROOVE_PITCH_NAMES)
  return metric_ops


def get_metrics(features, labels, frame_probs, onset_probs, frame_predictions,
                onset_predictions, offset_predictions, velocity_values,
                hparams):
  """Return metrics values ops."""
  if hparams.drums_only:
    return _drums_only_metric_ops(
        features=features,
        labels=labels,
        frame_probs=frame_probs,
        onset_probs=onset_probs,
        frame_predictions=frame_predictions,
        onset_predictions=onset_predictions,
        offset_predictions=offset_predictions,
        velocity_values=velocity_values,
        hparams=hparams)
  else:
    return metrics.define_metrics(
        frame_probs=frame_probs,
        onset_probs=onset_probs,
        frame_predictions=frame_predictions,
        onset_predictions=onset_predictions,
        offset_predictions=offset_predictions,
        velocity_values=velocity_values,
        length=features.length,
        sequence_label=labels.note_sequence,
        frame_labels=labels.labels,
        sequence_id=features.sequence_id,
        hparams=hparams)


def _predict_sequences(frame_probs, onset_probs, frame_predictions,
                       onset_predictions, offset_predictions, velocity_values,
                       hparams):
  """Predict a batch of sequences."""

  def predict_sequence(frame_probs, onset_probs, frame_predictions,
                       onset_predictions, offset_predictions, velocity_values,
                       hparams):
    """Predict a single sequence."""
    if hparams.drums_only:
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
    else:
      sequence_prediction = infer_util.predict_sequence(
          frame_probs=frame_probs,
          onset_probs=onset_probs,
          frame_predictions=frame_predictions,
          onset_predictions=onset_predictions,
          offset_predictions=offset_predictions,
          velocity_values=velocity_values,
          min_pitch=constants.MIN_MIDI_PITCH,
          hparams=hparams)
    return sequence_prediction.SerializeToString()

  sequences = []
  for i in range(frame_predictions.shape[0]):
    sequence = tf.py_func(
        functools.partial(predict_sequence, hparams=hparams),
        inp=[
            frame_probs[i],
            onset_probs[i],
            frame_predictions[i],
            onset_predictions[i],
            offset_predictions[i],
            velocity_values[i],
        ],
        Tout=tf.string,
        stateful=False)
    sequence.set_shape([])
    sequences.append(sequence)
  return tf.stack(sequences)


def get_estimator_spec(hparams, mode, length, labels, frame_logits,
                       onset_logits, offset_logits, velocity_values,
                       offset_network=True):
  """Create TPUEstimatorSpec."""
  predictions = None
  if mode in (tf.estimator.ModeKeys.EVAL, tf.estimator.ModeKeys.PREDICT):
    frame_probs = tf.sigmoid(frame_logits)
    onset_probs = tf.sigmoid(onset_logits)
    if offset_network:
      offset_probs = tf.sigmoid(offset_logits)
    else:
      offset_probs = tf.zeros_like(onset_probs)
    frame_predictions = frame_probs > hparams.predict_frame_threshold
    onset_predictions = onset_probs > hparams.predict_onset_threshold
    offset_predictions = offset_probs > hparams.predict_offset_threshold

    if hparams.drum_prediction_map:
      map_predictions = functools.partial(
          drum_mappings.map_pianoroll,
          mapping_name=hparams.drum_prediction_map,
          reduce_mode='any',
          min_pitch=constants.MIN_MIDI_PITCH)
      frame_predictions = tf.map_fn(map_predictions, frame_predictions)
      onset_predictions = tf.map_fn(map_predictions, onset_predictions)
      offset_predictions = tf.map_fn(map_predictions, offset_predictions)
      map_values = functools.partial(
          drum_mappings.map_pianoroll,
          mapping_name=hparams.drum_prediction_map,
          reduce_mode='max',
          min_pitch=constants.MIN_MIDI_PITCH)
      velocity_values = tf.map_fn(map_values, velocity_values)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'frame_probs': frame_probs,
            'onset_probs': onset_probs,
            'frame_predictions': frame_predictions,
            'onset_predictions': onset_predictions,
            'offset_predictions': offset_predictions,
            'velocity_values': velocity_values,
        }
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
  else:
    raise ValueError('Unsupported mode: %s' % mode)
