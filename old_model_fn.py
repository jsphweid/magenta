
def model_fn(features, labels, mode, params, config):
    """Builds the acoustic model."""
    del config
    hparams = params

    # length = features.length
    # length = tf.placeholder(dtype=tf.int32, shape=[1, ], name="lmao")  # for now hard code to 1 length
    length = tf.constant([1])  # arbitrary but whatever

    spec = features["feature"]  # should be a tensor containing all 229 bins
    # spec = features.spec

    if hparams.stop_activation_gradient and not hparams.activation_loss:
        raise ValueError('If stop_activation_gradient is true, activation_loss must be true.')

    with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=False):
        with tf.variable_scope('onsets'):
            onset_outputs = acoustic_model(
                spec,
                hparams,
                lstm_units=hparams.onset_lstm_units,
                lengths=length)
            onset_probs = slim.fully_connected(
                onset_outputs,
                constants.MIDI_PITCHES,
                activation_fn=tf.sigmoid,
                scope='onset_probs')

            # onset_probs_flat is used during inference.
            onset_probs_flat = flatten_maybe_padded_sequences(onset_probs, length)
        with tf.variable_scope('offsets'):
            offset_outputs = acoustic_model(
                spec,
                hparams,
                lstm_units=hparams.offset_lstm_units,
                lengths=length)
            offset_probs = slim.fully_connected(
                offset_outputs,
                constants.MIDI_PITCHES,
                activation_fn=tf.sigmoid,
                scope='offset_probs')

            # offset_probs_flat is used during inference.
            offset_probs_flat = flatten_maybe_padded_sequences(offset_probs, length)

        with tf.variable_scope('velocity'):
            velocity_outputs = acoustic_model(
                spec,
                hparams,
                lstm_units=hparams.velocity_lstm_units,
                lengths=length)
            velocity_values = slim.fully_connected(
                velocity_outputs,
                constants.MIDI_PITCHES,
                activation_fn=None,
                scope='onset_velocities')

            velocity_values_flat = flatten_maybe_padded_sequences(
                velocity_values, length)

        with tf.variable_scope('frame'):
            if not hparams.share_conv_features:
                # TODO(eriche): this is broken when hparams.frame_lstm_units > 0
                activation_outputs = acoustic_model(
                    spec,
                    hparams,
                    lstm_units=hparams.frame_lstm_units,
                    lengths=length)
                activation_probs = slim.fully_connected(
                    activation_outputs,
                    constants.MIDI_PITCHES,
                    activation_fn=tf.sigmoid,
                    scope='activation_probs')
            else:
                activation_probs = slim.fully_connected(
                    onset_outputs,
                    constants.MIDI_PITCHES,
                    activation_fn=tf.sigmoid,
                    scope='activation_probs')

            probs = []
            if hparams.stop_onset_gradient:
                probs.append(tf.stop_gradient(onset_probs))
            else:
                probs.append(onset_probs)

            if hparams.stop_activation_gradient:
                probs.append(tf.stop_gradient(activation_probs))
            else:
                probs.append(activation_probs)

            if hparams.stop_offset_gradient:
                probs.append(tf.stop_gradient(offset_probs))
            else:
                probs.append(offset_probs)

            combined_probs = tf.concat(probs, 2)

            if hparams.combined_lstm_units > 0:
                outputs = lstm_layer(
                    combined_probs,
                    hparams.combined_lstm_units,
                    lengths=length if hparams.use_lengths else None,
                    stack_size=hparams.combined_rnn_stack_size,
                    use_cudnn=hparams.use_cudnn,
                    bidirectional=hparams.bidirectional)
            else:
                outputs = combined_probs

            frame_probs = slim.fully_connected(
                outputs,
                constants.MIDI_PITCHES,
                activation_fn=tf.sigmoid,
                scope='frame_probs')

        frame_probs_flat = flatten_maybe_padded_sequences(frame_probs, length)

    frame_predictions = frame_probs_flat > hparams.predict_frame_threshold
    onset_predictions = onset_probs_flat > hparams.predict_onset_threshold
    offset_predictions = offset_probs_flat > hparams.predict_offset_threshold

    frame_predictions = tf.expand_dims(frame_predictions, axis=0)
    onset_predictions = tf.expand_dims(onset_predictions, axis=0)
    offset_predictions = tf.expand_dims(offset_predictions, axis=0)
    velocity_values = tf.expand_dims(velocity_values_flat, axis=0)

    def predict_sequence():
        """Convert frame predictions into a sequence (TF)."""

        def _predict(frame_probs, onset_probs, frame_predictions, onset_predictions,
                     offset_predictions, velocity_values):
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
            return sequence.SerializeToString()

        sequence = tf.py_func(
            _predict,
            inp=[
                frame_probs[0],
                onset_probs[0],
                frame_predictions[0],
                onset_predictions[0],
                offset_predictions[0],
                velocity_values[0],
            ],
            Tout=tf.string,
            stateful=False)
        sequence.set_shape([])
        return tf.expand_dims(sequence, axis=0)

    predictions = {
        'frame_probs': frame_probs,
        'onset_probs': onset_probs,
        'frame_predictions': frame_predictions,
        'onset_predictions': onset_predictions,
        'offset_predictions': offset_predictions,
        'velocity_values': velocity_values,
        'sequence_predictions': predict_sequence(),
        # Include some features and labels in output because Estimator 'predict'
        # API does not give access to them.
        # 'sequence_ids': features.sequence_id,  # shouldnt' need probably,
        # 'sequence_labels': labels.note_sequence,
        # 'frame_labels': labels.labels,
        # 'onset_labels': labels.onsets,
    }

    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
