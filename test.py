import tensorflow as tf
import grpc

tf.compat.v1.enable_eager_execution()

from preprocess import prepare_input, prepare_output
from tensorflow_serving.apis import prediction_service_pb2_grpc, predict_pb2

# channel = grpc.insecure_channel("192.168.1.18:8500")
channel = grpc.insecure_channel("127.0.0.1:8500")
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

f = tf.saved_model.load("export/1619559542", tags=["serve"]).signatures["serving_default"]


def get_grpc_prediction(t):
    request = predict_pb2.PredictRequest()
    request.model_spec.name = "onsets-and-frames-piano-transcriber"
    request.model_spec.signature_name = 'serving_default'
    request.inputs['input'].CopyFrom(tf.make_tensor_proto(t.numpy()))
    result = stub.Predict(request)
    keys = ["frame_probs", "onset_probs", "frame_predictions", "onset_predictions", "offset_predictions", "velocity_values"]
    return {k: tf.make_ndarray(result.outputs[k]) for k in keys}


input_data = prepare_input("alkan-short.wav")
result = get_grpc_prediction(input_data)
# result = f(input_data)
final_result = prepare_output(result)
# print("okay", result)
