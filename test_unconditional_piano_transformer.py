import tensorflow as tf

# f = tf.saved_model.load("export/1619751595", tags=["serve"]).signatures["serving_default"]

with tf.Session(graph=tf.Graph()) as session:
    serve = tf.saved_model.load(
        session, tags=['serve'], export_dir="./export/1619751595")

    # tags = extract_tags(serve.signature_def, session.graph)
    # model = tags['serving_default']