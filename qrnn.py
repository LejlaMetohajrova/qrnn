#!/usr/bin/python3

import tensorflow as tf
import os

from gensim.models import KeyedVectors
from gensim.parsing.preprocessing import STOPWORDS
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.utils import simple_preprocess


class ImdbDataset:
    def __init__(self, dataset_path, word2vec_model):
        self.X = []
        self.Y = []
        self.dataset_path = dataset_path
        self.word2vec_model = word2vec_model
        self.load_data()


    def tokenize(self, text):
        return [token for token in simple_preprocess(text) \
                if token not in STOPWORDS]


    def vectorize(self, tokens):
        vectors = []
        for token in tokens:
            try:
                vectors.append(self.word2vec_model[token])
            except Exception:
                pass
        return vectors


    def load_by_label(path, label):
        for filename in os.listdir(path):
            # Text of a single review
            text = open(path+filename, 'r').read()
            # Tokenize (list of words)
            tokens = self.tokenize(text)
            # Get word vectors (dim 300)
            vectors = self.vectorize(tokens)
            self.X.append(vectors)
            self.Y.append(label)


    def load_data(self):
        self.load_by_label(self.dataset_path+'neg/', label=0)
        self.load_by_label(self.dataset_path+'pos/', label=1)


class QRNN:
    def __init__(self, hidden_units):
        self.hidden_units = hidden_units


    def forward(self, inputs):
        """
        inputs: (batch_size x timestamps x word2vec_size)
        """
        # Candidate vectors
        z = tf.layers.conv1d(inputs, filters=self.hidden_units, kernel_size=5,
                             padding='same', activation=tf.tanh)
        # Forget gate
        f = tf.layers.conv1d(inputs, filters=self.hidden_units, kernel_size=5,
                             padding='same', activation=tf.sigmoid)
        # Output gate
        o = tf.layers.conv1d(inputs, filters=self.hidden_units, kernel_size=5,
                             padding='same', activation=tf.sigmoid)
        # Init context
        c = tf.zeros([int(inputs.get_shape()[1]), self.hidden_units],
                     dtype=tf.float32)

        hidden = []
        # fo-pool
        for i in range(int(z.get_shape()[0])):
            c = tf.multiply(f[i], c) + tf.multiply(1 - f[i], z[i])
            h = tf.multiply(o[i], c)
            hidden.append(h)

        return tf.convert_to_tensor(hidden)


    def qrnn_model_fn(features, labels, mode):
        """
        features: (batch_size x timestamps x word2vec_size)
        labels: (batch_size)
        mode: one of (TRAIN, EVAL, PREDICT)
        """
        # Quasi-recurrent layer1
        qr_layer1 = self.forward(features)
        # Quasi-recurrent layer2
        qr_layer2 = self.forward(qr_layer1)
        # Flatten output
        flat_layer2 = tf.reshape(qr_layer2, [qr_layer2.get_shape()[0],
                               qr_layer2.get_shape()[1] * qr_layer2.get_shape()[2]])
        # Dense layer
        dense = tf.layers.dense(inputs=flat_layer2, units=self.hidden_units,
                                activation=tf.nn.relu)
        dropout = tf.layers.dropout(
            inputs=dense, rate=0.3, training=mode == tf.estimator.ModeKeys.TRAIN)
        # Logits layer
        logits = tf.layers.dense(inputs=dropout, units=2)

        predictions = {
          # Generate predictions (for PREDICT and EVAL mode)
          "classes": tf.argmax(input=logits, axis=1),
          # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
          # `logging_hook`.
          "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
        }

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

        # Calculate Loss (for both TRAIN and EVAL modes)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

        # Configure the Training Op (for TRAIN mode)
        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.RMSPropOptimizer(learning_rate=0.001)
            train_op = optimizer.minimize(
                loss=loss,
                global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

        # Add evaluation metrics (for EVAL mode)
        eval_metric_ops = {
          "accuracy": tf.metrics.accuracy(
              labels=labels, predictions=predictions["classes"])}
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


if __name__ == "__main__":
    if not os.path.exists('aclImdb'):
        prnt('Downloading data.')
        os.system('./download.sh')

    # Load embeddings model
    print('Converting GloVe to Word2Vec...')
    glove2word2vec('glove.42B.300d.txt', 'word2vec.txt')
    print('Loading Word2Vec model...')
    word2vec_model = KeyedVectors.load_word2vec_format('word2vec.txt')

    # Load training data
    train_data = ImdbDataset('aclImdb/train/', word2vec_model)

    X = tf.keras.preprocessing.sequence.pad_sequences(train_data.X, maxlen=200,
                                                      dtype='int32')
    x_train = tf.convert_to_tensor(np.array(X))
    y_train = tf.convert_to_tensor(np.array(train_data.Y))

    qrnn = QRModel()

    qrnn_classifier = tf.estimator.Estimator(
    model_fn=qrnn.qrnn_model_fn, model_dir="/tmp/qrnn_model")

    # Set up logging for predictions
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": x_train},
        y=y_train,
        batch_size=25,
        num_epochs=None,
        shuffle=True)
    qrnn_classifier.train(
        input_fn=train_input_fn,
        steps=150,
        hooks=[logging_hook])

    # Load validation data
    test_data = ImdbDataset('aclImdb/test/', word2vec_model)

    X = tf.keras.preprocessing.sequence.pad_sequences(test_data.X, maxlen=200,
                                                      dtype='int32')
    x_test = tf.convert_to_tensor(np.array(X))
    y_test = tf.convert_to_tensor(np.array(train_data.Y))

    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": x_test},
        y=y_test,
        num_epochs=1,
        shuffle=False)
    eval_results = qrnn_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)
