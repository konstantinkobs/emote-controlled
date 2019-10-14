import tensorflow as tf
import numpy as np
import os
import time
import datetime
from tensorflow.contrib import learn
import csv
from collections import Counter

from BaseClassifier import SentimentClassifier
from SentenceCNN.SentenceCNN import SentenceCNN
import argparse
import pandas as pd
from sklearn.metrics import precision_score, f1_score, accuracy_score, recall_score, precision_recall_fscore_support
from pathlib import Path
import gensim
from Tokenizer import TwitchTokenizer

from tensorpack.dataflow.base import DataFlow
from tensorpack.dataflow.common import BatchData, TestDataSpeed, PrintData, MapData
from tensorpack.dataflow import MultiProcessMapData
from tensorpack.dataflow.parallel import MultiProcessPrefetchData
from tensorpack.dataflow.raw import DataFromList

def softmax(a):
    e = np.exp(a)
    softmaxed = e/np.sum(e, axis=1, keepdims=True)
    return [np.array(message) for message in softmaxed]

class CNNBasedClassifier(SentimentClassifier):
    def __init__(self,
                 embedding_file="embedding/embedding",
                 cnn_file="model/model-9500"):
        self.wv = gensim.models.word2vec.Word2Vec.load(str(embedding_file)).wv
        self.zero_index = len(self.wv.vocab)
        self.cnn = cnn_file
        self.tokenizer = TwitchTokenizer()


    def prepare_message(self, message):
        one_hot_list = [self.zero_index]*502
        tokens = self.tokenizer.tokenize(message)
        for i in range(len(tokens)):
            try:
                one_hot_list[i] = (self.wv.vocab[tokens[i]].index)
            except:
                pass
        return one_hot_list


    def predict_batch(self, message_list, cnn=None):
        message_list = []
        for message in ground_truth["message"]:
            one_hot_list = [self.zero_index] * 502
            tokens = self.tokenizer.tokenize(message)
            for i in range(len(tokens)):
                try:
                    one_hot_list[i] = (self.wv.vocab[tokens[i]].index)
                except:
                    pass
            message_list.append(one_hot_list)
        if cnn is not None:
            self.cnn = cnn
        print("predicting using:", self.cnn)
        graph = tf.Graph()
        with graph.as_default():
            session_conf = tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=False)
            sess = tf.Session(config=session_conf)
            with sess.as_default():
                # Load the saved meta graph and restore variables
                saver = tf.train.import_meta_graph("{}.meta".format(self.cnn))
                saver.restore(sess, self.cnn)

                # Get the placeholders from the graph by name
                input_x = graph.get_operation_by_name("input_x").outputs[0]
                input_y = graph.get_operation_by_name("input_y").outputs[0]
                dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

                # Tensors we want to evaluate
                predictions = graph.get_operation_by_name("output/predictions").outputs[0]
                batch_predictions = sess.run(predictions,
                                             {input_x: message_list, dropout_keep_prob: 1.0})

                return [val - 1 for val in batch_predictions]

    def find_best(self, x_test, y_test):
        print("\nEvaluating...\n")

        best_f1 = -1
        best_model = ""
        for filename in (Path(args.model_dir) / "cnn").glob('**/checkpoint'):
            print("")
            print("filename:", filename)

            if not "delete" in str(filename) or not str(filename):
                continue

            # checkpoint_file = tf.train.latest_checkpoint(str(filename.parent)) # [:-4] + "1500"
            checkpoint_file = \
            tf.train.get_checkpoint_state(str(filename.parent)).all_model_checkpoint_paths[0]

            print("trained until:", tf.train.latest_checkpoint(str(filename.parent)))
            print("but choosing:", checkpoint_file)

            batch_predictions = self.predict_batch(x_test, cnn=checkpoint_file)

            print(Counter(batch_predictions))
            print(Counter(y_test))

            prec, rec, f1, _ = precision_recall_fscore_support(y_test, batch_predictions,
                                                               average="macro")
            acc = accuracy_score(y_test, batch_predictions)

            if f1 > best_f1:
                best_f1 = f1
                best_model = checkpoint_file

            print("acc: {:5.3f}\tprec: {:5.3f}\trec: {:5.3f}\tf1: {:5.3f}".format(acc * 100, prec * 100, rec * 100, f1 * 100), flush=True)

        print("Best model:", best_model)
        print("Best F1:", best_f1)
        self.cnn = best_model


    def classify_df(self, df, message_row="message", batch_size=1000):
        messages = df[message_row].values.tolist()
        l = len(messages)

        results = []
        results_scores = []
        print("predicting using:", self.cnn)
        graph = tf.Graph()
        with graph.as_default():
            session_conf = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False)
            sess = tf.Session(config=session_conf)
            with sess.as_default():

                # Load the saved meta graph and restore variables
                saver = tf.train.import_meta_graph("{}.meta".format(self.cnn))
                saver.restore(sess, self.cnn)

                # Get the placeholders from the graph by name
                input_x = graph.get_operation_by_name("input_x").outputs[0]
                input_y = graph.get_operation_by_name("input_y").outputs[0]
                dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

                # Tensors we want to evaluate
                predictions = graph.get_operation_by_name("output/predictions").outputs[0]
                scores = graph.get_operation_by_name("output/scores").outputs[0]

                for ndx in range(0, l, batch_size):
                    begin = ndx
                    end = min(ndx + batch_size, l)
                    print("working on data: [{},{}]".format(begin, end), flush=True)
                    batch = [self.prepare_message(val) for val in messages[begin:end]]
                    #print(batch)
                    batch_predictions, batch_scores = sess.run([predictions, scores], {input_x: batch, dropout_keep_prob: 1.0})
                    results.extend(batch_predictions)
                    results_scores.extend(batch_scores)

                df["sentiment"] = [val-1 for val in results]
                df["sent_softmax"] = softmax(results_scores)
                return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="generating an embedding")
    parser.add_argument("--data_dir", dest="data_dir", action="store", type=str, help="where to find the data to be read")
    parser.add_argument("--model_dir", dest="model_dir", action="store", type=str, help="where to find the model")
    args = parser.parse_args()

    # read data
    ground_truth = pd.read_csv("data/labeled_dataset.csv")
    ground_truth.rename({"sentiment": "sentiment_ground_truth"}, axis="columns", inplace=True)

    classifier = CNNBasedClassifier()
    pred = classifier.classify_df(ground_truth)
    
    print("Macro Recall:\t" + "%.4f" % (
        recall_score(pred["sentiment_ground_truth"], pred["sentiment"], average="macro")))
    print("Macro F1:\t" + "%.4f" % (
        f1_score(pred["sentiment_ground_truth"], pred["sentiment"], average="macro")))
    print("Accuracy:\t" + "%.4f" % (accuracy_score(pred["sentiment_ground_truth"], pred["sentiment"])))
