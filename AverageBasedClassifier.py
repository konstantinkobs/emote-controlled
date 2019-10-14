import pandas as pd
import numpy as np
import os
from sklearn.metrics import precision_score, f1_score, accuracy_score, recall_score, \
    precision_recall_fscore_support
from BaseClassifier import SentimentClassifier
from Tokenizer import TwitchTokenizer


class AverageBasedClassifier(SentimentClassifier):
    def __init__(self):

        self.lexicon = self.get_lexicon()
        self.tokenizer = TwitchTokenizer()

    def predict(self, sentence: str):
        """
        :param sentence: un-tokenized message string
        :type sentence: String
        :return: tuple of tokenized string, sentiment class, certainty;
            certainty = 0 means that no token in the message was found in the lexicon
        :rtype: (str, int, int)
        """
        msg = self.tokenizer.tokenize(sentence)
        return self.predict_tokens(msg)

    def predict_tokens(self, msg):
        msg_sentiment = 0
        total = 1
        certainty = 0

        for word in msg:
            if word.endswith("_MULTIPLE"):
                word = word.replace("_MULTIPLE", "")
            if word.endswith("_NEG"):
                word = word.replace("_NEG", "")
                if self.lexicon.get(word) is not None:
                    msg_sentiment -= self.lexicon.get(word)
                    total += 1
            else:
                if self.lexicon.get(word) is not None:
                    msg_sentiment += self.lexicon.get(word)
                    total += 1

        if total > 1:  # Avoid division by 0
            certainty = 1
            total -= 1

        score = 0
        if msg_sentiment / total < -0.33:
            score = -1
        elif msg_sentiment / total > 0.33:
            score = 1

        return msg, score, certainty

    def classify_df(self, twitch_data: pd.DataFrame, analyze_predictions=True) -> pd.DataFrame:
        sentiment = self.predict_dataframe(twitch_data, analyze_predictions)

        twitch_data["sentiment"] = [sent[1] for sent in sentiment]
        twitch_data["certainty"] = [sent[2] for sent in sentiment]

        return twitch_data

    def predict_dataframe(self, twitch_data: pd.DataFrame, analyze_predictions=True):
        prediction_tuples = []
        sentiment = []
        inferred = []
        for row in twitch_data.itertuples():
            msg = getattr(row, "message")

            if msg is not np.nan and isinstance(msg, str):
                pred = self.predict(msg)
            else:
                raise ValueError("Message of the following row could not be extracted:", row)

            prediction_tuples.append((row, pred[1], pred[2]))
            sentiment.append(pred[1])
            inferred.append(pred[2])

        if analyze_predictions:
            self.analyze_predictions(prediction_tuples)

        return pd.DataFrame({"id": twitch_data.id.values,
                             "message": twitch_data.message.values,
                             "sentiment": sentiment,
                             "inferred": inferred})

    def evaluate(self, twitch_data: pd.DataFrame):
        true_sentiment = list(twitch_data["sentiment"])

        pred = self.predict_dataframe(twitch_data, analyze_predictions=True)

        print("Macro Recall:\t" + "%.4f" % (
            recall_score(true_sentiment, pred["sentiment"], average="macro")))
        print("Macro F1:\t" + "%.4f" % (
            f1_score(true_sentiment, pred["sentiment"], average="macro")))
        print("Accuracy:\t" + "%.4f" % (accuracy_score(true_sentiment, pred["sentiment"])))

    # Analyze and print generated predictions for pos-neg-neu distribution
    def analyze_predictions(self, predictions):
        pos_count, neg_count, neu_count = 0, 0, 0
        sentiment_list = []
        for item in predictions:
            if item[1] == 1:
                pos_count += 1
                sentiment_list.append(1)
            elif item[1] == -1:
                neg_count += 1
                sentiment_list.append(-1)
            else:
                neu_count += 1
                sentiment_list.append(0)

        print("# Positive: ", pos_count)
        print("# Negative: ", neg_count)
        print("# Neutral: ", neu_count)

    # sliding window
    def moving_average(self, values, window):
        weights = np.repeat(1.0, window) / window
        sma = np.convolve(values, weights, 'valid')
        return sma

    def get_lexicon(self):
        emotes = "lexica/emote_average.tsv"
        emojis = "lexica/emoji_average.tsv"
        vader = "lexica/vader_average.tsv"

        lexicon = {}
        with open(emotes) as f:
            lines = [l.strip().split("\t") for l in f.readlines()[1:]]
            for l in lines:
                if l[0] not in lexicon:
                    lexicon[l[0]] = float(l[1])

        with open(emojis) as f:
            lines = [l.strip().split("\t") for l in f.readlines()[1:]]
            for l in lines:
                if l[0] not in lexicon:
                    lexicon[l[0]] = float(l[1])

        with open(vader) as f:
            lines = [l.strip().split("\t") for l in f.readlines()[1:]]
            for l in lines:
                if l[0] not in lexicon:
                    lexicon[l[0]] = float(l[1])

        return lexicon



if __name__ == '__main__':
    abc = AverageBasedClassifier()

    eval_data = "data/labeled_dataset.csv"
    df = pd.read_csv(eval_data)
    abc.evaluate(df)
