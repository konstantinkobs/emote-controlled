import os
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, f1_score, accuracy_score, recall_score, \
    precision_recall_fscore_support

from BaseClassifier import SentimentClassifier
from Tokenizer import TwitchTokenizer

EMOTES_PATH = "lexica/emote_distribution.tsv"
EMOJIS_PATH = "lexica/emoji_distribution.tsv"
VADER_PATH = "lexica/vader_distribution.tsv"


class DistributionBasedClassifier(SentimentClassifier):

    def __init__(self, lexica=None):
        self.lexicon = self.get_lexicon(lexica)
        self.tokenizer = TwitchTokenizer()

    def get_lexicon(self, lexica_to_use: dict):
        if lexica_to_use == None:
            lexica_to_use = {"emotes": True, "emojis": True, "vader": True}

        emotes_labeled = pd.read_table(EMOTES_PATH)
        emojis_labeled = pd.read_table(EMOJIS_PATH)
        vader_lexicon = pd.read_table(VADER_PATH)

        lexica = []
        if lexica_to_use["vader"]:
            lexica.append(vader_lexicon)
        if lexica_to_use["emojis"]:
            lexica.append(emojis_labeled)
        if lexica_to_use["emotes"]:
            lexica.append(emotes_labeled)

        # merge labeled data and convert to dict
        labeled_data = pd.concat(lexica, ignore_index=True, sort=False)
        labeled_data.drop_duplicates(subset="word", keep="last", inplace=True)

        lexicon = labeled_data.set_index("word").to_dict("index")

        for token, value in lexicon.items():
            lexicon[token] = [lexicon[token]["negative"], lexicon[token]["neutral"],
                              lexicon[token]["positive"]]

        return lexicon

    """ Main classification algorithm.
    
        :param sentence: un-tokenized message string
        :return: 3 dim. tupel consisting of: at index 0: classification -1,0,1 correspond to
        neg,neu,pos respectively ; at index 1: 3 dim. numpy array containing the prob. of the
        classes neg,neu,pos at index 0,1,2 respectively, at index 2: boolean, true if message
        was classified based on labels of the lexicon, else false.
        """

    def classify_message(self, message):
        msg_tokens = self.tokenizer.tokenize(message)
        return self.classify_tokens(msg_tokens)

    def classify_tokens(self, msg_tokens):
        prob_list = []
        if type(msg_tokens) is list:
            for token in msg_tokens:
                if token.endswith("_NEG"):
                    token = token.replace("_NEG", "")
                if token in self.lexicon:
                    prob_list.append(self.lexicon[token])

        if len(prob_list) == 0:
            return 0, [0.33, 0.34, 0.33], False
        prob_matrix = np.matrix(prob_list)
        class_products = np.asarray(prob_matrix.prod(axis=0))
        max_class = class_products.argmax(1).item(0)
        max_class_prob = class_products[0][max_class] / class_products.sum()
        return max_class - 1, class_products[0] / class_products.sum(), True

    def classify_df(self, df, with_id: bool = False):
        cs, senti, infer = [], [], []
        for index, row in df.iterrows():
            result = self.classify_message(row.message)
            senti.append(result[0])
            cs.append(result[1])
            infer.append(result[2])

        df["sentiment"] = senti
        df["inferred"] = infer
        df["class_scores"] = cs

        return df

    def evaluate(self, twitch_data: pd.DataFrame):
        true_sentiment = list(twitch_data["sentiment"])

        pred = self.classify_df(twitch_data)

        print("Macro Recall:\t" + "%.4f" % (
            recall_score(true_sentiment, pred["sentiment"], average="macro")))
        print("Macro F1:\t" + "%.4f" % (
            f1_score(true_sentiment, pred["sentiment"], average="macro")))
        print("Accuracy:\t" + "%.4f" % (accuracy_score(true_sentiment, pred["sentiment"])))


if __name__ == '__main__':
    dbc = DistributionBasedClassifier()

    eval_data = "data/labeled_dataset.csv"
    df = pd.read_csv(eval_data)
    dbc.evaluate(df)
