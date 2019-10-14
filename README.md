# Emote-Controlled

In this repository you find the code and data for the paper "Emote-Controlled: Obtaining Implicit Viewer Feedback through Emote based Sentiment Analysis on Comments of Popular Twitch.tv Channels" by Konstantin Kobs et al., published in "Transactions on Social Computing":

> In recent years, streaming platforms for video games have seen increasingly large interest, as so-called “esports” have developed into a lucrative branch of business. Like for other sports, watching esports has become a new kind of entertainment medium, which is possible due to platforms that allow gamers to live stream their gameplay, the most popular platform being [Twitch.tv](http://twitch.tv). On these platforms, users can comment on streams in real-time and thereby express their opinion about the events in the stream. Due to the popularity of Twitch.tv, this can be a valuable source of feedback for streamers aiming to improve their reception in a gaming-oriented audience. In this work, we explore the possibility of deriving feedback for video streams on Twitch.tv by analyzing the sentiment of live text comments made by stream viewers in highly active channels. Automatic sentiment analysis on these comments is a challenging task, as one can compare the language used in Twitch.tv with that used by an audience in a stadium, shouting as loud as possible in sometimes non-organized ways. This language is very different from common English, mixing Internet slang and gaming-related language with abbreviations, intentional and unintentional grammatical and orthographic mistakes as well as emoji- like images called emotes. Classic lexicon based sentiment analysis techniques therefore fail when applied to Twitch comments.
> In order to overcome the challenge posed by the non-standard language, we propose two unsupervised lexicon based approaches that make heavy use of the information encoded in emotes, as well as a weakly supervised neural network based classifier trained on the lexicon based outputs, that is supposed to help generalization to unknown words by use of domain-specific word embeddings. To enable better understand- ing of Twitch.tv comments, we analyze a large dataset of comments, uncovering specific properties of their language and provide a smaller set of comments labeled with sentiment information by crowd sourcing.
> We present two case studies showing the effectiveness of our methods in generating sentiment trajectories for events live-streamed on Twitch.tv that correlate well with specific topics in the given stream. This allows for a new kind of implicit real-time feedback gathering for Twitch streamers and companies producing games or streaming content on Twitch.

# Code

The code in this repository should be sufficient to replicate the main results of our paper.

## Using Classifiers

The code for querying our three classifiers are present in `AverageBasedClassifier.py`, `DistributionBasedClassifier.py`, and `CNNBasedClassifier.py`, respectively. After installing the required packages given in `requirements.txt` you can run `python3 AverageBasedClassifier.py` to replicate the sentiment analysis results from the paper on the test set. The same applies to the other classifiers as well. However, to use the `CNNBasedClassifier`, you need to first download the embedding and model weights (see Downloads section). Extract the files into folders named `embedding` and `model`, respectively, into the main directory.

Each of the three classifiers has a `classify_df` method that classifies a Pandas `DataFrame` containing a `message` column that includes the texts to be classified. This can be used to easily estimate the sentiment of a batch of chat messages.

## Querying Embedding

The trained embedding can not only be used in the CNNBasedClassifier. We can also query it like we did in Section 6.3 of the paper. For this, you can run `python3 query_embedding.py` and change the desired inputs to the queries. Examples can be found in the code.


# Data

We provide all the data we used in this work, including unlabeled Twitch comment data for three months, a small labeled dataset of Twitch comments, as well as the lexica we used or created for our classifiers.

## Lexica

In our work, we used three sentiment lexica:

1. Emoji Sentiment Lexicon by [Kralj Novak et al.](http://kt.ijs.si/data/Emoji_sentiment_ranking/)
2. VADER Sentiment Lexicon by [Hutto and Gilbert](https://github.com/cjhutto/vaderSentiment)
3. **A Self-Labeled Emote Lexicon**

The emote lexicon was built by conducting a survey among active Twitch users.
We let them label the 100 most used emotes at the time of writing the paper.

All three lexica are available in two versions: an average based version including only a value between -1 (very negative) and 1 (very positive) and a distribution based version that gives a distribution across the classes positive, neutral, and negative.
You can find them as .tsv files in `lexica`.

## Labeled Twitch Data

For evaluating the performance of our prediction models, we sampled 2000 Twitch comments from very active Twitch channels.
We then let three annotators assess the sentiment of each comment.
Please refer to the paper for more information on the sampling and labeling process.

The labeled dataset can be found as a .csv file in `data`.

## Unlabeled Twitch Data

We also provide three months of all Twitch comments.
These are not labeled and mostly suited for analyzing the language used on Twitch.
You can find the files in the Downloads section.
The .csv files do not have a header row, but the structure is similar to the labeled dataset: `date,channel,game,user,mod,subscriber,message`.

**Note:** Usernames are anonymized and replaced with ids. These ids are coherent across all data files so you still can track the commenting history of users.

# Downloads

Here is a list of all provided downloads. The password for the download links is always `twitch-sentiment`.

- [Embedding](https://oc.informatik.uni-wuerzburg.de/s/YxmjCW3dtBDeTeT) (2.2GB)
- [CNN model weights](https://oc.informatik.uni-wuerzburg.de/s/mtwNqLytjfHEfYW) (3.3GB)
- Unlabeled dataset (each approx. 30GB packed, approx. 100GB unpacked)
    - [April 2018](https://oc.informatik.uni-wuerzburg.de/s/9ZmWz8RB6YKiKtX)
    - [May 2018](https://oc.informatik.uni-wuerzburg.de/s/esRkffKgp2fnjy4)
    - [June 2018](https://oc.informatik.uni-wuerzburg.de/s/c7kmDmtNWrSLKQ8)


# Citation

If you use our code or data, please cite us:

```
TODO
```

# License

You are not allowed to use the provided code or data for commercial purposes.
