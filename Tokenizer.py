import re
import emoji
import pandas as pd
from itertools import groupby
from nltk.corpus import stopwords
import os

"""
Twitch tokenizer. As a code basis the NLTK TwitterTokenizer and the NLTK mark_negation()-method were used.
The basic logic is this:

1. Replace instances of type url, numbers, usernames, chatbot commands, mails with tags.

2. Tokenize text considering e.g. words, various emoticons and twitch emotes.

3. Lowercase tokens except of emoticons and tags.

4. Shortening: normalizing of words with chacters occuring more than twice in succession e.g. "looooove" -> "loove"

5. "_NEG"-Tagging for negated words (see "mark_negation()" impl. for details)

6. Remove all non-alphabetical characters, keep line emoticons, unicode-emoji & emotes

"""

#pattern to match emoticons.
EMOTICONS = r"""
    (?:
      [<>3Oo0|]?
      [:;=8Xx%]                     # eyes
      [']?                        # optional tear
      [\-o\*\']?                 # optional nose
      [\)\]\(\[dDpP/\:\}\{@\|\\X><c3$LSÞ] # mouth
      |
      [\)\]\(\[dDpP/\:\}\{@\|\\X><c3$LSÞ] # mouth
      [\-o\*\']?                 # optional nose
      [']?                        # optional tear
      [:;=8Xx%]                     # eyes
      [<>Oo0|]?
      |<3|<\/3|<\\3
      |
     \( ͡° ͜ʖ ͡°\)                 # lenny face
    |
    ¯\\_\(ツ\)_/¯                  # meh
      |
      >_>|<_<
      |
      @};-|@}->--|@}‑;‑|@>‑‑>‑‑     #rose
      |
      O_O|o\‑o|O_o|o_O|o_o|O\-O     #schock
      |
      >.<|v.v|>>|<<
      |
      \(>_<\)|\^\^|\^_\^|\(-__-\)|\(-_-\)|\(/◕ヮ◕\)/|\(\^o\^\)丿
      |\('_'\)|\(/_;\)|\(T_T\)|\(;_;\)|\(=\^·\^=\)|\(\*_\*\)|\(\+_\+\)|\(@_@\)
      |\(ง •̀_•́\)ง
    )"""

#pattern to match urls.
URL = r"(?:http(s)?:\/\/)?[\w.-]+(?:\.[\w\.-]+)+[\w\-\._~:/?#[\]@!\$&'\(\)\*\+,;=.]+"

# The components of the tokenizer:
REGEXPS = (
    # ASCII Emoticons
    EMOTICONS
    ,
    # HTML tags:
    r"""<[^>\s]+>"""
    ,
    # ASCII Arrows
    r"""[\-]+>|<[\-]+"""
    ,
    # Twitter like hashtags:
    r"""(?:\#+[\w_]+[\w\'_\-]*[\w_]+)"""
    ,
    # Remaining word types:
    r"""
    #(?:[^\W\d_](?:[^\W\d_]|['\-_])+[^\W\d_]) # Words with apostrophes or dashes.
	(?:[^\W_](?:[^\W\d_]|['\-_\d])+[^\W_]) # Words with apostrophes or dashes. (modified)
    |
    (?:[+\-]?\d+[,/.:-]\d+[+\-]?)  # Numbers, including fractions, decimals.
    |
    (?:[\w_]+)                     # Words without apostrophes or dashes.
    |
    (?:\.(?:\s*\.){1,})            # Ellipsis dots.
    |
    (?:\S)                         # Everything else that isn't whitespace.
    """
    )

# Regular expression for negation by Christopher Potts
NEGATION = r"""
    (?:
        ^(?:never|no|nothing|nowhere|noone|none|not|
            havent|hasnt|hadnt|cant|couldnt|shouldnt|
            wont|wouldnt|dont|doesnt|didnt|isnt|arent|aint
        )$
    )
    |
    n't"""

URL_RE = re.compile(URL)
NUM = re.compile("(?<=^|(?<=\s)|(?<=\())#{,1}\d{1,}(?=$|(?=\s)|(?=\)))")
USERNAME = re.compile("(?<=^|(?<=\s))@\w+(?=$|(?=\s))")
COMMAND = re.compile("(?<=^|(?<=\s))!#?[a-zA-Z]+(?=$|(?=\s))")
MAIL = re.compile("[\w.+-]+@[\w-]+\.(?:[\w-]\.?)+[\w-]")

SHORTENING_OF =  re.compile("(.)\\1{2,}")

# pattern to find punctuation
CLAUSE_PUNCT = r'^[.:;!?]$'

# pattern to find non alphabetical chars
NON_ALPHABETICAL = re.compile('[^A-Za-z ]')

# create lexicon of stoppwords
#en_stopwords = set(stopwords.words('english'))
#en_stopwords.add("i'm")
#en_stopwords.add("i've")
#en_stopwords.add("can't")
#stripped_stopwords = [word.replace("'", "") for word in en_stopwords]  # add stopwords without apostrophes
#negated_stopwords = [word+"_NEG" for word in en_stopwords] # add negated stopwords
#[en_stopwords.add(word) for word in stripped_stopwords if word not in en_stopwords]
#[en_stopwords.add(word) for word in negated_stopwords]

# create lexicon of emoji
emoji_lexicon = emoji.UNICODE_EMOJI

######################################################################
# This is the core tokenizing regex:

WORD_RE = re.compile(r"""(%s)""" % "|".join(REGEXPS), re.VERBOSE | re.I
                     | re.UNICODE)

# WORD_RE performs poorly on these patterns:
HANG_RE = re.compile(r'([^a-zA-Z0-9])\1{3,}')

# The emoticon string gets its own regex so that we can preserve case for
# them as needed:
EMOTICON_RE = re.compile(EMOTICONS, re.VERBOSE | re.I | re.UNICODE)

# These are for regularizing HTML entities to Unicode:
ENT_RE = re.compile(r'&(#?(x?))([^&;\s]+);')

#negation and punctuation matching patterns
NEGATION_RE = re.compile(NEGATION, re.VERBOSE)
CLAUSE_PUNCT_RE = re.compile(CLAUSE_PUNCT)
######################################################################
# Functions for converting html entities
######################################################################

def load_labeled_emotes():
    emotes = set(pd.read_table("lexica/emote_average.tsv")["word"])
    return emotes

def mark_negation(token_list, emotes, double_neg_flip=False):
    """
    Append _NEG suffix to words that appear in the scope between a negation
    and a punctuation mark or twitch emote.

    :param token_list: a list of words/tokens
    :param double_neg_flip: if True, double negation is considered affirmation
        (we activate/deactivate negation scope everytime we find a negation).

    """
    neg_scope = False
    for i, word in enumerate(token_list):
        if NEGATION_RE.search(word):
            if not neg_scope or (neg_scope and double_neg_flip):
                neg_scope = not neg_scope
                continue
            else:
                token_list[i] += '_NEG'
        elif neg_scope and (CLAUSE_PUNCT_RE.search(word) or word in emotes):
            neg_scope = not neg_scope
        elif neg_scope and not CLAUSE_PUNCT_RE.search(word):
            token_list[i] += '_NEG'

    return token_list


######################################################################

class TwitchTokenizer:

    def __init__(self, preserve_case=True):
        self.preserve_case = preserve_case
        self.emotes = load_labeled_emotes()

    def tokenize(self, text):
        """
        :param text: str
        :rtype: list(str)
        :return: a tokenized list of strings; concatenating this list returns\
        the original string if `preserve_case=False`
        """
        text = URL_RE.sub("URL", text)

        # Shorten problematic sequences of characters
        safe_text = HANG_RE.sub(r'\1\1\1', text)
        # Tokenize:
        words = WORD_RE.findall(safe_text)
        # Possibly alter the case, but avoid changing emoticons like :D into :d:
        if not self.preserve_case:
            words = list(map((lambda x : x if EMOTICON_RE.search(x) else
                              x.lower()), words))
        token_list = []
        for word in words:
            if word not in self.emotes and not re.match(EMOTICON_RE,word):
                if word not in ["URL", "NUM", "USERNAME", "COMMAND","MAIL"]: # do not lowercase tags
                    word = word.lower()
                # shortening: normalizing of words with chacters occuring more than twice in succession e.g. "looooove" -> "loove"
                word = re.sub(SHORTENING_OF, r'\\1\\1', word)
            token_list.append(word)

        # remove all non-alphabetical characters, keep line emoticons, unicode-emoji & emotes
        def keep_token(token):
            if re.match(NON_ALPHABETICAL,token) and not re.match(EMOTICON_RE,token) and token not in self.emotes \
            and token not in emoji_lexicon:
                return False
            else:
                return True

        return token_list
