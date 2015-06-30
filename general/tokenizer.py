#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Johannes'

import re
from util.datacleaning import killgremlins


class TokenType:
    EMOHAPPY = "emohappy"
    EMOSAD = "emosad"
    EMOSHOCK = "emoshock"
    EMOCHEEKY = "emocheeky"
    EMOUNDECIDED = "emoundec"
    EMOOTHER = "emoother"
    EMOASTERISK = "emoaster"

    NUMBBASIC = "numbbasic"
    NUMBPRICE = "numbprice"

    TWTRNAME = "twtrname"
    TWTRHASH = "twtrhash"

    WEBADR = "webadr"

    ENFQUEST = "enfquest"
    ENFEXCL = "enfexcl"
    ENFCONF = "enfconf"
    ENFDOTS = "enfdots"

    WORD = "word"


emoticon_happy = r"""
    (?P<eha>
      [:8xX=]
      [\']?
      [\-oc\^]?
      [\)\]\}D3>\*]
      |
      [\(\[\{\*]
      [\-o\^]?
      [\']?
      [:8xX=]
      |
      [\^]{2,}
    )"""

emoticon_sad = r"""
    (?P<esa>
      [>]?
      [:]
      [\']?
      [\-]?
      [\(\[\{Cc<(\|\|)@]
      |
      [\)\]\}D>(\|\|)]
      [\-]?
      [\']?
      [:;=8X]
      [<]?
    )"""

emoticon_shock = r"""
    (?P<esh>
      [>]?
      [:8]
      [\-]?
      [Oo0]
      |
      [Oo0]
      [\-]?
      [:8]
      [<]?
      |
      [oO]
      [\-\_\.]?
      [oO]
    )"""

emoticon_cheeky = r"""
    (?P<ech>
      [<>]?
      [:;8Xx=]
      [\-\^]?
      [pPb]
      |
      [qd]
      [\-\^]?
      [:8Xx=]
      [<>]?
      |
      [;\*]
      [\-\^]?
      [\)\],D]
    )"""

emoticon_undecided = r"""
    (?P<eun>
      [>]?
      [:=]
      [\-]?
      [\\/LS\|\.]
      |
      [\\/S\|\.]
      [\-]?
      [:=]
      [<]?
    )"""

emoticon_other = r"""
    (?P<eot>
      [:;]
      [\-]?
      [Xx\#\$&]
      |
      [Xx\#\$&]
      [\-]?
      [:]
    )"""
emoticon_asterisk = r"""
    (?P<eas>
      \*
      [\w]+
      \*
    )"""

number = r"""
    (?P<num>
      [\+\-]?
      [0-9]+
      (?:
        [\-\s',\./:]?
        [0-9]+
      )*
    )"""

number_price = r"""
    (?P<numbprice>
      (
        [EUR|USD|\$\p{Sc}]
        \s?
        [\+\-]?
        [0-9]+
        (
          [\-\s',\./:]?
          [0-9]+
        )*
      )
      |
      (
        [\+\-]?
        [0-9]+
        (
          [\-\s',\./:]?
          [0-9]+
        )*
      )
    )"""

# Twitter usernames:
twitter_uname = r"""
    (?P<tnm>
      @[\w_]+
    )"""

# Twitter hashtags:
twitter_hashtag = r"""
    (?P<tsh>
      \#+
      [\w_]+
      [\w\'_\-]*
      [\w_]+
    )"""

web_adress = r"""
    (?P<adr>
      \b[\w\-/]+\.com\b
      |
      \b[\w\-/]+\.org\b
      |
      \b[\w\-/]+\.net\b
      |
      \b[\w\-/]+\.de\b
    )"""

newline = r"""(?P<newline>\n)"""

enforce_question = r"""(?P<que>[\?][\?\s]*[\?](?![\!]))"""
enforce_exclamation = r"""(?P<exc>[\!][\!\s]*[\!](?![\?]))"""
enforce_confusion = r"""(?P<cfd>(?:\![\!\.\s]*\?+[\!\?\.\s]*)|(?:[\?][\?\.\s]*[\!]+[!\?\.\s]*))"""
enforce_dots = r"""(?P<dts>[\.](?:\s*[\.]){1,})"""

words = r"""
    (?P<word>
      \b\w+(?=n't\b)             # Prefix of words ending with n't
      |
      \b\w+(?='[a-z]{0,2}\b)\b     # Prefix of words ending with 've, ', 'm 'am 'll etc
      |
      n't\b                      # Suffix n't
      |
      '[a-z]{0,2}\b              # Suffixes like 've, ', 'am, 'll
      |
      \w(?:[\-/_]?\w+)*  # Normal words without apostrophes
      |
      ["'´\(\[\{\-]                       # Any other cipher except whitespace !, ?, ., ,, etc
      |
      [\-\}\]\)"'´]
      |
      [\.,:;\?\!\$]
    )"""


class ITokenizer(object):
    """ Tokenizer Interface """
    def __init__(self):
        pass

    """ A tokenizer takes a character string and returns a list of tokens split by whitespaces. """
    def normalize(self, str):
        raise NotImplementedError("This is an interface method. Implement it in subclass.")


class Tokenizer(ITokenizer):
    def __init__(self, preserve_case=True):
        super(Tokenizer, self).__init__()
        self.preserve_case = preserve_case
        # The components of the tokenizer
        self.regex_strings = (
            newline,
            # Emoticons
            emoticon_happy,
            emoticon_sad,
            emoticon_cheeky,
            emoticon_shock,
            emoticon_undecided,
            emoticon_other,
            emoticon_asterisk,
            # Twitter
            twitter_hashtag,
            twitter_uname,
            # Numbers
            number,
            # Enforcement and Repetition
            enforce_question,
            enforce_exclamation,
            enforce_confusion,
            enforce_dots,
            # Webadress
            web_adress,

            # Words
            words
        )
        self.word_re = re.compile(r"""(%s)""" % "|".join(self.regex_strings), re.VERBOSE | re.I | re.UNICODE)

    def normalize(self, s):
        if not self.preserve_case:
            s = s.lower()
        words = []

        for tok in self.word_re.finditer(s):
            tokType = tok.groupdict()
            for key in tokType:
                if tokType[key] is not None:
                    if key == "word":
                        words.append(tokType[key])
                    else:
                        words.append('*'+key+'*')

        return words


class WikiExtractorTokenizer(ITokenizer):
    def __init__(self, preserve_case=True):
        super(WikiExtractorTokenizer, self).__init__()
        self.preserve_case = preserve_case
        # The components of the tokenizer
        self.regex_strings = (
            #newline,
            # Emoticons
            #emoticon_happy,
            #emoticon_sad,
            #emoticon_cheeky,
            #emoticon_shock,
            #emoticon_undecided,
            #emoticon_other,
            #emoticon_asterisk,
            # Twitter
            #twitter_hashtag,
            #twitter_uname,
            # Numbers
            number,
            # Enforcement and Repetition
            enforce_question,
            enforce_exclamation,
            enforce_confusion,
            enforce_dots,
            # Webadress
            web_adress,

            # Words
            words
        )
        self.word_re = re.compile(r"""(%s)""" % "|".join(self.regex_strings), re.VERBOSE | re.I | re.UNICODE)

    def normalize(self, s):

        if not self.preserve_case:
            s = s.lower()

        words = []

        if not s.startswith("<"):
            for tok in self.word_re.finditer(s):
                tokType = tok.groupdict()
                for key in tokType:
                    if tokType[key] is not None:
                        if key == "word":
                            words.append(tokType[key])
                        else:
                            words.append('*'+key+'*')

        return words



###############################################################################

if __name__ == '__main__':

    # tok = Tokenizer(preserve_case=False)
    # list = ["!!!", "????", "....", "!?!??!", ":-)", ":-(", ":)", ":(", "*whatever*", "Oo", "^^", "#machinelearning", "@root"]
    # list2 = ["i'm", "you're", "she's", "i've", "we'd", "don't", "Ben's house", "non-linear", "in     di   vid ual"]
    # sample = "Anyway, why didn't they remove that!?!??:/"
    #
    # for (u, n) in zip(list2, [tok.normalize(s) for s in list2]):
    #     print "\\texttt{%s} & \\texttt{%s}" % (u, " ".join(n))
    # for (u, n) in zip(list, [tok.normalize(s) for s in list]):
    #     print "\\texttt{%s} & \\texttt{%s}" % (u, " ".join(n))
    #
    # print "\\texttt{%s} & \\texttt{%s}" % (sample, " ".join(tok.normalize(sample)))
    #
    # for s in list2:
    #     print tok.normalize(s)
    # print tok.normalize(sample)
    # for s in list:
    #     print tok.normalize(s)

    tok = WikiExtractorTokenizer(preserve_case=True)

    for l in open("../util/extracted/AA/wiki_00", "r"):
        print tok.normalize(l)