__author__ = 'johannesjurgovsky'

from gensim.models.word2vec import Word2Vec
import numpy as np
import cPickle
from os import path
from gensim import matutils
import mwparserfromhell


def getEmbeddingsAndVocab(w2vModelFilename, rebuild=False):
    if path.exists(w2vModelFilename):
        p, f = path.split(w2vModelFilename)
        fName = f.split('.')[0]
        matFile = path.join(p, fName + "-mat.npy")
        vocFile = path.join(p, fName + "-voc.pkl")
        if not path.exists(matFile) or not path.exists(vocFile):
            model = Word2Vec.load_word2vec_format(w2vModelFilename, binary=False)
            np.save(matFile, model.syn0)
            cPickle.dump(model.vocab, open(vocFile, "w"))
        m = np.load(matFile)
        v = cPickle.load(open(vocFile, "r"))
        return m, v


if __name__ == "__main__":

    # m, v = getEmbeddingsAndVocab("../../data/language/corpora/wikientities/wikientitymodel.seq", rebuild=True)
    # print m.shape
    # print len(v)
    # print np.dot(matutils.unitvec(m[v["World_War_I"].index]), matutils.unitvec(m[v["World_War_II"].index]))

    text = "I has a template! {{foo|bar|baz|eggs=spam}} [[World_War_I]] See it?"
    text2 = r"<page> <title>AdA</title> <id>11</id> <revision> <id>15898946</id> <timestamp>2002-09-22T16:02:58Z</timestamp> <contributor> <username>Andre Engels</username> <id>300</id> </contributor> <minor /> <text xml:space='preserve'>#REDIRECT [[Ada programming language]]</text> </revision> </page> <page> <title>Anarchism</title> <id>12</id> <revision> <id>42136831</id> <timestamp>2006-03-04T01:41:25Z</timestamp> <contributor> <username>CJames745</username> <id>832382</id> </contributor><minor /> <comment>/* Anarchist Communism */  too many brackets</comment> <text xml:space='preserve'>{{Anarchism}}; '''Anarchism''' originated as a term of abuse first used against early [[working class]] [[radical]]s including the [[Diggers]] of the [[English Revolution]] an d the [[sans-culotte|''sans-culottes'']] of the [[French Revolution]].[http://uk.encarta.msn.com/encyclopedia_761568770/Anarchism.html] Whilst the term is still; used in a pejorative way to describe ''&quot;any act that used violent means to; destroy the organization of society&quot;''&lt;ref&gt;[http://www.cas.sc.edu/so; cy/faculty/deflem/zhistorintpolency.html History of International Police Coopera; tion], from the final protocols of the &quot;International Conference of Rome fo; r the Social Defense Against Anarchists&quot;, 1898&lt;/ref&gt;, it has also been taken up as a positive label by self-defined anarchists.The word '''anarchism''' is [[etymology|derived from]] the [[Greek language|Gree; k]] ''[[Wiktionary:&amp;#945;&amp;#957;&amp;#945;&amp;#961;&amp;#967;&amp;#943;& amp;#945;|&amp;#945;&amp;#957;&amp;#945;&amp;#961;&amp;#967;&amp;#943;&amp;#945;]]'' (&quot;without [[archon]]s (ruler, chief, king)&quot;). Anarchism as a [[political philosophy]], is the belief that ''rulers'' are"
    wikicode = mwparserfromhell.parse(text2)
    print wikicode.strip_code()
    print wikicode