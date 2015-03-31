#!/usr/bin/env python

import string, numpy, gzip
from textSNE.tsneOrig import tsne

o = gzip.open("testdata/english-embeddings.turian.txt.gz", "rb")
titles, x = [], []
for l in o:
    toks = string.split(l)
    titles.append(toks[0])
    x.append([float(f) for f in toks[1:]])
x = numpy.array(x)
print x.shape

#from tsne import tsne
#out = tsne(x, no_dims=2, perplexity=30, initial_dims=30, USE_PCA=False)
#out = tsne(x, no_dims=2, perplexity=30, initial_dims=30, use_pca=False)
out = tsne(x, no_dims=2, perplexity=30, initial_dims=50, use_pca=False)

import render
render.render([(title, point[0], point[1]) for title, point in zip(titles, out)], "test-output.rendered.png", width=3000, height=1800)
