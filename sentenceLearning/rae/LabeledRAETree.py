__author__ = 'Johannes'

import sys
from RAENode import RAENode


class LabeledRAETree(object):

    def __init__(self, sentenceLength, wordsEmbedded, label=None):
        # The tree as list of its constituent nodes
        self.T = []
        # A list of (int, int)-pairs keeping track of all nodes' left and right children.
        # A tuple (a, b) at index i indicates that node i has node a as left child and node b as right child.
        self.structure = []
        self.sentenceLength = sentenceLength
        self.treeSize = 2 * sentenceLength - 1
        self.treeSizeInternal = sentenceLength - 1
        self.targetLabel = label
        self.predictionLabel = None
        self.totalScore = 0
        self.ceScore = 0
        self.reScore = 0

        for i in range(self.treeSize):
            self.T.append(RAENode(nodeIndex=i, sentenceLength=sentenceLength, wordsEmbedded=wordsEmbedded))
            self.structure.append((-1, -1))

    def getNodes(self):
        return self.T

    def getLabel(self):
        return self.label

    def printy(self):
        print "treeSize: %d" % self.treeSize
        print "treeNodes:"
        for n in self.T:
            n.printy()
        print "structure:"
        for s in self.structure:
            print s

    def getStructureString(self):
        parents = [-1] * self.treeSize
        for i in xrange(self.treeSize-1, -1, -1):
            lc = self.structure[i][0]
            rc = self.structure[i][1]
            if lc != -1 and rc != -1:
                if parents[lc] != -1 or parents[rc] != -1:
                    print >> sys.stderr, "TreeStructure is messed up!"
                parents[lc] = i
                parents[rc] = i
        return parents