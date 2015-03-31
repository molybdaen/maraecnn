__author__ = 'Johannes'

import sys

import numpy as np

from general.config import Config


class RAENode(object):

    def __init__(self, nodeIndex, sentenceLength, wordsEmbedded):

        self.nodeName = nodeIndex
        self.subtreeSize = 1

        self.parent = None
        self.leftChild = None
        self.rightChild = None

        self.features = np.zeros((Config.RAE_EMBED_SIZE, 1))
        self.unnormalizedFeatures = np.zeros((Config.RAE_EMBED_SIZE, 1))

        # The delta coming from the upper node. To be calculated during backpropagation while traversing the tree downwards.
        # LaTeX: \delta_q
        self.parentDelta = None

        # The change of the cost function with respect to the inputs to reconstruction units. Separately for left and right child reconstruction.
        # LaTeX: \gamma = \frac{\partial{J}}{\partial{e}}
        self.DeltaOut1 = None
        self.DeltaOut2 = None

        # The change of the cost function with respect to the inputs to the softmax layer.
        # LaTeX: \zeta = \frac{\partial{J}}{\partial{g}}
        self.catDelta = None

        # Softmax output P(y_i | x)
        self.predictionLabels = None

        # The change of the hidden layer non-linearity function with respect to its inputs.
        # LaTeX: \frac{\partial{f}}{\partial{a}}
        self.gfa = None

        # The change of the reconstruction error with respect to the original input vectors.
        # The hidden layer activation of the Autoencoder can either be the compositional representation of two inputs (influencing the overall cost function via this combined representation)
        # or by itself one of two inputs to the layer above (influencing the overall cost function by serving as inputs).
        self.Y1C1 = None
        self.Y2C2 = None

        if nodeIndex < sentenceLength:
            self.features = np.asarray(wordsEmbedded[:,nodeIndex]).reshape((Config.RAE_EMBED_SIZE,1))
            self.unnormalizedFeatures = np.asarray(wordsEmbedded[:,nodeIndex]).reshape((Config.RAE_EMBED_SIZE,1))

    def isLeaf(self):
        if self.leftChild is None and self.rightChild is None:
            return True
        if self.leftChild is not None and self.rightChild is not None:
            return False
        print >> sys.stderr, "Broken tree, node has one child %d" % self.nodeName
        return False

    def getSubtreeSize(self):
        subtreeSize = 0
        if self.isLeaf():
            subtreeSize = 1
        else:
            if isinstance(self.leftChild, RAENode):
                subtreeSize += self.leftChild.subtreeSize
            if isinstance(self.rightChild, RAENode):
                subtreeSize += self.rightChild.subtreeSize
        return subtreeSize

    def getSubtreeWordIndices(self):
        subtreeIndices = []
        if self.isLeaf():
            subtreeIndices.append(self.nodeName)
        else:
            if isinstance(self.leftChild, RAENode) and isinstance(self.rightChild, RAENode):
                subtreeIndices += self.leftChild.getSubtreeWordIndices()
                subtreeIndices += self.rightChild.getSubtreeWordIndices()
            else:
                print >> sys.stderr, "Broken tree, node types not consistent in subtree %d" % self.nodeName
        return subtreeIndices

    def printy(self):
        lc = -1
        rc = -1
        if self.leftChild is not None:
            lc = self.leftChild.nodeName
        if self.rightChild is not None:
            rc = self.rightChild.nodeName
        print "(NN: %d NS: %d LC: %d RC: %d LABEL: %s)" % (self.nodeName, self.subtreeSize, lc, rc, str(self.predictionLabels))

    def getFeatures(self):
        return self.features

    def __str__(self):
        lc = -1
        rc = -1
        if self.leftChild is not None:
            lc = self.leftChild.nodeName
        if self.rightChild is not None:
            rc = self.rightChild.nodeName
        return "(NN: %d NS: %d LC: %d RC: %d LABEL: %s)" % (self.nodeName, self.subtreeSize , lc, rc, str(self.predictionLabels))