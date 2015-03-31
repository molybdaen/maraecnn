__author__ = 'Johannes'

import sys

import numpy as np
import theano

from LabeledRAETree import LabeledRAETree
from general.config import Config


class RAEPropagation(object):

    def __init__(self, params):

        self.theta = params
        self.cost = 0
        self.num_nodes = 0

        self.gTheta = self.theta.initGradientTheta()
        self.gL = np.zeros((np.shape(params.embeddings)[0], Config.RAE_EMBED_SIZE))

    def reInit(self, newTheta):
        self.theta = newTheta
        self.cost = 0
        self.num_nodes = 0
        for a in self.gTheta:
            a.fill(0.0)
        self.gL.fill(0.0)

    def _getLabelRepresentation(self, labels):
        numDataItems = len(labels)
        labelRep = np.zeros((2, numDataItems))
        for i in xrange(0, numDataItems):
            if labels[i] < 0.0 or labels[i] >= 2:
                print >> sys.stderr, "Error: Bad label datatype. Check 0 <= label < categorySize!"
            else:
                labelRep[int(labels[i]), i] = 1.0
        return labelRep

    def forwardPropagate(self, example, wordsEmbedded, wordLabels, treeLabel):

        sentenceLength = len(example)
        nrInnerNodes = sentenceLength - 1
        tree = LabeledRAETree(sentenceLength, wordsEmbedded, label=treeLabel)
        collapsedSentence = range(0, sentenceLength, 1)

        labelRep = self._getLabelRepresentation(wordLabels)
        innerNodeLabel = np.zeros(1) + treeLabel
        innerNodeLabelRep = self._getLabelRepresentation(innerNodeLabel)
        softmaxInput = np.dot(self.theta.Wlabel, wordsEmbedded) + self.theta.blabel
        (error, predictions, gradient) = self.mySoftmax(softmaxInput, labelRep)

        for i in xrange(0, tree.treeSize):
            currentNode = tree.T[i]
            if i < sentenceLength:
                currentNode.predictionLabels = predictions[:,[i]]
                currentNode.catDelta = (1. - Config.RAE_ALPHA_CAT) * gradient[:,[i]]
                tree.totalScore += (1. - Config.RAE_ALPHA_CAT) * error[i]
            else:
                numComponents = np.shape(wordsEmbedded)[1]

                C1 = np.asarray(wordsEmbedded[:, xrange(0, numComponents-1, 1)])
                C2 = np.asarray(wordsEmbedded[:, xrange(1, numComponents, 1)])

                # leftChildIndices  = [collapsedSentence[a] for a in xrange(0, len(collapsedSentence)-1, 1)]
                # rightChildIndices = [collapsedSentence[b] for b in xrange(1, len(collapsedSentence), 1)]
                # leftChildSubtreeSizes = [tree.getNodes()[a].subtreeSize for a in leftChildIndices]
                # rightChildSubtreeSizes = [tree.getNodes()[b].subtreeSize for b in rightChildIndices]
                # n1 = np.asarray(leftChildSubtreeSizes, dtype=float)
                # n2 = np.asarray(rightChildSubtreeSizes, dtype=float)

                # c1w = (n1 / (n1 + n2))
                # c2w = (n2 / (n1 + n2))

                c1w = 0.5
                c2w = 0.5

                ActivationInput = np.dot(self.theta.Wc1, C1) + np.dot(self.theta.Wc2, C2) + self.theta.bh

                (P, PNorm) = self._tanhNorm(ActivationInput)

                y1Input = np.dot(self.theta.Wr1, PNorm) + self.theta.br1
                y2Input = np.dot(self.theta.Wr2, PNorm) + self.theta.br2

                (Y1, Y1Norm) = self._tanhNorm(y1Input)
                (Y2, Y2Norm) = self._tanhNorm(y2Input)

                Y1MinusC1 = (Y1Norm - C1)
                Y2MinusC2 = (Y2Norm - C2)

                JErec = (c1w * np.sum(Y1MinusC1**2.0, axis=0) + c2w * np.sum(Y2MinusC2**2.0, axis=0))
                J_min = np.min(JErec)
                J_minpos = np.argmin(JErec)

                # alpha * c1w * d(norm(YNorm-C)**2.0)/dYNorm
                # dErec_dY1Norm = Config.RAE_ALPHA_CAT * c1w[:,[J_minpos]] * (2.0 * Y1MinusC1[:,[J_minpos]])
                # dErec_dY2Norm = Config.RAE_ALPHA_CAT * c2w[:,[J_minpos]] * (2.0 * Y2MinusC2[:,[J_minpos]])
                dErec_dY1Norm = Config.RAE_ALPHA_CAT * c1w * (2.0 * Y1MinusC1[:,[J_minpos]])
                dErec_dY2Norm = Config.RAE_ALPHA_CAT * c2w * (2.0 * Y2MinusC2[:,[J_minpos]])

                # alpha * c1w * d(norm(YNorm-C)**2.0)/dC
                dErec_dC1 = - dErec_dY1Norm
                dErec_dC2 = - dErec_dY2Norm

                dY1Norm_dy1 = self._jacTanhNorm(y1Input[:,[J_minpos]])
                dY2Norm_dy2 = self._jacTanhNorm(y2Input[:,[J_minpos]])

                softmaxInput = np.dot(self.theta.Wlabel, PNorm[:,[J_minpos]]) + self.theta.blabel
                (CEErr, predictions, gradient) = self.mySoftmax(softmaxInput, innerNodeLabelRep)

                error = Config.RAE_ALPHA_CAT * J_min + (1. - Config.RAE_ALPHA_CAT) * CEErr[0]

                newParentIndex = i
                newParent = tree.T[newParentIndex]
                newParent.Y1C1 = dErec_dC1
                newParent.Y2C2 = dErec_dC2

                newParent.DeltaOut1 = np.dot(dY1Norm_dy1, dErec_dY1Norm)
                newParent.DeltaOut2 = np.dot(dY2Norm_dy2, dErec_dY2Norm)
                newParent.features = PNorm[:,[J_minpos]]
                newParent.unnormalizedFeatures = P[:,[J_minpos]]
                newParent.gfa = self._jacTanhNorm(ActivationInput[:,[J_minpos]])
                newParent.predictionLabels = predictions
                newParent.catDelta = (1. - Config.RAE_ALPHA_CAT) * gradient
                tree.totalScore += error

                leftChildIndex = collapsedSentence[J_minpos]
                rightChildIndex = collapsedSentence[J_minpos+1]
                leftChild = tree.T[leftChildIndex]
                rightChild = tree.T[rightChildIndex]
                newParent.leftChild = leftChild
                newParent.rightChild = rightChild
                newParent.subtreeSize = leftChild.getSubtreeSize() + rightChild.getSubtreeSize()
                leftChild.parent = newParent
                rightChild.parent = newParent
                tree.structure[newParentIndex] = (leftChildIndex, rightChildIndex)

                wordsEmbedded = np.delete(wordsEmbedded, J_minpos+1, 1)
                wordsEmbedded[:,[J_minpos]] = PNorm[:,[J_minpos]]
                del collapsedSentence[J_minpos+1]
                collapsedSentence[J_minpos] = newParentIndex

        tree.predictionLabel = tree.T[tree.treeSize-1].predictionLabels
        return tree

    def backPropagate(self, example, tree):

        sentenceLength = len(example)

        GL = np.zeros((sentenceLength, Config.RAE_EMBED_SIZE))

        treeStack = []
        root = tree.T[tree.treeSize-1]
        root.parentDelta = np.zeros((Config.RAE_EMBED_SIZE, 1))
        treeStack.append((root, 0, None))

        W = [np.zeros((Config.RAE_EMBED_SIZE, Config.RAE_EMBED_SIZE)), self.theta.Wc1, self.theta.Wc2]
        Y0C0 = np.zeros((Config.RAE_EMBED_SIZE, 1))

        while len(treeStack) > 0:
            (currentNode, leftOrRight, parentNode) = treeStack.pop()
            if parentNode is None:
                YCSelector = [Y0C0, None, None]
            else:
                YCSelector = [Y0C0, parentNode.Y1C1, parentNode.Y2C2]
            NodeW = W[leftOrRight]
            delta = YCSelector[leftOrRight]

            if not currentNode.isLeaf():
                treeStack.append((currentNode.leftChild, 1, currentNode))
                treeStack.append((currentNode.rightChild, 2, currentNode))

                A1Norm = currentNode.features
                ND1 = currentNode.DeltaOut1
                ND2 = currentNode.DeltaOut2
                PD = currentNode.parentDelta
                CD = currentNode.catDelta

                Activation = np.dot(self.theta.Wr1.T, ND1) + np.dot(self.theta.Wr2.T, ND2) + np.dot(NodeW.T, PD) + np.dot(self.theta.Wlabel.T, CD) + delta

                currentDelta = np.dot(currentNode.gfa, Activation)

                currentNode.leftChild.parentDelta = currentDelta
                currentNode.rightChild.parentDelta = currentDelta

                gWc1 = np.dot(currentDelta, currentNode.leftChild.features.T)
                gWc2 = np.dot(currentDelta, currentNode.rightChild.features.T)
                gWr1 = np.dot(ND1, A1Norm.T)
                gWr2 = np.dot(ND2, A1Norm.T)
                gbh = currentDelta
                gbr1 = ND1
                gbr2 = ND2
                gWcat = np.dot(CD, A1Norm.T)
                gbcat = CD

                # accumulate all gradients, then you have the gradients of this example
                self.accumulate([gWc1, gWc2, gWr1, gWr2, gbh, gbr1, gbr2, gWcat, gbcat])

            else:
                self.accumulateCat(np.dot(currentNode.catDelta, currentNode.features.T), currentNode.catDelta)
                GL[currentNode.nodeName] = (np.dot(NodeW.T, currentNode.parentDelta) + np.dot(self.theta.Wlabel.T, currentNode.catDelta) + delta).flatten()
                # GL[currentNode.nodeName] = (np.dot(NodeW.T, currentNode.parentDelta) + delta).flatten()

        self.incrementWordEmbedding(GL, example)

    def forwardSequential(self, example, wordsEmbedded, wordLabels, treeLabel):
        sentenceLength = len(example)
        nrInnerNodes = sentenceLength - 1
        tree = LabeledRAETree(sentenceLength, wordsEmbedded, label=treeLabel)
        collapsedSentence = range(0, sentenceLength, 1)

        labelRep = self._getLabelRepresentation(wordLabels)
        innerNodeLabel = np.zeros(1) + treeLabel
        innerNodeLabelRep = self._getLabelRepresentation(innerNodeLabel)
        softmaxInput = np.dot(self.theta.Wlabel, wordsEmbedded) + self.theta.blabel
        (error, predictions, gradient) = self._soch_Softmax(softmaxInput, labelRep)
        for i in xrange(0, tree.treeSize):
            currentNode = tree.T[i]
            if i < sentenceLength:
                currentNode.predictionLabels = predictions[:,[i]]
                # currentNode.catDelta = (1. - Config.RAE_ALPHA_CAT) * gradient[:,[i]]
                currentNode.catDelta = gradient[:,[i]]
                # tree.totalScore += (1. - Config.RAE_ALPHA_CAT) * error[i]
                tree.totalScore += error[i]
                # tree.ceScore += (1. - Config.RAE_ALPHA_CAT) * error[i]
                tree.ceScore += error[i]
            else:

                numComponents = np.shape(wordsEmbedded)[1]
                C1 = np.asarray(wordsEmbedded[:, [0]])
                C2 = np.asarray(wordsEmbedded[:, [1]])

                n1 = tree.T[collapsedSentence[0]].subtreeSize
                n2 = tree.T[collapsedSentence[1]].subtreeSize
                c1w = (n1 / float(n1 + n2))
                c2w = (n2 / float(n1 + n2))

                ActivationInput = np.dot(self.theta.Wc1, C1) + np.dot(self.theta.Wc2, C2) + self.theta.bh

                (P, PNorm) = self._tanhNorm(ActivationInput)

                y1Input = np.dot(self.theta.Wr1, PNorm) + self.theta.br1
                y2Input = np.dot(self.theta.Wr2, PNorm) + self.theta.br2

                (Y1, Y1Norm) = self._tanhNorm(y1Input)
                (Y2, Y2Norm) = self._tanhNorm(y2Input)

                Y1MinusC1 = (Y1Norm - C1)
                Y2MinusC2 = (Y2Norm - C2)

                JErec = 0.5 * (c1w * np.sum(Y1MinusC1**2.0, axis=0) + c2w * np.sum(Y2MinusC2**2.0, axis=0))
                J_min = JErec[0]

                # alpha * c1w * d(norm(YNorm-C)**2.0)/dYNorm
                # dErec_dY1Norm = Config.RAE_ALPHA_CAT * c1w[:,[J_minpos]] * (2.0 * Y1MinusC1[:,[J_minpos]])
                # dErec_dY2Norm = Config.RAE_ALPHA_CAT * c2w[:,[J_minpos]] * (2.0 * Y2MinusC2[:,[J_minpos]])
                dErec_dY1Norm = Config.RAE_ALPHA_CAT * c1w * Y1MinusC1
                dErec_dY2Norm = Config.RAE_ALPHA_CAT * c2w * Y2MinusC2

                # alpha * c1w * d(norm(YNorm-C)**2.0)/dC
                dErec_dC1 = - dErec_dY1Norm
                dErec_dC2 = - dErec_dY2Norm

                # dY1Norm_dy1 = self._jacTanhNorm(y1Input)
                # dY2Norm_dy2 = self._jacTanhNorm(y2Input)
                dY1Norm_dy1 = self._soch_f_prime(Y1)
                dY2Norm_dy2 = self._soch_f_prime(Y2)

                softmaxInput = np.dot(self.theta.Wlabel, PNorm) + self.theta.blabel
                # (CEErr, predictions, gradient) = self.mySoftmax(softmaxInput, innerNodeLabelRep)
                (CEErr, predictions, gradient) = self._soch_Softmax(softmaxInput, innerNodeLabelRep)

                # error = Config.RAE_ALPHA_CAT * J_min + (1. - Config.RAE_ALPHA_CAT) * CEErr[0]
                error = Config.RAE_ALPHA_CAT * J_min + CEErr[0]
                tree.reScore += Config.RAE_ALPHA_CAT * J_min
                # tree.ceScore += (1. - Config.RAE_ALPHA_CAT) * CEErr[0]
                tree.ceScore += CEErr[0]

                newParentIndex = i
                newParent = tree.T[newParentIndex]
                newParent.Y1C1 = dErec_dC1
                newParent.Y2C2 = dErec_dC2

                newParent.DeltaOut1 = np.dot(dY1Norm_dy1, dErec_dY1Norm)
                newParent.DeltaOut2 = np.dot(dY2Norm_dy2, dErec_dY2Norm)
                newParent.features = PNorm
                newParent.unnormalizedFeatures = P
                # newParent.gfa = self._jacTanhNorm(ActivationInput)
                newParent.gfa = self._soch_f_prime(P)
                newParent.predictionLabels = predictions
                # newParent.catDelta = (1. - Config.RAE_ALPHA_CAT) * gradient
                newParent.catDelta = gradient
                tree.totalScore += error

                leftChildIndex = collapsedSentence[0]
                rightChildIndex = collapsedSentence[1]
                leftChild = tree.T[leftChildIndex]
                rightChild = tree.T[rightChildIndex]
                newParent.leftChild = leftChild
                newParent.rightChild = rightChild
                newParent.subtreeSize = leftChild.getSubtreeSize() + rightChild.getSubtreeSize()
                leftChild.parent = newParent
                rightChild.parent = newParent
                tree.structure[newParentIndex] = (leftChildIndex, rightChildIndex)
                wordsEmbedded = np.delete(wordsEmbedded, 1, 1)
                wordsEmbedded[:,[0]] = PNorm
                del collapsedSentence[1]
                collapsedSentence[0] = newParentIndex

        tree.predictionLabel = tree.T[tree.treeSize-1].predictionLabels
        return tree

    def backwardSequential(self, example, tree):


        sentenceLength = len(example)

        GL = np.zeros((sentenceLength, Config.RAE_EMBED_SIZE))

        treeStack = []
        root = tree.T[tree.treeSize-1]
        root.parentDelta = np.zeros((Config.RAE_EMBED_SIZE, 1))
        treeStack.append((root, 0, None))

        W = [np.zeros((Config.RAE_EMBED_SIZE, Config.RAE_EMBED_SIZE)), self.theta.Wc1, self.theta.Wc2]
        Y0C0 = np.zeros((Config.RAE_EMBED_SIZE, 1))

        while len(treeStack) > 0:
            (currentNode, leftOrRight, parentNode) = treeStack.pop()
            if parentNode is None:
                YCSelector = [Y0C0, None, None]
            else:
                YCSelector = [Y0C0, parentNode.Y1C1, parentNode.Y2C2]
            NodeW = W[leftOrRight]
            delta = YCSelector[leftOrRight]

            if not currentNode.isLeaf():
                treeStack.append((currentNode.leftChild, 1, currentNode))
                treeStack.append((currentNode.rightChild, 2, currentNode))

                A1Norm = currentNode.features
                ND1 = currentNode.DeltaOut1
                ND2 = currentNode.DeltaOut2
                PD = currentNode.parentDelta
                CD = currentNode.catDelta

                Activation = np.dot(self.theta.Wr1.T, ND1) + np.dot(self.theta.Wr2.T, ND2) + np.dot(NodeW.T, PD) + np.dot(self.theta.Wlabel.T, CD) + delta

                currentDelta = np.dot(currentNode.gfa, Activation)

                currentNode.leftChild.parentDelta = currentDelta
                currentNode.rightChild.parentDelta = currentDelta

                gWc1 = np.dot(currentDelta, currentNode.leftChild.features.T)
                gWc2 = np.dot(currentDelta, currentNode.rightChild.features.T)
                gWr1 = np.dot(ND1, A1Norm.T)
                gWr2 = np.dot(ND2, A1Norm.T)
                gbh = currentDelta
                gbr1 = ND1
                gbr2 = ND2
                gWcat = np.dot(CD, A1Norm.T)
                gbcat = CD

                # accumulate all gradients, then you have the gradients of this example
                self.accumulate([gWc1, gWc2, gWr1, gWr2, gbh, gbr1, gbr2, gWcat, gbcat])

            else:
                self.accumulateCat(np.dot(currentNode.catDelta, currentNode.features.T), currentNode.catDelta)
                GL[currentNode.nodeName] = (np.dot(NodeW.T, currentNode.parentDelta) + np.dot(self.theta.Wlabel.T, currentNode.catDelta) + delta).flatten()
                # GL[currentNode.nodeName] = (np.dot(NodeW.T, currentNode.parentDelta) + delta).flatten()
        self.incrementWordEmbedding(GL, example)

    def forwardUnsupervised(self, example, wordsEmbedded):
        sentenceLength = len(example)
        nrInnerNodes = sentenceLength - 1
        tree = LabeledRAETree(sentenceLength, wordsEmbedded, label=None)
        collapsedSentence = range(0, sentenceLength, 1)
        for i in xrange(0, nrInnerNodes):
            numComponents = np.shape(wordsEmbedded)[1]

            C1 = np.asarray(wordsEmbedded[:, range(0, numComponents-1, 1)]).reshape((Config.EMBED_SIZE, numComponents-1))
            C2 = np.asarray(wordsEmbedded[:, range(1, numComponents, 1)]).reshape((Config.EMBED_SIZE, numComponents-1))

            n1 = tree.T[collapsedSentence[0]].subtreeSize
            n2 = tree.T[collapsedSentence[1]].subtreeSize
            c1w = (n1 / float(n1 + n2))
            c2w = (n2 / float(n1 + n2))

            ActivationInput = np.dot(self.theta.Wc1, C1) + np.dot(self.theta.Wc2, C2) + self.theta.bh

            (P, PNorm) = self._tanhNorm(ActivationInput)

            y1Input = np.dot(self.theta.Wr1, PNorm) + self.theta.br1
            y2Input = np.dot(self.theta.Wr2, PNorm) + self.theta.br2

            (Y1, Y1Norm) = self._tanhNorm(y1Input)
            (Y2, Y2Norm) = self._tanhNorm(y2Input)

            Y1MinusC1 = (Y1Norm - C1)
            Y2MinusC2 = (Y2Norm - C2)

            JErec = 0.5 * ((c1w * np.sum(Y1MinusC1**2.0, axis=0) + c2w * np.sum(Y2MinusC2**2.0, axis=0)))
            J_min = np.min(JErec)
            J_minpos = np.argmin(JErec)

            dErec_dY1Norm = Config.RAE_ALPHA_CAT * c1w * Y1MinusC1[:,[J_minpos]]
            dErec_dY2Norm = Config.RAE_ALPHA_CAT * c2w * Y2MinusC2[:,[J_minpos]]

            # alpha * c1w * d(norm(YNorm-C)**2.0)/dC
            dErec_dC1 =  - dErec_dY1Norm
            dErec_dC2 =  - dErec_dY2Norm

            dY1Norm_dy1 = self._soch_f_prime(Y1[:,[J_minpos]])
            dY2Norm_dy2 = self._soch_f_prime(Y2[:,[J_minpos]])

            error = J_min

            newParentIndex = sentenceLength + i
            newParent = tree.T[newParentIndex]
            newParent.Y1C1 = dErec_dC1
            newParent.Y2C2 = dErec_dC2

            newParent.DeltaOut1 = np.dot(dY1Norm_dy1, dErec_dY1Norm)
            newParent.DeltaOut2 = np.dot(dY2Norm_dy2, dErec_dY2Norm)
            newParent.features = PNorm[:,[J_minpos]]
            newParent.unnormalizedFeatures = P[:,[J_minpos]]
            newParent.gfa = self._soch_f_prime(P[:,[J_minpos]])
            tree.totalScore += error

            leftChildIndex = collapsedSentence[J_minpos]
            rightChildIndex = collapsedSentence[J_minpos+1]
            leftChild = tree.T[leftChildIndex]
            rightChild = tree.T[rightChildIndex]
            newParent.leftChild = leftChild
            newParent.rightChild = rightChild
            newParent.subtreeSize = leftChild.getSubtreeSize() + rightChild.getSubtreeSize()
            leftChild.parent = newParent
            rightChild.parent = newParent
            tree.structure[newParentIndex] = (leftChildIndex, rightChildIndex)

            wordsEmbedded = np.delete(wordsEmbedded, J_minpos+1, 1)
            wordsEmbedded[:,[J_minpos]] = PNorm[:,[J_minpos]]
            del collapsedSentence[J_minpos+1]
            collapsedSentence[J_minpos] = newParentIndex
        return tree

    def forwardSupervised(self, example, wordsEmbedded, wordLabels, treeLabel, tree):
        sentenceLength = len(example)
        labelRep = self._getLabelRepresentation(wordLabels)
        innerNodeLabel = np.zeros(1) + treeLabel
        innerNodeLabelRep = self._getLabelRepresentation(innerNodeLabel)
        softmaxInput = np.dot(self.theta.Wlabel, wordsEmbedded) + self.theta.blabel
        (error, predictions, gradient) = self._soch_Softmax(softmaxInput, labelRep)

        for i in xrange(0, tree.treeSize):
            currentNode = tree.T[i]
            if i < sentenceLength:
                currentNode.predictionLabels = predictions[:,[i]]
                # currentNode.catDelta = (1. - Config.RAE_ALPHA_CAT) * gradient[:,[i]]
                currentNode.catDelta = gradient[:,[i]]
                # tree.totalScore += (1. - Config.RAE_ALPHA_CAT) * error[i]
                tree.ceScore += error[i]
            else:
                leftChildIndex = tree.structure[i][0]
                rightChildIndex = tree.structure[i][1]
                leftChild = tree.T[leftChildIndex]
                rightChild = tree.T[rightChildIndex]
                C1 = leftChild.features
                C2 = rightChild.features

                currentNode.leftChild = leftChild
                currentNode.rightChild = rightChild
                currentNode.leftChild.parent = currentNode
                currentNode.rightChild.parent = currentNode

                ActivationInput = np.dot(self.theta.Wc1, C1) + np.dot(self.theta.Wc2, C2) + self.theta.bh

                (P, PNorm) = self._tanhNorm(ActivationInput)

                currentNode.unnormalizedFeatures = P
                currentNode.features = PNorm

                softmaxInput = np.dot(self.theta.Wlabel, PNorm) + self.theta.blabel
                (CEErr, predictions, gradient) = self._soch_Softmax(softmaxInput, innerNodeLabelRep)

                currentNode.predictionLabels = predictions
                # Beta constant? why?
                # beta = 0.5
                # currentNode.catDelta = (1. - Config.RAE_ALPHA_CAT) * beta * gradient
                # tree.totalScore += (1. - Config.RAE_ALPHA_CAT) * beta * CEErr[0]
                currentNode.catDelta = gradient
                tree.ceScore += CEErr[0]

        tree.predictionLabel = tree.T[tree.treeSize-1].predictionLabels
        return tree

    def backpropUnsupervised(self, example, tree):

        sentenceLength = len(example)

        GL = np.zeros((sentenceLength, Config.EMBED_SIZE))

        treeStack = []
        root = tree.T[tree.treeSize-1]
        root.parentDelta = np.zeros((Config.EMBED_SIZE, 1))
        treeStack.append((root, 0, None))

        W = [np.zeros((Config.EMBED_SIZE, Config.EMBED_SIZE)), self.theta.Wc1, self.theta.Wc2]
        Y0C0 = np.zeros((Config.EMBED_SIZE, 1))

        while len(treeStack) > 0:
            (currentNode, leftOrRight, parentNode) = treeStack.pop()
            if parentNode is None:
                YCSelector = [Y0C0, None, None]
            else:
                YCSelector = [Y0C0, parentNode.Y1C1, parentNode.Y2C2]
            NodeW = W[leftOrRight]
            delta = YCSelector[leftOrRight]

            if not currentNode.isLeaf():
                treeStack.append((currentNode.leftChild, 1, currentNode))
                treeStack.append((currentNode.rightChild, 2, currentNode))

                A1Norm = currentNode.features
                ND1 = currentNode.DeltaOut1
                ND2 = currentNode.DeltaOut2
                PD = currentNode.parentDelta
                CD = currentNode.catDelta

                Activation = np.dot(self.theta.Wr1.T, ND1) + np.dot(self.theta.Wr2.T, ND2) + np.dot(NodeW.T, PD) + delta

                currentDelta = np.dot(currentNode.gfa, Activation)

                currentNode.leftChild.parentDelta = currentDelta
                currentNode.rightChild.parentDelta = currentDelta

                gWc1 = np.dot(currentDelta, currentNode.leftChild.features.T)
                gWc2 = np.dot(currentDelta, currentNode.rightChild.features.T)
                gWr1 = np.dot(ND1, A1Norm.T)
                gWr2 = np.dot(ND2, A1Norm.T)
                gbh = currentDelta
                gbr1 = ND1
                gbr2 = ND2

                # accumulate all gradients, then you have the gradients of this example
                self.accumulate([gWc1, gWc2, gWr1, gWr2, gbh, gbr1, gbr2])

            else:
                GL[currentNode.nodeName] = (np.dot(NodeW.T, currentNode.parentDelta) + delta).flatten()

        self.incrementWordEmbedding(GL, example)

    def backpropSupervised(self, example, tree):

        sentenceLength = len(example)

        GL = np.zeros((sentenceLength, Config.EMBED_SIZE))

        treeStack = []
        root = tree.T[tree.treeSize-1]
        root.parentDelta = np.zeros((Config.EMBED_SIZE, 1))
        treeStack.append((root, 0, None))

        W = [np.zeros((Config.EMBED_SIZE, Config.EMBED_SIZE)), self.theta.Wc1, self.theta.Wc2]
        Y0C0 = np.zeros((Config.EMBED_SIZE, 1))

        while len(treeStack) > 0:
            (currentNode, leftOrRight, parentNode) = treeStack.pop()
            if parentNode is None:
                YCSelector = [Y0C0, None, None]
            else:
                YCSelector = [Y0C0, parentNode.Y1C1, parentNode.Y2C2]
            NodeW = W[leftOrRight]
            delta = YCSelector[leftOrRight]

            if not currentNode.isLeaf():
                treeStack.append((currentNode.leftChild, 1, currentNode))
                treeStack.append((currentNode.rightChild, 2, currentNode))

                A1Norm = currentNode.features
                ND1 = currentNode.DeltaOut1
                ND2 = currentNode.DeltaOut2
                PD = currentNode.parentDelta
                CD = currentNode.catDelta

                Activation = np.dot(self.theta.Wr1.T, ND1) + np.dot(self.theta.Wr2.T, ND2) + np.dot(NodeW.T, PD) + np.dot(self.theta.Wlabel.T, CD) + delta

                currentDelta = np.dot(currentNode.gfa, Activation)

                currentNode.leftChild.parentDelta = currentDelta
                currentNode.rightChild.parentDelta = currentDelta

                gWc1 = np.dot(currentDelta, currentNode.leftChild.features.T)
                gWc2 = np.dot(currentDelta, currentNode.rightChild.features.T)
                gWr1 = np.dot(ND1, A1Norm.T)
                gWr2 = np.dot(ND2, A1Norm.T)
                gbh = currentDelta
                gbr1 = ND1
                gbr2 = ND2
                gWcat = np.dot(CD, A1Norm.T)
                gbcat = CD

                # accumulate all gradients, then you have the gradients of this example
                self.accumulate([gWc1, gWc2, gWr1, gWr2, gbh, gbr1, gbr2, gWcat, gbcat])

            else:
                self.accumulateCat(np.dot(currentNode.catDelta, currentNode.features.T), currentNode.catDelta)
                GL[currentNode.nodeName] = (np.dot(NodeW.T, currentNode.parentDelta) + np.dot(self.theta.Wlabel.T, currentNode.catDelta) + delta).flatten()

        self.incrementWordEmbedding(GL, example)

    def predict(self, example, wordsEmbedded):
        # print "predicting: "
        # print example
        sentenceLength = len(example)
        nrInnerNodes = sentenceLength - 1
        tree = LabeledRAETree(sentenceLength, wordsEmbedded)
        collapsedSentence = range(0, sentenceLength, 1)

        for i in xrange(sentenceLength, sentenceLength+nrInnerNodes):
            numComponents = np.shape(wordsEmbedded)[1]
            C1 = wordsEmbedded[:, xrange(0, numComponents-1, 1)]
            C2 = wordsEmbedded[:, xrange(1, numComponents, 1)]

            leftChildIndices  = [collapsedSentence[a] for a in xrange(0, len(collapsedSentence)-1, 1)]
            rightChildIndices = [collapsedSentence[b] for b in xrange(1, len(collapsedSentence), 1)]
            leftChildSubtreeSizes = [tree.getNodes()[a].subtreeSize for a in leftChildIndices]
            rightChildSubtreeSizes = [tree.getNodes()[b].subtreeSize for b in rightChildIndices]
            n1 = np.asarray(leftChildSubtreeSizes, dtype=theano.config.floatX)
            n2 = np.asarray(rightChildSubtreeSizes, dtype=theano.config.floatX)

            c1w = (n1 / (n1 + n2))
            c2w = (n2 / (n1 + n2))
            r = self.raeGraph.predict(C1, c1w, C2, c2w)

            (minErrIdx, error, normedParent, parent, prediction) = r
            # print "node %d:" % i
            # print prediction
            tree.totalScore += error
            newParentIndex = i
            newParent = tree.T[newParentIndex]
            newParent.predictionLabels = np.asarray(prediction).reshape((self.raeGraph.params.categorySize, 1))
            newParent.features = np.asarray(normedParent).reshape((Config.EMBED_SIZE,1))
            newParent.unnormalizedFeatures = np.asarray(parent).reshape((Config.EMBED_SIZE,1))

            leftChildIndex = collapsedSentence[minErrIdx]
            rightChildIndex = collapsedSentence[minErrIdx+1]
            leftChild = tree.T[leftChildIndex]
            rightChild = tree.T[rightChildIndex]
            newParent.leftChild = leftChild
            newParent.rightChild = rightChild
            newParent.subtreeSize = leftChild.getSubtreeSize() + rightChild.getSubtreeSize()
            leftChild.parent = newParent
            rightChild.parent = newParent
            tree.structure[newParentIndex] = (leftChildIndex, rightChildIndex)

            wordsEmbedded = np.delete(wordsEmbedded, minErrIdx+1, 1)
            wordsEmbedded[:,minErrIdx] = normedParent
            del collapsedSentence[minErrIdx+1]
            collapsedSentence[minErrIdx] = newParentIndex

        tree.predictionLabel = tree.T[tree.treeSize-1].predictionLabels
        return tree

    def _columnWiseNormalize(self, mat):
        return mat / np.sum(mat**2.0, axis=0)**(1./2.)

    def _tanhNorm(self, mat):
        tanh = np.tanh(mat)
        normTanh = np.sqrt(np.sum(tanh**2.0, axis=0))
        return (tanh, (tanh/normTanh))

    def _jacTanhNorm(self, v):
        size = np.shape(v)[0]
        tanh = np.tanh(v)
        normTanh = np.sqrt(np.sum(tanh**2.0, axis=0))
        sech2 = 1.0 - (tanh**2.0)#(1.0 / np.cosh(v))**2.0
        dtanh_v = np.diag(sech2.flatten())
        a = ((1. / normTanh) * np.identity(size))
        b = ((1. / (normTanh**3.0)) * np.dot(tanh, tanh.T))
        df_v = np.dot(dtanh_v, (a - b))
        return df_v

    def mySoftmax(self, x, labels):
        e = np.exp(x)
        es = np.sum(e, axis=0)
        pred = e / es
        cEErr = -np.sum((np.log(pred) * labels), axis=0)
        return (cEErr, pred, (pred-labels))

    def _soch_Softmax(self, x, labels):
        beta = 0.5
        sm = 1.0 / (1.0 + np.exp(-x))
        sm_prime = sm * (1.0 - sm)
        lbl_sm = beta * (1.0 - Config.RAE_ALPHA_CAT) * (labels - sm);
        grad = -lbl_sm * sm_prime
        J = 0.5 * np.sum(lbl_sm * (labels - sm), axis=0)
        return (J, sm, grad)

    def _soch_f_prime(self, x):
        # x = tanh(x)./norm(tanh(x))
        nrm = np.sqrt(np.sum(x**2.0, axis=0))
        y = (x - x**3.0)
        out = (np.diag((1.0 - x**2.0).flatten()) / nrm) - (np.dot(y, x.T) / nrm**3.0);
        return out

    def accumulate(self, *gradients):
        for i in xrange(len(gradients[0])):
            self.gTheta[i] += gradients[0][i]

    def accumulateCat(self, gWcat, gbcat):
        self.gTheta[-2] += gWcat
        self.gTheta[-1] += gbcat

    def incrementWordEmbedding(self, embeddings, wordIndices):
        for (sentIdx, wIdx) in enumerate(wordIndices):
            self.gL[wIdx] += embeddings[sentIdx]

    def getGradient(self):
        return (self.gTheta, self.gL)
