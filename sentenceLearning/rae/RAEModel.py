__author__ = 'Johannes'

import cPickle
from multiprocessing import Process, Manager
import time
import logging

import numpy as np
from scipy import optimize

from RAESoftmaxClassifier import RAESoftmaxClassifier
from RAEParameters import Parameters
from RAEPropagation import RAEPropagation
from general.config import Config, ModelEnum
from general.analysis import Accuracy, SignalLogger
from util.movingaverage import MovingAverage

class _par_RAECost():
    def __init__(self, theta):
        self.theta = theta
        self.propagator = RAEPropagation(theta)
        self.cost = 0
        self.num_nodes = 0

    def reInit(self, newTheta):
        self.theta = newTheta
        self.propagator.reInit(newTheta)
        self.cost = 0
        self.num_nodes = 0

    def replicateLabel(self, sequence, label):
        return np.zeros(len(sequence)) + label

    def embed(self, sequence):
        seq = [self.theta.embeddings[s] for s in sequence]
        st = np.vstack(seq).T
        return st

    def compute(self, datum):
        tree = self.propagator.forwardPropagate(datum[0], self.embed(datum[0]), self.replicateLabel(datum[0], datum[1]), datum[1])
        self.propagator.backPropagate(datum[0], tree)
        self.cost += tree.totalScore
        self.num_nodes += tree.treeSize
        return tree

    def get_par_CostnGrad(self):
        (gTheta, gL) = self.propagator.getGradient()
        return (self.cost, self.num_nodes, gTheta, gL)


def par_compute(data, costObj, resultQueue):
    print len(data)
    for (idx, seq) in enumerate(data):
        costObj.compute(seq)
    (cost, num_nodes, grad, Lgrad) = costObj.get_par_CostnGrad()
    resultQueue.append((cost, num_nodes, grad, Lgrad))


class RAEModel(object):

    def __init__(self, dictionary, categorySize, BFGS_iter=130, theta=None):

        self.dictionary = dictionary
        self.categorySize = categorySize
        if theta is not None:
            self.parameters = theta
        else:
            self.parameters = Parameters(embeddingMatrix=dictionary.embeddings, catSize=categorySize)
        self.bfgsIter = BFGS_iter
        self.propagation = RAEPropagation(self.parameters)
        self.raeCost = self.RAECost(self, self.parameters)
        self.count = 0

    def embed(self, sequence):
        seq = [self.parameters.embeddings[s] for s in sequence]
        return np.vstack(seq).T

    def unbed(self, sequence, gradientWords):
        for (idx, widx) in enumerate(sequence):
            self.parameters.embeddings[widx] -= 0.5 * gradientWords[idx]

    def replicateLabel(self, sequence, label):
        return np.zeros(len(sequence)) + label

    def predict(self, sequenceDataBatch):
        trees = []
        for seq in sequenceDataBatch:
            tree = self.propagation.forwardUnsupervised(seq[0], self.embed(seq[0]))
            trees.append(tree)
        return trees

    def __getstate__(self):
        return ((self.dictionary, self.categorySize, self.parameters, self.bfgsIter, self.count),)

    def __setstate__(self, state):
        (self.dictionary, self.categorySize, self.parameters, self.bfgsIter, self.count) = state[0]
        self.propagation = RAEPropagation(self.parameters)
        self.raeCost = self.RAECost(self, self.parameters)

    def learnFeatures(self, trainData):
        logging.info("PROGRESS: Starting minBFGS")
        costComputer = self.buildCost(trainData)
        iniTheta = costComputer.iniTheta.pack()
        (x, f, d) = optimize.fmin_l_bfgs_b(costComputer.computeCost, iniTheta, fprime=costComputer.computePrime, maxfun=self.bfgsIter)
        iniTheta = x

        finalTheta = Parameters(initTheta=iniTheta, catSize=self.categorySize)
        self.parameters = finalTheta
        self.propagation = RAEPropagation(finalTheta)
        return costComputer.signals

    def extractFeatures(self, data):
        reprs = []
        preds = []
        labels = []
        for (i, d) in enumerate(data):
            tree = self._predict(d)
            repr = tree.T[tree.treeSize-1].features
            pred = tree.predictionLabel
            reprs.append(repr)
            preds.append(int(np.argmax(pred.flatten())))
            labels.append(int(d[1]))
        return reprs, preds, labels

    def extractFeatures_single(self, wordSequence):
        wordIndices = [self.dictionary.vocab.id(word) for word in wordSequence]
        tree = self._predict([wordIndices, 0.0])
        (repr, pred) = (tree.T[tree.treeSize-1].features, tree.predictionLabel)
        labelPred = int(np.argmax(pred.flatten()))
        return repr, pred, labelPred

    class RAECost(object):
        def __init__(self, rae, theta):
            self.rae = rae
            self.theta = theta
            self.propagator = RAEPropagation(theta)
            self.cost = 0
            self.reCost = 0
            self.ceCost = 0
            self.num_nodes = 0
            self.reNodes = 0
            self.ceNodes = 0

        def reInit(self, newTheta):
            self.theta = newTheta
            self.propagator.reInit(newTheta)
            self.cost = 0
            self.num_nodes = 0

        def embed(self, sequence):
            seq = [self.theta.embeddings[s] for s in sequence]
            st = np.vstack(seq).T
            return st

        def compute(self, datum):
            tree = self.propagator.forwardPropagate(datum[0], self.embed(datum[0]), self.rae.replicateLabel(datum[0], datum[1]), datum[1])
            self.propagator.backPropagate(datum[0], tree)
            self.cost += tree.totalScore
            self.num_nodes += tree.treeSize
            return tree

        def getCostnGrad(self):
            WNormConst = (self.theta.Wc1**2.0).sum() + (self.theta.Wc2**2.0).sum()
            WNormReconst = (self.theta.Wr1**2.0).sum() + (self.theta.Wr2**2.0).sum()
            totalCost = (1.0 / (self.num_nodes)) * (self.cost\
                                 + 0.5 * Config.RAE_LAMBDA_W * WNormConst \
                                 + 0.5 * Config.RAE_LAMBDA_W * WNormReconst \
                                 + 0.5 * Config.RAE_LAMBDA_L * (self.theta.embeddings**2.0).sum() \
                                 + 0.5 * Config.RAE_LAMBDA_CAT * (self.theta.Wlabel**2.0).sum())
            (gTheta, gL) = self.propagator.getGradient()
            currentTheta = self.theta.getThetaAsList()
            fGradTheta = [None] * 9
            fGradTheta[0] = (1.0 / (self.num_nodes)) * (gTheta[0] + Config.RAE_LAMBDA_W * currentTheta[0])
            fGradTheta[1] = (1.0 / (self.num_nodes)) * (gTheta[1] + Config.RAE_LAMBDA_W * currentTheta[1])
            fGradTheta[2] = (1.0 / (self.num_nodes)) * (gTheta[2] + Config.RAE_LAMBDA_W * currentTheta[2])
            fGradTheta[3] = (1.0 / (self.num_nodes)) * (gTheta[3] + Config.RAE_LAMBDA_W * currentTheta[3])
            fGradTheta[4] = (1.0 / (self.num_nodes)) * gTheta[4]
            fGradTheta[5] = (1.0 / (self.num_nodes)) * gTheta[5]
            fGradTheta[6] = (1.0 / (self.num_nodes)) * gTheta[6]
            fGradTheta[7] = (1.0 / (self.num_nodes)) * (gTheta[7] + Config.RAE_LAMBDA_CAT * currentTheta[7])
            fGradTheta[8] = (1.0 / (self.num_nodes)) * gTheta[8]
            fGradL = (1.0 / (self.num_nodes)) * (gL + Config.RAE_LAMBDA_L * self.theta.embeddings)
            return (totalCost, fGradTheta, fGradL)

        def get_par_CostnGrad(self):
            (gTheta, gL) = self.propagator.getGradient()
            return (self.cost, self.num_nodes, gTheta, gL)

    class RAEFeatureCost(object):
        def __init__(self, rae, theta):
            self.rae = rae
            self.theta = theta
            self.propagator = RAEPropagation(theta)
            self.cost = 0
            self.num_nodes = 0

        def reInit(self, newTheta):
            self.theta = newTheta
            self.propagator.reInit(newTheta)
            self.cost = 0
            self.num_nodes = 0

        def compute(self, datum):
            # pass example forward through network. Build tree via Reconstruction Error and calculate deltas for each node.
            tree = self.propagator.forwardUnsupervised(datum[0], self.embed(datum[0]))
            # Take the constructed tree and backprop the Reconstruction error backwards through the tree structure. Calculates Gradients for W-Matrices, Biases, and Embeddings.
            self.propagator.backpropUnsupervised(datum[0], tree)
            # All gradients from this example are accumulated inside self.propagation
            # The reconstruction costs induced by collapsing words into inner nodes add to the total costs of this batch
            self.cost += tree.totalScore
            # In order to properly calculate the average of the gradient of this batch later on, we need to keep track of the number of times we collapsed words into inner nodes. This is 'sentenceLength-1'.
            self.num_nodes += len(datum[0]) - 1
            # Now we built a tree in accordance to the minimum reconstruction error for each inner node. Aaaand, via backpropagation we have already calculated the gradients of the Reconstruction Error with respect to W-, b-, and L-Matrices.
            return tree

        def embed(self, sequence):
            seq = [self.theta.embeddings[s] for s in sequence]
            return np.vstack(seq).T

        def getCostnGrad(self):
            lamdbaW = Config.RAE_ALPHA_CAT * Config.RAE_LAMBDA_W
            lambdaL = Config.RAE_ALPHA_CAT * Config.RAE_LAMBDA_L
            wNormSquared = (self.theta.Wc1**2.0).sum()\
                           + (self.theta.Wc2**2.0).sum()\
                           + (self.theta.Wr1**2.0).sum()\
                           + (self.theta.Wr2**2.0).sum()

            cost = (1.0 / self.num_nodes) * self.cost\
                      + 0.5 * lamdbaW * wNormSquared \
                      + 0.5 * lambdaL * (self.theta.embeddings**2.0).sum()
            (gTheta, gL) = self.propagator.getGradient()
            currentTheta = self.theta.getThetaAsList()
            fGradTheta = [None] * 7
            fGradTheta[0] = (1.0 / self.num_nodes) * gTheta[0] + lamdbaW * currentTheta[0]
            fGradTheta[1] = (1.0 / self.num_nodes) * gTheta[1] + lamdbaW * currentTheta[1]
            fGradTheta[2] = (1.0 / self.num_nodes) * gTheta[2] + lamdbaW * currentTheta[2]
            fGradTheta[3] = (1.0 / self.num_nodes) * gTheta[3] + lamdbaW * currentTheta[3]
            fGradTheta[4] = (1.0 / self.num_nodes) * gTheta[4]
            fGradTheta[5] = (1.0 / self.num_nodes) * gTheta[5]
            fGradTheta[6] = (1.0 / self.num_nodes) * gTheta[6]
            fGradL = (1.0 / self.num_nodes) * gL + lambdaL * self.theta.embeddings
            return (cost, fGradTheta, fGradL)

    def _testExample(self, datum):
        propagator = RAEPropagation(self.parameters)
        tree = propagator.forwardSequential(datum[0], self.embed(datum[0]), self.replicateLabel(datum[0], datum[1]), datum[1])
        propagator.backwardSequential(datum[0], tree)


    class RAEClassificationCost(object):
        def __init__(self, rae, theta):
            self.rae = rae
            self.theta = theta
            self.propagator = RAEPropagation(theta)
            self.cost = 0
            self.num_nodes = 0

        def reInit(self, newTheta):
            self.theta = newTheta
            self.propagator.reInit(newTheta)
            self.cost = 0
            self.num_nodes = 0

        def compute(self, datum, unsupTree):
            # The classification cost comes from the Cross Entropy Error. So, we pass the example forward through the given tree structure and calculate the crossEntError and the corresponding delta values for each node. The given tree gets populated with these delta values.
            supTree = self.propagator.forwardSupervised(datum[0], self.embed(datum[0]), self.replicateLabel(datum[0], datum[1]), datum[1], unsupTree)
            # Take the populated tree and backprop the delta values through the W-, b- and L-Matrices.
            self.propagator.backpropSupervised(datum[0], supTree)
            # The Cross Entropy costs induced by collapsing words into inner nodes add to the total costs of this batch
            self.cost += supTree.ceScore
            # Cross Entropy error had also been calculated for input nodes not only for inner ones. That's why we take into account all nodes of the tree - including leaf nodes.
            self.num_nodes += supTree.treeSize
            return supTree

        def replicateLabel(self, sequence, label):
            return np.zeros(len(sequence)) + label

        def embed(self, sequence):
            seq = [self.theta.embeddings[s] for s in sequence]
            return np.vstack(seq).T

        def getCostnGrad(self):
            lamdbaW = (1.0 - Config.RAE_ALPHA_CAT) * Config.RAE_LAMBDA_W
            lambdaC = (1.0 - Config.RAE_ALPHA_CAT) * Config.RAE_LAMBDA_CAT
            lambdaL = (1.0 - Config.RAE_ALPHA_CAT) * Config.RAE_LAMBDA_L
            wNormSquared = (self.theta.Wc1**2.0).sum()\
                           + (self.theta.Wc2**2.0).sum()\
                           + (self.theta.Wr1**2.0).sum()\
                           + (self.theta.Wr2**2.0).sum()
            cost = (1.0 / self.num_nodes) * self.cost + 0.5 * lamdbaW * wNormSquared \
                                 + 0.5 * lambdaL * (self.theta.embeddings**2.0).sum() \
                                 + 0.5 * lambdaC * (self.theta.Wlabel**2.0).sum()
            (gTheta, gL) = self.propagator.getGradient()
            currentTheta = self.theta.getThetaAsList()
            fGradTheta = [None] * 9
            fGradTheta[0] = (1.0 / self.num_nodes) * gTheta[0] + lamdbaW * currentTheta[0]
            fGradTheta[1] = (1.0 / self.num_nodes) * gTheta[1] + lamdbaW * currentTheta[1]
            fGradTheta[2] = (1.0 / self.num_nodes) * gTheta[2] + lamdbaW * currentTheta[2]
            fGradTheta[3] = (1.0 / self.num_nodes) * gTheta[3] + lamdbaW * currentTheta[3]
            fGradTheta[4] = (1.0 / self.num_nodes) * gTheta[4]
            fGradTheta[5] = (1.0 / self.num_nodes) * gTheta[5]
            fGradTheta[6] = (1.0 / self.num_nodes) * gTheta[6]
            fGradTheta[7] = (1.0 / self.num_nodes) * gTheta[7] + lambdaC * currentTheta[7]
            fGradTheta[8] = (1.0 / self.num_nodes) * gTheta[8]
            fGradL = (1.0 / self.num_nodes) * gL + lambdaL * self.theta.embeddings
            return (cost, fGradTheta, fGradL)

    class CostComputer(object):
        def __init__(self, rae, data):
            self.rae = rae
            self.data = data
            self.iniTheta = Parameters(catSize=self.rae.categorySize)
            self.totalCostFunction = rae.RAECost(rae, self.iniTheta)
            self.prevTheta = None
            self.gradient = None
            self.cost = None
            self.epochCount = 1
            self.batchCount = 1
            self.error = MovingAverage()
            self.signals = SignalLogger("epoch", "batch", "error")

        def prepareEpoch(self, epochNr):
            self.epochCount = epochNr

        def _par_computeCost(self, packedTheta):
            if np.array_equal(packedTheta, self.prevTheta):
                return self.cost

            self.iteration += 1
            nTheta = Parameters(initTheta=packedTheta, catSize=2)
            print "%d: Evaluate J for %d data items." % (self.iteration, len(self.data))

            if self.iteration % 10 == 0:
                cPickle.dump(nTheta, open((r'C:\MA\params_%d.pkl' % self.iteration), "wb"))
            strtTime = time.time()

            # Parallel execution
            nrProcs = 1
            jobs = []
            if len(self.data) < nrProcs:
                nrProcs = len(self.data)
            split = len(self.data) / nrProcs
            manager = Manager()
            resultQueue = manager.list()
            for i in xrange(nrProcs):
                print "For proc %d: startData %d up to endData %d" % (i, split*i, split*(i + 1))
                costObj = _par_RAECost(nTheta)
                p = Process(target=par_compute, args=(self.data[split*i: split*(i + 1)], costObj, resultQueue,))
                jobs.append(p)

            for p in jobs:
                p.start()

            for p in jobs:
                p.join()

            for r in resultQueue:
                (_par_cost, _par_num_nodes, _par_grad, _par_Lgrad)= r
                print "proc %d:"
                print _par_cost
                print _par_num_nodes
            endTime = time.time()
            print "Took Time %.3f seconds" % (endTime-strtTime)

        def computeCost(self, packedTheta):
            if np.array_equal(packedTheta, self.prevTheta):
                return self.cost

            nTheta = Parameters(initTheta=packedTheta, catSize=self.rae.categorySize)

            print "%d: Evaluate J for %d data items." % (self.batchCount, self.data.getSize())

            self.totalCostFunction.reInit(nTheta)
            for seq in self.data:
                self.totalCostFunction.compute(seq)

            (totalCost, totalGrad, totalLGrad) = self.totalCostFunction.getCostnGrad()

            print "TotalCost: %.4f" % totalCost
            self.error.add(totalCost)
            self.signals.add(self.epochCount, self.batchCount, totalCost)
            logging.info("PROGRESS: Trained epoch %d / %d" % (self.batchCount, self.rae.bfgsIter))
            logging.info("STATS: Error: %.5f" % (totalCost))
            packedGrad = np.asarray([])
            for i in xrange(9):
                packedGrad = np.hstack((packedGrad, totalGrad[i].flatten()))
            packedGrad = np.hstack((packedGrad, totalLGrad.flatten()))

            self.cost = totalCost
            self.gradient = packedGrad
            self.prevTheta = np.copy(packedTheta)

            self.epochCount += 1
            self.batchCount += 1

            return totalCost

        def computePrime(self, packedTheta):
            print "Compute prime J"
            if np.array_equal(packedTheta, self.prevTheta):
                print "..already known"
                return self.gradient
            else:
                self.computeCost(packedTheta)
                return self.gradient


    def minibatchGradientStep(self, sequenceDataBatch):

        for (idx, seq) in enumerate(sequenceDataBatch):
            self.raeCost.compute(seq)
        self.count += 100
        print "Trained Examples: %d" % self.count
        (JCost, fGradTheta, fGradL) = self.raeCost.getCostnGrad()
        totalCost = JCost

        currentTheta = self.raeCost.theta
        currentTheta.Wc1 -= Config.RAE_LEARNING_RATE * fGradTheta[0]
        currentTheta.Wc2 -= Config.RAE_LEARNING_RATE * fGradTheta[1]
        currentTheta.Wr1 -= Config.RAE_LEARNING_RATE * fGradTheta[2]
        currentTheta.Wr2 -= Config.RAE_LEARNING_RATE * fGradTheta[3]
        currentTheta.bh -= Config.RAE_LEARNING_RATE * fGradTheta[4]
        currentTheta.br1 -= Config.RAE_LEARNING_RATE * fGradTheta[5]
        currentTheta.br2 -= Config.RAE_LEARNING_RATE * fGradTheta[6]

        currentTheta.Wlabel -= Config.RAE_LABEL_LEARNING_RATE * fGradTheta[7]
        currentTheta.blabel -= Config.RAE_LABEL_LEARNING_RATE * fGradTheta[8]

        currentTheta.embeddings -= Config.RAE_EMB_LEARNING_RATE * fGradL
        self.raeCost.reInit(currentTheta)
        return (totalCost, currentTheta)


    def buildCost(self, data):
        JCost = self.CostComputer(self, data)
        return JCost

    def _predict(self, d):
        tree = self.propagation.forwardPropagate(d[0], self.embed(d[0]), self.replicateLabel(d[0], d[1]), d[1])
        return tree

    def validate(self, data, theta):
        self.parameters = theta
        self.propagation = RAEPropagation(theta)
        predictions = []
        labels = []
        for (idx, d) in enumerate(data):
            if len(d[0]) > 1:
                tree = self._predict(d)
                predictions.append(tree.predictionLabel)
                labels.append(d[1])
        npPred = np.asarray(predictions).reshape((len(predictions), self.parameters.categorySize))
        npLabels = np.asarray(labels, dtype=int).reshape((len(labels),))
        fPredictions = np.argmax(npPred, axis=1)
        return Accuracy(fPredictions, npLabels, 2)

    def validateFC(self, data, theta):
        self.parameters = theta
        self.propagation = RAEPropagation(theta)
        trees = []
        labels = []
        for (idx, d) in enumerate(data):
            if len(d[0]) > 1:
                tree = self._predict(d)
                feats = tree.T[tree.treeSize-1].features
                treeFeats = np.zeros((Config.EMBED_SIZE,1))
                for n in xrange(tree.treeSize):
                    treeFeats += tree.T[n].features
                treeFeats = (1. / tree.treeSize) * treeFeats
                finFeats = np.vstack((feats, treeFeats))
                # finFeats = feats
                labl = None
                if d[1] == 0.0:
                    labl = np.asarray([1.0, 0.0]).reshape((2, 1))
                else:
                    labl = np.asarray([0.0, 1.0]).reshape((2, 1))
                trees.append(finFeats)
                labels.append(labl)
        return (trees, labels)

    def trainSoftmax(self, trees, labels):
        softmaxClassifier = RAESoftmaxClassifier()
        for epoch in xrange(50):
            count = 0
            print "Train epoch %d" % epoch
            xbatch = []
            ybatch = []
            for (idx, t) in enumerate(trees):
                if count == 50:
                    npxbatch = np.hstack(xbatch)
                    npybatch = np.hstack(ybatch)
                    (err, pred) = softmaxClassifier.fit(npxbatch, npybatch)
                    print "Err: %.4f" % np.mean(err)
                    xbatch = []
                    ybatch = []
                    count = 0
                else:
                    xbatch.append(t)
                    ybatch.append(labels[idx])
                    count += 1

        return softmaxClassifier

    def predictSoftmax(self, classifier, trees):
        results = []
        for t in trees:
            results.append(classifier.predict(t))
        return results

    def __str__(self):
        return "<ModelName: %s, CategorySize: %d, BFGSIterations: %d>" % (ModelEnum.rae.name, self.categorySize, self.bfgsIter)