__author__ = 'Johannes'

import math

import numpy

from general.config import Config


class Parameters:

    def __init__(self, initTheta=None, embeddingMatrix=None, embedSize=Config.RAE_EMBED_SIZE, catSize=0, seed=481516):

        self.embedSize = embedSize
        self.init_embed_range = 0.1
        self.scale_init_weights_by = 1.0
        self.supervised = catSize > 0
        self.categorySize = catSize

        self.r = 0.05
        self.r1 = numpy.sqrt(6) / numpy.sqrt(embedSize+embedSize+1)
        self.r2 = 0.001

        if initTheta is not None:
            self.unpack(initTheta)
        else:
            if embeddingMatrix is None:
                self.embeddings = 1e-3 * numpy.asarray((((numpy.random.rand(Config.VOCABULARY_SIZE, self.embedSize) * 2.0 * self.r1) - self.r1)))
            else:
                self.embeddings = embeddingMatrix

            self.Wc1 = numpy.asarray(random_weights(self.embedSize, self.embedSize, scale_by=self.scale_init_weights_by))
            self.Wc2 = numpy.asarray(random_weights(self.embedSize, self.embedSize, scale_by=self.scale_init_weights_by))
            self.Wr1= numpy.asarray(random_weights(self.embedSize, self.embedSize, scale_by=self.scale_init_weights_by))
            self.Wr2 = numpy.asarray(random_weights(self.embedSize, self.embedSize, scale_by=self.scale_init_weights_by))

            self.bh = numpy.asarray(numpy.zeros((self.embedSize, 1)))
            self.br1 = numpy.asarray(numpy.zeros((self.embedSize, 1)))
            self.br2 = numpy.asarray(numpy.zeros((self.embedSize, 1)))

            self.Wlabel = numpy.asarray(random_weights(self.categorySize, self.embedSize, scale_by=self.scale_init_weights_by))
            self.blabel = numpy.asarray(numpy.zeros((self.categorySize,1)))


    def getThetaAsList(self):
        theta = [self.Wc1, self.Wc2, self.Wr1, self.Wr2, self.bh, self.br1, self.br2, self.Wlabel, self.blabel]
        return theta

    def initGradientTheta(self):
        gTheta = []
        gTheta.append(numpy.zeros((self.embedSize, self.embedSize)))
        gTheta.append(numpy.zeros((self.embedSize, self.embedSize)))
        gTheta.append(numpy.zeros((self.embedSize, self.embedSize)))
        gTheta.append(numpy.zeros((self.embedSize, self.embedSize)))
        gTheta.append(numpy.zeros((self.embedSize, 1)))
        gTheta.append(numpy.zeros((self.embedSize, 1)))
        gTheta.append(numpy.zeros((self.embedSize, 1)))
        gTheta.append(numpy.zeros((self.categorySize, self.embedSize)))
        gTheta.append(numpy.zeros((self.categorySize, 1)))
        return gTheta

    def pack(self):
        theta = self.getThetaAsList()
        thetaArray = numpy.asarray([])
        for t in theta:
            thetaArray = numpy.hstack((thetaArray, t.flatten()))
        thetaArray = numpy.hstack((thetaArray, self.embeddings.flatten()))
        return numpy.copy(thetaArray)

    def unpack(self, theta):
        strt = 0
        end = strt + (self.embedSize * self.embedSize)
        self.Wc1 = theta[strt:end].reshape((self.embedSize, self.embedSize))
        strt = end
        end = strt + (self.embedSize * self.embedSize)
        self.Wc2 = theta[strt:end].reshape((self.embedSize, self.embedSize))
        strt = end
        end = strt + (self.embedSize * self.embedSize)
        self.Wr1 = theta[strt:end].reshape((self.embedSize, self.embedSize))
        strt = end
        end = strt + (self.embedSize * self.embedSize)
        self.Wr2 = theta[strt:end].reshape((self.embedSize, self.embedSize))
        strt = end
        end = strt + self.embedSize
        self.bh = theta[strt:end].reshape((self.embedSize,1))
        strt = end
        end = strt + self.embedSize
        self.br1 = theta[strt:end].reshape((self.embedSize,1))
        strt = end
        end = strt + self.embedSize
        self.br2 = theta[strt:end].reshape((self.embedSize,1))

        strt = end
        end = strt + (self.categorySize * self.embedSize)
        self.Wlabel = theta[strt:end].reshape((self.categorySize, self.embedSize))
        strt = end
        end = strt + self.categorySize
        self.blabel = theta[strt:end].reshape((self.categorySize, 1))

        strt = end
        end = strt + (Config.VOCABULARY_SIZE * self.embedSize)
        self.embeddings = theta[strt:end].reshape((Config.VOCABULARY_SIZE, self.embedSize))


sqrt3 = math.sqrt(3.0)
mr1 = numpy.sqrt(6) / numpy.sqrt(Config.RAE_EMBED_SIZE+Config.RAE_EMBED_SIZE+1)
mr2 = 0.001


def random_weights(nin, nout, scale_by=1./sqrt3, power=0.5):
    return ((numpy.random.rand(nin, nout) * 2.0 * mr1) - mr1)
