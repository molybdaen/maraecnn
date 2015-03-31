__author__ = 'Johannes'

import math

import numpy
import theano

from general.config import Config


class CNNParameters:

    def __init__(self, initTheta=None, embeddingMatrix=None, embedSize=Config.CNN_EMBED_SIZE, catSize=0, seed=481516):

        self.embedSize = embedSize
        self.categorySize = catSize
        self.init_embed_range = 0.001
        self.r1 = numpy.sqrt(6) / numpy.sqrt(embedSize+embedSize+1)
        self.r2 = 0.001
        self.scale_init_weights_by = 1.0

        if initTheta is not None:
            self.unpack(initTheta)
        else:
            if embeddingMatrix is None:
                self.embeddings = numpy.asarray((numpy.random.rand(Config.CNN_VOCABULARY_SIZE, Config.CNN_EMBED_SIZE) - 0.5) * 2.0 * self.init_embed_range, dtype=theano.config.floatX)
            else:
                self.embeddings = embeddingMatrix

            self.W = numpy.asarray(random_weights(Config.CNN_HIDDEN_SIZE, Config.CNN_WINDOW_SIZE*Config.CNN_EMBED_SIZE, scale_by=self.scale_init_weights_by))

            self.bh = numpy.asarray(numpy.zeros((Config.CNN_HIDDEN_SIZE, 1)))#, broadcastable=(False, True)

            self.Wlabel = numpy.asarray(random_weights(self.categorySize, Config.CNN_HIDDEN_SIZE, scale_by=self.scale_init_weights_by))
            self.blabel = numpy.asarray(numpy.zeros((self.categorySize,1)))

    def getThetaAsList(self):
        theta = [self.W, self.bh, self.Wlabel, self.blabel]
        return theta

    def initGradientTheta(self):
        gTheta = []
        gTheta.append(numpy.zeros((self.embedSize, self.embedSize)))
        gTheta.append(numpy.zeros((self.embedSize, Config.RSA_EMBED_SIZE)))
        gTheta.append(numpy.zeros((self.embedSize, self.embedSize)))
        gTheta.append(numpy.zeros((Config.RSA_EMBED_SIZE, self.embedSize)))
        gTheta.append(numpy.zeros((self.embedSize, 1)))
        gTheta.append(numpy.zeros((self.embedSize, 1)))
        gTheta.append(numpy.zeros((Config.RSA_EMBED_SIZE, 1)))
        gTheta.append(numpy.zeros((self.categorySize, self.embedSize)))
        gTheta.append(numpy.zeros((self.categorySize, 1)))
        return gTheta

    def initGradientDictionary(self):
        return numpy.zeros((Config.CNN_VOCABULARY_SIZE, Config.CNN_EMBED_SIZE))

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
        end = strt + (self.embedSize * Config.RSA_EMBED_SIZE)
        self.Wc2 = theta[strt:end].reshape((self.embedSize, Config.RSA_EMBED_SIZE))
        strt = end
        end = strt + (self.embedSize * self.embedSize)
        self.Wr1 = theta[strt:end].reshape((self.embedSize, self.embedSize))
        strt = end
        end = strt + (Config.RSA_EMBED_SIZE * self.embedSize)
        self.Wr2 = theta[strt:end].reshape((Config.RSA_EMBED_SIZE, self.embedSize))
        strt = end
        end = strt + self.embedSize
        self.bh = theta[strt:end].reshape((self.embedSize,1))
        strt = end
        end = strt + self.embedSize
        self.br1 = theta[strt:end].reshape((self.embedSize,1))
        strt = end
        end = strt + Config.RSA_EMBED_SIZE
        self.br2 = theta[strt:end].reshape((Config.RSA_EMBED_SIZE,1))

        strt = end
        end = strt + (self.categorySize * self.embedSize)
        self.Wlabel = theta[strt:end].reshape((self.categorySize, self.embedSize))
        strt = end
        end = strt + self.categorySize
        self.blabel = theta[strt:end].reshape((self.categorySize, 1))

        strt = end
        end = strt + (Config.VOCABULARY_SIZE * Config.RSA_EMBED_SIZE)
        self.embeddings = theta[strt:end].reshape((Config.VOCABULARY_SIZE, Config.RSA_EMBED_SIZE))


sqrt3 = math.sqrt(3.0)
mr1 = numpy.sqrt(6) / numpy.sqrt(Config.EMBED_SIZE+Config.EMBED_SIZE+1)
mr2 = 0.001

def random_weights(nin, nout, scale_by=1./sqrt3, power=0.5):
    return (numpy.random.rand(nin, nout) * 2.0 - 1.0) * scale_by * sqrt3 / pow(nin,power)
    # return (numpy.random.rand(nin, nout) * 2.0 - 1.0)# / math.sqrt(nin)
    # return ((numpy.random.rand(nin, nout) * 2.0 * mr1) - mr1)

