__author__ = 'Molybdaen'

from general.config import Config, DataEnum, ModelEnum, ModelType
from util import exceptions as e
from sentenceLearning.cnn.CNNModel import CNNModel
from sentenceLearning.rae.RAEModel import RAEModel
from sentenceLearning.rsa.RSAModel import RSAModel
import time
import logging
from os.path import join
import cPickle


class SentenceRepresentationModel(object):
    def __init__(self, dictionary, categorySize, modelId=ModelEnum.cnn, loadExistingModel=None, epochs=10, BFGS_iter=130):
        if not isinstance(modelId, ModelType):
            raise e.NoValidTypeException("Supply valid model type. See config.ModelType.")
        self.modelId = modelId
        self.dictionary = dictionary
        self.bfgsIter = BFGS_iter
        self.model = None
        if loadExistingModel is not None:
            self.load(loadExistingModel)
        else:
            if modelId == ModelEnum.cnn:
                self.model = CNNModel(dictionary, categorySize=categorySize, epochs=epochs)
            elif modelId == ModelEnum.rae:
                self.model = RAEModel(dictionary, categorySize=categorySize, BFGS_iter=BFGS_iter)
            elif modelId == ModelEnum.rsa:
                self.model = RSAModel(dictionary, categorySize=categorySize, BFGS_iter=BFGS_iter)
            else:
                raise NotImplementedError("No implementation of model %s specified." % modelId.name)

        # logging.basicConfig(filename=join(modelId.getPath(), Config.FILE_LOGGING), filemode='w', format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    def train(self, trainDataset, prefix):
        logging.basicConfig(filename=join(self.modelId.getPath(), ("logging_%d.log" % trainDataset.splitNumber)), filemode='w', format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        logging.info("MODEL: %s" % str(self.model))
        logging.info("DATASET: %s" % str(trainDataset))
        logging.info("RUN: %s" % prefix)
        logging.info("PROGRESS: Training model...")
        strtTime = time.time()
        errSignals = self.model.learnFeatures(trainDataset)
        endTime = time.time()
        logging.info("PROGRESS: Finished! Training Time %.3f seconds" % (endTime-strtTime))
        return (errSignals, self.model)

    def extract(self, dataset):
        logging.info("PROGRESS: Extracting features...")
        strtTime = time.time()
        (features, predictions, labels) = self.model.extractFeatures(dataset)
        endTime = time.time()
        logging.info("PROGRESS: Finished! Feature Extraction Time %.3f seconds" % (endTime-strtTime))
        return (features, predictions, labels)

    def save(self, prefix):
        cPickle.dump(self.model, open(join(self.modelId.getPath(), (prefix + Config.FILE_MODEL)), "w"))

    def load(self, prefix):
        self.model = cPickle.load(open(join(self.modelId.getPath(), (prefix + Config.FILE_MODEL)), "r"))
        self.dictionary = self.model.dictionary