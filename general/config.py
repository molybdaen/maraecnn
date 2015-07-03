__author__ = 'Johannes'

from os.path import join


class T_Model_Category:
    language_model = "language model"
    sentence_representation_model = "sentence representation model"
    classification_model = "classification model"


class ModelType:
    def __init__(self, name, dirName, mType):
        self.name = name
        self.dirName = dirName
        self.mType = mType

    def getPath(self):
        if self.mType == T_Model_Category.language_model:
            return join(Config.ROOT_DIR, Config.DIRNAME_LANGUAGE, Config.DIRNAME_MODELS, self.dirName)
        elif self.mType == T_Model_Category.sentence_representation_model:
            return join(Config.ROOT_DIR, Config.DIRNAME_SENTIMENT, Config.DIRNAME_MODELS, self.dirName)
        elif self.mType == T_Model_Category.classification_model:
            return join(Config.ROOT_DIR, Config.DIRNAME_CLASSIFICATION, Config.DIRNAME_MODELS, self.dirName)
        else:
            raise NotImplementedError("Model category %s not yet implemented." % self.mType)

    def __str__(self):
        return "<ModelType name: %s, dirName: %s, dType: %s>" % (self.name, self.dirName, self.mType)


class T_Data_Category:
    unlabeled_corpus = "unlabeled text corpus"
    labeled_sentence_dataset = "labeled sentence dataset"


class DataType:
    def __init__(self, name, dirName, dType):
        self.name = name
        self.dirName = dirName
        self.dType = dType

    def getPath(self):
        if self.dType == T_Data_Category.unlabeled_corpus:
            return join(Config.ROOT_DIR, Config.DIRNAME_LANGUAGE, Config.DIRNAME_LANGUAGE_CORPORA, self.dirName)
        elif self.dType == T_Data_Category.labeled_sentence_dataset:
            return join(Config.ROOT_DIR, Config.DIRNAME_SENTIMENT, Config.DIRNAME_SENTIMENT_DATASETS, self.dirName)
        else:
            raise NotImplementedError("Dataset category %s not yet implemented." % self.dType)

    def isLabeled(self):
        return self.dType == T_Data_Category.labeled_sentence_dataset

    def __str__(self):
        return "<DataType name: %s, dirName: %s, dType: %s>" % (self.name, self.dirName, self.dType)


class DataEnum:
    amazon = DataType('Amazon Product Reviews', 'amazon', T_Data_Category.unlabeled_corpus)
    wikipedia = DataType('Wikipedia (english)', 'wikipedia', T_Data_Category.unlabeled_corpus)
    bopang = DataType('Bo Pang Movie Reviews', 'bopang', T_Data_Category.unlabeled_corpus)
    turian = DataType('Turian et al. Wikipedia CW', 'turian', T_Data_Category.unlabeled_corpus)
    amazonreviews = DataType('Amazon Product Reviews', 'amazon-productreviews', T_Data_Category.labeled_sentence_dataset)
    moviereviews = DataType('Movie Reviews Rotten Tomatoes - Bo Pang et al', 'bopang-moviereviews', T_Data_Category.labeled_sentence_dataset)
    treebank = DataType('Sentiment Treebank - Stanford NLP', 'sentiment-treebank', T_Data_Category.labeled_sentence_dataset)
    twitter = DataType('Twitter Sentiment', 'twitter', T_Data_Category.labeled_sentence_dataset)


class ModelEnum:
    cw = ModelType('NNLM - Collobert and Weston', 'cw', T_Model_Category.language_model)
    w2v = ModelType('Skip-Gram - Mikoholov et al.', 'w2v', T_Model_Category.language_model)
    cnn = ModelType('Convolution net - IBM', 'cnn', T_Model_Category.sentence_representation_model)
    rae = ModelType('Recursive autoencoder - Socher et al.', 'rae', T_Model_Category.sentence_representation_model)
    rsa = ModelType('Recursive sequence associator - Homebrew', 'rsa', T_Model_Category.sentence_representation_model)


class TokenizerEnum:
    sentiment = "sentiment"


class Config(object):

    def __init__(self):
        pass

    RUN_ID = 1

    VOCABULARY_SIZE = 14000
    INCLUDE_UNKNOWN_WORD = True
    CASE_SENSITIVE = True
    MAX_TOKENS = 1000000000

    EMBED_SIZE = 100

    #
    DATA_FILE_EXT = ".gz"
    DATA_FILE_EXT_BZ2 = ".bz2"
    SNAPSHOT_FILE_EXT = ".pkl"
    CONFIG_FILE_EXT = ".pkl"
    INOUT_FILE_EXT = '.gz'

    TRAIN_SENTENCES_PREFIX = r'train'
    CONFIG_PREFIX = r'config'

    INTERRUPT_VERYFREQUENT = 1000
    INTERRUPT_FREQUENT = 100000
    INTERRUPT_MEDIUM =  1000000
    INTERRUPT_RARE =   10000000

    ROOT_DIR = r'../data'

    RUN_NAME = r'amazon'

    DIRNAME_DATA = r'data'
    DIRNAME_DATA_RAW = r'raw'
    DIRNAME_DATA_TRAIN = r'train'
    DIRNAME_RUN = r'run'
    DIRNAME_RUN_LMCW = r'lmcw'
    DIRNAME_RUN_RAE = r'rae'
    DIRNAME_INPUT = r'input'
    DIRNAME_OUTPUT = r'output'
    DIRNAME_SNAPSHOT_STREAM = r'stream'
    DIRNAME_SNAPSHOT_MODEL = r'model'

    DIRNAME_LANGUAGE = r'language'
    DIRNAME_LANGUAGE_CORPORA = r'corpora'
    DIRNAME_SENTIMENT = r'sentiment'
    DIRNAME_SENTIMENT_DATASETS = r'datasets'
    DIRNAME_CLASSIFICATION = r'classification'
    DIRNAME_MODELS = r'models'
    DIRNAME_LMCW = r'cw'
    DIRNAME_W2V = r'w2v'
    DIRNAME_CNN = r'cnn'
    DIRNAME_RAE = r'rae'
    DIRNAME_RSA = r'rsa'

    FILE_LOGGING = r'logging.log'
    FILE_LANGUAGE_DICTIONARY = r'languageDictionary.pkl'
    FILE_MODEL = r'model.pkl'
    FILE_MODEL_DATA = r'modelData.pkl'
    FILE_TRAINSPLIT = r'trainsplit'

    EXPERIMENT_DIR = join(ROOT_DIR, RUN_NAME)

    DIR_DATA_RAW = join(ROOT_DIR, RUN_NAME, DIRNAME_DATA, DIRNAME_DATA_RAW)
    DIR_DATA_TRAIN = join(ROOT_DIR, RUN_NAME, DIRNAME_DATA, DIRNAME_DATA_TRAIN)
    DIR_INPUT = join(ROOT_DIR, RUN_NAME, DIRNAME_INPUT)
    DIR_OUTPUT = join(ROOT_DIR, RUN_NAME, DIRNAME_OUTPUT)

    DIR_RUN_LMCW = join(ROOT_DIR, RUN_NAME, DIRNAME_RUN, DIRNAME_RUN_LMCW)
    DIR_RUN_LMCW_SNAPSHOT_STREAM = join(DIR_RUN_LMCW, DIRNAME_SNAPSHOT_STREAM)
    DIR_RUN_LMCW_SNAPSHOT_MODEL = join(DIR_RUN_LMCW, DIRNAME_SNAPSHOT_MODEL)

    DIR_RUN_RAE = join(ROOT_DIR, RUN_NAME, DIRNAME_RUN, DIRNAME_RUN_RAE)
    DIR_RUN_RAE_SNAPSHOT_STREAM = join(DIR_RUN_RAE, DIRNAME_SNAPSHOT_STREAM)
    DIR_RUN_RAE_SNAPSHOT_MODEL = join(DIR_RUN_RAE, DIRNAME_SNAPSHOT_MODEL)

    FILE_VOCABULARY = join(EXPERIMENT_DIR, r'vocabulary_size=%d.txt' % VOCABULARY_SIZE)
    FILE_WORDMAP = join(EXPERIMENT_DIR, r'wordmap_size=%d_include-unknown=%s.pkl' % (VOCABULARY_SIZE, INCLUDE_UNKNOWN_WORD))
    FILE_CONFIG = join(EXPERIMENT_DIR, r'config%s' % CONFIG_FILE_EXT)
    FILE_LMCW_IN = join(DIR_INPUT, r'embeddings-scaled.EMBEDDING_SIZE=%d.txt%s' % (EMBED_SIZE, INOUT_FILE_EXT))
    FILE_LMCW_OUT = join(DIR_OUTPUT, r'lmcw_embedsize=%d_vocabsize=%d.txt%s' % (EMBED_SIZE, VOCABULARY_SIZE, INOUT_FILE_EXT))
    FILE_RAE_IN = join(DIR_INPUT, r'lmcw_embedsize=%d_vocabsize=%d.txt%s' % (EMBED_SIZE, VOCABULARY_SIZE, INOUT_FILE_EXT))
    FILE_RAE_OUT = join(DIR_OUTPUT, r'rae_embedsize=%d_vocabsize=%d.txt%s' % (EMBED_SIZE, VOCABULARY_SIZE, INOUT_FILE_EXT))

    #############################################################################
    # Following are configs for language models
    #############################################################################

    # According to Turian et al.: Train for 50 epochs.
    # ...and do about 2'270'000'000 updates in total (~45'000'000 Updates per epoch)
    # ...which requires about 220 MB of (compressed) text
    WINDOW_SIZE = 5
    MINIBATCH_SIZE = 3
    # According to Turian et al.: Validaton score doesn't improve much after 50 dimensional Embeddings.
    HIDDEN_SIZE = 100
    # According to Turian et al.: ELR in [1000:32000] * LR
    LEARNING_RATE = 0.000011
    EMBEDDING_LEARNING_RATE = 0.034
    # According to Turian et al.: Initialize embeddings uniformly in range [-0.01, 0.01]
    # For Normalization use scaling factor of 0.1 to keep stddev(embedding_i) = 0.1
    NORMALIZE_EMBEDDINGS = False
    # see turian et al. for justification. After training, embedding values are not bound to be within a certain interval. Thus, we have to rescale them.
    EMBED_SCALING = 0.1
    # Embeddings are initialized randomly from a uniform distribution [-1, +1] * INIT_EMBED_RANGE
    INIT_EMBED_RANGE = 0.01
    INIT_EMBEDDINGS = False


    #############################################################################
    # Following are configs for Recursive Autoencoder (see R. Socher for details)
    #############################################################################

    RAE_EMBED_SIZE = EMBED_SIZE

    RAE_LEARNING_RATE = 0.1
    RAE_EMB_LEARNING_RATE = 0.01
    RAE_LABEL_LEARNING_RATE = 0.1
    RAE_MINIBATCH_SIZE = 20

    # According to R. Socher's Java Implementation
    RAE_ALPHA_CAT =     0.2
    RAE_BETA =          0.5
    RAE_LAMBDA_W =      1e-06
    RAE_LAMBDA_W_REC =  1e-06
    RAE_LAMBDA_L =      1e-08
    RAE_LAMBDA_CAT =    1e-05
    RSA_EMBED_SIZE = 25

    RSA_LAMBDA_W =      1e-05
    RSA_LAMBDA_L =      5e-07
    RSA_LAMBDA_CAT =    1e-04

    #############################################################################
    # Following are configs for a Simple Convolutional Network
    #############################################################################

    CNN_VOCABULARY_SIZE = VOCABULARY_SIZE
    CNN_EPOCHS = 11
    CNN_EMBED_SIZE = EMBED_SIZE
    CNN_HIDDEN_SIZE = 100
    CNN_WINDOW_SIZE = 5
    CNN_PAD_SENTENCE = True
    CNN_LR = 0.25
    CNN_ELR = 0.25
    CNN_LAMBDA = 0.01
    CNN_LAMBDA_L = 0.001


    @staticmethod
    def getLMDataPath(corpusName):
        return join(Config.ROOT_DIR, Config.DIRNAME_LANGUAGE, Config.DIRNAME_LANGUAGE_CORPORA, corpusName)

    @staticmethod
    def getSentDataPath(datasetName):
        return join(Config.ROOT_DIR, Config.DIRNAME_SENTIMENT, Config.DIRNAME_SENTIMENT_DATASETS, datasetName)

    @staticmethod
    def getLMModelPath(modelName):
        return join(Config.ROOT_DIR, Config.DIRNAME_LANGUAGE, Config.DIRNAME_MODELS, modelName)

    @staticmethod
    def getSentModelPath(modelName):
        return join(Config.ROOT_DIR, Config.DIRNAME_SENTIMENT, Config.DIRNAME_MODELS, modelName)

    @staticmethod
    def getDataPath(dataIdent):
        if dataIdent.isLabeledDataset:
            return join(Config.ROOT_DIR, Config.DIRNAME_SENTIMENT, Config.DIRNAME_SENTIMENT_DATASETS, dataIdent.name)
        if not dataIdent.isLabeledDataset:
            return join(Config.ROOT_DIR, Config.DIRNAME_LANGUAGE, Config.DIRNAME_LANGUAGE_CORPORA, dataIdent.name)

    @staticmethod
    def getModelPath(modelIdent):
        if modelIdent.isLabeledDataset:
            return join(Config.ROOT_DIR, Config.DIRNAME_SENTIMENT, Config.DIRNAME_MODELS, modelIdent.name)

    @staticmethod
    def getConfig():
        myConf = {}
        for i in Config.__dict__.keys():
            if not i.startswith('__') and type(Config.__dict__[i]) != staticmethod:
                myConf[i] = Config.__dict__[i]
        print myConf
        state = ((myConf),)
        return state

    @staticmethod
    def setConfig(conf):
        for k in conf[0]:
            setattr(Config, k, conf[0][k])