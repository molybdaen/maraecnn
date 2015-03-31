__author__ = 'Johannes'

from os.path import join
import cPickle
import os
import logging

from lmcw.config import Config
import util.utils
import util.exceptions as e


def syncConfig(runName, runId=0):
    Config.RUN_NAME = runName
    Config.RUN_ID = runId
    currentId = 0
    fLoadedConfig = False
    for f in os.listdir(Config.CONFIG_DIR):
        if f.startswith(Config.CONFIG_PREFIX):
            currentId = int(f.split("_")[-1].split(".")[0])
            if Config.RUN_ID == currentId:
                logging.basicConfig(filename=join(Config.CONFIG_DIR, "logging_%d.log" % Config.RUN_ID), level=logging.INFO)
                logging.info("Load Config for %s and runId %d..." % (runName, runId))
                confTuple = cPickle.load(open(join(Config.CONFIG_DIR, f), "r"))
                logging.info("Found ConfigFile %s: %s" % (str(f), str(confTuple)))
                Config.setConfig(confTuple)
                fLoadedConfig = True

    if not fLoadedConfig:
        Config.RUN_ID = currentId + 1
        stConfigFile = join(Config.CONFIG_DIR, "%s_%d%s" % (Config.CONFIG_PREFIX, Config.RUN_ID, Config.CONFIG_FILE_EXT))
        logging.basicConfig(filename=join(Config.CONFIG_DIR, "logging_%d.log" % Config.RUN_ID), level=logging.INFO)
        logging.info("No ConfigFile found for runId %d. Storing current Config in file %s." % (runId, str(stConfigFile)))
        conf = Config.getConfig()
        cPickle.dump(conf, open(stConfigFile, "wb"))
        logging.info("ConfigFile: %s" % str(conf))


def saveRAE(model, stream, epoch, cnt):
    snapId = util.utils.createSnapshotId()
    if not os.path.exists(Config.DIR_RUN_RAE_SNAPSHOT_MODEL):
        os.makedirs(Config.DIR_RUN_RAE_SNAPSHOT_MODEL)
    if not os.path.exists(Config.DIR_RUN_RAE_SNAPSHOT_STREAM):
        os.makedirs(Config.DIR_RUN_RAE_SNAPSHOT_STREAM)
    cPickle.dump(model, open(join(Config.DIR_RUN_RAE_SNAPSHOT_MODEL, r'model_%s%s' % (snapId, Config.SNAPSHOT_FILE_EXT)), "wb"))
    cPickle.dump((stream, epoch, cnt), open(join(Config.DIR_RUN_RAE_SNAPSHOT_STREAM, r'stream_%s%s' % (snapId, Config.SNAPSHOT_FILE_EXT)), "wb"))
    logging.info("SNAPSHOT: Saved Model and Stream (epoch %d - minibatch %d)." % (epoch, stream.count))
    print "Saved Snapshot (epoch %d - minibatch %d)." % (epoch, stream.count)

def saveRAEModel(model):
    snapId = util.utils.createSnapshotId()
    if not os.path.exists(Config.DIR_RUN_RAE_SNAPSHOT_MODEL):
        os.makedirs(Config.DIR_RUN_RAE_SNAPSHOT_MODEL)
    cPickle.dump(model, open(join(Config.DIR_RUN_RAE_SNAPSHOT_MODEL, r'model_%s%s' % (snapId, Config.SNAPSHOT_FILE_EXT)), "wb"))
    logging.info("SNAPSHOT: Saved Model")
    print "Saved Model."

def loadRAEModel():
    print "loading RAE-Model..."
    filename = "?unknown?"
    try:
        filename = util.utils.getRecentSnapshotFile(Config.DIR_RUN_RAE_SNAPSHOT_MODEL)
        model = cPickle.load(open(filename, "r"))
        print "RAE-model successfully loaded from file: %s" % filename
    except:
        raise e.SnapshotRAEModelLoad("Error: Unable to load RAE-Model from file: %s" % filename)
    logging.info("SNAPSHOT: Loaded Model from file: %s" % filename)
    return model

def loadRAEStream():
    print "loading RAE-Stream..."
    filename = "?unknown?"
    try:
        filename = util.utils.getRecentSnapshotFile(Config.DIR_RUN_RAE_SNAPSHOT_STREAM)
        (stream, epoch, cnt) = cPickle.load(open(filename, "r"))
        print "RAE-stream successfully loaded from file: %s" % filename
    except:
        raise e.SnapshotRAEStreamLoad("Error: Unable to load RAE-Stream from file: %s" % filename)
    logging.info("SNAPSHOT: Loaded Stream from file: %s - (Continue at epoch %d - minibatch %d)." % (filename, epoch, stream.count))
    return (stream, epoch, cnt)