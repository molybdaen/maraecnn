__author__ = 'Johannes'


class DataFileException(Exception):
    pass


class NoValidPreprocessorException(Exception):
    pass

class NoValidTypeException(Exception):
    pass

class SnapshotRAEModelLoad(Exception):
    pass

class SnapshotRAEStreamLoad(Exception):
    pass

class SnapshotRAELoadException(Exception):
    pass