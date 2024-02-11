import logging

class Logger():
    def __init__(self, file_path):
        self.logger = logging.getLogger(file_path)
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers = []

        sh = logging.StreamHandler()
        sh.setLevel(logging.DEBUG)
        sh.setFormatter(CustomFormatter())
        self.logger.addHandler(sh)

        fh = logging.FileHandler(file_path, encoding='utf-8')
        fh.setFormatter(CustomFormatter())
        fh.setLevel(logging.CRITICAL)
        self.logger.addHandler(fh)

    def get_logger(self):
        return self.logger


class CustomFormatter(logging.Formatter):

    grey = "\x1b[38;20m"
    blue = "\x1b[1;34m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    green = "\x1b[32m"
    reset = "\x1b[0m"
    format = "%(asctime)s:%(levelname)s: "

    FORMATS = {
        logging.DEBUG: blue + format + reset + "%(message)s",
        logging.INFO: blue + format + reset + "%(message)s",
        logging.WARNING: yellow + format + reset + "%(message)s",
        logging.ERROR: red + format + reset + "%(message)s",
        logging.CRITICAL: green + format + reset + "%(message)s"
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)
