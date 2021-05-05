import logging
import os
from logging.handlers import RotatingFileHandler
from time import strftime


class Logger():
    def __init__(self, log_dirrectory):
        handler = RotatingFileHandler(filename=log_dirrectory, maxBytes=100000, backupCount=10)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(handler)

    def write(self, message=""):
        dt = strftime("[%Y-%b-%d %H:%M:%S]")
        self.logger.info(f"{dt} {message}")
