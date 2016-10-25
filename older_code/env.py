import os
from getpass import getuser
from time import time
from os.path import isdir, basename, join
import __main__ as main
import logging


def mkdir_cd(path):
    """Change working directory to `path` creating it if necessary"""
    if not isdir(path):
        os.makedirs(path)
    os.chdir(path)


def set_workspace(username=True, timestamp=False, scriptname=True, name=None, persistent=False):
    """Set the current working directory to a new dedicated one."""
    if persistent:
        tmpd = "/cl/work/{}/RunningExperiments".format(getuser())
        if not os.path.isdir(tmpd):
            tmpd = os.path.expanduser("~/RunningExperiments")
    else:
        tmpd = "/tmp"

    path = []
    if username:
        path.append(getuser())
    if name:
        path.append(name)
    if scriptname:
        path.append(basename(getattr(main, '__file__', 'ipython')))
    if timestamp:
        path.append(str(int(time())))
    paths = join(tmpd, *path)
    mkdir_cd(paths)
    return paths


def config_logging(filename):
    format = '%(asctime)s [%(levelname)s] %(message)s'
    formatter = logging.Formatter(format)
    logging.basicConfig(format=format, filename=filename, level=logging.NOTSET)
    logger = logging.getLogger()
    stderr_log_handler = logging.StreamHandler()
    logger.addHandler(stderr_log_handler)
    stderr_log_handler.setFormatter(formatter)
