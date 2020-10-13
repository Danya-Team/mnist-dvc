
import numpy as np
import matplotlib.image as mpimg
from matplotlib import pyplot as plt
from tensorflow.keras.utils import plot_model
import tensorflow as tf
from keras import layers
import keras
import platform
import subprocess
import os
from datetime import datetime

import logging
import coloredlogs

coloredlogs.install(level='INFO')
log = logging.getLogger('mlutils')

def timestamp():
    return datetime.now().strftime("%Y-%d-%m_%H-%M-%S-%f-")


def git_version():
    def _minimal_ext_cmd(cmd):
        # construct minimal environment
        env = {}
        for k in ['SYSTEMROOT', 'PATH', 'HOME']:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        # LANGUAGE is used on win32
        env['LANGUAGE'] = 'C'
        env['LANG'] = 'C'
        env['LC_ALL'] = 'C'
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, env=env)
        return out

    try:
        out = _minimal_ext_cmd(['git', 'rev-parse', '--short', 'HEAD'])
        GIT_REVISION = out.strip().decode('ascii')
    except (subprocess.SubprocessError, OSError):
        GIT_REVISION = "Unknown"

    if not GIT_REVISION:
        # this shouldn't happen but apparently can (see gh-8512)
        GIT_REVISION = "Unknown"

    return GIT_REVISION


def mlversion():
    from IPython.core.interactiveshell import InteractiveShell

    InteractiveShell.ast_node_interactivity = "all"
    from keras import models

    strategy = tf.distribute.MirroredStrategy()
    log.info("python version       :" + platform.python_version())
    log.info("keras version        :" + keras.__version__)
    log.info("tensorflow version   :" + tf.__version__)
    log.info("Number of GPU devices: {}".format(strategy.num_replicas_in_sync))

    log.info(tf.config.list_physical_devices('GPU'))
    log.info(tf.sysconfig.get_lib())
