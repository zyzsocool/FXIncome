# -*- coding: utf-8; py-indent-offset:4 -*-
###############################################################################
#
# Copyright (C) 2021 Mark Cheung, Dominic Hong
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
###############################################################################

import pandas as pd
import numpy as np
import datetime
import os
import joblib
import matplotlib.pyplot as plt
import logging

formatter = logging.Formatter(
    fmt="%(asctime)s-%(levelname)s - %(message)s", datefmt="%Y-%m-%d,%H:%M:%S"
)

logger = logging.getLogger("console_logger")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)

f_logger = logging.getLogger("file_logger")
f_logger.setLevel(logging.DEBUG)

f_handler = logging.FileHandler(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "spread_log.txt")
)
f_handler.setFormatter(formatter)
f_logger.addHandler(f_handler)
