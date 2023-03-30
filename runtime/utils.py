# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import multiprocessing
import os
import pickle
import shutil
import sys
from functools import wraps
from pathlib import Path
import numpy as np


def get_task_code(task, dim):
    return f"{task}_{dim}d_tf2"


def make_empty_dir(path, force=False):
    path = Path(path)
    if path.exists():
        if not path.is_dir():
            print(f"Output path {path} exists and is not a directory." "Please remove it and try again.")
            sys.exit(1)
        else:
            if not force:
                decision = input(f"Output path {path} exists. Continue and replace it? [Y/n]: ")
                if decision.strip().lower() not in ["", "y"]:
                    sys.exit(1)
            shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True)
