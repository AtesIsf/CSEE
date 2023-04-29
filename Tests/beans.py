import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import Models.models as models
import Data.beans as beans

x_train = np.zeros((len(beans.ds["train"]), beans.SHAPE[0], beans.SHAPE[1], beans.SHAPE[2]))
for i in range(len(beans.ds["train"])):
    x_train[i] = beans.ds["train"][i]["image"]

x_test = np.zeros((len(beans.ds["test"]), beans.SHAPE[0], beans.SHAPE[1], beans.SHAPE[2]))
for i in range(len(beans.ds["test"])):
    x_test[i] = beans.ds["test"][i]["image"]

y_train = np.zeros(len(beans.ds["train"]))
for i in range(len(beans.ds["train"])):
    y_train[i] = beans.ds["train"][i]["labels"]
    
y_test = np.zeros(len(beans.ds["test"]))
for i in range(len(beans.ds["test"])):
    y_test[i] = beans.ds["test"][i]["labels"]

def input_fn(t):
    return x_train[int(t / 0.001) % len(beans.ds)].flatten()

ann, snn = models.get_ann_model(len(beans.labels))

