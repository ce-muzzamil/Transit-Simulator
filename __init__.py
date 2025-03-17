import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from topology import Topology
from node import Node
from bus import Bus
from passenger import Passenger

def softmax(x, axis):
    """
    Safe implementation of softmax function
    """
    x = np.asarray(x)
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    sum_e_x = np.sum(e_x, axis=axis, keepdims=True)
    return e_x / sum_e_x
