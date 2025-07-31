import pandas as pd
import numpy as np
import random

def shuffle_data( data ):
    order = list(range(len(data)))
    random.shuffle(order)
    