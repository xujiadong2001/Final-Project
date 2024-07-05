import cv2
import os
import pickle
from scipy import signal
import numpy as np
import pandas as pd
from tqdm import tqdm

class Analyser():
    def __init__(self, path):
        self.i = None
        self.timeseriesPath = path
    
    def get_labels(self, i):
        self.i = i
        filename = f"sample_{self.i}.pkl"

        with open(os.path.join(self.timeseriesPath, filename), 'rb') as handle:
            data = pickle.load(handle)

        Fx = data["fx"]
        Fy = data["fy"]
        Fz = data["fz"]

        # Try without filtering, just take last value:
        Fx_final = Fx[-1]
        Fy_final = Fy[-1]
        Fz_final = Fz[-1]

        return Fx_final, Fy_final, Fz_final

