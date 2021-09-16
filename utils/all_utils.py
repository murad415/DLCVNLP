import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
import os
from matplotlib.colors import ListedColormap
# plt.style.use("fivethirtyeight")


def prepare_data(df):
    x=df.drop("y",axis=1)
    y=df["y"]
    return x, y



def save_mode(filename,model):
    model_dir="model"
    os.makedirs(model_dir,exist_ok=True)
    filepath=os.path.join(model_dir,filename)
    joblib.dump(model, filepath)