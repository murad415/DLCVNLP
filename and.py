from utils.model import Perceptron
from utils.all_utils import prepare_data, save_mode
import pandas as pd

AND = {
    "x1": [0,0,1,1],
    "x2": [0,1,0,1],
    "y": [0,0,0,1],
}

df = pd.DataFrame(AND)

x,y=prepare_data(df)

epoch=10
eta=0.2
model=Perceptron(eta=eta,epoch=epoch)
model.fit(x,y)
save_mode("and.mode",model)
