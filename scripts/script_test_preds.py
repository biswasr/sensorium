#%%
import pandas as pd
import pyarrow
import fastparquet
import matplotlib.pyplot as plt
import numpy as np
import json
#%%
parquet_file = "~/Documents/GitHub/sensorium/data/predictions/true_batch_001/final_test_main/predictions_final_main.parquet.brotli"
df = pd.read_parquet(parquet_file, engine='pyarrow')
# %%
df['prediction'][0].shape
#%%
df.groupby('mouse')
# %%
thisfilepath = "/home/ucsf/Documents/GitHub/sensorium/data/predictions/true_batch_001/final_test_main/dynamic29515-10-12-Video-9b4f6a1a067fe51e15306b9628efea20/4.npy"
df1 = np.load(thisfilepath)
# %%
df1
# %%
thisfilepath = "/home/ucsf/Documents/GitHub/sensorium/data/predictions/true_batch_001/out-of-fold/evaluate_new.json"
f = open(thisfilepath,)
df2 = json.load(thisfilepath)
f.close()
#%%
import os
os.getcwd()
os.chdir('/home/ucsf/Documents/GitHub/sensorium')
# %%
from src import constants
import numpy as np
import pandas as pd

from src.responses import ResponseNormalizer
from src.data import get_mouse_data
from src.metrics import corr
experiment = 'true_batch_001'
prediction_dir = constants.predictions_dir / experiment / "out-of-fold"
correlations = dict()
dataset = 'new'
mouse = constants.dataset2mice[dataset][0]
# %%
mouse_data = get_mouse_data(mouse=mouse, splits=constants.folds_splits)
#%%        
mouse_prediction_dir = prediction_dir / mouse
# %%

def cut_responses_for_submission(prediction: np.ndarray):
    prediction = prediction[..., :constants.submission_limit_length]
    prediction = prediction[..., constants.submission_skip_first:]
    if constants.submission_skip_last:
        prediction = prediction[..., :-constants.submission_skip_last]
    return prediction

#%%
predictions = []
targets = []

for trial_data in mouse_data["trials"]:
    trial_data = mouse_data["trials"][0]
    trial_id = trial_data['trial_id']
    prediction = np.load(str(mouse_prediction_dir / f"{trial_id}.npy"))
    target = np.load(trial_data["response_path"])[..., :trial_data["length"]]
    prediction = cut_responses_for_submission(prediction)
    target = cut_responses_for_submission(target)
    predictions.append(prediction)
    targets.append(target)
#%%
corrneuro = np.zeros(prediction.shape[1])
for i in range(corrneuro.shape[0]):
    corrneuro[i] = corr(prediction[i,:],target[i,:])
# %%
correlation = corr(np.concatenate(predictions, axis=1),np.concatenate(targets, axis=1),axis=1)
# %%
import matplotlib.pyplot as plt
# %%
plt.hist(correlation)
plt.title("Histogram of Corr(predicted, observed signal) per neuron")
plt.xlabel("Correlation")
# %%
nidx = np.where(correlation>0.7)[0]
# %%
neuron_ids = mouse_data["neuron_ids"].tolist()
#%%
neuron_ids = np.array(neuron_ids)
#%%
neuron_ids[nidx]
# %%
coords = mouse_data["cell_motor_coordinates"]
# %%
plt.title("Histogram of depth of neurons with corr(pred,obs) > 70%")
plt.hist(coords[nidx,2])
plt.xlabel("Depth")
# %%
plt.hist(coords[:,2])

# %%
len(nidx)
# %%
