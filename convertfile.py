import pyarrow as pa
import pandas as pd
import numpy as np
import os
import warnings
from typing import Mapping, Sequence, cast

import datasets
import numpy as np

from imitation.data import huggingface_utils
from imitation.data.types import AnyPath, Trajectory, TrajectoryWithRew
from imitation.util import util
from imitation.data import serialize



path='/home/zhong/quickstart/rl/rollouts/final.npz'
csv_file_path ='/home/zhong/quickstart/rl/csv_file.csv'

data=serialize.load(path)
obs_to_save = []
#print("*****",data[0])
# Print content of data (assuming it's a TrajectoryDatasetSequence)
for i, trajectory in enumerate(data):
        print(f"Trajectory {i+1}:",trajectory)
        extracted_obs = trajectory.obs
        print("obs size:",extracted_obs.shape[0], len(extracted_obs))
        extracted_action=trajectory.acts
        dim = extracted_action.size
        print("act size:",len(extracted_action))
        extracted_term=trajectory.terminal
        extracted_infos=np.empty((dim, 1)) #create placeholder for infos
        extracted_rews=trajectory.rews
        
        reshaped_acts = extracted_action.reshape(dim, 1)
        reshaped_infos = extracted_infos
        reshaped_rews = extracted_rews.reshape(dim,1)
        
        padding_size = extracted_obs.shape[0] - extracted_action.shape[0]
        padding_array = np.full((padding_size, 1), np.nan)
        joined_array = np.vstack([reshaped_acts, padding_array])
        combined_data = np.concatenate((extracted_obs, np.vstack([reshaped_acts, padding_array]), np.vstack([reshaped_infos, padding_array]), np.vstack([reshaped_rews, padding_array])), axis=1) 
        #print("OBS=", extracted_obs)
        #print(f"  Observation: {observation}")
        #print(f"  Action: {action}")
        break

df = pd.DataFrame(combined_data)  # Adjust column names as needed
df["terminated"] = extracted_term
# Save to CSV

df.to_csv(csv_file_path, index=False)  # Omit the index column

print(f"Observations saved to CSV file: {csv_file_path}")