#by default takes the rollouts from following command and convert it to CSV file.
#Then the CSV file can be edited manually
#python -m imitation.scripts.train_rl with cartpole logging.log_dir=quickstart/rl/

import pandas as pd
import numpy as np
import numpy as np
from imitation.data import serialize

input_path='/home/zhong/quickstart/rl/rollouts/final.npz'
csv_file_path ='/home/zhong/quickstart/rl/csv_file.csv'

data=serialize.load(input_path)
combined_data=None
#print("*****",data[0])
# Print content of data (assuming it's a TrajectoryDatasetSequence)
for i, trajectory in enumerate(data):
        print(f"Trajectory {i+1}:",trajectory)
        extracted_obs = trajectory.obs
        num_obs_cols = extracted_obs.shape[1]
        obs_names = [f"obs{i+1}" for i in range(num_obs_cols)]
        print(f"obs size: {extracted_obs.shape[0]} = {len(extracted_obs)}, number of features = {num_obs_cols}" )
        extracted_action=trajectory.acts
        extracted_rews=trajectory.rews
        dim = extracted_action.shape[0]
        reshaped_acts = extracted_action.reshape(dim, 1)
        reshaped_rews = extracted_rews.reshape(dim,1)
        #
        extracted_terms=np.full((dim+1,1),trajectory.terminal)
        extracted_infos=np.empty((dim+1,1)) #create placeholder for infos
        traj_id = np.full((dim+1,1),i)

        padding_size = extracted_obs.shape[0] - extracted_action.shape[0]
        padding_array = np.full((padding_size, 1), np.nan)
        current_combined_data = np.concatenate((extracted_obs, np.vstack([reshaped_acts, padding_array]), 
                                        extracted_infos, np.vstack([reshaped_rews, padding_array]),
                                        extracted_terms, traj_id), axis=1) 
        if combined_data is None:
          combined_data = current_combined_data  # Initialize on first iteration
        else:
          combined_data = np.concatenate((combined_data, current_combined_data), axis=0)  # Append to existing data
        #break #break after first loop to keep keep it simple for now
        
df = pd.DataFrame(combined_data)  # Adjust column names as needed
col_names = obs_names+['action']+['info']+['rewards']+['terminals']+['traj_id']
df.columns = col_names
#df.columns = ['obs1', 'obs2','obs3', 'obs4','action','info','rewards','terminal','traj_id']
# Save to CSV

df.to_csv(csv_file_path, index=False)  # Omit the index column

print(f"Observations saved to CSV file: {csv_file_path}")