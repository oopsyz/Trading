#read from CSV and save in Huggingface format to be used by imitation training
import pandas as pd
from imitation.data import huggingface_utils
from imitation.data.types import AnyPath, Trajectory, TrajectoryWithRew
from imitation.util import util
from imitation.data import serialize
import numpy as np

csv_file_path ='/home/zhong/quickstart/rl/csv_file.csv'
hug_file_path ='/home/zhong/quickstart/rl/test/'
# Load the CSV data
df = pd.read_csv(csv_file_path)

# Split into trajectories (adapt if trajectory boundaries are already defined)
# use IMEI, MDN, or customer ID as trajectory_id
#df["trajectory_id"] = df["action"].diff().ne(0).cumsum()  # Assuming new trajectory starts with a new action
#trajectories = df.groupby("trajectory_id")

# Create Trajectory objects
'''
trajectory_objects = []
for trajectory_id, trajectory_data in trajectories:
    observations = trajectory_data.iloc[:, :4].to_numpy()  # Extract observations (columns 1-4)
    actions = trajectory_data["action"].to_numpy()
    rewards = trajectory_data["reward"].to_numpy()

    # Assuming you have a Trajectory class defined:
    trajectory_objects.append(Trajectory(observations, actions, rewards))
'''
trajectory_objects = []
for traj_id, trajectory_data in df.groupby('traj_id'):
    observations = trajectory_data.iloc[:, :4].to_numpy()  # Extract observations (columns 1-4)
    print("length=",len(observations))
    len_obs=len(observations)
    num_of_obs=4
    #VSC file format 
    actions = trajectory_data.iloc[:len_obs-1,num_of_obs] #Trajectory() expect observations to have 1 additional row than actions
    infos = trajectory_data.iloc[:len_obs-1,num_of_obs+1]
    rewards = trajectory_data.iloc[:len_obs-1,num_of_obs+2]
    terminal = trajectory_data.iloc[0,num_of_obs+3]  #whether the trajectory was terminated, it just need one value
    # create Trajectory with Rewards and and to trajectory_objects in case there are multiple trajactories
    trajectory_objects.append(TrajectoryWithRew(observations, actions, infos, terminal, rewards))

# Create TrajectoryDatasetSequence
#dataset = huggingface_utils.trajectories_to_dataset(trajectory_objects)
serialize.save(hug_file_path,trajectory_objects)