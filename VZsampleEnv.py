import gym
from gym import spaces
import random
from enum import Enum


#Action list
class Actions(Enum):
    AddrValidation=0
    DeviceValidation=1
    SimValidation=2
    Activation=3
    AskZip=4
    AskSim=5
    AskIMEI=6


# Define environment logic
class MobilePhoneCarrierEnv(gym.Env):

    # Device Information
    _device_type = spaces.Discrete(3)  # Assuming 4 device types: smartphone, smartwatch, other
    _manufacturer = spaces.Discrete(10)  # Assuming 10 major manufacturers: apple, samsung, google, ....
    _os = spaces.Discrete(3)  # iOS, Android, other

    # Sim Information
    _sim1_type = spaces.Discrete(3)  # 3 SIM types: physical, eSIM, embedded
    _sim1_status = spaces.Discrete(3)  # 3 SIM statuses: active, inactive, suspended
    #sim2_type = spaces.Discrete(3)  # 3 SIM types: physical, eSIM, embedded
    #sim2_status = spaces.Discrete(3)  # 3 SIM statuses: active, inactive, suspended

    # Network config status
    _network_type = spaces.Discrete(4) #4G, 5G, undefined, other
    _OCS_status = spaces.Discrete(4) #configured, not configured, pending, failed
    _HLR_status = spaces.Discrete(4) #configured, not configured, pending, failed
    _ROAMGW_status = spaces.Discrete(4) #configured, not configured, pending, failed
    _TASPI_status = spaces.Discrete(4) #configured, not configured, pending, failed

    # Customer info
    _customer_status = spaces.Discrete(4) #active, inactive, new, other
    _payment_status = spaces.Discrete(4) #paid, pastdue, pending, new, other
    _sub_expiration_date = spaces.Discrete(31) #31 days in a month
    _today = spaces.Discrete(31) #31 days in a month

    # Validations
    _address_validation_status = spaces.Discrete(2) # Address
    _device_validation_status = spaces.Discrete(2)  # Device
    _sim_validation_status = spaces.Discrete(2)  # SIM

    #group related info together
    _device_info = spaces.Tuple((_device_type, _manufacturer, _os))
    _sim_info = spaces.Tuple((_sim1_type, _sim1_status))  # Fill in SIM information dimensions
    _network_info = spaces.Tuple((_network_type,_OCS_status,_HLR_status,_ROAMGW_status,_TASPI_status))  # Fill in network information dimensions
    _verification_status = spaces.Tuple((_address_validation_status, _device_validation_status, _sim_validation_status))
    _customer_info = spaces.Tuple((_customer_status,_payment_status,_sub_expiration_date,_today))


    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Dict({
            "device_info": self._device_info,
            "sim_info": self._sim_info,
            "network_info": self._network_info,
            "verification_status": self._verification_status,
            "customer_info": self._customer_info
        })
        self.action_space = spaces.Discrete(len(Actions))
        self.current_state = self.observation_space.sample()
        # change validation status
        self.current_state["verification_status"] = (0, 0, 0)
        #self.current_state["verification_status"] = (random.randint(0, 1), random.randint(0, 1), random.randint(0, 1))
        

    def reset(self):
        # Reset the environment to its initial state (replace with your logic)
        self.current_state = self.observation_space.sample()  # Example initialization
        # Reset verification status and return initial observation
        self.current_state["verification_status"] = (0, 0, 0)
        return self.current_state

    def step(self, action):
        # Reward based on action and update verification status
        reward = 0
        done = False
        action=0
        if ((action==Actions.AddrValidation.value) and (self.current_state["verification_status"][0]==0)):
                self.current_state["verification_status"]=(1, self.current_state["verification_status"][1], self.current_state["verification_status"][2])
                reward +1
        elif (action==Actions.DeviceValidation.value and self.current_state["verification_status"][1]==0):
                self.current_state["verification_status"]=(self.current_state["verification_status"][0], 1,self.current_state["verification_status"][2])
                reward +=1
        elif (action==Actions.SimValidation.value and self.current_state["verification_status"][2]==0):
                self.current_state["verification_status"]=(self.current_state["verification_status"][0], self.current_state["verification_status"][1], 1)
                reward +=1
        elif (action==Actions.Activation.value and all(self.current_state["verification_status"])):  # Activate
            reward += 5
            done = True  # Complete order, end episode
        else:
            done = False

        # Reward for asking for missing information
        if action[4][0] and not self.verification_status[0]:
            reward += 0.5  

        new_state =  self._get_obs()
        info = {}  # Optional: Additional information
        return new_state, reward, done, info
    
    def _get_obs(self):
        return {}