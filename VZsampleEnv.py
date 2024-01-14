import gymnasium as gym
from gymnasium import spaces
import random
from enum import Enum


#Action list
class Actions(Enum):
    AddrValidation=0
    DeviceValidation=1
    SimValidation=2
    Activation=3
#    AskZip=4
#    AskSim=5
#    AskIMEI=6

total_rewards=0

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

    reward=0

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Dict({
            "device_type": self._device_type,
            "manufacturer": self._manufacturer,
            "os": self._os,
            "sim1_type": self._sim1_type,
            "sim1_status": self._sim1_status,
            "network_type": self._network_type,
            "OCS_status": self._OCS_status,
            "HLR_status": self._HLR_status,
            "ROAMGW_status": self._ROAMGW_status,
            "TASPI_status": self._TASPI_status,
            "customer_status": self._customer_status,
            "payment_status": self._payment_status,
            "sub_expiration_date": self._sub_expiration_date,
            "today": self._today,
            "address_validation_status": self._address_validation_status,
            "device_validation_status": self._device_validation_status,
            "sim_validation_status": self._sim_validation_status
        })

        self.action_space = spaces.Discrete(len(Actions))
        self.current_state = self.observation_space.sample()
        #set all validation status to 0
        self.current_state["address_validation_status"] = 0
        self.current_state["device_validation_status"] = 0
        self.current_state["sim_validation_status"] = 0
        self.reward=0
        #self.current_state["verification_status"] = (random.randint(0, 1), random.randint(0, 1), random.randint(0, 1))
        

    def reset(self, seed=None):
        if seed is not None:
             self.np_random.seed(seed)
        # Reset the environment to its initial state (replace with your logic)
        new_state = self.observation_space.sample()  # Example initialization
      
        #self.current_state[key] = np.array([value], dtype=int)
        new_state["address_validation_status"] = 0
        new_state["device_validation_status"] = 0
        new_state["sim_validation_status"] = 0
        self.current_state=new_state
        self.reward=0
        return self.current_state, {}

    def step(self, action):
        new_state = self.current_state.copy()  # Create a copy of the current state
        # Reward based on action and update verification status
        done = False

        if (action==Actions.AddrValidation.value and new_state["address_validation_status"] ==0):
                new_state["address_validation_status"] = 1  # Mark address as validated
                self.reward +=1
        elif (action==Actions.DeviceValidation.value and new_state["device_validation_status"]==0):
                new_state["device_validation_status"]=1
                self.reward +=1
        elif (action==Actions.SimValidation.value and new_state["sim_validation_status"]==0):
                new_state["sim_validation_status"]=1
                self.reward +=1
        elif (action==Actions.Activation.value and
               new_state["address_validation_status"] ==1 and 
               new_state["device_validation_status"]==1 and 
               new_state["sim_validation_status"]==1):  # Activate
                self.reward += 5
                done = True  # Complete order, end episode
        else:
            self.reward -=1
            done = False

        # Reward for asking for missing information
        #if action[4][0] and not self.verification_status[0]:
        #    reward += 0.5 
        terminated = False
        info = {}  # Optional: Additional information
        self.current_state = new_state
        return self.current_state, self.reward, done, terminated, info
    
    def _get_obs(self):
        return {}