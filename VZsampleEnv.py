import gymnasium as gym
from gymnasium import spaces
import numpy as np
from enum import Enum

#Action list
class Actions(Enum):
    AddrValidation=0
    DeviceValidation=1
    SimValidation=2
    ServiceActivation=3
    ReserveMDN=4
    DeactivateOld=5
    MakePayment=6
    #ActivateVAS=7
    #ServiceQualification=5
    #ServiceCatalogLookup=6
    #ServiceInventoryMgt=7
    def get_action_type(value):
        for action in Actions:
            if action.value == value:
                return action
        return None


total_rewards=0

# Define environment logic
class MobilePhoneCarrierEnv(gym.Env):
    num_envs = 1 #required by Stable Baselines3

    # Device Information
    _device_type = spaces.Discrete(3)  # Assuming 4 device types: smartphone, smartwatch, other
    _os = spaces.Discrete(3)  # iOS, Android, other

    # Sim Information
    _sim1_type = spaces.Discrete(2)  # 2 SIM types: physical, eSIM
    _sim1_status = spaces.Discrete(3)  # 3 SIM statuses: active, inactive, suspended
    _sim2_type = spaces.Discrete(2)  # 2 SIM types: physical, eSIM
    _sim2_status = spaces.Discrete(3)  # 3 SIM statuses: active, inactive, suspended

    #MDN Info
    _mdn_status = spaces.Discrete(4) # 0: na; 1: in use with sim1; 2: in use with sim 2 ; 3: reserved
    _useExistingMdn = spaces.Discrete(2) # 0: no; 1: yes 

    # Network config status
    _network_type = spaces.Discrete(4) #4G, 5G, undefined, other
    _OCS_status = spaces.Discrete(4) #configured, not configured, pending, failed
    _HLR_status = spaces.Discrete(4) #configured, not configured, pending, failed
    _ROAMGW_status = spaces.Discrete(4) #configured, not configured, pending, failed
    _TASPI_status = spaces.Discrete(4) #configured, not configured, pending, failed

    # Payment info
    _payment_status = spaces.Discrete(4) #0: pastdue, 1: paid; 2: pending, 3: new

    # Validations
    _address_validation_status = spaces.Discrete(2) # Address
    _device_validation_status = spaces.Discrete(2)  # Device
    _sim_validation_status = spaces.Discrete(2)  # SIM

    _vas_status = spaces.Discrete(3) # 0: No subscription; 1: active; 2: not active

    reward=0
    steps=0
    done = False
    
    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Dict({
            "device_type": self._device_type,
            "os": self._os,

            "sim_src_type": self._sim1_type,
            "sim_src_status": self._sim1_status,
            "sim_target_type":self._sim2_type,
            "sim_target_status":self._sim2_status,

            "network_type": self._network_type,
            "OCS_status": self._OCS_status,
            "HLR_status": self._HLR_status,
            "ROAMGW_status": self._ROAMGW_status,
            "TASPI_status": self._TASPI_status,

            "payment_status": self._payment_status,

            "address_validation_status": self._address_validation_status,
            "device_validation_status": self._device_validation_status,
            "sim_validation_status": self._sim_validation_status,
            "existing_mdn_status": self._mdn_status,
            "new_mdn_status": self._mdn_status,
            #"vas_status": self._vas_status,
            "use_existing_mdn": self._useExistingMdn

        })
        self.action_space = spaces.Discrete(len(Actions))
        self.current_state = self._get_obs()
        #self.current_state["verification_status"] = (random.randint(0, 1), random.randint(0, 1), random.randint(0, 1))
        

    def reset(self, seed=None):
        if seed is not None:
             self.np_random.seed(seed)
        # Reset the environment to its initial state (replace with your logic)
        self.current_state=self._get_obs()
        return self.current_state, {}

    def step(self, action):
        """
        Update the state of the environment based on the chosen action and calculate the reward.

        Args:
            action (int): An integer representing the chosen action.

        Returns:
            tuple: A tuple containing the updated state, reward, done flag, and additional information.
        """
        new_state = self.current_state.copy()

        if action == Actions.AddrValidation.value:
            self._address_validation(new_state)
        elif action == Actions.DeviceValidation.value:
            self._device_validation(new_state)
        elif action == Actions.SimValidation.value:
            self._sim_validation(new_state)
        elif action == Actions.ReserveMDN.value:
            self._reserve_mdn(new_state)
        elif action == Actions.ServiceActivation.value:
            self._service_activation(new_state)
        elif action == Actions.DeactivateOld.value:
            self._deactivate_old_phone(new_state)
        elif action == Actions.MakePayment.value:
            self._make_payment(new_state)
        else:
            self.reward -= 1

        self._update_done_flag(new_state)

        terminated = False
        info = {}
        self.current_state = new_state
        return self.current_state, self.reward, self.done, terminated, info

    def _address_validation(self, new_state):
        if new_state["address_validation_status"] == 0:
            new_state["address_validation_status"] = 1
            self.reward += 1
        else:
            self.reward -=1

    def _device_validation(self, new_state):
        if new_state["device_validation_status"] == 0:
            new_state["device_validation_status"] = 1
            self.reward += 1
        else:
            self.reward -=1

    def _sim_validation(self, new_state):
        if new_state["sim_validation_status"] == 0:
            new_state["sim_validation_status"] = 1
            self.reward += 1
        else:
            self.reward -=1

    def _reserve_mdn(self, new_state):
        if new_state["use_existing_mdn"] == 1:
            if new_state["existing_mdn_status"] in [1, 2, 3]:
                self.reward -= 1
            elif new_state["existing_mdn_status"] == 0:
                new_state["existing_mdn_status"] = 3
                self.reward += 1
            else:
                self.reward -= 1
        else:
            if new_state["new_mdn_status"] in [1, 2, 3]:
                self.reward -= 1
            elif new_state["new_mdn_status"] == 0:
                new_state["new_mdn_status"] = 3
                self.reward += 1
            else:
                self.reward -= 1


    def _service_activation(self, new_state):
        if not self.is_everything_valid(new_state):
            self.reward -= 1
        elif new_state["payment_status"] != 1:
            self.reward -= 1
        else:
            if new_state["use_existing_mdn"] == 0:
                if new_state["new_mdn_status"] == 3:
                    self._activate(new_state)
                else:
                    self.reward -= 1
            elif new_state["use_existing_mdn"] == 1:
                if new_state['existing_mdn_status'] == 3:
                    self._activate(new_state)
                elif new_state['existing_mdn_status'] == 2:
                    self.reward = 8
                else:
                    self.reward -= 1
            else:
                print("how did I get here?")

    def _deactivate_old_phone(self, new_state):
        if new_state["use_existing_mdn"] == 1:
            if new_state["existing_mdn_status"] == 1:
                new_state["existing_mdn_status"] = 0
                self.reward += 1
            else:
                self.reward -= 1
        else:
            self.reward -= 1

    def _make_payment(self, new_state):
        PAID = 1
        if new_state["payment_status"] == PAID:
            self.reward -= 1
        else:
            new_state["payment_status"] = PAID
            self.reward += 1

    def _update_done_flag(self, new_state):
        if new_state["use_existing_mdn"] == 0:
            if new_state['new_mdn_status'] == 2:
                self.done = True
        elif new_state["use_existing_mdn"] == 1:
            if new_state['existing_mdn_status'] == 2:
                self.done = True
        else:
            self.done = False
    
    def is_everything_valid(self, new_state):
        return (
            new_state["address_validation_status"] == 1
            and new_state["device_validation_status"] == 1
            and new_state["sim_validation_status"] == 1
        )
    def _get_obs(self):
        new_state = self.observation_space.sample()  # Example initialization
        #self.current_state[key] = np.array([value], dtype=int)
        new_state["address_validation_status"] = 0
        new_state["device_validation_status"] = 0
        new_state["sim_validation_status"] = 0
        new_state["new_mdn_status"] = 0
        self.reward=0
        self.done=False
        #new_state=self._test_case1()
        return new_state

    def set_test_data(self, testdata):
        self.current_state=testdata

    def _test_case1(self):
        print("Using ********** test case****")
        new_state = self.observation_space.sample()
        new_state["address_validation_status"] = 1
        new_state["device_validation_status"] = 1
        new_state["sim_validation_status"] = 1
        #
        new_state["new_mdn_status"] = 3
        new_state["existing_mdn_status"] = 0
        new_state["payment_status"] = 1
        new_state["use_existing_mdn"] = 1
        return new_state

    def _activate(self, new_state):
        if(new_state["use_existing_mdn"]==1):
            new_state["existing_mdn_status"]=2
        else:
            new_state["new_mdn_status"]=2
        self.reward += 10
        #print("activation using existint ",new_state["use_existing_mdn"])

    def render(self, mode='human'):
        if mode == 'human':
            print("---------- Activation Status ----------")

            print(f"Existing MDN status: {self.current_state['existing_mdn_status']}")
            print(f"New MDN status: {self.current_state['new_mdn_status']}")
            print(f"Use Existing MDN: {self.current_state['use_existing_mdn']}\n")

            print(f"Payment status: {self.current_state['payment_status']}")
            print(f"Validation address: {self.current_state['address_validation_status']}")
            print(f"Validation device: {self.current_state['device_validation_status']}")
            print(f"Validation sim: {self.current_state['sim_validation_status']}") 
            #print(f"Vas status: {self.current_state['vas_status']}")
            print(f"Reward: {self.reward}")
            if(self.done==True):
                print("***************Mobile Service Done***************")
            # ... (print other relevant status information)
            print("----------------------------------")