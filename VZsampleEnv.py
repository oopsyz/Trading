import gym
from gym import spaces
import random

# Define observation and action spaces
observation_space = spaces.Dict({
    "customer_details": spaces.Tuple((spaces.Box(low=0, high=10000, shape=(1,), dtype=int),  # ID
                                    spaces.MultiDiscrete([26] * 40),  # Name
                                    spaces.MultiDiscrete([10] * 10 + [26] * 50)  # Contact info
                                    )),
    "order_details": spaces.Tuple((spaces.Discrete(5),  # Order type
                                  spaces.MultiDiscrete([26] * 20),  # Device model
                                  spaces.MultiDiscrete([10] * 15)  # SIM number
                                  )),
    "verification_status": spaces.Tuple((spaces.Discrete(2),  # Address
                                        spaces.Discrete(2),  # Device
                                        spaces.Discrete(2)  # SIM
                                        )),
})

action_space = spaces.Tuple((spaces.Discrete(1),  # Validate address
                              spaces.Discrete(1),  # Validate device
                              spaces.Discrete(1),  # Validate SIM
                              spaces.Discrete(1),  # Activate
                              spaces.Tuple((spaces.Discrete(1),  # Ask for ZIP code
                                           spaces.Discrete(1),  # Ask for SIM
                                           spaces.Discrete(1)  # Ask for IMEI
                                           ))
                              ))

# Define environment logic
class MobilePhoneCarrierEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        # Generate random customer details and order details
        self.customer_id = random.randint(1, 10000)
        self.customer_name = "".join([chr(random.randint(65, 90)) for _ in range(40)])
        self.contact_info = "".join([str(random.randint(0, 9)) for _ in range(10)] +
                                    [chr(random.randint(65, 90)) for _ in range(50)])
        self.order_type = random.randint(0, 4)
        self.device_model = "".join([chr(random.randint(65, 90)) for _ in range(20)])
        self.sim_number = "".join([str(random.randint(0, 9)) for _ in range(15)])
        self.verification_status = (random.randint(0, 1), random.randint(0, 1), random.randint(0, 1))

    def reset(self):
        # Reset verification status and return initial observation
        self.verification_status = (0, 0, 0)
        return {
            "customer_details": (self.customer_id, self.customer_name, self.contact_info),
            "order_details": (self.order_type, self.device_model, self.sim_number),
            "verification_status": self.verification_status
        }

    def step(self, action):
        # Reward based on action and update verification status
        reward = 0
        if action[0] and not self.verification_status[0]:  # Validate address
            self.verification_status = (1, self.verification_status[1], self.verification_status[2])
            reward += 1
        if action[1] and not self.verification_status[1]:  # Validate device
            self.verification_status = (self.verification_status[0], 1, self.verification_status[2])
            reward += 1
        if action[2] and not self.verification_status[2]:  # Validate SIM
            self.verification_status = (self.verification_status[0], self.verification_status[1], 1)
            reward += 1
        if action[3] and all(self.verification_status):  # Activate
            reward += 5
            done = True  # Complete order, end episode
        else:
            done = False

        # Reward for asking for missing information
        if action[4][0] and not self.verification_status[0]:
            reward += 0.5  