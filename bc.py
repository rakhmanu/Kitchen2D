import gym
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from gym import spaces
from kitchen2d.kitchen_stuff import Kitchen2D
import active_learners.helper as helper
import kitchen2d.kitchen_stuff as ks
from kitchen2d.gripper import Gripper
import os
import matplotlib.pyplot as plt

class KitchenEnv(gym.Env):
    def __init__(self, setting):
        super(KitchenEnv, self).__init__()
        self.setting = setting
        self.kitchen = Kitchen2D(**self.setting)
        self.gripper = None
        self.cup1 = None
        self.cup2 = None
        self.liquid = None
        
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -np.pi]), 
            high=np.array([1.0, 1.0, np.pi]), 
            dtype=np.float32
        )
        
        # Observation space now includes cup size (width, height)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32  
        )  
        
        self.objects_created = False
        self.demonstration_data = []
        self._create_objects()  

    def reset(self, seed=None, **kwargs):
        np.random.seed(seed)
        
        if not self.objects_created:
            self._create_objects()
            self.objects_created = True
        
        # Get initial state including cup size
        state = self._get_state()
        info = {}
        return state, info

    def step(self, action):
        """Take a step in the environment based on the action."""
        dx, dy, dtheta = action  
        
        new_x = np.clip(self.gripper.position[0] + dx, -30, 30)
        new_y = np.clip(self.gripper.position[1] + dy, -10, 20)
        new_theta = np.clip(self.gripper.angle + dtheta, -np.pi, np.pi)
        self.gripper.find_path((new_x, new_y), new_theta)  
        self.gripper.release()
        print(f"Trying to grasp at: {self.gripper.position}, Cup1 at: {self.cup1.position}")
        grasp_successful = self.gripper.grasp(self.cup1, action[:2])
        
        self.render()
        print(f"Gripper Position: {self.gripper.position}, Cup1 Position: {self.cup1.position}")
        print(f"Grasp successful? {grasp_successful}")

        if grasp_successful:
            print("Grasp successful!")
            gp_pour, c_pour = helper.process_gp_sample(self.expid_pour, exp='pour', is_adaptive=False, flag_lk=False)
            grasp, rel_x, rel_y, dangle, *_ = gp_pour.sample(c_pour)
            dangle *= np.sign(rel_x)
            print(f"Pour attempt: rel_x={rel_x}, rel_y={rel_y}, dangle={dangle}")
            pour_successful, pos_ratio = self.gripper.pour(self.cup2, (rel_x, rel_y), dangle)

            print(f"Pouring result: {pour_successful}, Position Ratio: {pos_ratio}")

            if pour_successful and pos_ratio > 0.9:
                reward = 10
                done = True
                print("Pouring successful! Reward assigned.")
                self.gripper.place((15, 0), 0)
                self.kitchen.liquid.remove_particles_in_cup(self.cup2)
                self.kitchen.gen_liquid_in_cup(self.cup1, N=10, userData='water')
                print("Cup1 refilled with liquid again.")
                
            else:
                reward = -10
                done = True
                print("Pouring failed. Penalty assigned.")
                self.kitchen.liquid.remove_particles_in_cup(self.cup2)
                self._reset_cup1()
                self._reset_gripper()
        else:
            reward = -10
            done = True
            print("Grasp failed. Penalty assigned.")
            self._reset_gripper()

        self.render()
        state = np.array([
            self.gripper.position[0], self.gripper.position[1], self.gripper.angle, 
            self.cup1.position[0], self.cup1.position[1], 
            self.cup2.position[0], self.cup2.position[1], 
            len(self.kitchen.liquid.particles)
        ])
        info = {}
        return state, reward, done, False, info

    def _get_state(self):
        # Get number of liquid particles in cup1
        num_liquid_particles = len(self.kitchen.liquids) if hasattr(self.kitchen, "liquids") else 0
        cup1_size = getattr(self.cup1, "userData", {}).get("size", (0, 0))
        cup2_size = getattr(self.cup2, "userData", {}).get("size", (0, 0))

        return np.array([
            self.gripper.position[0], self.gripper.position[1], self.gripper.angle,
            self.cup1.position[0], self.cup1.position[1], cup1_size[0], cup1_size[1],
            self.cup2.position[0], self.cup2.position[1], cup2_size[0], cup2_size[1],
            num_liquid_particles  # Add number of liquid particles
        ])



    def _create_objects(self):
        cup_sizes = [(5, 7), (6, 8), (7, 9)]  # Multiple cup sizes
        size_idx = np.random.randint(len(cup_sizes))
        w1, h1 = cup_sizes[size_idx]
        w2, h2 = cup_sizes[np.random.randint(len(cup_sizes))]
        
        self.gripper = Gripper(self.kitchen, (0, 8), 0)
        self.cup1 = ks.make_cup(self.kitchen, (15, 0), 0, w1, h1, 0.5)
        self.cup1.userData = {"size": (w1, h1)}
        self.cup2 = ks.make_cup(self.kitchen, (-25, 0), 0, w2, h2, 0.5)
        self.cup2.userData = {"size": (w2, h2)}
        self.kitchen.gen_liquid_in_cup(self.cup1, N=10, userData='water')  

    def save_demonstrations(self, filename="demonstrations.pkl"):
        with open(filename, "wb") as f:
            pickle.dump(self.demonstration_data, f)
        print(f"Demonstration data saved to {filename}")


class BCNet(nn.Module):
    def __init__(self, input_dim=12, output_dim=3):
        super(BCNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

def prepare_data(demo_filename="demonstrations.pkl"):
    with open(demo_filename, "rb") as f:
        demonstrations = pickle.load(f)
    
    states = np.array([demo[0] for demo in demonstrations])
    actions = np.array([demo[1] for demo in demonstrations])
    
    states_tensor = torch.tensor(states, dtype=torch.float32)
    actions_tensor = torch.tensor(actions, dtype=torch.float32)
    
    dataset = TensorDataset(states_tensor, actions_tensor)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    return dataloader

def train_behavior_cloning(dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BCNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    for epoch in range(10):
        for states, actions in dataloader:
            states, actions = states.to(device), actions.to(device)  
            optimizer.zero_grad()
            predicted_actions = model(states)
            loss = criterion(predicted_actions, actions)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")
    
    torch.save(model.state_dict(), "behavior_cloning_model.pth")
    print("Behavior Cloning model saved.")

if __name__ == "__main__":
    setting = {
        'do_gui': False,
        'left_table_width': 50.,
        'right_table_width': 50.,
        'planning': False,
        'overclock': 5 
    }
    
    env = KitchenEnv(setting)
    for episode in range(10):  
        state, info = env.reset()
        done = False
        while not done:
            action = np.random.uniform(low=-1.0, high=1.0, size=(3,)) 
            state, reward, done, _, info = env.step(action)
    
    env.save_demonstrations()
    dataloader = prepare_data()
    train_behavior_cloning(dataloader)
