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
    def __init__(self, setting, test_mode=False):
        super(KitchenEnv, self).__init__()
        self.setting = setting
        self.kitchen = Kitchen2D(**self.setting)
        self.test_mode = test_mode  

        self.gripper = None
        self.cup1 = None
        self.cup2 = None
        self.liquid = None

        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -np.pi]), 
            high=np.array([1.0, 1.0, np.pi]), 
            dtype=np.float32
        )

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32)

        self.expid_pour = 0
        self.expid_scoop = 0
        self.objects_created = False
        self._create_objects()

        self.fig, self.ax = plt.subplots()
        self.render_initialized = False
        self.demonstration_data = []

    def reset(self, seed=None, **kwargs):
        np.random.seed(seed)
        if not self.objects_created:
            self._create_objects()
            self.objects_created = True
        
        state = np.zeros(self.observation_space.shape)  
        info = {}  
        return state, info

    def step(self, action):
        dx, dy, dtheta = action  
        
        new_x = np.clip(self.gripper.position[0] + dx, -30, 30)
        new_y = np.clip(self.gripper.position[1] + dy, -10, 20)
        new_theta = np.clip(self.gripper.angle + dtheta, -np.pi, np.pi)
        self.gripper.find_path((new_x, new_y), new_theta)  
        self.gripper.release()
        grasp_successful = self.gripper.grasp(self.cup1, action[:2])
        self.render()
        state = np.zeros(self.observation_space.shape)
        self.demonstration_data.append((state, action))  

        if grasp_successful:
            gp_pour, c_pour = helper.process_gp_sample(self.expid_pour, exp='pour', is_adaptive=False, flag_lk=False)
            grasp, rel_x, rel_y, dangle, *_ = gp_pour.sample(c_pour)
            dangle *= np.sign(rel_x)

            pour_successful, pos_ratio = self.gripper.pour(self.cup2, (rel_x, rel_y), dangle)

            if pour_successful and pos_ratio > 0:
                reward = 10
                done = True
                self.gripper.place((15, 0), 0)
                self.kitchen.liquid.remove_particles_in_cup(self.cup2)
                self.kitchen.gen_liquid_in_cup(self.cup1, N=10, userData='water')
                
            else:
                reward = -10
                done = True
                self._reset_cup1()
                self._reset_gripper()
        else:
            reward = -10
            done = True
            self._reset_gripper()

        self.render()
        info = {}
        return np.zeros(self.observation_space.shape), reward, done, False, info
    def render(self):
        if not self.render_initialized:
            self.ax.set_xlim(-30, 30)
            self.ax.set_ylim(-10, 20)
            self.gripper_plot, = self.ax.plot([], [], "ro", label="Gripper") 
            self.cup1_plot, = self.ax.plot([], [], "yo", label="Cup1")        
            self.cup2_plot, = self.ax.plot([], [], "go", label="Cup2")        
            self.water_plot = []
            self.ax.legend()
            self.render_initialized = True

        self.gripper_plot.set_data(self.gripper.position[0], self.gripper.position[1])
        self.cup1_plot.set_data(self.cup1.position[0], self.cup1.position[1])
        self.cup2_plot.set_data(self.cup2.position[0], self.cup2.position[1])

        for water_particle in self.water_plot:
            water_particle.remove() 

        self.water_plot = []  

        for particle in self.kitchen.liquid.particles:
            pos = particle.position  
            water_particle = self.ax.plot(pos[0], pos[1], "bo", markersize=2)  
            self.water_plot.append(water_particle[0]) 

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def _create_objects(self):
        if self.cup1 is None:
            if self.test_mode:
            
                pour_from_w, pour_from_h = np.random.uniform(2, 5), np.random.uniform(3, 6)  
                pour_to_w, pour_to_h = np.random.uniform(3, 6), np.random.uniform(4, 7) 
            else:
              
                pour_from_w, pour_from_h, pour_to_w, pour_to_h = helper.process_gp_sample(
                    self.expid_pour, exp='pour', is_adaptive=False, flag_lk=False
                )[1]

            holder_d = 0.5
            self.gripper = Gripper(self.kitchen, (0, 8), 0)
            self.cup1 = ks.make_cup(self.kitchen, (15, 0), 0, pour_from_w, pour_from_h, holder_d)
            self.cup2 = ks.make_cup(self.kitchen, (-25, 0), 0, pour_to_w, pour_to_h, holder_d)
            self.kitchen.gen_liquid_in_cup(self.cup1, N=10, userData='water')

    def _reset_gripper(self):
        self.gripper.position = (0, 8)

    def _reset_cup1(self):
        self.cup1.position = (15, 0)
        self.kitchen.gen_liquid_in_cup(self.cup1, N=10, userData='water') 

class BCNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BCNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def evaluate_model(env, model, num_episodes=10):
    success_count = 0
    total_reward = 0

    for episode in range(num_episodes):
        env = KitchenEnv(env.setting, test_mode=True) 
        state, info = env.reset()
        done = False
        episode_reward = 0

        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32).to(device).unsqueeze(0)
            with torch.no_grad():
                action = model(state_tensor).cpu().numpy().squeeze()

            state, reward, done, _, info = env.step(action)
            episode_reward += reward

        total_reward += episode_reward
        if episode_reward > 0:
            success_count += 1

    success_rate = success_count / num_episodes
    avg_reward = total_reward / num_episodes
    print(f"Success Rate on Different Cup Sizes: {success_rate * 100:.2f}%")
    print(f"Average Reward: {avg_reward:.2f}")
    return success_rate, avg_reward

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BCNet(input_dim=12, output_dim=3).to(device)
model.load_state_dict(torch.load("behavior_cloning_model.pth", map_location=device))
model.eval()
setting = {
    'do_gui': True,
    'left_table_width': 50.,
    'right_table_width': 50.,
    'planning': False,
    'overclock': 5 
}

env = KitchenEnv(setting, test_mode=True)  
evaluate_model(env, model, num_episodes=10)
