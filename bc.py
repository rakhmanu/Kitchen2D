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

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)

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

            if pour_successful and pos_ratio > 0.9:
                reward = 10
                done = True
                self.gripper.place((15, 0), 0)
                self.kitchen.liquid.remove_particles_in_cup(self.cup2)
                self.kitchen.gen_liquid_in_cup(self.cup1, N=10, userData='water')
                
            else:
                reward = -10
                done = True
                self.kitchen.liquid.remove_particles_in_cup(self.cup2)
                self.kitchen.liquid.remove_particles_outside_cup()
                self._reset_cup1()
                self._reset_gripper()
        else:
            reward = -10
            done = True
            self.kitchen.liquid.remove_particles_in_cup(self.cup2)
            self._reset_gripper()

        self.render()
        info = {}
        state = np.array([
            self.gripper.position[0], self.gripper.position[1], self.gripper.angle, 
            self.cup1.position[0], self.cup1.position[1], 
            self.cup2.position[0], self.cup2.position[1], 
            len(self.kitchen.liquid.particles)
        ])
        return state, reward, done, False, info

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
            pour_from_w, pour_from_h, pour_to_w, pour_to_h = helper.process_gp_sample(self.expid_pour, exp='pour', is_adaptive=False, flag_lk=False)[1]
            holder_d = 0.5
            self.gripper = Gripper(self.kitchen, (0, 8), 0)
            self.cup1 = ks.make_cup(self.kitchen, (15, 0), 0, pour_from_w, pour_from_h, holder_d)
            self.cup2 = ks.make_cup(self.kitchen, (-25, 0), 0, pour_to_w, pour_to_h, holder_d)
            liquid = ks.Liquid(self.kitchen, radius=0.2, liquid_frequency=5.0) 
            self.kitchen.gen_liquid_in_cup(self.cup1, N=10, userData='water')  

    

    def _reset_gripper(self):
        self.gripper.position = (0, 8)
        print("Gripper reset to starting position")

    def _reset_cup1(self):
        self.cup1.position = (15, 0)
        self.kitchen.liquid.remove_particles_in_cup(self.cup1)
        self.kitchen.gen_liquid_in_cup(self.cup1, N=10, userData='water') 
        print("Gripper reset to starting position")

    def save_demonstrations(self, filename="demonstrations.pkl"):
        with open(filename, "wb") as f:
            pickle.dump(self.demonstration_data, f)
        print(f"Demonstration data saved to {filename}")

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

from torch.utils.tensorboard import SummaryWriter  

def train_behavior_cloning(dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = BCNet(input_dim=8, output_dim=3).to(device) 
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    writer = SummaryWriter("runs/behavior_cloning")  

    for epoch in range(100):
        epoch_loss = 0.0  
        for i, (states, actions) in enumerate(dataloader):
            states, actions = states.to(device), actions.to(device)  

            optimizer.zero_grad()
            predicted_actions = model(states)
            loss = criterion(predicted_actions, actions)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            
            writer.add_scalar("Loss/batch", loss.item(), epoch * len(dataloader) + i)

        avg_epoch_loss = epoch_loss / len(dataloader) 
        writer.add_scalar("Loss/epoch", avg_epoch_loss, epoch + 1)  
        print(f"Epoch {epoch+1}, Loss: {avg_epoch_loss}")

    

    torch.save(model.state_dict(), "behavior_cloning_model.pth")
    print("Behavior Cloning model saved.")

    writer.close()  



if __name__ == "__main__":
    setting = {
        'do_gui': True,
        'left_table_width': 50.,
        'right_table_width': 50.,
        'planning': False,
        'overclock': 5 
    }
    
    env = KitchenEnv(setting)
    for episode in range(200):  
        state, info = env.reset()
        done = False
        while not done:
            action = np.random.uniform(low=-1.0, high=1.0, size=(3,)) 
            state, reward, done, _, info = env.step(action)
    
    env.save_demonstrations()
    dataloader = prepare_data()
    train_behavior_cloning(dataloader)