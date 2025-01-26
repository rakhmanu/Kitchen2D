import gym
import numpy as np
from gym import spaces
from kitchen2d.kitchen_stuff import Kitchen2D
import active_learners.helper as helper
import kitchen2d.kitchen_stuff as ks
from kitchen2d.gripper import Gripper
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

class KitchenEnv(gym.Env):
    def __init__(self, setting):
        super(KitchenEnv, self).__init__()

        # Environment settings
        self.setting = setting
        self.kitchen = Kitchen2D(**self.setting)
        
        # Initialize variables to hold objects (use `None` as default)
        self.gripper = None
        self.cup1 = None
        self.cup2 = None

        # Define the action and observation spaces
        self.action_space = spaces.Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)

        # Initialize objects
        self.expid_pour = 0
        self.expid_scoop = 0

        # Ensure objects are only created once
        self.objects_created = False
        self._create_objects()

        # Initialize matplotlib figure and axes for rendering
        self.fig, self.ax = plt.subplots()
        self.render_initialized = False

    def reset(self, seed=None, **kwargs):
        """Reset the environment and return the initial state."""
        np.random.seed(seed)
        
        # Only create objects once, avoid re-initializing them unnecessarily
        if not self.objects_created:
            self._create_objects()
            self.objects_created = True
        
        # Reset environment objects and state
        state = np.zeros(self.observation_space.shape)  # Modify based on your actual state representation
        info = {}  # You can populate this with additional information if needed
        return state, info

    def step(self, action):
        """Take a step in the environment based on the action."""
        state = np.zeros(self.observation_space.shape)  # Modify based on action
        reward = 0  # Default reward
        done = False  # Default termination condition
        truncated = False  # Placeholder truncated condition
        
        # Process the action (this could be any logic you want to apply)
        self.gripper.find_path((15, 10), 0)  # Update this based on your agent's actions
        
        # Try to grasp the cup (assuming `grasp` method returns a boolean indicating success)
        grasp_successful = self.gripper.grasp(self.cup1, action)
        
        if grasp_successful:
            print("Grasp successful!")
            gp_pour, c_pour = helper.process_gp_sample(self.expid_pour, exp='pour', is_adaptive=False, flag_lk=False)
            grasp, rel_x, rel_y, dangle, *_ = gp_pour.sample(c_pour)
            dangle *= np.sign(rel_x)
            self.gripper.get_liquid_from_faucet(5)
            
            # Pouring action and result validation
            pour_successful, pos_ratio = self.gripper.pour(self.cup2, (rel_x, rel_y), dangle)  # Expecting a tuple from `pour`
            print(f"Pouring result: {pour_successful}, Position Ratio: {pos_ratio}")

            if pour_successful and pos_ratio > 0:  # Ensure pouring was successful and some liquid was transferred
                reward = 10  # Reward for success
                self.gripper.place((15, 0), 0)
                done = True
                print("Pouring successful! Reward assigned.")
            else:
                reward = -10  # Penalty for failed pour
                done = True
                print("Pouring failed. Penalty assigned.")
        else:
            reward = -10  # Penalty for failed grasp
            done = True
            print("Grasp failed. Penalty assigned.")

        
        # Render the environment during each step
        self.render()

        info = {}  # You can populate this with additional information

        # Return 5 values as expected by Stable Baselines3
        return state, reward, done, truncated, info



    def render(self):
        """Render the environment using matplotlib."""
        if not self.render_initialized:
            # Set up the matplotlib figure and elements
            self.ax.set_xlim(-30, 30)
            self.ax.set_ylim(-10, 20)
            
            # Gripper, Cup1, and Cup2 plots
            self.gripper_plot, = self.ax.plot([], [], "ro", label="Gripper")  # Gripper in red
            self.cup1_plot, = self.ax.plot([], [], "bo", label="Cup1")       # Cup1 position in blue
            self.cup2_plot, = self.ax.plot([], [], "go", label="Cup2")       # Cup2 position in green
            
            # Initialize empty water plots
            self.water1_plot = None
            self.water2_plot = None
            
            self.ax.legend()
            self.render_initialized = True

        # Update gripper and cup positions
        self.gripper_plot.set_data(*self.gripper.position)
        self.cup1_plot.set_data(*self.cup1.position)
        self.cup2_plot.set_data(*self.cup2.position)

        # Clear old water plots
        if self.water1_plot:
            self.water1_plot.remove()
        if self.water2_plot:
            self.water2_plot.remove()

        # Draw water in Cup1
        water1_height = self.cup1.water_level if hasattr(self.cup1, "water_level") else 0
        if water1_height > 0:
            self.water1_plot = self.ax.fill_between(
                [self.cup1.position[0] - 1, self.cup1.position[0] + 1], 
                self.cup1.position[1], 
                self.cup1.position[1] + water1_height,
                color="blue", alpha=0.5
            )
        
        # Draw water in Cup2
        water2_height = self.cup2.water_level if hasattr(self.cup2, "water_level") else 0
        if water2_height > 0:
            self.water2_plot = self.ax.fill_between(
                [self.cup2.position[0] - 1, self.cup2.position[0] + 1], 
                self.cup2.position[1], 
                self.cup2.position[1] + water2_height,
                color="blue", alpha=0.5
            )

        # Redraw the updated figure
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()



    def _create_objects(self):
        """Create and initialize objects like gripper, cups, etc."""
        if self.cup1 is None:
            # Get dimensions of objects from GP samples
            pour_from_w, pour_from_h, pour_to_w, pour_to_h = helper.process_gp_sample(self.expid_pour, exp='pour', is_adaptive=False, flag_lk=False)[1]
            scoop_w, scoop_h = helper.process_gp_sample(self.expid_scoop, exp='scoop', is_adaptive=True, flag_lk=False)[1]
            holder_d = 0.5

            # Initialize objects only if they haven't been created yet
            self.gripper = Gripper(self.kitchen, (0, 8), 0)
            self.cup1 = ks.make_cup(self.kitchen, (15, 0), 0, pour_from_w, pour_from_h, holder_d)
            self.cup2 = ks.make_cup(self.kitchen, (-25, 0), 0, pour_to_w, pour_to_h, holder_d)

    def close(self):
        """Close the environment."""
        self.kitchen.close()
        plt.close(self.fig)


def make_env():
    setting = {
        'do_gui': True,
        'sink_w': 10.,
        'sink_h': 5.,
        'sink_d': 1.,
        'sink_pos_x': -3.,
        'left_table_width': 50.,
        'right_table_width': 50.,
        'faucet_h': 12.,
        'faucet_w': 5.,
        'faucet_d': 0.5,
        'planning': False,
        'overclock': 50
    }
    env = KitchenEnv(setting)
    return env


def collect_expert_data(env, model, num_episodes=50, save_path="expert_data.npz"):
    observations = []
    actions = []

    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False

        while not done:
            action, _ = model.predict(obs)  # Expert predicts actions
            observations.append(obs)
            actions.append(action)
            obs, _, done, _, _ = env.step(action)

    # Save data
    np.savez(save_path, observations=np.array(observations), actions=np.array(actions))
    print(f"Expert data collected and saved to {save_path}")


# 3. Behavior Cloning Model
class BehaviorCloningModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BehaviorCloningModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.net(x)


# 4. Train the BC Model
def train_bc_model(data_path="expert_data.npz", model_save_path="behavior_cloning_model.pth", epochs=100, batch_size=64):
    # Load expert data
    data = np.load(data_path)
    observations = data["observations"]
    actions = data["actions"]

    # Split data
    obs_train, obs_test, act_train, act_test = train_test_split(observations, actions, test_size=0.2, random_state=42)

    # Define model, loss, and optimizer
    input_dim = obs_train.shape[1]
    output_dim = act_train.shape[1]
    model = BehaviorCloningModel(input_dim, output_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Convert data to PyTorch tensors
    obs_train_tensor = torch.FloatTensor(obs_train)
    act_train_tensor = torch.FloatTensor(act_train)
    obs_test_tensor = torch.FloatTensor(obs_test)
    act_test_tensor = torch.FloatTensor(act_test)

    # Training loop
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # Forward pass
        predictions = model(obs_train_tensor)
        loss = criterion(predictions, act_train_tensor)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Evaluate on test data
        model.eval()
        with torch.no_grad():
            test_predictions = model(obs_test_tensor)
            test_loss = criterion(test_predictions, act_test_tensor)

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}, Test Loss: {test_loss.item()}")

    # Save the trained model
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")


# 5. Evaluate the BC Model
def evaluate_bc_model(env, model_path="behavior_cloning_model.pth", num_episodes=50):
    # Load the BC model
    model = BehaviorCloningModel(input_dim=10, output_dim=2)  # Adjust dimensions as needed
    model.load_state_dict(torch.load(model_path))
    model.eval()

    total_rewards = []
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0

        while not done:
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)  # Add batch dimension
            action = model(obs_tensor).detach().numpy().squeeze(0)  # Predict action
            obs, reward, done, _, _ = env.step(action)
            episode_reward += reward

        total_rewards.append(episode_reward)
        print(f"Episode {episode + 1}: Reward = {episode_reward}")

    average_reward = sum(total_rewards) / num_episodes
    print(f"Average Reward: {average_reward}")


# 6. Main Function to Run the Pipeline
def main():
    # 6.1. Train SAC Model (Expert)
    print("Training SAC expert...")
    env = DummyVecEnv([make_env for _ in range(8)])  # Parallel environments
    log_dir = os.path.join(os.getcwd(), "kitchen2d_tensorboard")
    model = SAC('MlpPolicy', env, verbose=1, tensorboard_log=log_dir)
    model.learn(total_timesteps=600)
    model.save("pour_sac_model-ep-600")
    print("SAC training completed.")

    # 6.2. Collect Expert Data
    print("Collecting expert data...")
    env = make_env()  # Single environment for data collection
    collect_expert_data(env=env, model=model, num_episodes=50, save_path="expert_data.npz")

    # 6.3. Train BC Model
    print("Training behavior cloning model...")
    train_bc_model(data_path="expert_data.npz", model_save_path="behavior_cloning_model.pth", epochs=100)

    # 6.4. Evaluate BC Model
    print("Evaluating behavior cloning model...")
    env = make_env()
    evaluate_bc_model(env=env, model_path="behavior_cloning_model.pth", num_episodes=50)


if __name__ == "__main__":
    main()
