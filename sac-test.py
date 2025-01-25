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

# Initialize and wrap the environment with DummyVecEnv for Stable Baselines3
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

def train_sac():
    # Create environment wrapped in DummyVecEnv for Stable Baselines3
    num_envs = 8  # Use 8 environments in parallel
    env = DummyVecEnv([make_env for _ in range(num_envs)])
     # env = DummyVecEnv([make_env])

    log_dir = os.path.join(os.getcwd(), "kitchen2d_tensorboard")
    model = SAC('MlpPolicy', env, verbose=1, tensorboard_log=log_dir)

    # Start training
    model.learn(total_timesteps=200)  # You can adjust the number of timesteps

    # Save the model after training
    model.save("pour_sac_model-ep-200")


def evaluate_model(model_path="pour_sac_model-ep-200"):
    # Load the trained model
    model = SAC.load(model_path)

    # Create the environment wrapped in DummyVecEnv
    env = DummyVecEnv([make_env])

   # Define evaluation parameters
    num_episodes = 50
    total_rewards = []
    success_threshold = 50  # Example threshold for success
    success_episodes = 0  # Track successful episodes

    # Evaluation loop
    for episode in range(num_episodes):
        obs = env.reset()  # Reset environment at the start of each episode
        done = False
        episode_reward = 0

        while not done:
            action, _states = model.predict(obs)  # Predict action
            obs, reward, done, info = env.step(action)

            reward = reward[0]  # Convert reward to scalar (if using DummyVecEnv)
            episode_reward += reward

        total_rewards.append(episode_reward)

        print(f"Episode {episode + 1}: Total Reward = {episode_reward}")  # Debug print

        # Increment success count if the reward exceeds the threshold
        if episode_reward >= success_threshold:
            success_episodes += 1

    # Calculate statistics
    average_reward = sum(total_rewards) / num_episodes
    success_rate = success_episodes / num_episodes

    print(f"Average reward over {num_episodes} episodes: {average_reward}")
    print(f"Success rate: {success_rate * 100}%")



# Main function
def main():
    print("Training the model with GUI...")
    train_sac()
    print("Training completed. Now evaluating the model...")
    evaluate_model()

if __name__ == "__main__":
    main()


