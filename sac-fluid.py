import gym
import numpy as np
from gym import spaces
from kitchen2d.kitchen_stuff import Kitchen2D
import active_learners.helper as helper
import kitchen2d.kitchen_stuff as ks
from kitchen2d.gripper import Gripper
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
import os
import random
import time
from Box2D import b2

class Cup:
    def __init__(self, b2_body, water_level=0, max_water_particles=100, world=None):
        self.body = b2_body  # b2Body from Kitchen2D
        self.water_level = water_level  # Custom water level attribute
        self.max_water_particles = max_water_particles  # Max number of water particles
        self.water_particles = []  # List of water particles
        self.world = world  # PyBox2D world
        
        # Initially, no water particles are created, the cup is empty
        self.water_particles.clear()  # Ensure no particles are present initially.

    def create_water_particles(self):
        """Generate water particles based on the current water level."""
        num_particles = int(np.clip(self.water_level, 0, self.max_water_particles))
        
        # Clear any existing particles (they will be recreated with the updated water level)
        self.water_particles.clear()

        # Create new particles inside the cup
        for i in range(num_particles):
            particle = self.create_water_particle(i)
            self.water_particles.append(particle)

    def create_water_particle(self, index):
        """Create a single water particle (small circle) at a specific height inside the cup."""
        particle_radius = 0.2  # Smaller particle radius
        x_pos = self.body.position[0]
        y_pos = self.body.position[1] + (index * 0.1)  # Stack particles vertically inside the cup

        # Create a circular particle body in the Box2D world
        particle = self.world.CreateDynamicBody(position=(x_pos, y_pos))
        particle.CreateCircleFixture(radius=particle_radius, density=1, friction=0, restitution=0)

        return particle

    def reset(self):
        """Reset the water level and particles."""
        self.water_level = 0
        self.water_particles.clear()  # Clear any existing particles
        # No particles initially, cup is empty
        self.create_water_particles()  # No water particles created as the cup starts empty
        self.body.position = (0, 0)  # Example reset position


class KitchenEnv(gym.Env):
    def __init__(self, setting):
        super(KitchenEnv, self).__init__()

        # Initialize the Kitchen2D environment
        self.setting = setting
        self.kitchen = Kitchen2D(**self.setting)

        # Initialize objects
        self.gripper = None
        self.cup1 = None
        self.cup2 = None
        self.expid_pour = 0
        
        # Create objects and environment state
        self.objects_created = False
        self._create_objects()

        # Action space: [-1, -1] to [1, 1] for gripper movement and pouring actions
        self.action_space = spaces.Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32)

        # Observation space: Gripper position, cup1/cup2 water levels, and gripper velocity (total 6)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)

    def step(self, action):
        """Perform an action and update the environment."""
        reward = 0
        done = False
        truncated = False

        # Step 1: Gripper moves based on action (without holding anything yet)
        self.gripper.find_path((15, 10), 0)  # Move to target (update this based on your agent's actions)

        # Step 2: The gripper holds cup1
        grasp_successful = self.gripper.grasp(self.cup1.body, action)  # Grasp cup1

        if grasp_successful:
            print("Grasp successful.")
            
            # Step 3: Get liquid from faucet and add particles to cup1
            self.gripper.get_liquid_from_faucet(5)  # Simulate the gripper getting liquid
            
            # Update the water level in cup1 and create water particles
            self.cup1.water_level += 5  # Add 5 units of water (or any desired amount)
            self.cup1.create_water_particles()  # Create particles based on the new water level

            # Ensure the gripper is attached before pouring
            assert self.gripper.attached, "Gripper must be attached to cup1 before pouring!"

            # Step 4: Pour liquid into cup2 (pour action)
            gp_pour, c_pour = helper.process_gp_sample(self.expid_pour, exp='pour', is_adaptive=False, flag_lk=False)
            grasp, rel_x, rel_y, dangle, _, _, _, _ = gp_pour.sample(c_pour)

            # Adjust dangle based on the sign of rel_x
            dangle *= np.sign(rel_x)

            # Pour into cup2
            pour_successful = self.gripper.pour(self.cup2.body, (rel_x, rel_y), dangle)  # Pour into cup2

            if pour_successful:
                # Update water levels (transfer liquid)
                transfer_amount = min(self.cup1.water_level, 10)  # Transfer 10 units per pour
                self.cup1.water_level -= transfer_amount
                self.cup2.water_level += transfer_amount

                reward = 10  # Reward for successful pour
                done = True  # End episode after successful pour
            else:
                reward = -10  # Penalty for failed pour
                done = True
        else:
            reward = -5  # Penalty for failed grasp
            done = True

        # Observation: Gripper position, water levels, and gripper velocity
        state = np.array(
            [self.gripper.position[0], self.gripper.position[1], self.cup1.water_level, self.cup2.water_level,
             self.gripper.velocity[0], self.gripper.velocity[1]], dtype=np.float32
        )
        info = {}

        return state, reward, done, truncated, info


    def show_water_levels(self):
        """Method to print the water levels in the cups and number of water particles."""
        print(f"Water Levels - Cup 1: {self.cup1.water_level} (Particles: {len(self.cup1.water_particles)}), "
              f"Cup 2: {self.cup2.water_level} (Particles: {len(self.cup2.water_particles)})")


    def _create_objects(self):
        """Initialize gripper and cups."""
        if not self.objects_created:
            pour_from_w, pour_from_h, pour_to_w, pour_to_h = helper.process_gp_sample(
                self.expid_pour, exp="pour", is_adaptive=False, flag_lk=False
            )[1]
            holder_d = 0.5

            # Create the gripper
            self.gripper = Gripper(self.kitchen, (0, 8), 0)

            # Create cups as b2Body objects
            self.cup1_body = ks.make_cup(self.kitchen, (15, 0), 0, pour_from_w, pour_from_h, holder_d)
            self.cup2_body = ks.make_cup(self.kitchen, (-25, 0), 0, pour_to_w, pour_to_h, holder_d)

            # Initialize Cup objects (with water level and particles)
            self.cup1 = Cup(self.cup1_body, water_level=10, world=self.kitchen.world)  # Initialize with a starting water level
            self.cup2 = Cup(self.cup2_body, water_level=0, world=self.kitchen.world)

            self.objects_created = True

    def reset(self, seed=None, **kwargs):
        """Reset the environment and return the initial state."""
        np.random.seed(seed)
        
        # Only create objects once, avoid re-initializing them unnecessarily
        if not self.objects_created:
            self._create_objects()
            self.objects_created = True
        
        # Reset environment objects and state
        state = np.zeros(self.observation_space.shape)  # Ensure this is (6,) shape
        info = {}  # You can populate this with additional information if needed
        return state, info


    def close(self):
        """Close the environment."""
        self.kitchen.close()


# Initialize and wrap the environment with DummyVecEnv for Stable Baselines3
def make_env():
    setting = {
        'do_gui': True,  # Ensure 'do_gui' is included here
        'sink_w': 10.,
        'sink_h': 5.,
        'sink_d': 1.,
        'sink_pos_x': -3.,  # Remove sink_pos_y since it's not valid in Kitchen2D
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
    num_envs = 4  # Use 4 environments in parallel
    env = DummyVecEnv([make_env for _ in range(num_envs)])

    log_dir = os.path.join(os.getcwd(), "kitchen2d_tensorboard")
    model = SAC('MlpPolicy', env, verbose=1, tensorboard_log=log_dir)

    # Start training
    model.learn(total_timesteps=10000)  # You can adjust the number of timesteps

    # Save the model after training
    model.save("pour_sac_model")


def evaluate_model(model_path="pour_sac_model"):
    # Load the trained model
    model = SAC.load(model_path)

    # Create the environment wrapped in DummyVecEnv
    env = DummyVecEnv([make_env])

    # Define evaluation parameters
    num_episodes = 5
    total_rewards = []
    success_count = 0
    success_threshold = 5  # Example threshold for success

    # Evaluation loop
    for episode in range(num_episodes):
        obs = env.reset()  # Reset environment at the start of each episode
        done = False
        episode_reward = 0

        while not done:
            action, _states = model.predict(obs)  # Predict action from the model
           
            # Get the result from env.step
            step_result = env.step(action)  # Step returns 4 values
            
            print(f"Step result: {step_result}")  # Debug print to see structure
            
            # Unpack the result into 4 values
            obs, reward, done, info = step_result  # Now unpack 4 values instead of 5

            episode_reward += reward

        total_rewards.append(episode_reward)

        # Track success (e.g., if the reward exceeds a threshold)
        if episode_reward >= success_threshold:
            success_count += 1

    # Print the evaluation results
    average_reward = sum(total_rewards) / num_episodes
    success_rate = success_count / num_episodes

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
