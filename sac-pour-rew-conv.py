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
from stable_baselines3.common.logger import configure
import torch
 
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

    def reset(self, seed=None, **kwargs):
        """Reset the environment and return the initial state."""
        np.random.seed(seed)
        
        if not self.objects_created:
            self._create_objects()
            self.objects_created = True
        
        state = np.zeros(self.observation_space.shape)  
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
            reward = 0
            particles_in_cup2 = 0
            particles_outside = 0
            for particle in self.kitchen.liquid.particles:
                if self.kitchen.liquid.is_particle_inside_cup(particle, self.cup2):
                    particles_in_cup2 += 1
                else:
                    particles_outside += 1

            reward += particles_in_cup2 * 1 
            reward -= particles_outside * 1  

            print(f"Particles in cup2: {particles_in_cup2}, Particles outside: {particles_outside}")
            print(f"Reward assigned: {reward}")

            done = True
            self.gripper.place((15, 0), 0)
            self.kitchen.liquid.remove_particles_in_cup(self.cup2)
            self.kitchen.liquid.remove_particles_outside_cup()
            self.kitchen.gen_liquid_in_cup(self.cup1, N=10, userData='water')
            print("Cup1 refilled with liquid again.")

        else:
            reward = -1  
            done = True
            print("Grasp failed. Penalty assigned.")
            self.kitchen.liquid.remove_particles_in_cup(self.cup2)
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

   
    def render(self):
        """Render the environment using matplotlib."""
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

        print(f"Total liquid particles in simulation: {len(self.kitchen.liquid.particles)}")
        for i, particle in enumerate(self.kitchen.liquid.particles):
            print(f"Particle {i}: Position {particle.position}")


        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

        #plt.pause(0.001) 

    def _create_objects(self):
        """Create and initialize objects like gripper, cups, etc."""
        if self.cup1 is None:
            pour_from_w, pour_from_h, pour_to_w, pour_to_h = helper.process_gp_sample(self.expid_pour, exp='pour', is_adaptive=False, flag_lk=False)[1]
            holder_d = 0.5
            self.gripper = Gripper(self.kitchen, (0, 8), 0)
            self.cup1 = ks.make_cup(self.kitchen, (15, 0), 0, pour_from_w, pour_from_h, holder_d)
            self.cup2 = ks.make_cup(self.kitchen, (-25, 0), 0, pour_to_w, pour_to_h, holder_d)
            liquid = ks.Liquid(self.kitchen, radius=0.2, liquid_frequency=5.0) 
            self.kitchen.gen_liquid_in_cup(self.cup1, N=10, userData='water') 
             
    def _reset_liquid(self):
        """Remove all particles and add exactly 10 new ones."""
        print(f"Clearing old particles. Before: {len(self.kitchen.liquid.particles)}")
        self.kitchen.liquid.remove_particles_in_cup(self.cup1)
        self.kitchen.liquid.remove_particles_in_cup(self.cup2)
        self.kitchen.gen_liquid_in_cup(self.cup1, N=10, userData='water')
        print(f"New total particles: {len(self.kitchen.liquid.particles)}")

    def _reset_gripper(self):
        self.gripper.position = (0, 8)
        print("Gripper reset to starting position")

    def _reset_cup1(self):
        self.cup1.position = (15, 0)
        self.kitchen.liquid.remove_particles_in_cup(self.cup1)
        self.kitchen.gen_liquid_in_cup(self.cup1, N=10, userData='water') 
        print("Gripper reset to starting position")


    def close(self):
        """Close the environment."""
        plt.close(self.fig)

def make_env():
    setting = {
        'do_gui': False,  
        'left_table_width': 50.,
        'right_table_width': 50.,
        'planning': False,
        'overclock': 5 
    }
    env = KitchenEnv(setting)
    return env

def train_sac():
    env = DummyVecEnv([make_env])
    log_dir = "./sac_logs"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")  
    model = SAC(
        'MlpPolicy', 
        env, 
        verbose=2, 
        tensorboard_log=log_dir, 
        learning_rate=1e-5,  
        batch_size=256,
        ent_coef="auto_0.1",
        device=device 
    )

    logger = configure(log_dir, ["stdout", "tensorboard"])
    model.set_logger(logger)

    episode_rewards = []
    total_episodes = 50000
    convergence_threshold = 1e-3 
    window_size = 1000 

    while True:
        state = env.reset()
        done = False
        episode_reward = 0

        while not done:
            action, _ = model.predict(state, deterministic=True)
            step_result = env.step(action)
            if len(step_result) == 5:
                state, reward, done, truncated, info = step_result
            else:  
                state, reward, done, info = step_result
                truncated = False 

            episode_reward += reward

        episode_rewards.append(episode_reward)
        print(f"Episode {len(episode_rewards)}: Reward = {episode_reward}")
        model.logger.record("train/episode_reward", episode_reward)
        if len(episode_rewards) > window_size:
            recent_rewards = episode_rewards[-window_size:]
            avg_reward_last_window = sum(recent_rewards) / window_size
            avg_reward_prev_window = sum(episode_rewards[-2 * window_size:-window_size]) / window_size

            if abs(avg_reward_last_window - avg_reward_prev_window) < convergence_threshold:
                print(f"Training converged at episode {len(episode_rewards)} with avg reward {float(avg_reward_last_window):.2f}")

                break

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(episode_rewards) + 1), episode_rewards, marker='o', linestyle='-', color='b')
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Reward Progression")
    plt.grid()
    plt.savefig("training_rewards_sac_conv.png", dpi=300)
    print("Training reward plot saved as training_rewards_conv.png")

    model.save("pour_sac_model_conv")

  

def evaluate_on_trained_env():
    setting = {
        'do_gui': False,  
        'left_table_width': 50.,
        'right_table_width': 50.,
        'planning': False,
        'overclock': 5 
    }
    env = KitchenEnv(setting)

    model_path = "pour_sac_model_conv"
    if not os.path.exists(model_path + ".zip"):
        raise FileNotFoundError("Trained SAC model not found.")

    model = SAC.load(model_path)
    
    num_episodes = 20
    rewards = []
    success_threshold = 20
    success_episodes = 0

    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        truncated = False

        while not (done or truncated):
            action, _ = model.predict(state, deterministic=False)
            state, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            env.render()

        rewards.append(episode_reward)
        print(f"Episode {episode + 1}: Total Reward = {episode_reward}")
        if episode_reward >= success_threshold:
            success_episodes += 1

    average_reward = sum(rewards) / num_episodes
    success_rate = success_episodes / num_episodes

    print(f"Average reward: {average_reward}")
    print(f"Success rate: {success_rate * 100}%")

    env.close()

    # Plot and save the rewards
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_episodes + 1), rewards, marker='o', linestyle='-', color='b', label="Episode Reward")
    plt.xlabel("Episode Number")
    plt.ylabel("Reward")
    plt.title("Reward per Episode")
    plt.legend()
    plt.grid()
    
    # Save the plot as a PNG file
    plt.savefig("reward_plot_sac_conv.png", dpi=300)
    print("Reward plot saved as reward_plot.png")


def main():
    print("Training the model with GUI...")
    train_sac()
    print("Training completed. Now evaluating the model...")
    evaluate_on_trained_env()

if __name__ == "__main__":
    main()