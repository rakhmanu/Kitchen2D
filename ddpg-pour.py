import gym
import numpy as np
from gym import spaces
from kitchen2d.kitchen_stuff import Kitchen2D
import active_learners.helper as helper
import kitchen2d.kitchen_stuff as ks
from kitchen2d.gripper import Gripper
from stable_baselines3 import DDPG
from stable_baselines3.common.vec_env import DummyVecEnv
import matplotlib.pyplot as plt
import os
 
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

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)

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
        grasp_successful = self.gripper.grasp(self.cup1, action[:2])
        self.render()
        print(f"Gripper Position: {self.gripper.position}, Cup1 Position: {self.cup1.position}")
        print(f"Grasp successful? {grasp_successful}")

        if grasp_successful:
            print("Grasp successful!")
            gp_pour, c_pour = helper.process_gp_sample(self.expid_pour, exp='pour', is_adaptive=False, flag_lk=False)
            grasp, rel_x, rel_y, dangle, *_ = gp_pour.sample(c_pour)
            dangle *= np.sign(rel_x)

            pour_successful, pos_ratio = self.gripper.pour(self.cup2, (rel_x, rel_y), dangle)

            print(f"Pouring result: {pour_successful}, Position Ratio: {pos_ratio}")

            if pour_successful and pos_ratio > 0:
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
                self._reset_cup1()
                self._reset_gripper()
        else:
            reward = -10
            done = True
            print("Grasp failed. Penalty assigned.")
            self._reset_gripper()

        self.render()
        info = {}
        return np.zeros(self.observation_space.shape), reward, done, False, info

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
            

    def _reset_gripper(self):
        self.gripper.position = (0, 8)
        print("Gripper reset to starting position")

    def _reset_cup1(self):
        self.cup1.position = (15, 0)
        self.kitchen.gen_liquid_in_cup(self.cup1, N=10, userData='water') 
        print("Gripper reset to starting position")


    def close(self):
        """Close the environment."""
        self.kitchen.close()
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

    log_dir = os.path.join(os.getcwd(), "kitchen2d_tensorboard")
    model = DDPG(
    'MlpPolicy', 
    env, 
    verbose=2, 
    tensorboard_log=log_dir, 
    learning_rate=1e-3,  
    batch_size=64        
)
    model.learn(total_timesteps=100000)  
    model.save("pour_ddpg_model")


class ModifiedKitchenEnv(KitchenEnv):
    def _create_objects(self):
        if self.cup1 is None:
            pour_from_w, pour_from_h, pour_to_w, pour_to_h = (4, 6, 8, 10)
            holder_d = 0.55
            self.gripper = Gripper(self.kitchen, (0, 8), 0)
            self.cup1 = ks.make_cup(self.kitchen, (15, 0), 0, pour_from_w, pour_from_h, holder_d)
            self.cup2 = ks.make_cup(self.kitchen, (-25, 0), 0, pour_to_w, pour_to_h, holder_d)


def evaluate_on_new_env():
    setting = {
        'do_gui': False,  
        'left_table_width': 50.,
        'right_table_width': 50.,
        'planning': False,
        'overclock': 5 
    }
    env = ModifiedKitchenEnv(setting)

    model_path = "pour_ddpg_model" 
    if not os.path.exists(model_path + ".zip"):
        raise FileNotFoundError("Trained SAC model not found.")

    model = DDPG.load(model_path)
    
    num_episodes = 50
    rewards = []
    success_threshold = 50
    success_episodes = 0

    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        truncated = False

        while not (done or truncated):
            action, _ = model.predict(state, deterministic=True)
            state, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            env.render()

        rewards.append(episode_reward)
        print(f"Episode {episode + 1}: Total Reward = {episode_reward}")
        if episode_reward >= success_threshold:
            success_episodes += 1

    average_reward = sum(rewards) / num_episodes
    success_rate = success_episodes / num_episodes

    print(f"Average reward over {num_episodes} episodes: {average_reward}")
    print(f"Success rate: {success_rate * 100}%")
    env.close()


def main():
    print("Training the model with GUI...")
    train_sac()
    #print("Training completed. Now evaluating the model...")
    #evaluate_on_new_env()

if __name__ == "__main__":
    main()
