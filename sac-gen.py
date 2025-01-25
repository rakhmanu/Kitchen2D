import gym
import numpy as np
from gym import spaces
from kitchen2d.kitchen_stuff import Kitchen2D
import active_learners.helper as helper
import kitchen2d.kitchen_stuff as ks
from kitchen2d.gripper import Gripper
from stable_baselines3 import SAC
import matplotlib.pyplot as plt
import os


class KitchenEnv(gym.Env):
    def __init__(self, setting):
        super(KitchenEnv, self).__init__()

        # Environment settings
        self.setting = setting
        self.kitchen = Kitchen2D(**self.setting)
        
        # Initialize variables to hold objects
        self.gripper = None
        self.cup1 = None
        self.cup2 = None

        # Define action and observation spaces
        self.action_space = spaces.Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)

        # Initialize objects
        self.expid_pour = 0
        self.expid_scoop = 0
        self.objects_created = False
        self._create_objects()

        # Initialize rendering
        self.fig, self.ax = plt.subplots()
        self.render_initialized = False

    def reset(self, seed=None, **kwargs):
        np.random.seed(seed)
        
        if not self.objects_created:
            self._create_objects()
            self.objects_created = True
        
        state = np.zeros(self.observation_space.shape)
        info = {}
        return state, info

    def step(self, action):
        state = np.zeros(self.observation_space.shape)  # Modify based on action
        reward = 0  # Default reward
        done = False  # Default termination condition
        truncated = False  # Placeholder for truncation

        self.gripper.find_path((15, 10), 0)  # Update this based on action
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


        self.render()
        info = {}
        return state, reward, done, truncated, info

    def render(self):
        if not self.render_initialized:
            self.ax.set_xlim(-30, 30)
            self.ax.set_ylim(-10, 20)
            self.gripper_plot, = self.ax.plot([], [], "ro", label="Gripper")
            self.cup1_plot, = self.ax.plot([], [], "bo", label="Cup1")
            self.cup2_plot, = self.ax.plot([], [], "go", label="Cup2")
            self.water1_plot = None
            self.water2_plot = None
            self.ax.legend()
            self.render_initialized = True

        self.gripper_plot.set_data(*self.gripper.position)
        self.cup1_plot.set_data(*self.cup1.position)
        self.cup2_plot.set_data(*self.cup2.position)

        if self.water1_plot:
            self.water1_plot.remove()
        if self.water2_plot:
            self.water2_plot.remove()

        water1_height = self.cup1.water_level if hasattr(self.cup1, "water_level") else 0
        if water1_height > 0:
            self.water1_plot = self.ax.fill_between(
                [self.cup1.position[0] - 1, self.cup1.position[0] + 1], 
                self.cup1.position[1], 
                self.cup1.position[1] + water1_height,
                color="blue", alpha=0.5
            )

        water2_height = self.cup2.water_level if hasattr(self.cup2, "water_level") else 0
        if water2_height > 0:
            self.water2_plot = self.ax.fill_between(
                [self.cup2.position[0] - 1, self.cup2.position[0] + 1], 
                self.cup2.position[1], 
                self.cup2.position[1] + water2_height,
                color="blue", alpha=0.5
            )

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def _create_objects(self):
        if self.cup1 is None:
            pour_from_w, pour_from_h, pour_to_w, pour_to_h = helper.process_gp_sample(self.expid_pour, exp='pour', is_adaptive=False, flag_lk=False)[1]
            holder_d = 0.5
            self.gripper = Gripper(self.kitchen, (0, 8), 0)
            self.cup1 = ks.make_cup(self.kitchen, (15, 0), 0, pour_from_w, pour_from_h, holder_d)
            self.cup2 = ks.make_cup(self.kitchen, (-25, 0), 0, pour_to_w, pour_to_h, holder_d)

    def close(self):
        #self.kitchen.close()
        plt.close(self.fig)


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
    env = ModifiedKitchenEnv(setting)

    model_path = "pour_sac_model-ep-200" 
    if not os.path.exists(model_path + ".zip"):
        raise FileNotFoundError("Trained SAC model not found.")

    model = SAC.load(model_path)
    
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


if __name__ == "__main__":
    evaluate_on_new_env()
