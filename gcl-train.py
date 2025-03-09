import gym
import numpy as np
from gym import spaces
from kitchen2d.kitchen_stuff import Kitchen2D
import active_learners.helper as helper
import kitchen2d.kitchen_stuff as ks
from kitchen2d.gripper import Gripper
import os
import gcl
from torch.utils.tensorboard import SummaryWriter
import torch
from gcl.train import train_vpg_on_session
from gcl.agentVPG import AgentVPG
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
 
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
        np.random.seed(seed)
        if not self.objects_created:
            self._create_objects()
            self.objects_created = True
        
        state = np.zeros(self.observation_space.shape, dtype=np.float32)  # Ensure correct type
        info = {}
        return state, info  # Ensure tuple (state, info)


    def step(self, action):
        """Take a step in the environment based on the action."""
        action = np.asarray(action).flatten()  # Flatten any extra dimensions

        if action.shape[0] != 3:
            raise ValueError(f"Expected action of shape (3,), but got {action.shape}")

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
                self.kitchen.liquid.remove_particles_outside_cup()
                self._reset_cup1()
                self._reset_gripper()
        else:
            reward = -10
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


        #self.fig.canvas.draw_idle()
        #self.fig.canvas.flush_events()

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

def train_gcl():
    env = make_env() 
    state_shape = env.observation_space.shape
    n_actions = env.action_space.shape[0]

    # Initialize GCL-VPG model
    model = AgentVPG(state_shape, n_actions, 'toy')  
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = StepLR(optimizer, step_size=1000, gamma=0.9)
    
    writer = SummaryWriter("./gcl_logs")

    mean_rewards = []
    total_iterations = 10

    for i in range(total_iterations):
        rewards = []
        
        for _ in range(10):  # 100 sessions per update
            session_rewards = train_vpg_on_session(model, env, *model.generate_session(env), optimizer)
            rewards.append(session_rewards)
        
        mean_reward = np.mean(rewards)
        mean_rewards.append(mean_reward)

        writer.add_scalar("TrainingReward/Mean", mean_reward, i)

        if i % 5 == 0:
            print(f"Iteration {i}: Mean reward = {mean_reward:.3f}")

        if mean_reward > 50:
            print("Stopping early: reward threshold reached.")
            break

    # After all iterations, show the final plot
    plt.clf()
    #plt.figure(figsize=[9, 6])
    plt.title("Mean reward per 10 games")
    plt.plot(mean_rewards)
    plt.grid()
    plt.savefig("mean-rew.png", dpi=300, bbox_inches="tight")
    #plt.show()
    
    writer.close()
    torch.save(model.state_dict(), 'model_vpg.pt')  

    num_demo = 50
    demo_samples = [model.generate_session(env) for i in range(num_demo)]
    new_model = gcl.AgentVPG(state_shape, n_actions, 'toy')
    cost = gcl.CostNN(state_shape)
    optimizer_model = torch.optim.Adam(new_model.parameters(), 1e-3)
    optimizer_cost = torch.optim.Adam(cost.parameters(), 1e-3)
    mean_rewards = []
    mean_costs = []
    size = 10
    samples = [new_model.generate_session(env) for _ in range(int(size/2))]
    for i in range(10):
        traj = [new_model.generate_session(env) for _ in range(int(size))]
        samples = samples + traj
        #generate samples
        demo_trajs_ids = np.random.choice(range(len(demo_samples)), size)
        demo_trajs = [demo_samples[i] for i in demo_trajs_ids]
        #sampled_trajs_ids = np.random.choice(range(len(samples)), size)
        sampled_trajs = traj#np.array(samples)[sampled_trajs_ids]
        rewards, costs = [],  []
        for (demo_traj, sampled_traj) in zip(demo_trajs, sampled_trajs):
            rew, cost_item = gcl.train_gcl_on_session(
                            new_model,
                            env,
                            cost, 
                            demo_traj,
                            sampled_traj,
                            optimizer_model, 
                            optimizer_cost,
                        )
        
        rewards.append(rew)
        costs.append(cost_item)
        mean_rewards.append(np.mean(rewards))
        mean_costs.append(np.mean(costs))
        
        if i % 5:    
            print("mean reward:%.3f" % (np.mean(rewards)))

        if np.mean(rewards) > 50:
            break
    plt.clf()   
    plt.figure(figsize=[16, 6])
    plt.subplot(1, 2, 1)
    plt.title(f"Mean reward per {size} games")
    plt.plot(mean_rewards)
    plt.grid()   
    plt.subplot(1, 2, 2)
    plt.title(f"Mean cost per {size} games")
    plt.plot(mean_costs)
    plt.grid()
    plt.savefig("mean-rew-cost.png", dpi=300, bbox_inches="tight")
    #plt.show()
    
def main():
    print("Training the model with GUI...")
    train_gcl()
   

if __name__ == "__main__":
    main()



