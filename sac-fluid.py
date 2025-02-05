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
from Box2D import b2

class Cup:
    def __init__(self, b2_body, water_level=0, max_water_particles=100, world=None):
        self.body = b2_body
        self.water_level = water_level
        self.max_water_particles = max_water_particles
        self.water_particles = []
        self.world = world

    def create_water_particles(self):
        """Create visual water particles inside the cup, but without affecting physics."""
        num_particles = int(np.clip(self.water_level, 0, self.max_water_particles))
        self.water_particles.clear()

        for i in range(num_particles):
            particle = self.create_water_particle(i)
            self.water_particles.append(particle)

    def create_water_particle(self, index):
        """Create a visual water particle inside the cup."""
        particle_radius = 0.1
        x_pos = self.body.position[0]
        y_pos = self.body.position[1] + (index * 0.1)

        # Create a lightweight dynamic body for visual effect, but set density to 0 so it doesn't affect physics
        particle = self.world.CreateDynamicBody(position=(x_pos, y_pos))
        fixture = particle.CreateCircleFixture(radius=particle_radius, density=0, friction=0, restitution=0)

        # No need to attach them to the cup fixture as they won't affect the cup's mass
        return particle

    def update_cup_mass(self):
        """Ensure the cup mass is correct and doesn't include the water particles."""
        base_density = 0.5  # The base density for the cup.
        for fixture in self.body.fixtures:
            fixture.density = base_density  # Ensure only cup’s density is used for mass calculation
        self.body.ResetMassData()  # Recalculate mass and inertia

    def disable_water_particles(self):
        """Temporarily deactivate water particles."""
        for particle in self.water_particles:
            particle.active = False

    def enable_water_particles(self):
        """Re-enable water particles."""
        for particle in self.water_particles:
            particle.active = True

class KitchenEnv(gym.Env):
    def __init__(self, setting):
        super(KitchenEnv, self).__init__()

        # Initialize the Kitchen2D environment
        self.setting = setting
        self.kitchen = Kitchen2D(**self.setting)

        self.gripper = None
        self.cup1 = None
        self.cup2 = None
        self.expid_pour = 0
        self.objects_created = False
        self._create_objects()

        self.action_space = spaces.Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)

    def step(self, action):
        """Perform an action and update the environment."""
        reward = 0
        done = False
        truncated = False

        # Ensure water particles don't interfere with gripper interaction
        self.cup1.disable_water_particles()

        # Attempt to grasp the cup
        grasp_successful = self.gripper.grasp(self.cup1.body, action)

        # Re-enable water particles after grasp attempt
        self.cup1.enable_water_particles()

        if grasp_successful:
            print("Grasp successful.")
            # Move to faucet and pour water
            faucet_position = (15, 10)
            self.gripper.find_path(faucet_position, 0)
            self.gripper.get_liquid_from_faucet(5)
            # Add water and update particles (no physics effect on the cup)
            self.cup1.water_level += 1
            self.cup1.create_water_particles()

            # Ensure gripper remains attached before moving to pour
            #assert self.gripper.attached, "Gripper must be attached to cup1 before pouring!"

            # Move to cup2 and pour
            pour_position = (self.cup2.body.position[0], self.cup2.body.position[1])
            self.gripper.find_path(pour_position, 0)
            pour_successful = self.gripper.pour(self.cup2.body, (0, -1), 0)

            if pour_successful:
                transfer_amount = min(self.cup1.water_level, 10)
                self.cup1.water_level -= transfer_amount
                self.cup2.water_level += transfer_amount
                reward = 10
                done = True
            else:
                reward = -10
                done = True
        else:
            print("Grasp failed.")
            reward = -5
            done = True

        state = np.array([
            self.gripper.position[0], self.gripper.position[1], self.cup1.water_level, self.cup2.water_level,
            self.gripper.velocity[0], self.gripper.velocity[1]
        ], dtype=np.float32)

        info = {}
        return state, reward, done, truncated, info

    def _create_objects(self):
        """Initialize gripper and cups."""
        if not self.objects_created:
            pour_from_w, pour_from_h, pour_to_w, pour_to_h = helper.process_gp_sample(
                self.expid_pour, exp="pour", is_adaptive=False, flag_lk=False
            )[1]
            holder_d = 0.5

            self.gripper = Gripper(self.kitchen, (0, 8), 0)
            self.gripper.strength = 150  

            self.cup1_body = ks.make_cup(self.kitchen, (15, 0), 0, pour_from_w, pour_from_h, holder_d)
            self.cup2_body = ks.make_cup(self.kitchen, (-25, 0), 0, pour_to_w, pour_to_h, holder_d)

            self.cup1 = Cup(self.cup1_body, water_level=10, world=self.kitchen.world)
            self.cup2 = Cup(self.cup2_body, water_level=0, world=self.kitchen.world)

            self.objects_created = True

    def reset(self, seed=None, **kwargs):
        """Reset the environment and return the initial state."""
        np.random.seed(seed)

        if not self.objects_created:
            self._create_objects()
            self.objects_created = True

        state = np.zeros(self.observation_space.shape)
        info = {}
        return state, info

    def close(self):
        """Close the environment."""
        self.kitchen.close()


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
    return KitchenEnv(setting)

def train_sac():
    env = DummyVecEnv([make_env])
    log_dir = os.path.join(os.getcwd(), "kitchen2d_tensorboard")
    model = SAC('MlpPolicy', env, verbose=1, tensorboard_log=log_dir)
    model.learn(total_timesteps=10000)
    model.save("pour_sac_model")

def main():
    print("Training the model...")
    train_sac()

if __name__ == "__main__":
    main()
