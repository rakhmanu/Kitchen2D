# Author: Ulzhalgas Rakhman

import kitchen2d.kitchen_stuff as ks
from kitchen2d.kitchen_stuff import Kitchen2D
from kitchen2d.gripper import Gripper
import numpy as np
import time
import active_learners.helper as helper

SETTING = {
    'do_gui': False,  
    'left_table_width': 50.,
    'right_table_width': 50.,
    'planning': False,
    'overclock': 5 
}

def main():
    kitchen = Kitchen2D(**SETTING)

    # Enable GUI and ensure liquid rendering is enabled
    print('Initializing GUI...')
    kitchen.enable_gui()

    expid_pour, expid_scoop = 0, 0

    # Load GP model for pouring actions
    gp_pour, c_pour = helper.process_gp_sample(expid_pour, exp='pour', is_adaptive=False, flag_lk=False)
    pour_to_w, pour_to_h, pour_from_w, pour_from_h = c_pour

    holder_d = 0.5

    # Create objects
    gripper = Gripper(kitchen, (0, 8), 0)
    cup1 = ks.make_cup(kitchen, (10, 0), 0, pour_from_w, pour_from_h, holder_d)
    cup2 = ks.make_cup(kitchen, (-25, 0), 0, pour_to_w, pour_to_h, holder_d)

    # Create and add liquid with higher particle count and visibility
    liquid = ks.Liquid(kitchen, radius=0.5, liquid_frequency=10.0) 
    kitchen.gen_liquid_in_cup(cup1, N=5, userData='water')  
    print(f"Liquid particles in Liquid object: {len(liquid.particles)}")  
    

    # Pick
    grasp, rel_x, rel_y, dangle, _, _, _, _ = gp_pour.sample(c_pour)
    dangle *= np.sign(rel_x)

    gripper.find_path((15, 10), 0)
    gripper.grasp(cup1, grasp)

    # Pour
    print(gripper.pour(cup2, (rel_x, rel_y), dangle))
   
    # Place
    gripper.place((10, 0), 0)

if __name__ == '__main__':
    main()
