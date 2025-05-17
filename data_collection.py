# Author: Ulzhalgas Rakhman

import kitchen2d.kitchen_stuff as ks
from kitchen2d.kitchen_stuff import Kitchen2D
from kitchen2d.gripper import Gripper
import numpy as np
import random
import h5py
import active_learners.helper as helper

SETTING = {
    'do_gui': False,
    'left_table_width': 50.,
    'right_table_width': 50.,
    'planning': False,
    'overclock': 5,
}

def collect_pour_sample(gp_model, constants, kitchen, cup1_pos, cup2_pos, gripper_pos):
    pour_to_w, pour_to_h, pour_from_w, pour_from_h = constants
    holder_d = 0.5
    cup_src = ks.make_cup(kitchen, cup1_pos, 0, pour_from_w, pour_from_h, holder_d)
    cup_dst = ks.make_cup(kitchen, cup2_pos, 0, pour_to_w, pour_to_h, holder_d)
    kitchen.gen_liquid_in_cup(cup_src, N=10, userData='water')
   
    grasp, rel_x, rel_y, dangle, *_ = gp_model.sample(constants)
    dangle *= np.sign(rel_x)
    gripper = Gripper(kitchen, gripper_pos, 0)
    gripper.find_path((15, 10), 0)
    try:
        success = gripper.grasp(cup_src, grasp)
    except Exception as e:
        print(f"Grasping failed due to motion planning error: {e}")
        return None

    if not success:
        print("Grasp failed.")
        return None

    try:
        pour_success, ratio = gripper.pour(cup_dst, (rel_x, rel_y), dangle)
    except AssertionError as e:
        print(f"Skipping trial: {e}")
        return None

    print(f"Pour success: {pour_success}, Ratio: {ratio:.2f}")

    gripper.place((10, 0), 0)

    return {
        'grasp': tuple(grasp) if hasattr(grasp, '__iter__') else (grasp,),
        'rel_x': float(rel_x),
        'rel_y': float(rel_y),
        'dangle': float(dangle),
        'success': int(pour_success),  # save as int (0 or 1)
        'ratio': float(ratio),
        'cup1_pos_x': float(cup1_pos[0]),
        'cup1_pos_y': float(cup1_pos[1]),
        'cup2_pos_x': float(cup2_pos[0]),
        'cup2_pos_y': float(cup2_pos[1]),
        'gripper_pos_x': float(gripper.position[0]),
        'gripper_pos_y': float(gripper.position[1]),
        'gripper_angle': float(gripper.angle) 
    }

def save_results_hdf5(results, filename='pour_data.h5'):
    grasp_array = np.array([r['grasp'] for r in results])
    rel_x_array = np.array([r['rel_x'] for r in results])
    rel_y_array = np.array([r['rel_y'] for r in results])
    dangle_array = np.array([r['dangle'] for r in results])
    success_array = np.array([r['success'] for r in results])
    ratio_array = np.array([r['ratio'] for r in results])
    cup1_x = np.array([r['cup1_pos_x'] for r in results])
    cup1_y = np.array([r['cup1_pos_y'] for r in results])
    cup2_x = np.array([r['cup2_pos_x'] for r in results])
    cup2_y = np.array([r['cup2_pos_y'] for r in results])
    grip_x = np.array([r['gripper_pos_x'] for r in results])
    grip_y = np.array([r['gripper_pos_y'] for r in results])
    grip_angle = np.array([r['gripper_angle'] for r in results])

    with h5py.File(filename, 'w') as f:
        f.create_dataset('grasp', data=grasp_array)
        f.create_dataset('rel_x', data=rel_x_array)
        f.create_dataset('rel_y', data=rel_y_array)
        f.create_dataset('dangle', data=dangle_array)
        f.create_dataset('success', data=success_array)
        f.create_dataset('ratio', data=ratio_array)
        f.create_dataset('cup1_pos_x', data=cup1_x)
        f.create_dataset('cup1_pos_y', data=cup1_y)
        f.create_dataset('cup2_pos_x', data=cup2_x)
        f.create_dataset('cup2_pos_y', data=cup2_y)
        f.create_dataset('gripper_pos_x', data=grip_x)
        f.create_dataset('gripper_pos_y', data=grip_y)
        f.create_dataset('gripper_angle', data=grip_angle)
    print(f"Saved results to {filename}")


def get_random_positions():
    TABLE_MARGIN = 5
    TABLE_WIDTH = 50
    MIN_SEPARATION = 5
    left_table_x = -20
    right_table_x = 20

    # Safe bounds for each table
    left_range = (left_table_x - (TABLE_WIDTH / 2 - TABLE_MARGIN),
                  left_table_x + (TABLE_WIDTH / 2 - TABLE_MARGIN))
    right_range = (right_table_x - (TABLE_WIDTH / 2 - TABLE_MARGIN),
                   right_table_x + (TABLE_WIDTH / 2 - TABLE_MARGIN))

    fixed_y = 0

    while True:
        left_x = np.random.uniform(*left_range)
        right_x = np.random.uniform(*right_range)

        if abs(left_x - right_x) >= MIN_SEPARATION:
            break
    if random.random() < 0.5:
        cup1_pos = (left_x, fixed_y)
        cup2_pos = (right_x, fixed_y)
    else:
        cup1_pos = (right_x, fixed_y)
        cup2_pos = (left_x, fixed_y)
    gripper_pos = (0, np.random.uniform(5, 15))

    return cup1_pos, cup2_pos, gripper_pos


def main():
    expid = 0
    gp_pour, c_pour = helper.process_gp_sample(expid, exp='pour', is_adaptive=False, flag_lk=False)

    num_trials = 100
    results = []

    for i in range(num_trials):
        print(f"\n=== Trial {i + 1} ===")
        kitchen = Kitchen2D(**SETTING)
        #kitchen.enable_gui()

        # Randomize positions
        #gripper_pos = (np.random.uniform(-10, 10), np.random.uniform(5, 15))
        #cup1_pos = (np.random.uniform(5, 15), np.random.uniform(-5, 5))
        #cup2_pos = (np.random.uniform(-30, -20), np.random.uniform(-5, 5))

        cup1_pos, cup2_pos, gripper_pos = get_random_positions()
        result = collect_pour_sample(gp_pour, c_pour, kitchen, cup1_pos, cup2_pos, gripper_pos)

        if result:
            results.append(result)
            print(f"\nCollected: trial {i + 1} ")
    if results:
        save_results_hdf5(results, 'pour_data.h5')

if __name__ == '__main__':
    main()
