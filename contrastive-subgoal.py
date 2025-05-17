import h5py
import numpy as np


def load_contrastive_data(filename='pour_data.h5'):
    with h5py.File(filename, 'r') as f:
        data = {
            'cup1': np.stack([f['cup1_pos_x'][:], f['cup1_pos_y'][:]], axis=1),
            'cup2': np.stack([f['cup2_pos_x'][:], f['cup2_pos_y'][:]], axis=1),
            'gripper': np.stack([f['gripper_pos_x'][:], f['gripper_pos_y'][:]], axis=1),
            'success': f['success'][:],
        }

    states = np.concatenate([data['cup1'], data['cup2'], data['gripper']], axis=1)  # shape (N, 6)
    labels = data['success']

    return states, labels
