import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import conv2d_size_out


class AgentVPG(nn.Module):
    def __init__(self, state_shape, n_actions, mode = 'toy'):
        
        super().__init__()
        self.state_shape = state_shape
        self.n_actions = n_actions
        if mode == 'toy':
            self.model = nn.Sequential(
              nn.Linear(in_features = state_shape[0], out_features = 128),
              nn.ReLU(),
              nn.Linear(in_features = 128 , out_features = 64),
              nn.ReLU(),
              nn.Linear(in_features = 64 , out_features = n_actions)
            )
        elif mode == 'atari':
            conv_output = conv2d_size_out(
                conv2d_size_out(
                    conv2d_size_out(
                        conv2d_size_out(64, 3, 2), 
                        3, 2), 
                    3, 2), 
                1, 1
            )         
            self.model = nn.Sequential(
              nn.Conv2d(in_channels=state_shape[0], out_channels=16, kernel_size=3, stride=2),
              nn.ReLU(),
              nn.Conv2d(16, 32, 3, stride=2),
              nn.ReLU(),
              nn.Conv2d(32, 64, 3, stride=2),
              nn.ReLU(),nn.Linear(conv_output , 256),
              nn.Linear(256 , n_actions),
                
            )
    def forward(self, x):
        logits = self.model(x)
        return logits
    
    # def predict_probs(self, states):
    #     states = torch.FloatTensor(states)
    #     logits = self.model(states).detach()
    #     probs = F.softmax(logits, dim = -1).numpy()
    #     return probs
    
    def predict_probs(self, states):
        states = np.array(states, dtype=np.float32)  # Ensure uniform NumPy array
        if states.ndim == 1:  
            states = np.expand_dims(states, axis=0)  # Add batch dimension

        states = torch.FloatTensor(states)  
        logits = self.model(states).detach()
        probs = F.softmax(logits, dim=-1).numpy()
        
        return probs

    def generate_session(self, env, t_max=1000):
        states, actions, rewards = [], [], []
        s, _ = env.reset()  # Ensure reset() returns tuple
        s = np.array(s, dtype=np.float32)  # Convert state

        for t in range(t_max):
            action_probs = self.predict_probs(s[np.newaxis, :])[0]  # Ensure shape
            print(f"Action probabilities: {action_probs}, Shape: {action_probs.shape}, Expected Actions: {self.n_actions}")

            if len(action_probs) != self.n_actions:
                raise ValueError(f"Mismatch: action_probs has {len(action_probs)} elements, expected {self.n_actions}")

            # Convert action index to one-hot vector
            a = np.zeros(self.n_actions, dtype=np.float32)
            a[np.random.choice(self.n_actions, p=action_probs)] = 1

            print(f"Selected action: {a}, Shape: {a.shape}")  # Debugging line

            new_s, r, done, _, _ = env.step(a)
            new_s = np.array(new_s, dtype=np.float32)  # Ensure type

            states.append(s)
            actions.append(a)  # Store one-hot action
            rewards.append(r)

            s = new_s
            if done:
                break

        return states, actions, rewards


    #def generate_session(self, env, t_max=1000):
    #    states, actions, rewards = [], [], []
    #    s, _ = env.reset()  # Ensure reset() returns tuple
    #    s = np.array(s, dtype=np.float32)  # Convert state

    #    for t in range(t_max):
    #        action_probs = self.predict_probs(s[np.newaxis, :])[0]  # Ensure shape
    #        print(f"Action probabilities: {action_probs}, Shape: {action_probs.shape}, Expected Actions: {self.n_actions}")

    #        if len(action_probs) != self.n_actions:
    #            raise ValueError(f"Mismatch: action_probs has {len(action_probs)} elements, expected {self.n_actions}")

    #        a = np.random.choice(self.n_actions, p=action_probs)

    #        new_s, r, done, _, _ = env.step(a)
    #        new_s = np.array(new_s, dtype=np.float32)  # Ensure type

    #        states.append(s)
    #        actions.append(a)
    #        rewards.append(r)

    #        s = new_s
    #        if done:
    #            break

    #    return states, actions, rewards


    
    #def generate_session(self, env, t_max=1000):
    #    states, actions, rewards = [], [], []
    #    s = env.reset()

    #    for t in range(t_max):
    #        action_probs = self.predict_probs(np.array([s]))[0]
    #        a = np.random.choice(self.n_actions,  p = action_probs)
    #        new_s, r, done, info = env.step(a)

    #        states.append(s)
    #        actions.append(a)
    #        rewards.append(r)

    #        s = new_s
    #        if done:
    #            break

    #    return states, actions, rewards