import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
# --- Dataset for Contrastive Learning ---
class ContrastiveGoalDataset(Dataset):
    def __init__(self, h5_path, subgoal_horizon=5):
        self.file = h5py.File(h5_path, 'r')
        # Load position arrays as states
        self.states = np.stack([
            self.file['cup1_pos_x'][:],
            self.file['cup1_pos_y'][:],
            self.file['cup2_pos_x'][:],
            self.file['cup2_pos_y'][:],
            self.file['gripper_pos_x'][:],
            self.file['gripper_pos_y'][:],
            self.file['gripper_angle'][:], 
        ], axis=1)

        self.subgoal_horizon = subgoal_horizon

        # We assume each index is a separate trajectory point (simplification)
        self.length = len(self.states) - self.subgoal_horizon

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Current state
        st = self.states[idx]

        # Goal state (last state in trajectory, simplified as idx + subgoal_horizon)
        g = self.states[idx + self.subgoal_horizon]

        # Positive subgoal (state at idx + h, h in [1, H])
        h = np.random.randint(1, self.subgoal_horizon + 1)
        s_pos = self.states[idx + h]

        # Negative subgoals: randomly sampled other states (avoid positives)
        neg_indices = np.random.choice(
            [i for i in range(len(self.states)) if i < idx or i > idx + self.subgoal_horizon],
            size=5,
            replace=False
        )
        s_negs = self.states[neg_indices]

        return torch.FloatTensor(st), torch.FloatTensor(g), torch.FloatTensor(s_pos), torch.FloatTensor(s_negs)

# --- Neural Network Encoders ---
class TaskEncoder(nn.Module):
    def __init__(self, input_dim=7, embed_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim*2, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim)
        )

    def forward(self, st, g):
        x = torch.cat([st, g], dim=-1)  # concatenate state and goal
        return F.normalize(self.net(x), dim=-1)  # normalize embeddings

class SubgoalEncoder(nn.Module):
    def __init__(self, input_dim=7, embed_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim)
        )

    def forward(self, s):
        return F.normalize(self.net(s), dim=-1)  # normalize embeddings

# --- InfoNCE Loss ---
def info_nce_loss(task_embed, pos_embed, neg_embeds, temperature=0.1):
    # Cosine similarity
    pos_sim = torch.sum(task_embed * pos_embed, dim=-1) / temperature  # [batch_size]
    neg_sim = torch.matmul(task_embed.unsqueeze(1), neg_embeds.permute(0,2,1)).squeeze(1) / temperature  # [batch_size, neg_count]

    numerator = torch.exp(pos_sim)
    denominator = numerator + torch.sum(torch.exp(neg_sim), dim=-1)
    loss = -torch.log(numerator / denominator)
    return loss.mean()

# --- Training Loop ---
def train_contrastive(h5_path, epochs=100, batch_size=32, lr=1e-3):
    dataset = ContrastiveGoalDataset(h5_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    task_encoder = TaskEncoder()
    subgoal_encoder = SubgoalEncoder()
    optimizer = torch.optim.Adam(list(task_encoder.parameters()) + list(subgoal_encoder.parameters()), lr=lr)
    loss_history = []
    for epoch in range(epochs):
        total_loss = 0
        for st, g, s_pos, s_negs in loader:
            optimizer.zero_grad()
            fq = task_encoder(st, g)           # [batch, embed_dim]
            phi_pos = subgoal_encoder(s_pos)  # [batch, embed_dim]
            phi_negs = subgoal_encoder(s_negs)  # [batch, neg_count, embed_dim]

            loss = info_nce_loss(fq, phi_pos, phi_negs)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    return task_encoder, subgoal_encoder, loss_history, dataset

def plot_loss(loss_history):
    plt.figure(figsize=(8,5))
    plt.plot(loss_history, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()

def find_best_subgoal(task_encoder, subgoal_encoder, st, g, candidate_states):
    task_encoder.eval()
    subgoal_encoder.eval()

    with torch.no_grad():
        zq = task_encoder(st.unsqueeze(0), g.unsqueeze(0))  # [1, embed_dim]
        phi_all = subgoal_encoder(candidate_states)          # [N, embed_dim]
        sim = torch.matmul(phi_all, zq.squeeze(0))           # [N]
        best_index = torch.argmax(sim)
        best_subgoal = candidate_states[best_index]

    return best_subgoal, best_index

def visualize_embeddings(task_encoder, subgoal_encoder, dataset, before_training=False):
    loader = DataLoader(dataset, batch_size=256, shuffle=True)
    st, g, s_pos, s_negs = next(iter(loader))

    with torch.no_grad():
        zq = task_encoder(st, g).numpy()         # task embeddings
        phi_pos = subgoal_encoder(s_pos).numpy() # positive subgoal embeddings
        # s_negs shape: [batch, neg_count, feat_dim]
        # Flatten negative embeddings into [batch*neg_count, feat_dim]
        batch_size, neg_count, feat_dim = s_negs.shape
        phi_neg = subgoal_encoder(s_negs.view(batch_size * neg_count, feat_dim)).numpy()

    # Combine embeddings for visualization
    combined = np.concatenate([zq, phi_pos, phi_neg], axis=0)
    if combined.shape[0] > 1000:
        combined = combined[:1000]

    reducer = TSNE(n_components=2, random_state=42)
    reduced = reducer.fit_transform(combined)

    n_task = zq.shape[0]
    n_pos = phi_pos.shape[0]
    n_neg = phi_neg.shape[0]

    plt.figure(figsize=(10, 6))
    plt.scatter(reduced[:n_task, 0], reduced[:n_task, 1], label='Task embedding (zq)', alpha=0.6, s=20)
    plt.scatter(reduced[n_task:n_task + n_pos, 0], reduced[n_task:n_task + n_pos, 1], label='Positive subgoal (phi_pos)', alpha=0.6, s=20)
    plt.scatter(reduced[n_task + n_pos:n_task + n_pos + n_neg, 0], reduced[n_task + n_pos:n_task + n_pos + n_neg, 1], 
                label='Negative subgoal (phi_neg)', alpha=0.6, s=20)

    plt.title("Embeddings Before Training" if before_training else "Embeddings After Training")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    h5_path = 'pour_data.h5'  

    dataset = ContrastiveGoalDataset(h5_path)

    # Visualize embeddings BEFORE training (using untrained encoders)
    print("Visualizing embeddings before training...")
    task_encoder_untrained = TaskEncoder()
    subgoal_encoder_untrained = SubgoalEncoder()
    visualize_embeddings(task_encoder_untrained, subgoal_encoder_untrained, dataset, before_training=True)

    # Train model and track loss
    task_encoder, subgoal_encoder, loss_history, dataset = train_contrastive(h5_path)

    # Plot training loss
    plot_loss(loss_history)

    # Visualize embeddings AFTER training
    print("Visualizing embeddings after training...")
    visualize_embeddings(task_encoder, subgoal_encoder, dataset, before_training=False)
    st, g, s_pos, _ = dataset[0]  # example task
    all_states = torch.FloatTensor(dataset.states)

    best_subgoal, best_idx = find_best_subgoal(task_encoder, subgoal_encoder, st, g, all_states)
    print("\nBest subgoal index:", best_idx.item())
    print("Best subgoal state:", best_subgoal.numpy())