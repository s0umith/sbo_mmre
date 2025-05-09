"""Continuous action policy for the Rover environment."""

from typing import Tuple, List
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

class ActorNetwork(nn.Module):
    """Neural network for the actor (policy) in continuous action space."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_size: int):
        """Initialize actor network.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_size: Size of hidden layers
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim * 2)  # Mean and log_std
        )
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the network.
        
        Args:
            state: Input state tensor
            
        Returns:
            Tuple of mean and log_std of action distribution
        """
        output = self.net(state)
        mean, log_std = torch.chunk(output, 2, dim=-1)
        log_std = torch.clamp(log_std, -20, 2)  # Prevent numerical instability
        return mean, log_std

class CriticNetwork(nn.Module):
    """Neural network for the critic (value function) in continuous action space."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_size: int):
        """Initialize critic network.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_size: Size of hidden layers
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.
        
        Args:
            state: Input state tensor
            action: Input action tensor
            
        Returns:
            Value estimate
        """
        x = torch.cat([state, action], dim=-1)
        return self.net(x)

class ContinuousActionPolicy:
    """Policy for continuous action space using actor-critic architecture."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_size: int = 128,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        exploration_noise: float = 0.1,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """Initialize policy.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_size: Size of hidden layers
            learning_rate: Learning rate for optimization
            gamma: Discount factor
            tau: Target network update rate
            exploration_noise: Standard deviation of exploration noise
            device: Device to run the model on
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.exploration_noise = exploration_noise
        
        # Initialize networks
        self.actor = ActorNetwork(state_dim, action_dim, hidden_size).to(device)
        self.critic = CriticNetwork(state_dim, action_dim, hidden_size).to(device)
        self.target_critic = CriticNetwork(state_dim, action_dim, hidden_size).to(device)
        
        # Copy weights to target network
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        # Initialize optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)
        
        # Initialize replay buffer
        self.replay_buffer: List[Tuple] = []
        self.max_buffer_size = 10000
        self.batch_size = 64
        
    def get_action(self, state: np.ndarray) -> np.ndarray:
        """Get action from policy.
        
        Args:
            state: Current state
            
        Returns:
            Selected action
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            mean, log_std = self.actor(state_tensor)
            std = log_std.exp()
            
            # Sample action from normal distribution
            normal = Normal(mean, std)
            action = normal.sample()
            
            # Add exploration noise
            action = action + torch.randn_like(action) * self.exploration_noise
            
            # Clip action to [-1, 1]
            action = torch.clamp(action, -1.0, 1.0)
            
            return action.cpu().numpy()[0]
            
    def update(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """Update policy with new experience.
        
        Args:
            state: Current state
            action: Taken action
            reward: Received reward
            next_state: Next state
            done: Whether episode is done
        """
        # Add to replay buffer
        self.replay_buffer.append((state, action, reward, next_state, done))
        if len(self.replay_buffer) > self.max_buffer_size:
            self.replay_buffer.pop(0)
            
        # Only update if we have enough samples
        if len(self.replay_buffer) < self.batch_size:
            return
            
        # Sample batch
        indices = np.random.choice(len(self.replay_buffer), self.batch_size)
        batch = [self.replay_buffer[i] for i in indices]
        
        # Convert to tensors
        states = torch.FloatTensor([b[0] for b in batch]).to(self.device)
        actions = torch.FloatTensor([b[1] for b in batch]).to(self.device)
        rewards = torch.FloatTensor([b[2] for b in batch]).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor([b[3] for b in batch]).to(self.device)
        dones = torch.FloatTensor([b[4] for b in batch]).unsqueeze(1).to(self.device)
        
        # Update critic
        with torch.no_grad():
            next_actions_mean, next_actions_log_std = self.actor(next_states)
            next_actions_std = next_actions_log_std.exp()
            next_actions = Normal(next_actions_mean, next_actions_std).sample()
            next_values = self.target_critic(next_states, next_actions)
            target_values = rewards + (1 - dones) * self.gamma * next_values
            
        current_values = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_values, target_values)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update actor
        actions_mean, actions_log_std = self.actor(states)
        actions_std = actions_log_std.exp()
        actions_dist = Normal(actions_mean, actions_std)
        actions_sample = actions_dist.rsample()
        actor_loss = -self.critic(states, actions_sample).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update target networks
        for param, target_param in zip(
            self.critic.parameters(), self.target_critic.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            ) 