"""
Implementing the same structure as given in the Playing Atari
with Deep Reinforcement Learning
(Not using the convolutions as mentioned as its of no use here)
"""

import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
import torch.optim as optim
import torch 

gen = torch.Generator('cuda:0')

class DeepQNetwork(nn.Module):
    def __init__(self,lr ,  input_dims, fc1_dims, n_actions):
        super().__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.n_actions = n_actions
        self.fc1  = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.n_actions)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        self.loss = nn.MSELoss()
        self.optim = optim.Adam(self.parameters(), lr=lr)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        actions = self.fc2(x)
        return actions
    

# the agent would look open all the functions that are carried on
# This includes the experience relay 
class Agent():
    def __init__(self,input_dims,  n_actions, lr, epsilon, gamma,  batch_size, epsilon_decr = 5e-4, eps_end = 0.01, 
                 max_mem_size=10000):
        self.lr = lr
        self.epsilon =  epsilon
        self.epsilon_decr = epsilon_decr
        self.gamma = gamma
        self.mem_size = max_mem_size
        self.batch_size = batch_size 
        self.action_space  = [i for i in range(n_actions)]
        self.eps_end = eps_end 
        self.Q_eval = DeepQNetwork(self.lr, n_actions=n_actions, fc1_dims=128, input_dims = input_dims)
        self.mem_cntr = 0  # We are not using a deque here, this is the reason using a cntr
        # Implementing the experience relay  
        self.state_memory = np.zeros((self.mem_size, *input_dims), dtype= np.float32)
        self.reward_memory = np.zeros(self.mem_size, dtype = np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype = np.int32)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims), dtype = np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype = np.bool_)
        

    def store_experience(self, state, action, reward, state_, terminated, truncated):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = state_
        if not terminated and not truncated:
            self.terminal_memory[index] =  terminated
        else:
            self.terminal_memory[index] = terminated if terminated else truncated
        

        self.mem_cntr +=1
    
    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            states = torch.tensor(observation, dtype=torch.float32).to(self.Q_eval.device)
            action = torch.argmax(self.Q_eval(states)).item()
        else:
            action = np.random.choice(self.action_space)
        
        return action
    
    def learn(self):
        if self.mem_cntr < self.mem_size:
            return
        else:
           max_mem = min(self.mem_cntr, self.mem_size)
           batch = np.random.choice(max_mem, self.batch_size, replace=False)
           batch_index = np.arange(self.batch_size, dtype=np.int32)
           
           state_batch = torch.tensor(self.state_memory[batch]).to(self.Q_eval.device)
           action_batch = torch.tensor(self.action_memory[batch]).to(self.Q_eval.device)
           reward_batch = torch.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
           new_state_batch = torch.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)
           terminal_batch = torch.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)

           q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
           q_next = self.Q_eval.forward(new_state_batch) # selecting the max_element
           q_next[terminal_batch] = 0.0
           q_target = reward_batch + self.gamma* torch.max(q_next, dim=1)[0]
           terminal_batch = 0
           loss = self.Q_eval.loss(q_eval, q_target).to(self.Q_eval.device)
           loss.backward()
           self.Q_eval.optim.step()
           self.epsilon = self.epsilon - self.epsilon_decr if self.epsilon>self.eps_end \
            else self.eps_end
           

        
        
        
    
