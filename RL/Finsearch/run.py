"""
This is the file which runs the animation after we have trained the model
Don't forget to use the saved weights
"""

import torch 
from inverted_pendulum import Agent
import gymnasium  as gym 
import numpy as np 
from inverted_pendulum import Agent
from utilities import plot_learning_curve
import torch

env = gym.make("CartPole-v1", render_mode = 'human')
agent = Agent(gamma = 0.99, epsilon=0.0, batch_size=64, n_actions =2, 
                  epsilon_decr= 1e-5, input_dims=(4,), lr=0.003)
scores, eps_history = [],[]
agent.Q_eval.load_state_dict(torch.load('agent_model.pth'))
observation = env.reset()
done = False
n_games = 40
for i in range(n_games):
    while not done:
        with torch.no_grad():
            env.render()
            action = agent.choose_action(observation if len(observation)==4 else observation[0]) #some bug is causing a problem so the code here
            observation_, reward, terminated, truncated, info =env.step(action)
            observation = observation_
            done = terminated or truncated