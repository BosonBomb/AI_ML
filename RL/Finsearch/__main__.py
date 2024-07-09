"""
Contains the training part
"""
import gymnasium  as gym 
import numpy as np 
from inverted_pendulum import Agent
from utilities import plot_learning_curve
import torch
filename = 'cartpole.png'
if __name__ == '__main__':
    env = gym.make("CartPole-v1")
    agent = Agent(gamma = 0.99, epsilon=1.0, batch_size=64, n_actions =2, 
                  epsilon_decr= 1e-5, input_dims=(4,), lr=0.003)
    n_games = 3000
    scores, eps_history = [],[]
    #Use below line if u want to use the saved weights  
    # agent.Q_eval.load_state_dict(torch.load('agent_model.pth'))
    for i in range(n_games):
        score = 0
        terminated = False
        truncated = False
        done = False
        observation = env.reset()
        while not done:
            action = agent.choose_action(observation if len(observation)==4 else observation[0]) #some bug is causing a problem so the code here
            observation_, reward, terminated, truncated, info =env.step(action)
            score +=reward
            agent.store_experience(observation if len(observation)==4 else observation[0], action, reward, observation_ if len(observation_)==4 else observation_[0] , terminated, truncated)
            agent.Q_eval.optim.zero_grad()
            agent.learn()
            observation = observation_
            done = terminated or truncated
            
        scores.append(score)
        avg_score = np.mean(scores[-100:])
        eps_history.append(agent.epsilon)

        print('episode', i, 'score %.2f' % score,
              'average score %.2f' % avg_score,
              'epsilon_score %.2f' % agent.epsilon
              )
        x= [i+1 for i in range(n_games)]
    #Use the below line if you want to save the files, else not required
    # torch.save(agent.Q_eval.state_dict(), 'agent_model.pth')
    plot_learning_curve(x, scores, eps_history, filename)