#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 09:59:46 2022

@author: avik
"""

import os
import gym
import time
import math
import random
import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import namedtuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable


Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))

class ExperienceReplay(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        
    def push(self, *args):
        # if memory isn't full, add a new experience
        if len(self.memory) < self.capacity: 
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, bs):
        return random.sample(self.memory, bs)
    
    def __len__(self):
        return len(self.memory)

#deep Q network implementation
#class DQN(nn.Module):
#    def __init__(self, in_size, out_size):
#        super(DQN, self).__init__()
#        self.layer1 = nn.Linear(in_size, 128)
#        self.layer2 = nn.Linear(128, 64)
#        self.layer3 = nn.Linear(64, out_size)
#        self.dropout = nn.Dropout(0.7)
#    def forward(self, x):
#        # x = Variable(torch.from_numpy(x).float().unsqueeze(0)).to(device) 
#        x = F.relu(self.layer1(x))
#        x = self.dropout(F.relu(self.layer2(x)))
#        x = F.relu(self.layer3(x))
#        return x
    
class DQN(nn.Module):
    def __init__(self, in_size, out_size):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(in_size, 16)
        self.layer2 = nn.Linear(16, 32)
        self.layer3 = nn.Linear(32, 32)
        self.layer4 = nn.Linear(32, out_size)
    def forward(self, x):
        # x = Variable(torch.from_numpy(x).float().unsqueeze(0)).to(device) 
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        return x


class DQNAgent():
    def __init__(self, env, dqn, loss, args, logs = "runs"):
        self.writer = SummaryWriter(logs)
        self.logs = logs
        self.device = args.device
        self.learner = dqn
        self.target = dqn
        
        self.env = env
        self.n_a = args.n_a

        self.target.load_state_dict(self.learner.state_dict())
        self.target.eval()

        self.lr = args.lr
        self.optimizer = optim.Adam(self.learner.parameters(), lr = args.lr)
        self.loss = loss
        self.episodes = args.episodes
        self.max_epilen = args.max_epilen
        self.smooth = args.smooth
        self.memory = ExperienceReplay(args.memlen)
        self.bs = args.bs
        self.eps_start = args.eps_start
        self.eps_end = args.eps_end
        self.eps_decay = args.eps_decay
        self.target_update = args.target_update
        self.gamma = args.gamma
        self.step = 0
        self.plots = {"Loss": [], "Reward": [], "Mean Reward": [], "Test Reward": [], "Epsilon": []}
        self.savepath = args.savepath
    
    def select_action(self, state):
        #select an action based on the state
        sample = random.random()
        #get a decayed epsilon threshold
        
        if sample > self.eps:
            with torch.no_grad():
                #select the optimal action based on the maximum expected return
                action = torch.argmax(self.learner(state)).view(1,1)
            return action
        else:
            return torch.tensor([[np.random.randint(self.n_a)]], device =self.device, dtype=torch.long)
    
    def train_inner(self):
        if len(self.memory) < self.bs:
            return 0
        
        sample_transitions = self.memory.sample(self.bs)
        batch = Transition(*zip(*sample_transitions))
        
        #get a list that is True where the next state is not "done"
        has_next_state = torch.tensor(list(map(lambda s: s is not None, batch.next_state)), device = self.device, dtype=torch.bool)
        next_states = torch.cat([s for s in batch.next_state if s is not None])
        
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        
        pred_values = self.learner(state_batch).gather(1, action_batch)
        
        next_state_values = torch.zeros(self.bs, device = self.device)
        #get the max along each row using the target network, then detach
        next_state_values[has_next_state] = self.target(next_states).max(1)[0].detach()
        
        #Q(s, a) = reward(s, a) + Q(s_t+1, a_t+1)* gamma
        target_values = (next_state_values*self.gamma) + reward_batch
        
        loss = self.loss(pred_values, target_values.unsqueeze(1))
        
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.learner.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        
        return loss
    
    def env_step(self, action):
        state, reward, done, log = self.env.step(action)
        return torch.FloatTensor(np.expand_dims(state,0)).to(self.device), torch.FloatTensor(np.expand_dims(reward,0)).to(self.device), done, log
    
    def train(self):
        smoothed_reward = []
        for episode in range(self.episodes):
            self.episode = episode
            #self.eps = min(max(1 - ((1 - 0.05)/(2000 - 500))*(self.episode - 500), 0.05), 1)
            self.eps = max(1/np.sqrt(1 + (self.episode/self.eps_decay)**4), 0.35)
            c_loss = 0
            c_samples = 0
            rewards = 0
            
            state = self.env.reset()
            state = Variable(torch.from_numpy(state).float().unsqueeze(0)).to(self.device)
            for i in range(self.max_epilen):
                action  = self.select_action(state)
                next_state, reward, done, _ = self.env_step(action.item())
                
                if done:
                    next_state = None
                
                self.memory.push(state, action, next_state, reward)
                state = next_state
                self.step += 1
                #self.env.render()
                loss = self.train_inner()
                rewards += reward.detach().item()

                if done:
                    break
                
                c_samples += self.bs
                c_loss += loss
            
            smoothed_reward.append(rewards)
            test_reward = self.test(self.episode >= 4500 and self.episode%50 == 0)
            if len(smoothed_reward) > self.smooth: 
                smoothed_reward = smoothed_reward[-1*self.smooth: -1]
            
            self.writer.add_scalar("Loss", c_loss/c_samples, self.step)
            self.writer.add_scalar("Reward", rewards, self.episode)  
            self.writer.add_scalar("Mean Reward", np.mean(smoothed_reward), self.episode)
            
            if torch.is_tensor(loss):
                loss = loss.item()
            self.plots["Loss"].append(loss)
            self.plots["Reward"].append(rewards)
            self.plots["Mean Reward"].append(np.mean(smoothed_reward))
            self.plots["Test Reward"].append(test_reward)
            self.plots["Epsilon"].append(self.eps)

            if self.episode % 50 == 0:
                print("\tEpisode {} \t Epsilon: {:.2f} \t Final reward {:.2f} \t Average reward: {:.2f} \t Test reward: {:.2f}".format(episode, self.eps, rewards, np.mean(smoothed_reward), test_reward))
            
            if i % self.target_update == 0:
                self.target.load_state_dict(self.learner.state_dict())
            
        self.env.close()
    
    def plot(self):
        plt.figure()
        plt.plot(np.arange(len(self.plots["Loss"])), self.plots["Loss"])
        plt.title("DQN Gradient Loss")
        plt.xlabel("Episodes")
        plt.ylabel("Loss")
        plt.savefig(self.savepath+"plot_loss.jpg")

        plt.figure()
        plt.plot(np.arange(len(self.plots["Reward"])), self.plots["Reward"], label="Reward")
        plt.plot(np.arange(len(self.plots["Mean Reward"])), self.plots["Mean Reward"], label = "Mean Reward")
        plt.legend()
        plt.title("DQN Gradient Rewards")
        plt.xlabel("Episodes")
        plt.ylabel("Rewards")
        plt.savefig(self.savepath+"plot_rewards.jpg")
        
        torch.save(self.plots, self.savepath+"plot_data.pt")
    
    def save(self):
        torch.save(self.learner.state_dict(),self.savepath+'model.pt')
    
    def test(self, render):
        dqn = self.target
        reward = 0
        s_next = self.env.reset()
        done = False
        while not done:
            s = Variable(torch.from_numpy(s_next).float().unsqueeze(0))
            a = torch.argmax(dqn(s)).view(1,1)
            s_next, r, done, _ = self.env.step(a.item())
            if render:
                self.env.render()
            reward += r
        self.env.close()
        return reward

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=0) #cuda device
    parser.add_argument('--memlen', type=int, default=10000000) #size of replay buffer
    parser.add_argument('--verbose', type=int, default=1) #printing preferences
    parser.add_argument('--load', type=bool, default = False) #if loading an existing model
    parser.add_argument('--save', type=bool, default = False) #if saving an existing model
    parser.add_argument('--plot', type=bool, default = True) #if plotting an existing model
    parser.add_argument('--smooth', type=int, default=10) #length of running average window
    parser.add_argument('--n_a', type=int, default=2) # number of discrete actions
    parser.add_argument('--bs', type=int, default=128) # batch size
    parser.add_argument('--eps_start', type=float, default=1)
    parser.add_argument('--eps_end', type=float, default=0.2)
    parser.add_argument('--eps_decay', type=int, default=2000)
    parser.add_argument('--target_update', type=int, default=40) #interval to update target
    
    parser.add_argument('--model', type=str, default='./save/model.pt') #model - currently supports resnet and alexnet, with more to come
    parser.add_argument('--runtype', type=str, default='train_run',
		                choices=('train', 'run', 'train_run')) #runtype: train only or train and validate
    parser.add_argument('--lr', type=float, default=0.001)  #learning rate
    parser.add_argument('--episodes', type=int, default=5000) #number of episodes
    parser.add_argument('--max_epilen', default=500, type=int, help="Number of maximum episodes of the task")
    parser.add_argument('--gamma', type=float, default=0.99) #discount factor
    parser.add_argument('--test', type=bool, default=False) #to test or to train
    args = parser.parse_args()
    
    args = parser.parse_args()
    args.savepath = "./save/"
    
    if not os.path.isdir(args.savepath):
        os.makedirs(args.savepath)
    return args


def main():
    
    args = args_parser()
    env = gym.make("CartPole-v1")

    args.device = "cuda: %s"%(args.device) if torch.cuda.is_available() else "cpu"
    print("[Device]\tDevice selected: ", args.device)

    dqn = DQN(env.observation_space.shape[0], args.n_a).to(args.device)
    loss = nn.MSELoss()
    
    #if we're loading a model
    if args.load or args.test:
        dqn.load_state_dict(torch.load(args.model))
    
    if args.test:
        runner = DQNAgent(env, dqn, loss, args, logs = "dqn_cartpole")
        average_reward = runner.test()
        print("Average reward per episode {:.4f}".format(average_reward))
    else:
        runner = DQNAgent(env, dqn, loss, args, logs = "dqn_cartpole")
        
        if "train" in ['train']:
            print("[Train]\tTraining Beginning ...")
            runner.train()
    
            if True:
                print("[Plot]\tPlotting Training Curves ...")
                runner.plot()
    
        if True: 
            print("[Save]\tSaving Model ...")
            runner.save()

if __name__ == '__main__':
    main()
