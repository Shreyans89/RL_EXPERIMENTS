import torch
import torch.nn as nn


class pi_net(nn.Module):
    """ Simple Convnet architecture replresents a policy network,i.e the probability of taking one of 2 actions (for pong)
    given a state (concatenation of 4 consecutive atari frames"""
    
    def __init__(self,n_actions,act=nn.ReLU(),kframes=2):
        super(pi_net, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(kframes, 16, 8, 4),act)
      
        self.conv2 = nn.Sequential(nn.Conv2d(16, 32, 4, 2),act)
        self.pool1=nn.AdaptiveAvgPool2d(1)
        self.fc1=nn.Sequential(nn.Linear(32,256),act)
        self.fc2=nn.Linear(256,n_actions)
        
    
    def forward(self, x):
        out=self.conv1(x)
        out=self.conv2(out)
        out=self.pool1(out)
        out=out.squeeze()
        out=self.fc1(out)
        out=self.fc2(out)
        return out
        
        
class Q_net(nn.Module):
    """ Simple Convnet architecture replresents a policy network,i.e the probability of taking one of 4 actions (for breakout)
    given a state (concatenation of 4 consecutive atari frames"""
    
    def __init__(self,n_actions=4,act=nn.ReLU(),kframes=4):
        super(Q_net, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(kframes, 16, 8, 4),act)
      
        self.conv2 = nn.Sequential(nn.Conv2d(16, 32, 4, 2),act)
        self.pool1=nn.AdaptiveAvgPool2d(1)
        self.fc1=nn.Sequential(nn.Linear(32,256),act)
        self.fc2=nn.Linear(256,n_actions)
        
    
    def forward(self, x):
        out=self.conv1(x)
        out=self.conv2(out)
        out=self.pool1(out)
        out=out.squeeze()
        out=self.fc1(out)
        out=self.fc2(out)
        return out