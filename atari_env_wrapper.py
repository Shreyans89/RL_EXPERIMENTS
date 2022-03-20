import numpy as np
import torch
import gym
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from gym.core import Env




class atari_env_wrapper(Env):
    
    """ Wrapper Class to wrap around actual env and 
    return processed next state,done and reward info
    so Qnet can ingest processed environment state to produce
    actions. Example:
    
    In Atari games the Kframes argument sets the number of frames
    to be used by the agent to produce an action.the agent repeats
    the action chosen for kframes=4 consecutive atari frames
    and concats the 4 produced observation into one tensor(frames) of
    shape 4 by 84 by 84 -4 greyscale images of size 84 by 84.
    
    The reset() method resets the wrapped atari env and also executes
    a random action kframes=4 times to produce the first video"""
    
   
    def __init__(self,env,kframes=4):
        self.env=env
        self.kframes=kframes
        self.action_space=self.env.action_space
        
        
    def preproc(self,frames):
        out=torch.tensor(np.transpose(frames,(0,3,1,2)))/255.
        out=TF.rgb_to_grayscale(out)
        out=out.squeeze()
        out=TF.resize(out,(110,84))
        out=TF.crop(out,top=110-84,left=0,height=84,width=84)
        return out
        
    def step(self,action):
        frames,rews,dones=[],0,[]
        for i in range(self.kframes):
            frame,rew,done,_=self.env.step(action)
            frames.append(frame)
            rews+=rew
            dones.append(done)
        return (self.preproc(np.stack(frames)),rews,any(dones))
    
    ## take 1 random step to produce k=4 frames for input to the Qnet
    def reset(self):
        self.env.reset()
        init_act=self.env.action_space.sample()
        return self.step(init_act)
        
        