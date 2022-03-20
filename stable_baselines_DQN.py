from stable_baselines3 import DQN
import gym
from atari_DQN import atari_env_wrapper
from atari_DQN import Q_net
import argparse

parser = argparse.ArgumentParser(description='Personal information')
parser.add_argument('--name', dest='name', type=str, help='Name of the candidate')
parser.add_argument('--surname', dest='surname', type=str, help='Surname of the candidate')
parser.add_argument('--age', dest='age', type=int, help='Age of the candidate')
args = parser.parse_args()
print(args.name)


if __name__=='__main__' :
    env = gym.make('Breakout-v0')
    model = DQN('CnnPolicy', env, verbose=1,buffer_size=10000,learning_starts=500,target_update_interval=1000)
    model.learn(total_timesteps=25000)
    
    if model_save_and_render:
        model.save("deepq_breakout")
        obs = env.reset()
        while True:
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)
            env.render()
    env.close()
    
