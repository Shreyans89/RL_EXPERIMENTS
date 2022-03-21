# RL_EXPERIMENTS
Implementations of a few deep RL algorithms (in pytorch) and running/testing on openai gym environments- ATARI games pong,breakout etc.
the goal is to maximize episodic (per game session)  reward by learning policies based on game frame pixel data.

steps to set up atari gym environment and dependencies :

# build docker container
1. docker build -f Dockerfile -t rl_container .
# run docker container
2. docker run -ti rl_container
# activate conda environment with dependencies installed
3. source activate rl
# cd to working (code) dir
4. cd src
# setup atari ROMS (required to run atari environments-pong,breakout etc.)
5. python atari_ROMS_setup.py

to run the experiments:
# for vanilla policy gradient (REINFORCE) algorithm on atari pong
python atari_REINFORCE_Pong.py
# for DQN (deep Q learning)
python atari_DQN_Breakout.py
 
