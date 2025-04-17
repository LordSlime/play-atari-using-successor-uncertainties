import argparse
import json
import os
import random
import sys
import time


import numpy as np
import torch
import torch.multiprocessing as _mp

import configs
import cv2
from gym import spaces
from collections import deque

import ale_py
from ale_py import ALEInterface, roms
#from ale_py.roms import Gravitar
import time
import models
from models.policies import GreedyPolicy, UncertaintyPolicy
import models.models
import play_env
import gym
import cv2
from PIL import Image
import imageio
class AtariArgParse(argparse.ArgumentParser):
    def __init__(self):
        super().__init__()
        self.add_argument('--algorithm', choices=('epsgreedy', 'successor'), default='successor')

        self.add_argument('--game', type=str, default='Pong',
                          help='Atari game to use. Openai gym lists environments in the form XXXXNoFrameskip-v4; '
                               'provide the XXXX part here.')
        self.add_argument('--total_num_steps', type=int, default=int(1e8),
                          help='Total number of training steps to perform.')
        self.add_argument('--learning_start_step', type=int, default=int(1e5),
                          help='At the start of training learning_start_step number of steps are '
                               'executed using a uniform random policy, with no training taking place.')

        """Arguments relating to case algorithm=epsgreedy"""
        self.add_argument('--eps_end_value', type=float, default=0.01,
                          help='Epsilon in [0, 1] for training.')
        self.add_argument('--eps_end_step', type=int, default=int(1e6),
                          help='Number of steps for epsilon to decay from 1.0 to eps_end_value.')

        """Arguments relating to case algorithm=successor"""
        self.add_argument('--beta', type=float, default=1e-3, help='Noise variance for linear Bayesian model.')
        self.add_argument('--decay_factor', type=float, default=1e-5,
                          help='Forgetting parameter determining the the rate with which the influence of older '
                               'observations on the Linear Bayesian Model mean and covariance estimates decays.')
        self.add_argument('--resample_interval', type=int, default=250,
                          help='Number of steps after which a new Q function estimate is sampled for use with '
                               'Posterior Sampling. A new sample is always drawn at the start of a new episode.')
        # Todo: we never looked at the effect of this!
        self.add_argument('--successor_size', type=int, default=64,
                          help='Dimensionality of state-action embeddings '
                               'and thus of the successor features themselves.')

        self.add_argument('--buffer_size', type=int, default=int(1e6))
        self.add_argument('--batch_size', type=int, default=32)
        self.add_argument('--lr', type=float, default=5e-5)
        self.add_argument('--grad_clip_norm', type=float, default=10.0)

        self.add_argument('--update_interval', type=int, default=int(1e4))
        self.add_argument('--train_interval', type=int, default=4)

        self.add_argument('--hidden_size', type=int, default=1024)

        self.add_argument('--name', type=str, default='play_test')
        self.add_argument('--action_size', type=int, default=18)

class TestArgParser(argparse.ArgumentParser):
    def __init__(self):
        super().__init__()
        self.add_argument('test_folder', type=str)
        self.add_argument('output_file', type=str)
        self.add_argument('--n', type=int, default=100)
        self.add_argument('--n_process', type=int, default=4)
        self.add_argument('--n_gpu', type=int, default=-1)
        self.add_argument('--n_thread', type=int, default=2)
        self.add_argument('--max-checkpoint', type=float, default=0)
        self.add_argument('--no_time_limit', action='store_true',
                          help='Removes 30min emulator time limit.')
        
class LazyFrames:
    def __init__(self, frames):
        """
        This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.
        """
        self._frames = frames
        self.dtype = np.float32

    def __array__(self):
        out = np.concatenate(self._frames)
        if self.dtype is not None:
            out = out.astype(self.dtype) / 255
        return out

    def torch(self):
        # noinspection PyTypeChecker
        return torch.from_numpy(np.array(self, copy=False))

def test_ALEinterface():
    config = AtariArgParse().parse_args()
    ale = ALEInterface()
    
    ale.setBool('display_screen', True)
    ale.setBool('sound', True)
    ale.loadROM("Gravitar")   ## game name
    ale.reset_game()
    print("Available actions:", ale.getMinimalActionSet())
    #ale.reset_game()
      # Check if the game is over

    state_dict = torch.load('logs/Gravitar1/checkpoints/195.0')[0]

    model = models.models.SFQNetwork(3, hidden_size=config.hidden_size,  #action_space.size
                                         successor_size=config.successor_size).cuda()
    model.load_state_dict(state_dict)

    policy = GreedyPolicy(3, model.q_fn)   ### action_space.size

    start_time = time.time()
    time_now = time.time()
    duration = time_now - start_time

    reward = ale.act(0)  # noop
    cumulative_reward = 0
    done = ale.game_over()
    while not done:
        
        screen_obs = ale.getScreenRGB()
        #print(screen_obs.shape)
        #max_frame = np.zeros((2,) + screen_obs.shape, dtype=np.uint8).max(axis=0)
        #screen_obs = spaces.Box(low=0, high=255,shape=(84, 84, 1), dtype=np.uint8)
        
        
        frame = cv2.cvtColor(screen_obs, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)
        frames = deque([], maxlen=4)
        for _ in range(4):
           frames.append(frame[None, :, :])
        #print(frames)
        frames = np.stack(frames, axis=0)
        #frames = frames / 255.0
        frames = LazyFrames(list(frames))
        
        #policy.start_new_episode()

        action = policy(state.torch().cuda())
        reward = ale.act(action)  # noop

        cumulative_reward+=reward
        time_now = time.time()
        duration = time_now - start_time
        print(f'cumulative_reward is {cumulative_reward}')
        done = ale.game_over()
    

def test():

    config = AtariArgParse().parse_args()

    action_size = config.action_size
    
    env = play_env.make_atari(f'{config.game}NoFrameskip-v4') #NoFrameskip-v4
    frame=env.render("rgb_array")
    frames = []
    obs = env.reset()
    episode_over = False

    checkpoint_folder = os.path.join(f'logs/{config.game}1/', 'checkpoints')
    checkpoints = os.listdir(checkpoint_folder)
    checkpoints = sorted(checkpoints, key=lambda x: float(x), reverse=True)
    state_dict = torch.load(f'logs/{config.game}1/checkpoints/{checkpoints[0]}')[0]
    model = models.models.SFQNetwork(action_size , hidden_size=config.hidden_size,        #action_size = action_space.size
                                         successor_size=config.successor_size).cuda()
    model.load_state_dict(state_dict)

    #policy = GreedyPolicy(18, model.q_fn)
    uncertainty_model = models.linear.SuccessorUncertaintyModel(
    input_size=config.successor_size, out_std=config.beta, bias=False,
    zero_mean_weights=True, decay_factor=config.decay_factor)

    policy = UncertaintyPolicy(action_size=action_size, uncertainty_model=uncertainty_model, q_fn=model.q_fn,
            local_embedding=model.local_embedding, global_embedding=model.global_embedding,
            resample_every=config.resample_interval)
    
    #cv2.imshow("Atari Game", frame)
    while not episode_over:
        action = policy(obs.torch().cuda())  # to implement - use `env.action_space.sample()` for a random policy
        frame = env.render("rgb_array")  # Get frame as RGB array
        frame = cv2.resize(frame, (frame.shape[1]*4 , frame.shape[0]*4 ))  
        frames.append(Image.fromarray(frame))
        #print(frame.shape)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert to OpenCV format (BGR)
        obs, reward, terminated, info= env.step(action)
        cv2.imshow("Atari Game", frame)
        if cv2.waitKey(50) & 0xFF == ord('q'):
            break
        
        
        episode_over = terminated
    env.close()
    frames[0].save(f'{config.game}.gif', save_all=True, append_images=frames[1:], loop=0, duration=6000)
    #imageio.mimsave('output.gif', frames, duration=10)
    
    

if __name__ == '__main__':
    test()