import argparse
from itertools import count

import gym
import gym.spaces
import scipy.optimize
import numpy as np
import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from models import *
from replay_memory import Memory
from running_state import ZFilter
from torch.autograd import Variable
from trpo import trpo_step
from utils import *
from loss import *

import matplotlib.pyplot as plt

import reacher
import ant
import swimmer
import driving
import panda_custom

import pickle

import time
import glob

import pdb

import IPython

import sys

torch.utils.backcompat.broadcast_warning.enabled = True
torch.utils.backcompat.keepdim_warning.enabled = True

torch.set_default_tensor_type('torch.DoubleTensor')
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--delta-s', type=float, default=None, help='parameter delta s')
parser.add_argument('--sigma', type=float, default=50., help='parameter sigma')
parser.add_argument('--gamma', type=float, default=0.995, metavar='G',
                    help='discount factor (default: 0.995)')
parser.add_argument('--env', type=str, default="Reacher-v1", metavar='G',
                    help='name of the environment to run')
parser.add_argument('--tau', type=float, default=0.97, metavar='G',
                    help='gae (default: 0.97)')
parser.add_argument('--l2-reg', type=float, default=1e-3, metavar='G',
                    help='l2 regularization regression (default: 1e-3)')
parser.add_argument('--max-kl', type=float, default=1e-2, metavar='G',
                    help='max kl value (default: 1e-2)')
parser.add_argument('--damping', type=float, default=1e-1, metavar='G',
                    help='damping (default: 1e-1)')
parser.add_argument('--seed', type=int, default=1111, metavar='N',
                    help='random seed (default: 1111')
parser.add_argument('--test_seed', type=int, default=2333, metavar='N',
                    help='test env random seed (default: 2333')
parser.add_argument('--batch-size', type=int, default=5000, metavar='N',
                    help='size of a single batch')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--eval-interval', type=int, default=1, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--fname', type=str, default='expert', metavar='F',
                    help='the file name to save trajectory')
parser.add_argument('--num-epochs', type=int, default=500, metavar='N',
                    help='number of epochs to train an expert')
parser.add_argument('--hidden-dim', type=int, default=100, metavar='H',
                    help='the size of hidden layers')
parser.add_argument('--lr', type=float, default=1e-3, metavar='L',
                    help='learning rate')
parser.add_argument('--optimality', action='store_true',
                    help='consider optimality or not')
parser.add_argument('--feasibility', action='store_true', 
                    help='consider feasibility or not')
parser.add_argument('--only', action='store_true',
                    help='only use labeled samples')
parser.add_argument('--noconf', action='store_true',
                    help='use only labeled data but without conf')
parser.add_argument('--vf-iters', type=int, default=30, metavar='V',
                    help='number of iterations of value function optimization iterations per each policy optimization step')
parser.add_argument('--vf-lr', type=float, default=3e-4, metavar='V',
                    help='learning rate of value network')
parser.add_argument('--noise', type=float, default=0.0, metavar='N')
parser.add_argument('--eval-epochs', type=int, default=3, metavar='E',
                    help='epochs to evaluate model')
parser.add_argument('--prior', type=float, default=0.2,
                    help='ratio of confidence data')
parser.add_argument('--ofolder', type=str, default='log')
parser.add_argument('--ifolder', type=str, default='demonstrations')
parser.add_argument('--demo_file_list', type=str, nargs='+')
parser.add_argument('--percent_list', type=float, nargs='+')
parser.add_argument('--test_episodes', type=int, help='Number of episodes')
parser.add_argument('--result_file', type=str, help='Result file name')
parser.add_argument('--snapshot_file', type=str, help='Snapshot file name')
args = parser.parse_args()

env = gym.make(args.env)

num_inputs = env.observation_space.shape[0]
num_actions = env.action_space.shape[0]

obs_len_init = num_inputs
env.reset()


env.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

policy_net = CloningPolicy(num_inputs, num_actions, args.hidden_dim).float()
value_net = Value(num_inputs, args.hidden_dim).float().to(device)

disc_criterion = nn.BCEWithLogitsLoss()
policy_optimizer = optim.Adam(policy_net.parameters(), args.lr)

def load_demos(file_list, percent_list):
    state_action_pairs = []
    confs = []
    sequences = []
    initial_reward = []
    for k in range(len(file_list)):
        fname = file_list[k]
        episodes = pickle.load(open(fname, 'rb'))
        len_epi_for_train = len(episodes) if len(episodes) < 1000 else 1000
        for j in range(int(len_epi_for_train*percent_list[k])):
            episode = episodes[j]
            sequences.append([])
            reward_sum = 0.
            for i in range(len(episode['action'])):
                sequences[-1].append(np.concatenate([episode['state'][i].squeeze(), episode['action'][i].reshape(-1)]))
                #state_action_pairs.append(np.concatenate([episode['state'][i].squeeze(), episode['action'][i].reshape(-1)]))
                if episode['action'][i].reshape(-1).shape[0] == 3:
                    state_action_pairs.append(np.concatenate([episode['state'][i].squeeze(), episode['action'][i].reshape(-1)[[0,2]]]))
                else:
                    state_action_pairs.append(np.concatenate([episode['state'][i].squeeze(), episode['action'][i].reshape(-1)]))
                #confs.append([np.exp(episode['reward'][i])])
                if type(episode['reward'][i]) == np.ndarray:
                    reward_sum += episode['reward'][i].squeeze()
                else:
                    reward_sum += episode['reward'][i]
            sequences[-1].append(np.concatenate([episode['state'][len(episode['action'])].squeeze(), episode['action'][len(episode['action'])-1].reshape(-1)]))
            for i in range(len(episode['action'])):
                confs.append([reward_sum])
            initial_reward.append(np.concatenate([episode['state'][0][0], np.array([reward_sum])]))
            sequences[-1] = np.array(sequences[-1])
    confs = np.array(confs)
    print(np.mean(confs))
    return np.array(state_action_pairs), confs, sequences, np.array(initial_reward)


def load_and_train_inverse_dynamic(args, env):
    feasible_seq = []
    feasible_traj = []
    file_list = glob.glob('demos/'+args.env+'_random_explore*.pkl')
    order = np.random.permutation(len(file_list))
    episodes = pickle.load(open(file_list[order[0]], 'rb'))
    for j in range(len(episodes)):
        episode = episodes[j]
        feasible_seq.append([])
        for i in range(len(episode['action'])):
            feasible_seq[-1].append(np.concatenate([episode['state'][i].squeeze(), episode['action'][i].reshape(-1)]))
    return feasible_seq
 
def predict_action_with_inverse(args, env, sequences, obs_len_init, num_inputs, state_action_len):
    norms = []
    state_action_pairs = []
    i = 0
    for sequence in sequences:
      try:
        env.reset()
        norms.append([])
        state = env.reset_with_obs(sequence[0][0:obs_len_init])
        for step in range(len(sequence)-1):
            action = env.inverse_dynamic(state[0:num_inputs], sequence[step+1][0:num_inputs])
            action = np.array([action])
            next_state, _, _, _ = env.step(action)
            if i >= state_action_len:
                state_action_pairs.append(np.concatenate([state[0:num_inputs], action], axis=0))
            norms[-1].append(np.linalg.norm(next_state - sequence[step+1][0:num_inputs]))
            state = next_state
        i += 1
      except Exception as reason:
        print(reason)
        pdb.set_trace()
    norms = np.array([sum(norms1) for norms1 in norms])
    return np.array(state_action_pairs), norms
   

expert_traj, expert_conf, sequences, initial_reward = load_demos(args.demo_file_list, args.percent_list)
num_label = expert_conf.shape[0]

if args.feasibility:
    feasible_seq = load_and_train_inverse_dynamic(args, env)
    all_sequences = feasible_seq + sequences
    state_action_pairs, norms = predict_action_with_inverse(args, env, all_sequences, obs_len_init, num_inputs, len(feasible_seq))
    len_list = []
    for seq in all_sequences:
        len_list.append(len(seq))
    len_list = np.array(len_list)
    norms = norms/len_list
    max_ = np.max(norms[0:len(feasible_seq)])
    min_ = np.min(norms[0:len(feasible_seq)])
    max_1 = np.max(norms[len(feasible_seq):])

    if max_*2 > max_1:
        upper_bound = max_1
    else:
        upper_bound = max_*2
    weight = (norms[len(feasible_seq):] - (min_))/(upper_bound-min_)
    weight[weight>1] = 1.0
    weight[weight<0] = 0.0
    weight_step = []
    for www in range(weight.shape[0]):
        weight_step += [weight[www]] * (len(sequences[www])-1)
    weight_step = np.array(weight_step)
    feas_weight = 1.-weight_step.reshape(-1,1)


if args.optimality:
    rectify_data = []
    rectify_label = []
    for i in range(1000):
        max_rew = -np.inf
        id_ = 0
        for j in range(1000):
            if np.linalg.norm(initial_reward[j, 0:2] - initial_reward[i, 0:2]) < 0.1:
                if max_rew < initial_reward[j, -1]:
                    max_rew = initial_reward[j, -1]
                    id_ = j
        rectify_data.append(initial_reward[i, 0:num_inputs])
        rectify_label.append(max_rew)
    rectify_data = np.array(rectify_data)
    rectify_label = np.array(rectify_label)
    reward_list = rectify_label
    rewards_all = []
    for www in range(reward_list.shape[0]):
        rewards_all += [reward_list[www]] * (len(sequences[www])-1)
    rectify_rewards = expert_conf.squeeze() - rewards_all
    rectify_rewards = np.exp(-np.square(rectify_rewards)/(2*(args.sigma**2)))
    rectify_rewards[rectify_rewards<0] = 0.
    rectify_rewards[rectify_rewards>1] = 1.
    conf_weight = rectify_rewards.reshape(-1, 1)

if (args.optimality and args.feasibility):
    weight = feas_weight * conf_weight
elif args.feasibility:
    weight = feas_weight
elif args.optimality:
    weight = conf_weight
else:
    weight = np.ones(expert_conf.shape)
weight /= np.sum(weight)
if np.sum(weight) < 1:
    weight[0] = 1-np.sum(weight[1:])
else:
    weight /= np.sum(weight)
if np.sum(weight) != 1:
    weight[1] = 1-weight[0]-np.sum(weight[2:])

mean_reward_list = []
min_reward_list = []
max_reward_list = []

all_idx = np.arange(0, expert_traj.shape[0])

snapshot_dir = os.path.dirname(args.snapshot_file)
os.makedirs(snapshot_dir, exist_ok=True)
result_dir = os.path.dirname(args.result_file)
os.makedirs(result_dir, exist_ok=True)
max_mean_reward = -1000000000

for i_episode in range(args.num_epochs):
    env.seed(int(time.time()))
    loss_count = 0
    loss_sum = 0
    for iter_num in range(expert_traj.shape[0]//64):
        policy_optimizer.zero_grad()
        idx = np.random.choice(all_idx, 64, p=weight.reshape(-1))
        expert_state_next_state = expert_traj[idx, :]
        expert_state_next_state = torch.Tensor(expert_state_next_state).float().to(device)
        e_action = policy_net(expert_state_next_state[:, 0:num_inputs])
        loss = torch.nn.SmoothL1Loss(reduction='none')(e_action.squeeze(), expert_state_next_state[:, num_inputs:].squeeze())
        loss = torch.mean(loss)
        loss.backward()
        policy_optimizer.step()
        loss_sum += loss.item()
        loss_count += 1
    print('Cloning Loss, ', loss_sum/loss_count)
    if i_episode % args.log_interval == 0:
        env.seed(args.test_seed)
        reward_list = []
        with torch.no_grad():
          for i in range(args.test_episodes):
            state = env.reset()
            reward_sum = 0
            while True:
                state = torch.from_numpy(state).float().unsqueeze(0)
                action = policy_net(state)
                action = action.data[0].numpy()
                action = np.clip(action, env.action_space.low, env.action_space.high)
                next_state, true_reward, done, infos = env.step(action)
                reward_sum += true_reward
                if done:
                    break
                state = next_state
            reward_list.append(reward_sum)
        print('Episode {}, Average reward: {:.3f}, Max reward: {:.3f}, Min reward: {:.3f}'.format(i_episode, np.mean(reward_list), max(reward_list), min(reward_list)))
        mean_reward_list.append(np.mean(reward_list))
        std_reward_list.append(reward_list)
        max_reward_list.append(max(reward_list))
        min_reward_list.append(min(reward_list))
        if max_mean_reward < np.mean(reward_list):
            max_mean_reward = np.mean(reward_list)
            torch.save({'policy_net':policy_net.cpu().state_dict(), 'value_net':value_net.cpu().state_dict()}, args.snapshot_file.replace('.tar', 'best.tar'))

        torch.save({'policy_net':policy_net.cpu().state_dict(), 'value_net':value_net.cpu().state_dict()}, args.snapshot_file)

        episode_id_list = np.array(list(range(((args.num_epochs-1) // args.log_interval) + 1))) * args.log_interval + 1
        pickle.dump((' '.join(sys.argv[1:]), episode_id_list, mean_reward_list, min_reward_list, max_reward_list, std_reward_list), open(args.result_file, 'wb'))

episode_id_list = np.array(list(range(((args.num_epochs-1) // args.log_interval) + 1))) * args.log_interval + 1
pickle.dump((' '.join(sys.argv[1:]), episode_id_list, mean_reward_list, min_reward_list, max_reward_list, std_reward_list), open(args.result_file, 'wb'))
