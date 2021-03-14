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
                    help='use optimality or not')
parser.add_argument('--inverse_weight', action='store_true', 
                    help='use feasibility or not')
parser.add_argument('--only', action='store_true',
                    help='only use labeled samples')
parser.add_argument('--noconf', action='store_true',
                    help='use only labeled data but without conf')
parser.add_argument('--vf-iters', type=int, default=30, metavar='V',
                    help='number of iterations of value function optimization iterations per each policy optimization step')
parser.add_argument('--vf-lr', type=float, default=3e-4, metavar='V',
                    help='learning rate of value network')
parser.add_argument('--demo_file_list', type=str, nargs='+')
parser.add_argument('--percent_list', type=float, nargs='+')
parser.add_argument('--test_episodes', type=int, help='Number of episodes')
parser.add_argument('--result_file', type=str, help='Result file name')
parser.add_argument('--snapshot_file', type=str, help='Snapshot file name')
args = parser.parse_args()

env = gym.make(args.env)

num_inputs = env.observation_space.shape[0]
num_actions = env.action_space.shape[0]


obs_len_init = 11

env.reset()


env.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

policy_net = Policy(num_inputs, num_actions, args.hidden_dim).float()
value_net = Value(num_inputs, args.hidden_dim).float().to(device)
discriminator = Discriminator(num_inputs + num_inputs, args.hidden_dim).float().to(device)
disc_criterion = nn.BCEWithLogitsLoss()
value_criterion = nn.MSELoss()
disc_optimizer = optim.Adam(discriminator.parameters(), args.lr)
value_optimizer = optim.Adam(value_net.parameters(), args.vf_lr)

inverse_model = InverseModel(num_inputs*2, args.hidden_dim, num_actions, 6).float()
inverse_optimizer = optim.Adam(inverse_model.parameters(), 0.01)

def select_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    action_mean, _, action_std = policy_net(state)
    action = torch.normal(action_mean, action_std)
    return action

def update_params(batch):
    rewards = torch.Tensor(batch.reward).float().to(device)
    masks = torch.Tensor(batch.mask).float().to(device)
    actions = torch.Tensor(np.concatenate(batch.action, 0)).float().to(device)
    states = torch.Tensor(batch.state).float().to(device)
    values = value_net(Variable(states))

    returns = torch.Tensor(actions.size(0),1).float().to(device)
    deltas = torch.Tensor(actions.size(0),1).float().to(device)
    advantages = torch.Tensor(actions.size(0),1).float().to(device)

    prev_return = 0
    prev_value = 0
    prev_advantage = 0
    for i in reversed(range(rewards.size(0))):
        returns[i] = rewards[i] + args.gamma * prev_return * masks[i]
        deltas[i] = rewards[i] + args.gamma * prev_value * masks[i] - values.data[i]
        advantages[i] = deltas[i] + args.gamma * args.tau * prev_advantage * masks[i]

        prev_return = returns[i, 0]
        prev_value = values.data[i, 0]
        prev_advantage = advantages[i, 0]

    targets = Variable(returns)

    batch_size = math.ceil(states.shape[0] / args.vf_iters)
    idx = np.random.permutation(states.shape[0])
    for i in range(args.vf_iters):
        smp_idx = idx[i * batch_size: (i + 1) * batch_size]
        smp_states = states[smp_idx, :]
        smp_targets = targets[smp_idx, :]
        
        value_optimizer.zero_grad()
        value_loss = value_criterion(value_net(Variable(smp_states)), smp_targets)
        value_loss.backward()
        value_optimizer.step()

    advantages = (advantages - advantages.mean()) / advantages.std()

    action_means, action_log_stds, action_stds = policy_net(Variable(states.cpu()))
    fixed_log_prob = normal_log_density(Variable(actions.cpu()), action_means, action_log_stds, action_stds).data.clone()

    def get_loss():
        action_means, action_log_stds, action_stds = policy_net(Variable(states.cpu()))
        log_prob = normal_log_density(Variable(actions.cpu()), action_means, action_log_stds, action_stds)
        action_loss = -Variable(advantages.cpu()) * torch.exp(log_prob - Variable(fixed_log_prob))
        return action_loss.mean()


    def get_kl():
        mean1, log_std1, std1 = policy_net(Variable(states.cpu()))

        mean0 = Variable(mean1.data)
        log_std0 = Variable(log_std1.data)
        std0 = Variable(std1.data)
        kl = log_std1 - log_std0 + (std0.pow(2) + (mean0 - mean1).pow(2)) / (2.0 * std1.pow(2)) - 0.5
        return kl.sum(1, keepdim=True)

    trpo_step(policy_net, get_loss, get_kl, args.max_kl, args.damping)

def expert_reward(states, actions):
    states = np.array(states).squeeze()
    actions = np.array(actions).squeeze()
    state_action = torch.Tensor(np.concatenate([states, actions], 1)).float().to(device)
    return -F.logsigmoid(discriminator(state_action)).cpu().detach().numpy()

def load_demos(file_list, percent_list):
    state_action_pairs = []
    confs = []
    sequences = []
    initial_reward = []
    for k in range(len(file_list)):
        fname = file_list[k]
        episodes = pickle.load(open(fname, 'rb'))
        len_epi_for_train = len(episodes)
        for j in range(int(len_epi_for_train*percent_list[k])):
            episode = episodes[j]
            sequences.append([])
            reward_sum = 0.
            for i in range(len(episode['action'])):
                sequences[-1].append(np.concatenate([episode['state'][i].squeeze()[0:num_inputs], episode['action'][i].reshape(-1)]))
                state_action_pairs.append(np.concatenate([episode['state'][i].squeeze()[0:num_inputs], episode['state'][i+1].squeeze()[0:num_inputs]]))
                reward_sum += episode['reward'][i].squeeze()
            sequences[-1].append(np.concatenate([episode['state'][i+1].squeeze()[0:num_inputs], episode['action'][i].reshape(-1)]))
            for i in range(len(episode['action'])):
                confs.append([reward_sum])
            initial_reward.append(np.concatenate([episode['state'][0].squeeze(), np.array([reward_sum])]))
            sequences[-1] = np.array(sequences[-1])
    confs = np.array(confs)
    print(np.mean(confs))
    return np.array(state_action_pairs), confs, sequences, np.array(initial_reward)


def train_inverse_dynamic(num_epochs, feasible_traj, inverse_model, action_dim, bs=128):
    training = inverse_model.training
    inverse_model.train()
    state_dim = (feasible_traj.shape[1]-action_dim)//2
    for ii in range(num_epochs):
        order = np.random.permutation(feasible_traj.shape[0])
        loss_epoch = 0
        for jj in range((len(feasible_traj)-1)//bs+1):
            idx = order[jj*bs:(jj+1)*bs]
            sampled_batch = feasible_traj[idx]
            if 'action1' in args.env:
                sampled_batch[:,-2] = np.clip(sampled_batch[:,-action_dim], -1, 0.)
                sampled_batch[:,-1] = 0.
            elif 'action2' in args.env:
                sampled_batch[:,-2] = np.clip(sampled_batch[:,-action_dim], 0, 1.)
                sampled_batch[:,-1] = 0.
            sampled_batch = torch.Tensor(sampled_batch).float().to(device)
            inverse_optimizer.zero_grad()
            output_action = inverse_model(torch.cat([sampled_batch[:,0:state_dim], sampled_batch[:,state_dim:state_dim*2]-sampled_batch[:,0:state_dim]], dim=1))
            loss = nn.SmoothL1Loss()(output_action, sampled_batch[:,-action_dim:].detach())
            loss.backward()
            inverse_optimizer.step()
            loss_epoch += loss.item()
        print(output_action[-1], sampled_batch[:,-action_dim:].detach()[-1])
        print('inverse loss', loss_epoch/((len(feasible_traj)-1)//bs+1))
    inverse_model.train(training)


try:
    expert_traj, expert_conf, sequences, initial_reward = load_demos(args.demo_file_list, args.percent_list)
except:
    print('Mixture demonstrations not loaded successfully.')
    assert False


if args.feasibility or args.optimality:
    feasible_seq = []
    feasible_traj = []
    file_list = glob.glob('demos/'+args.env+'_random_explore*.pkl')
    print(file_list)
    order = np.random.permutation(len(file_list))
    episodes = pickle.load(open(file_list[order[0]], 'rb'))
    for j in range(len(episodes)):
        episode = episodes[j]
        feasible_seq.append([])
        for i in range(len(episode['action'])):
            feasible_seq[-1].append(np.concatenate([episode['state'][i].squeeze()[0:num_inputs], episode['action'][i].reshape(-1)]))
    
    for k in range(10):
        order1 = np.random.permutation(order[1:10])
        for j in order1:
            feasible_traj = []
            episodes = pickle.load(open(file_list[order[j]], 'rb'))
            for episode in episodes:
                for i in range(len(episode['action'])):
                    feasible_traj.append(np.concatenate([episode['state'][i].squeeze()[0:num_inputs], episode['state'][i+1].squeeze()[0:num_inputs], episode['action'][i].reshape(-1)]))
            feasible_traj = np.array(feasible_traj)
            train_inverse_dynamic(1, feasible_traj, inverse_model, action_dim=env.action_space.shape[0])
    del feasible_traj
        
    all_sequences = feasible_seq + sequences
    len_list = []
    for seq in all_sequences:
        len_list.append(len(seq))
    len_list = np.array(len_list)
    norms = []
    for sequence in all_sequences:
        env.reset()
        norms.append([])
        state = env.reset_with_obs(sequence[0][0:obs_len_init])
        for step in range(len(sequence)-1):
            action = inverse_model(torch.from_numpy(np.concatenate([state[0:num_inputs], sequence[step+1][0:num_inputs]-state[0:num_inputs]])).float().to(device).unsqueeze(0))
            action = action.data[0].numpy()
            if args.delta_s is not None:
                action = 2*(np.random.rand(*action.shape)-0.5)*args.delta_s + action
            if 'action1' in args.env:
                action = np.clip(action, -1., 0.)
            elif 'action2' in args.env:
                action = np.clip(action, 0., 1.)
            next_state, _, _, _ = env.step(action)
            if 'reacher' in args.env:
                norms[-1].append(np.linalg.norm(next_state[0:4] - sequence[step+1][0:num_inputs][0:4]))
            else:
                norms[-1].append(np.linalg.norm(next_state[0:num_inputs] - sequence[step+1][0:num_inputs]))
            state = next_state
    norms = np.array([sum(norms1) for norms1 in norms])
    norms = norms/len_list
    max_ = np.max(norms[0:len(feasible_seq)])
    min_ = np.min(norms[0:len(feasible_seq)])
    max_1 = np.max(norms[len(feasible_seq):])
    if args.delta_s is None:
        if max_ < np.min(norms[len(feasible_seq):]):
            upper_bound = max_1
        else:
            upper_bound = max_
    else:
        upper_bound = max_
    weight = (norms[len(feasible_seq):] - min_)/(upper_bound-min_)
    weight[weight>1] = 1.0
    weight[weight<0] = 0.0
    weight_step = []
    for www in range(weight.shape[0]):
        weight_step += [weight[www]] * (len(sequences[www])-1)
    weight_step = np.array(weight_step)
    feas_weight = 1.-weight_step.reshape(-1,1)


if args.optimality:
    if 'action2' in args.env:
        bound_limit = 0.02
    elif 'action1' in args.env:
        bound_limit = 0.1
    rectify_data = []
    rectify_label = []
    id_list = []
    for i in range(1000):
        max_rew = initial_reward[i, -1]
        id_ = 0
        for j in range(1000):
            if np.linalg.norm(initial_reward[j, 4:6] - initial_reward[i, 4:6]) < bound_limit and 1-weight[j] > 0.1:
                if max_rew < initial_reward[j, -1]:
                    max_rew = initial_reward[j, -1]
                    id_ = j
        id_list.append(id_)
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
weight = weight / (np.sum(weight)+0.0000001)
weight[0] = 1-np.sum(weight[1:])

mean_reward_list = []
min_reward_list = []
max_reward_list = []
std_reward_list = []

all_idx = np.arange(0, expert_traj.shape[0])

snapshot_dir = os.path.dirname(args.snapshot_file)
os.makedirs(snapshot_dir, exist_ok=True)
result_dir = os.path.dirname(args.result_file)
os.makedirs(result_dir, exist_ok=True)

max_mean_reward = -1000000000

if 'Ant' in args.env:
    env = gym.make(args.env)

for i_episode in range(args.num_epochs):
    env.seed(int(time.time()))
    memory = Memory()

    num_steps = 0
    num_episodes = 0
    
    reward_batch = []
    states = []
    actions = []
    mem_actions = []
    mem_mask = []
    mem_next = []

    while num_steps < args.batch_size:
        state = env.reset()

        reward_sum = 0
        for t in range(10000): # Don't infinite loop while learning
            action = select_action(state[0:num_inputs])
            action = action.data[0].numpy()
            states.append(np.array([state[0:num_inputs]]))
            actions.append(np.array([action]))
            next_state, true_reward, done, _ = env.step(action)
            reward_sum += true_reward

            mask = 1
            if done:
                mask = 0

            mem_mask.append(mask)
            mem_next.append(next_state[0:num_inputs])

            if done:
                break

            state = next_state
        num_steps += (t-1)
        num_episodes += 1

        reward_batch.append(reward_sum)

    rewards = expert_reward(states, mem_next)
    for idx in range(len(states)):
        memory.push(states[idx][0], actions[idx], mem_mask[idx], mem_next[idx], \
                    rewards[idx][0])
    batch = memory.sample()
    update_params(batch)

    ### update discriminator ###
    mem_next = torch.from_numpy(np.array(mem_next).squeeze())
    states = torch.from_numpy(np.array(states).squeeze())
    
    idx = np.random.choice(all_idx, num_steps, p=weight.reshape(-1))
    expert_state_next_state = expert_traj[idx, :]
    weight_sample = weight[idx, :]
    expert_state_next_state = torch.Tensor(expert_state_next_state).float().to(device)
    weight_sample = torch.from_numpy(weight_sample).float().to(device)

    state_next_state = torch.cat((states, mem_next), 1).float().to(device)

    fake = discriminator(state_next_state)
    real = discriminator(expert_state_next_state)

    disc_optimizer.zero_grad()
    disc_loss = disc_criterion(fake, torch.ones(states.shape[0], 1).to(device)) + \
                disc_criterion(real, torch.zeros(expert_state_next_state.size(0), 1).to(device))
    disc_loss.backward()
    disc_optimizer.step()
    ############################
    if i_episode % args.log_interval == 0:
        env.seed(args.test_seed)
        reward_list = []
        with torch.no_grad():
          for i in range(args.test_episodes):
            state = env.reset()
            reward_sum = 0
            while True: # Don't infinite loop while learning
                action = select_action(state[0:num_inputs])
                action = action.data[0].numpy()
                action = np.clip(action, env.action_space.low, env.action_space.high)
                next_state, true_reward, done, infos = env.step(action)
                if 'reward_eval' in infos:
                    reward_sum += infos['reward_eval']
                else:
                    reward_sum += true_reward

                if done:
                    break

                state = next_state
            reward_list.append(reward_sum)
        print('Episode {}, Average reward: {:.3f}, Max reward: {:.3f}, Min reward: {:.3f}, Loss (disc): {:.3f}'.format(i_episode, np.mean(reward_list), max(reward_list), min(reward_list), disc_loss.item()))
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
