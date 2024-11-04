import os
import time
import torch
import pickle
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim

from MARL.Model import openai_actor, openai_critic


def get_eval_actor(env_agent, arglist):
    """init the trainers or load the old model"""
    actors_cur = [None for _ in range(env_agent)]

    for idx in range(env_agent):
        actors_cur[idx] = torch.load(arglist.old_model_name + 'a_c_{}.pt'.format(idx))
    return actors_cur


def get_plane_actor(env_agent, arglist):
    """init the trainers or load the old model"""
    actors_cur = [None for _ in range(env_agent)]

    for idx in range(env_agent):
        actors_cur[idx] = torch.load(arglist.plane_model_name + 'a_c_{}.pt'.format(idx % 4))
    return actors_cur



def get_trainers(env_agent, obs_shape_n, action_shape_n, arglist):
    """init the trainers or load the old model"""
    actors_cur = [None for _ in range(env_agent)]
    critics_cur = [None for _ in range(env_agent)]
    actors_tar = [None for _ in range(env_agent)]
    critics_tar = [None for _ in range(env_agent)]
    optimizers_c = [None for _ in range(env_agent)]
    optimizers_a = [None for _ in range(env_agent)]
    input_size_global = sum(obs_shape_n) + sum(action_shape_n)
    trainers_cur = []
    trainers_tar = []

    if arglist.restore == True:  # restore the model
        for idx in range(env_agent):
            actors_cur[idx] = torch.load(arglist.old_model_name + 'a_c_{}.pt'.format(idx))
            critics_cur[idx] = torch.load(arglist.old_model_name + 'c_c_{}.pt'.format(idx))
            actors_tar[idx] = torch.load(arglist.old_model_name + 'a_t_{}.pt'.format(idx))
            critics_tar[idx] = torch.load(arglist.old_model_name + 'c_t_{}.pt'.format(idx))
            optimizers_a[idx] = optim.Adam(actors_cur[idx].parameters(), arglist.lr_a)
            optimizers_c[idx] = optim.Adam(critics_cur[idx].parameters(), arglist.lr_c)
        print("Load Modeling......")
        # for idx in range(4):
        #     actors_cur[idx] = torch.load(arglist.old_model_name + '1/a_c_{}.pt'.format(idx))
        #     critics_cur[idx] = torch.load(arglist.old_model_name + '1/c_c_{}.pt'.format(idx))
        #     actors_tar[idx] = torch.load(arglist.old_model_name + '1/a_t_{}.pt'.format(idx))
        #     critics_tar[idx] = torch.load(arglist.old_model_name + '1/c_t_{}.pt'.format(idx))
        #     optimizers_a[idx] = optim.Adam(actors_cur[idx].parameters(), arglist.lr_a)
        #     optimizers_c[idx] = optim.Adam(critics_cur[idx].parameters(), arglist.lr_c)
        # for idx in range(4, 8):
        #     actors_cur[idx] = torch.load(arglist.old_model_name + '2/a_c_{}.pt'.format(idx % 4))
        #     critics_cur[idx] = torch.load(arglist.old_model_name + '2/c_c_{}.pt'.format(idx % 4))
        #     actors_tar[idx] = torch.load(arglist.old_model_name + '2/a_t_{}.pt'.format(idx % 4))
        #     critics_tar[idx] = torch.load(arglist.old_model_name + '2/c_t_{}.pt'.format(idx % 4))
        #     optimizers_a[idx] = optim.Adam(actors_cur[idx].parameters(), arglist.lr_a)
        #     optimizers_c[idx] = optim.Adam(critics_cur[idx].parameters(), arglist.lr_c)
        # for idx in arglist.restore_idxs:
        #     trainers_cur[idx] = torch.load(arglist.old_model_name + 'c_{}'.format(idx))
        #     trainers_tar[idx] = torch.load(arglist.old_model_name + 'c_{}'.format(idx))
    else:
        for i in range(env_agent):
            actors_cur[i] = openai_actor(obs_shape_n[i], action_shape_n[i], arglist).to(arglist.device)
            critics_cur[i] = openai_critic(sum(obs_shape_n), sum(action_shape_n), arglist).to(arglist.device)
            actors_tar[i] = openai_actor(obs_shape_n[i], action_shape_n[i], arglist).to(arglist.device)
            critics_tar[i] = openai_critic(sum(obs_shape_n), sum(action_shape_n), arglist).to(arglist.device)
            optimizers_a[i] = optim.Adam(actors_cur[i].parameters(), arglist.lr_a)
            optimizers_c[i] = optim.Adam(critics_cur[i].parameters(), arglist.lr_c)

        print("Init Modeling......")
    actors_tar = update_trainers(actors_cur, actors_tar, 1.0)  # update the target par using the cur
    critics_tar = update_trainers(critics_cur, critics_tar, 1.0)
    return actors_cur, critics_cur, actors_tar, critics_tar, optimizers_a, optimizers_c


def update_trainers(agents_cur, agents_tar, tao):
    """
    update the trainers_tar par using the trainers_cur
    This way is not the same as copy_, but the result is the same
    out:
    |agents_tar: the agents with new par updated towards agents_current
    """
    for agent_c, agent_t in zip(agents_cur, agents_tar):
        key_list = list(agent_c.state_dict().keys())
        state_dict_t = agent_t.state_dict()
        state_dict_c = agent_c.state_dict()
        for key in key_list:
            state_dict_t[key] = state_dict_c[key] * tao + (1 - tao) * state_dict_t[key]
        agent_t.load_state_dict(state_dict_t)
    return agents_tar


def agents_train(arglist, game_step, update_cnt, memory, obs_size, action_size,
                 actors_cur, actors_tar, critics_cur, critics_tar, optimizers_a, optimizers_c):
    """use this func to make the "main" func clean
    par:
    |input: the data for training
    |output: the data for next update"""

    # update all trainers, if not display or benchmark mode
    # if game_step >= arglist.learning_start_step and (game_step - arglist.learning_start_step) % arglist.learning_fre == 0:
    # if update_cnt == 0:
    #     print('\r=start training ...' + ' ' * 100)

    # update the target par using the cur
    update_cnt += 1
    # print('update_model:', update_cnt)

    # update every agent in different memory batch
    for agent_idx, (actor_c, actor_t, critic_c, critic_t, opt_a, opt_c) in enumerate(
            zip(actors_cur, actors_tar, critics_cur, critics_tar, optimizers_a, optimizers_c)):
        if opt_c is None:
            continue  # jump to the next model update
        # sample the experience #采样得到的是所有观测和
        _obs_n_o, _action_n, _rew_n, _obs_n_n, _done_n = memory.sample(arglist.batch_size,
                                                                       agent_idx)  # Note the func is not the same as others

        # --use the data tp update the CRITIC
        rew = torch.tensor(_rew_n, device=arglist.device, dtype=torch.float)  # set the rew to gpu
        done_n = torch.tensor(_done_n, dtype=torch.float, device=arglist.device)
        action_cur_o = torch.from_numpy(_action_n).to(arglist.device, torch.float)
        obs_n_o = torch.from_numpy(_obs_n_o).to(arglist.device, torch.float)
        obs_n_n = torch.from_numpy(_obs_n_n).to(arglist.device, torch.float)

        action_tar = torch.cat(
            [a_t(obs_n_n[:, obs_size[idx][0]:obs_size[idx][1]]).detach() for idx, a_t in enumerate(actors_tar)],
            dim=1)

        q = critic_c(obs_n_o, action_cur_o).reshape(-1)
        q_ = critic_t(obs_n_n, action_tar).reshape(-1)
        tar_value = q_ * arglist.gamma * done_n + rew
        loss_c = torch.nn.MSELoss()(q, tar_value)  # bellman equation
        opt_c.zero_grad()
        loss_c.backward()
        nn.utils.clip_grad_norm_(critic_c.parameters(), arglist.max_grad_norm)
        opt_c.step()

        # --use the data to update the ACTOR
        # There is no need to cal other agent's action
        model_out, policy_c_new = actor_c(obs_n_o[:, obs_size[agent_idx][0]:obs_size[agent_idx][1]],
                                          model_original_out=True)
        # update the action of this agent
        action_cur_o[:, action_size[agent_idx][0]:action_size[agent_idx][1]] = policy_c_new
        loss_pse = torch.mean(torch.pow(model_out, 2))
        loss_a = torch.mul(-1, torch.mean(critic_c(obs_n_o, action_cur_o)))
        opt_a.zero_grad()
        (1e-3 * loss_pse + loss_a).backward()
        nn.utils.clip_grad_norm_(actor_c.parameters(), arglist.max_grad_norm)
        opt_a.step()

        # update the tar par
        actors_tar = update_trainers(actors_cur, actors_tar, arglist.tao)
        critics_tar = update_trainers(critics_cur, critics_tar, arglist.tao)

    return update_cnt, actors_cur, actors_tar, critics_cur, critics_tar


def save_model_now(arglist, actors_cur, actors_tar, critics_cur, critics_tar):
    print(arglist.note)
    # 保存模型参数
    model_file_dir = os.path.join(arglist.save_dir, time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime()))
    print(model_file_dir)
    if not os.path.exists(model_file_dir):
        os.mkdir(model_file_dir)
    txt_file_path = model_file_dir+"/a_a_note.txt"
    with open(txt_file_path,'w') as f:
        strlist = ("note:{}\n"
                   "old_model_name:{}\n"
                   "model_file_dir:{}").format(arglist.note,arglist.old_model_name,model_file_dir)
        f.write(strlist)

    for agent_idx, (a_c, a_t, c_c, c_t) in enumerate(zip(actors_cur, actors_tar, critics_cur, critics_tar)):
        torch.save(a_c, os.path.join(model_file_dir, 'a_c_{}.pt'.format(agent_idx)))
        torch.save(a_t, os.path.join(model_file_dir, 'a_t_{}.pt'.format(agent_idx)))
        torch.save(c_c, os.path.join(model_file_dir, 'c_c_{}.pt'.format(agent_idx)))
        torch.save(c_t, os.path.join(model_file_dir, 'c_t_{}.pt'.format(agent_idx)))