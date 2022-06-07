import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model import RGCNDQNModel
from replay_buffer import ReplayBuffer
from torch.optim.lr_scheduler import StepLR
from random import choice
import os
class DQN(object):
    def __init__(self,args,number_nodes):
        self.eval_net, self.target_net = RGCNDQNModel(args['feat_dim'],
                             args['in_size'],
                             args['out_size'],
                             args['all_relations'],
                             args['num_bases'],
                             args['num_hidden_layers']).to(args['device']),\
                        RGCNDQNModel(args['feat_dim'],
                             args['in_size'],
                             args['out_size'],
                             args['all_relations'],
                             args['num_bases'],
                             args['num_hidden_layers']).to(args['device'])

        for name, param in self.eval_net.named_parameters():
                param.requires_grad = True

        self.learn_step_counter = 0
        self.memory_counter = 0
        self.args = args
        self.epsilon=args['epsilon']
        self.memory=ReplayBuffer( self.args['memory_capacity'], number_nodes)
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.args['learning_rate'])
        self.scheduler = StepLR(self.optimizer, step_size=self.args['step_size'], gamma=self.args['learning_rate_gamma'])
        self.loss_func = nn.MSELoss()


    def choose_action(self, g,s,num_rna):
        with torch.no_grad():
            action_index = [i for i, x in enumerate(s) if x == 1]
            f_mirna = torch.tensor([[-1.0]*self.args['in_size'] for o in range(g.number_of_nodes('mirna'))]).to(self.args['device'])
            f_mrna = torch.tensor([[-1.0]*self.args['in_size'] for o in range(g.number_of_nodes('mrna'))]).to(self.args['device'])
            f_lncrna = torch.tensor([[-1.0] *self.args['in_size'] for o in range(g.number_of_nodes('lncrna'))]).to(self.args['device'])
            for i in action_index:
                if num_rna[i][0]=='mirna':
                    f_mirna[num_rna[i][1]]=torch.ones(1,self.args['in_size']).to(self.args['device'])
                if num_rna[i][0]=='lncrna':
                    f_lncrna[num_rna[i][1]] = torch.ones(1, self.args['in_size']).to(self.args['device'])
                if num_rna[i][0]=='mrna':
                    f_mrna[num_rna[i][1]] = torch.ones(1, self.args['in_size']).to(self.args['device'])

            h = {'mirna': f_mirna, 'mrna': f_mrna, 'lncrna': f_lncrna}


            if np.random.uniform() < self.epsilon:
                actions_value = self.eval_net.forward(self.args,g,h)
                actions_value[action_index]=float("-inf")
                action = torch.argmax(actions_value).item()

            else:
                action = np.random.randint(0, g.number_of_nodes())
                while action in action_index:
                    action = np.random.randint(0, g.number_of_nodes())
            return action


    def store_transition(self,obs, action, reward, next_obs, done):
        self.memory.store_transition(obs, action, reward, next_obs, done)


    def can_sample(self):
        return self.memory.can_sample(self.args['batch_size'])


    def learn(self,g,num_rna):
        if self.learn_step_counter % self.args['target_replace_iter'] == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        b_s,b_a,b_r,b_s_,done = self.memory.sample(self.args['batch_size'])

        mrna_h_s = torch.tensor(
            [-1.0] * g.number_of_nodes('mrna') * self.args['in_size'] * self.args['batch_size']).view(
            g.number_of_nodes('mrna'), self.args['batch_size'], self.args['in_size']).to(self.args['device'])
        lncrna_h_s = torch.tensor(
            [-1.0] * g.number_of_nodes('lncrna') * self.args['in_size'] * self.args['batch_size']).view(
            g.number_of_nodes('lncrna'), self.args['batch_size'], self.args['in_size']).to(self.args['device'])
        mirna_h_s = torch.tensor(
            [-1.0] * g.number_of_nodes('mirna') * self.args['in_size'] * self.args['batch_size']).view(
            g.number_of_nodes('mirna'), self.args['batch_size'], self.args['in_size']).to(self.args['device'])
        lncrna_h_s = (b_s.T[:g.number_of_nodes('lncrna')] * 2 + lncrna_h_s.squeeze()).unsqueeze(2)
        mirna_h_s = (b_s.T[g.number_of_nodes('lncrna'):g.number_of_nodes('lncrna')+g.number_of_nodes('mirna')] * 2 + mirna_h_s.squeeze()).unsqueeze(2)
        mrna_h_s = (b_s.T[g.number_of_nodes('lncrna')+g.number_of_nodes('mirna'):] * 2 + mrna_h_s.squeeze()).unsqueeze(2)


        mrna_h_s_ = torch.tensor(
            [-1.0] * g.number_of_nodes('mrna') * self.args['in_size'] * self.args['batch_size']).view(
            g.number_of_nodes('mrna'), self.args['batch_size'], self.args['in_size']).to(self.args['device'])
        lncrna_h_s_ = torch.tensor(
            [-1.0] * g.number_of_nodes('lncrna') * self.args['in_size'] * self.args['batch_size']).view(
            g.number_of_nodes('lncrna'), self.args['batch_size'], self.args['in_size']).to(self.args['device'])
        mirna_h_s_ = torch.tensor(
            [-1.0] * g.number_of_nodes('mirna') * self.args['in_size'] * self.args['batch_size']).view(
            g.number_of_nodes('mirna'), self.args['batch_size'], self.args['in_size']).to(self.args['device'])
        lncrna_h_s_ = (b_s_.T[:g.number_of_nodes('lncrna')] * 2 + lncrna_h_s_.squeeze()).unsqueeze(2)
        mirna_h_s_ = (b_s_.T[g.number_of_nodes('lncrna'):g.number_of_nodes('lncrna')+g.number_of_nodes('mirna')] * 2 + mirna_h_s_.squeeze()).unsqueeze(2)
        mrna_h_s_ = (b_s_.T[g.number_of_nodes('lncrna')+g.number_of_nodes('mirna'):] * 2 + mrna_h_s_.squeeze()).unsqueeze(2)


        h_s = {'lncrna': lncrna_h_s, 'mirna': mirna_h_s, 'mrna': mrna_h_s}
        h_s_ = {'lncrna': lncrna_h_s_, 'mirna': mirna_h_s_, 'mrna': mrna_h_s_}

        q_eval = self.eval_net(self.args, g, h_s).squeeze(2).gather(0, b_a.long().unsqueeze(1)).squeeze(0)
        q_next=self.target_net(self.args, g,h_s_).squeeze(2).detach().max(dim=0)[0]

        for one_done,batch in zip(done,range(self.args['batch_size'])):
            if one_done:
                q_next[batch] = torch.tensor(0).to(self.args['device'])
        q_target=self.args['gamma'] * q_next + torch.tensor(b_r).to(self.args['device'])

        loss=self.loss_func(q_eval, q_target)
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()



