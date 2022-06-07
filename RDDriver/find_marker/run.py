from DQN import DQN
import mvc_env
from load import *
import warnings
import argparse
import torch
import numpy as np
import os
import time



warnings.filterwarnings("ignore")


def train():
    f_result=str(time.time())+'\n'
    env = mvc_env.MVC_env()
    init_graph=load_random_hetero_graph(args['path'],Specified_quantity={'lncrna':10,'mirna':3,'mrna':37})
    dqn = DQN(args, init_graph.number_of_nodes())
    print('\nCollecting experience...')
    for i_episode in range(args['maxiter']):
        action_choosed=[]
        g, s, num_rna = env.reset(init_graph)
        ep_r = 0
        while True:
            a = dqn.choose_action(g,s,num_rna)

            action_choosed+=[a]
            s_, r, done= env.step(a)
            if done:
                r=1
            dqn.store_transition(s, a, r, s_,done)

            ep_r += r

            if dqn.can_sample():
                dqn.learn(g,num_rna)
            if done:
                print('Ep: ', i_episode,
                          '| Ep_r: ', ep_r,
                          'len of choosed',len(action_choosed),
                          'choosed node',action_choosed)
                f_result+=str(i_episode)+"   "+str(ep_r)+"   "+str(action_choosed)+"\n"
            if done:
                break
            s = s_
    f = open("1.txt", mode="w")
    f_result+=str(time.time())
    f.write(f_result)
    f.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()


    parser.add_argument('--in_size', type=int, default=1)
    parser.add_argument('--out_size', type=int, default=1)
    parser.add_argument('--feat_dim', type=int, default=32)
    parser.add_argument('--num_bases', type=int, default=5)
    parser.add_argument('--num_hidden_layers', type=int, default=2)
    parser.add_argument('--all_relations', type=list,
                        default=['lnc_lnc', 'lnc_mi', 'm_lnc', 'mi_lnc', 'mi_m', 'mi_mi', 'lnc_m', 'm_m',
                                 'm_mi'])

    parser.add_argument('--memory_capacity', type=int, default=100000)
    parser.add_argument('--epsilon', type=float, default=0.8)
    parser.add_argument('--target_replace_iter', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--maxiter', type=int, default=10000)
    parser.add_argument('--path', type=str, default='../data/random_network/3')
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--step_size', type=int, default=50000)
    parser.add_argument('--learning_rate_gamma', type=float, default=1.0)

    args = parser.parse_args().__dict__
    args['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.cuda.set_device(0)

    seed = 1000
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    train()