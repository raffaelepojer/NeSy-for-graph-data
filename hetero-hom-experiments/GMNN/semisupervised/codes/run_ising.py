import sys
import os
import copy
import json
import datetime

settings = [(32, 0.5, 0.0, 0.4), (32, -0.5, 0.0, 0.4), (32, -0.4, 0.1, 0.4), (32, -0.7, 0.3, 0.4), (32, 0.9, 0.05, 0.4)]

for set in settings:
    N = set[0]
    J = set[1]
    Jb = set[2]
    temp = set[3]


    opt = dict()

    opt['dataset'] = f'../data/ising/{N}_{J}_{Jb}_{temp}'
    opt['hidden_dim'] = 16
    opt['input_dropout'] = 0.5
    opt['dropout'] = 0
    opt['optimizer'] = 'adam'
    opt['lr'] = 0.01
    opt['decay'] = 5e-4
    opt['self_link_weight'] = 1.0
    opt['pre_epoch'] = 200
    opt['epoch'] = 100
    opt['iter'] = 1
    opt['use_gold'] = 1
    opt['draw'] = 'smp'
    opt['tau'] = 0.1

    def generate_command(opt):
        cmd = 'python3 train.py'
        for opt, val in opt.items():
            cmd += ' --' + opt + ' ' + str(val)
        return cmd

    def run(opt):
        opt_ = copy.deepcopy(opt)
        os.system(generate_command(opt_))

    for k in range(5):
        seed = k + 1
        opt['seed'] = seed
        run(opt)
