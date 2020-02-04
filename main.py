import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # to import shared utils
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import time
from tqdm import tqdm
from IPython import embed
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s')
logFormatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
rootLogger = logging.getLogger()

from DataLoader import VQADataLoader
from model.net import XNMNet
from utils.misc import todevice
from validate import validate
###############Hyper-Params###########
lr = 0.0008
num_epoch = 50
######################################

train_loader_kwargs = {
    'question_pt': 'data/files/train_questions.pt',
    'vocab_json': 'data/files/vocab.json',
    'feature_h5': 'data/files/trainval_feature.h5',
    'batch_size': 256,
    'spatial': False,
    'num_workers': 2,
    'shuffle': True
}

val_loader_kwargs = {
    'question_pt': 'data/files/val_questions.pt',
    'vocab_json': 'data/files/vocab.json',
    'feature_h5': 'data/files/trainval_feature.h5',
    'batch_size': 256,
    'spatial': False,
    'num_workers': 2,
    'shuffle': False
}

model_kwargs = {
    'dim_v': 512,
    'dim_word': 300,
    'dim_hidden': 1024,
    'dim_vision': 2048,
    'dim_edge': 256,
    'cls_fc_dim': 1024,
    'dropout_prob': 0.5,
    'T_ctrl': 3,
    'glimpses': 2,
    'stack_len': 4,
    'spatial': False,
    'use_gumbel': False,
    'use_validity': True,
}

def train():
    train_loader = VQADataLoader(**train_loader_kwargs)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_kwargs.update({'vocab': train_loader.vocab,'device': device})
    val_loader = VQADataLoader(**val_loader_kwargs)
    model = XNMNet(**model_kwargs).to(device)
    train_loader.glove_matrix = torch.FloatTensor(train_loader.glove_matrix).to(device)
    with torch.no_grad():
        model.token_embedding.weight.set_(train_loader.glove_matrix)
    ################################################################
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(parameters, lr, weight_decay=0)
    for epoch in range(num_epoch):
        model.train()
        i = 0
        for batch in tqdm(train_loader, total=len(train_loader)):
            progress = epoch + i / len(train_loader)
            coco_ids, answers, *batch_input = [todevice(x, device) for x in batch]
            logits, others = model(*batch_input)
            ##################### loss #####################
            nll = -nn.functional.log_softmax(logits, dim=1)
            loss = (nll * answers / 10).sum(dim=1).mean()
            #################################################
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_value_(parameters, clip_value=0.5)
            optimizer.step()
            if (i + 1) % (len(train_loader) // 50) == 0:
                logging.info("Progress %.3f  ce_loss = %.3f" % (progress, loss.item()))
            i+=1
        train_acc,train_loss = validate(model, train_loader, device,withLossFlag = True,func = nn.functional)
        logging.info('\n ~~~~~~ Epoch: %.4f ~~~~~~~\n' % epoch)
        logging.info('\n ~~~~~~ Train Accuracy: %.4f ~~~~~~~\n' % train_acc)
        logging.info('\n ~~~~~~ Train Loss: %.4f ~~~~~~~\n' % train_loss)
        valid_acc,valid_loss = validate(model, val_loader, device,withLossFlag = True,func = nn.functional)
        logging.info('\n ~~~~~~ Valid Accuracy: %.4f ~~~~~~~\n' % valid_acc)
        logging.info('\n ~~~~~~ Valid Loss: %.4f ~~~~~~~\n' % valid_loss)
    # torch.save(model.state_dict(), 'model.pkl')

def main():
    fileHandler = logging.FileHandler('output3/stdout.log', 'w+')
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)
    # set random seed
    torch.manual_seed(666)
    np.random.seed(666)
    train()
    torch.cuda.empty_cache()

if __name__ == '__main__':
    main()