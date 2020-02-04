import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # to import shared utils
import torch
from tqdm import tqdm
from DataLoader import VQADataLoader
from model.net import XNMNet
from utils.misc import todevice

device = 'cuda'
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
    'device': device,
    'spatial': False,
    'use_gumbel': False,
    'use_validity': True,
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

def batch_accuracy(predicted, true):
    """ Compute the accuracies for a batch of predictions and answers """
    _, predicted_index = predicted.max(dim=1, keepdim=True)
    agreeing = true.gather(dim=1, index=predicted_index)
    '''
    Acc needs to be averaged over all 10 choose 9 subsets of human answers.
    While we could just use a loop, surely this can be done more efficiently (and indeed, it can).
    There are two cases for the 1 chosen answer to be discarded:
    (1) the discarded answer is not the predicted answer => acc stays the same
    (2) the discarded answer is the predicted answer => we have to subtract 1 from the number of agreeing answers

    There are (10 - num_agreeing_answers) of case 1 and num_agreeing_answers of case 2, thus
    acc = ((10 - agreeing) * min( agreeing      / 3, 1)
           +     agreeing  * min((agreeing - 1) / 3, 1)) / 10

    Let's do some more simplification:
    if num_agreeing_answers == 0:
        acc = 0  since the case 1 min term becomes 0 and case 2 weighting term is 0
    if num_agreeing_answers >= 4:
        acc = 1  since the min term in both cases is always 1
    The only cases left are for 1, 2, and 3 agreeing answers.
    In all of those cases, (agreeing - 1) / 3  <  agreeing / 3  <=  1, so we can get rid of all the mins.
    By moving num_agreeing_answers from both cases outside the sum we get:
        acc = agreeing * ((10 - agreeing) + (agreeing - 1)) / 3 / 10
    which we can simplify to:
        acc = agreeing * 0.3
    Finally, we can combine all cases together with:
        min(agreeing * 0.3, 1)
    '''
    return (agreeing * 0.3).clamp(max=1)

def validate(model, data, device, withLossFlag=False, func=None):
    print('validate...')
    model.eval()
    total_acc, count = 0, 0
    total_loss = 0
    for batch in tqdm(data, total=len(data)):
        coco_ids, answers, *batch_input = [todevice(x, device) for x in batch]
        logits, others = model(*batch_input)
        if withLossFlag:
            nll = -func.log_softmax(logits, dim=1)
            loss = (nll * answers / 10).sum(dim=1).mean()
            total_loss += loss.item()
        acc = batch_accuracy(logits, answers)
        total_acc += acc.sum().item()
        count += answers.size(0)
    acc = total_acc / count
    if withLossFlag:
        return acc, total_loss / len(data)
    return acc