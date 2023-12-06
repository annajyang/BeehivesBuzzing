import math
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as sched
import torch.utils.data as data
import torchvision.datasets as dset
import torchvision.transforms as T
import torchvision.models as models

from collections import OrderedDict
from json import dumps
from tensorboardX import SummaryWriter
from tqdm import tqdm
from ujson import load as json_load
import time

from jmodels import BaselineCNN, BaselineFusionCNN
import util
from args import get_train_args


def main(args):
    # Set up logging and devices
    args.save_dir = util.get_save_dir(args.save_dir, args.name, training=True)
    log = util.get_logger(args.save_dir, args.name)
    tbx = SummaryWriter(args.save_dir)
    device, args.gpu_ids  = util.get_available_devices()
    log.info(f'Args: {dumps(vars(args), indent=4, sort_keys=True)}')
    args.batch_size *= max(1, len(args.gpu_ids))

    # Set random seed
    log.info(f'Using random seed {args.seed}...')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    num_aux = len(args.aux_var)
    # Get model
    log.info('Building model...')
    if args.model_type == 'BaselineCNN' and num_aux == 0:
        model = BaselineCNN()
    elif args.model_type == 'BaselineCNN' and num_aux > 0:
        model = BaselineFusionCNN(num_aux)
    else:
        log.info('Wrong model type.')

    if args.load_path:
        log.info(f'Loading checkpoint from {args.load_path}...')
        model, step = util.load_model(model, args.load_path, args.gpu_ids)
    else:
        step = 0
    
    model = model.to(device)
    model.train()
    model = model.float()
    ema = util.EMA(model, args.ema_decay)
    dtype = torch.float32

    # Get saver
    saver = util.CheckpointSaver(args.save_dir,
                                 max_checkpoints=args.max_checkpoints,
                                 metric_name=args.metric_name,
                                 maximize_metric=args.maximize_metric,
                                 log=log)


    model_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(model_params, betas=(0.9, 0.99), eps=1e-07,
                           lr=args.lr, weight_decay=args.l2_wd)

    # Get data loader
    log.info('Building dataset...')
    #preprocess = T.Compose([T.Resize(args.input_size),T.RandomHorizontalFlip(),T.RandomAffine(30),T.GaussianBlur((3,3))])
    #preprocess = T.Compose([T.Resize(args.input_size)])
    train_dataset = util.MelSpectData(args.train_dir)

    # #GET SUBSET
    # train_dataset = torch.utils.data.Subset(train_dataset, np.random.choice(len(train_dataset), 64000, replace=False))

    train_loader = data.DataLoader(train_dataset,
                                   batch_size=args.batch_size,
                                   shuffle=True,
                                   num_workers=args.num_workers)
    dev_dataset = util.MelSpectData(args.dev_dir)
    dev_loader = data.DataLoader(dev_dataset,
                                 batch_size=args.batch_size,
                                 shuffle=False,
                                 num_workers=args.num_workers)
    
    # Train
    log.info('Training...')
    steps_till_eval = args.eval_steps
    epoch = step // len(train_dataset)
    while epoch != args.num_epochs:
        epoch += 1
        log.info(f'Starting epoch {epoch}...')
        with torch.enable_grad(), \
                tqdm(total=len(train_loader.dataset)) as progress_bar:
            for row, img, label in train_loader:
                #print(row)
                if len(label) != args.batch_size:
                    continue

                # Setup for forward
                batch_size = args.batch_size 
                optimizer.zero_grad()

                # Forward
                img = img.to(device)
                # row = torch.tensor(row)
                # row = row.to(device)
                if num_aux > 0:
                    y_pred = model(img,row)
                    for i,p in enumerate(y_pred):
                        if p != p:
                            y_pred[i]=0.0
                else:
                    y_pred = model(img)

                loss = F.binary_cross_entropy(y_pred.type(torch.FloatTensor), label.type(torch.FloatTensor).unsqueeze(1))
                accuracy = sum(np.round_(y_pred.cpu().detach().numpy()).squeeze() == label.numpy())/label.shape[0]
               
                loss_val = loss.item()

                # Backward
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                ema(model, step // batch_size)

                # Log info
                step += batch_size
                progress_bar.update(batch_size)
                progress_bar.set_postfix(epoch=epoch,
                                         Loss=loss_val,
                                         Accuracy=accuracy)
                tbx.add_scalar('train/Loss', loss_val, step)
                tbx.add_scalar('train/LR',
                               optimizer.param_groups[0]['lr'],
                               step)
                tbx.add_scalar('train/acc',accuracy, step)

                steps_till_eval -= batch_size

                if steps_till_eval <= 0:
                    steps_till_eval = args.eval_steps

                    # Evaluate and save checkpoint
                    log.info(f'Evaluating at step {step}...')
                    ema.assign(model)
                    results, pred_dict, gold_dict = evaluate(model, dev_loader, device, args.batch_size, num_aux)
                    if args.metric_name == 'Accuracy':
                        optimize = results['Accuracy']
                    elif args.metric_name == 'BCE':
                        optimize = results['BinaryCrossEntropyLoss']
                    else:
                        log.info('Unrecognized metric_name')
                    saver.save(step, model, optimize, device)
                    ema.resume(model)

                    # Log to console
                    results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in results.items())
                    log.info(f'Dev {results_str}')

                    # Log to TensorBoard
                    log.info('Visualizing in TensorBoard...')
                    for k, v in results.items():
                        tbx.add_scalar(f'dev/{k}', v, step)
                    util.visualize(tbx,
                                  pred_dict=pred_dict,
                                  gold_dict=gold_dict,
                                  step=step,
                                  split='dev',
                                  num_visuals=args.num_visuals)


def evaluate(model, data_loader, device, batch_size, num_aux):
    loss_meter = util.AverageMeter()
    acc_meter = util.AverageMeter()

    model.eval()
    pred_dict = {}
    gold_dict = {}
    dtype = torch.float32

    with torch.no_grad(), \
            tqdm(total=len(data_loader.dataset)) as progress_bar:
        for i,(row, img, label) in enumerate(data_loader):
            if len(label) != batch_size:
                continue
            # Forward
            img = img.to(device)
            # row = torch.tensor(row)
            # row = row.to(device)
            if num_aux > 0:
                y_pred = model(img,row)
                for i,p in enumerate(y_pred):
                    if p != p:
                        y_pred[i]=0.0
            else:
                y_pred = model(img)

            # loss
            loss = F.binary_cross_entropy(y_pred.type(torch.FloatTensor), label.type(torch.FloatTensor).unsqueeze(1))
            accuracy = sum(np.round_(y_pred.cpu().detach().numpy()).squeeze() == label.numpy())/label.shape[0]
            loss_meter.update(loss.item(),batch_size)
            acc_meter.update(accuracy,batch_size)

            # Todo accuracy, precision, recall, f1

            # Log info
            progress_bar.update(batch_size)
            progress_bar.set_postfix(Loss=loss_meter.avg,Acc=acc_meter.avg)

            pred_dict[i] = y_pred
            gold_dict[i] = {'answer':y_pred, 'img':img, 'label': label}


    model.train()

    results_list = [('BinaryCrossEntropyLoss', loss_meter.avg),('Accuracy',acc_meter.avg)]

    results = OrderedDict(results_list)

    return results, pred_dict, gold_dict


if __name__ == '__main__':
    main(get_train_args())