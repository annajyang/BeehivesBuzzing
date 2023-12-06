import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torch.utils.data as data
import util
import numpy as np
from skimage.color import lab2rgb, rgb2lab, rgb2gray, gray2rgb
from torchvision.utils import save_image

from args import get_test_args
from collections import OrderedDict
from json import dumps
from jmodels import BaselineCNN, BaselineFusionCNN
from os.path import join
from tensorboardX import SummaryWriter
from tqdm import tqdm
from ujson import load as json_load
from jmodels import *

def main(args):
    save_images = True
    batch_size = args.batch_size
    num_print = int(100/batch_size)

    # Set up logging
    # args.save_dir = util.get_save_dir(args.save_dir, args.name, training=False)
    #log = util.get_logger(args.save_dir, args.name)
    #log.info(f'Args: {dumps(vars(args), indent=4, sort_keys=True)}')
    device, args.gpu_ids = util.get_available_devices()
    args.batch_size *= max(1, len(args.gpu_ids))
    dtype = torch.float32
    num_aux = len(args.aux_var)
    print (num_aux)

    #log.info('Building model...')
    if args.model_type == 'BaselineCNN' and num_aux == 0:
        model = BaselineCNN()
    elif args.model_type == 'BaselineCNN' and num_aux > 0:
        model = BaselineFusionCNN(num_aux)
    else:
        print('Wrong model type.')

    #log.info(f'Loading checkpoint from {args.load_path}...')
    model = util.load_model(model, args.load_path, args.gpu_ids, return_step=False)
    model = model.to(device)
    model.eval()


    # Get data loader
    # log.info('Building dataset...')
    if args.split == 'dev':
        dataset = util.MelSpectData(args.dev_dir)
    elif args.split == 'test':
        dataset = util.MelSpectData(args.test_dir)
    else:
        dataset = util.MelSpectData(args.train_dir)
    
    data_loader = data.DataLoader(dataset,
                                 batch_size=args.batch_size,
                                 shuffle=False,
                                 num_workers=args.num_workers)

    # Evaluate
    # log.info(f'Evaluating on {args.split} split...')
    loss_meter = util.AverageMeter()
    acc_meter = util.AverageMeter()

    n = 0
    with torch.no_grad(), \
            tqdm(total=len(data_loader.dataset)) as progress_bar:
        for i,(row_og, img, label) in enumerate(data_loader):
            if len(label) != batch_size:
                continue;
            # Forward
            img = img.to(device=device, dtype=dtype)
            row = torch.tensor(row_og)
            row = torch.nan_to_num(row)
            row = row.to(device)
            # Forward
            if num_aux > 0:
                y_pred = model(img,row)
            else:
                y_pred = model(img)
            

            # loss
            loss = F.binary_cross_entropy(y_pred.type(torch.FloatTensor), label.type(torch.FloatTensor).unsqueeze(1))
            accuracy = sum(np.round_(y_pred.cpu().detach().numpy()).squeeze() == label.numpy())/label.shape[0]
            loss_meter.update(loss.item(),batch_size)
            acc_meter.update(accuracy,batch_size)

            # Log info
            progress_bar.update(batch_size)
            progress_bar.set_postfix(Loss=loss_meter.avg,Acc=acc_meter.avg)


    # Log results (except for test set, since it does not come with labels)
    results_list = [('BinaryCrossEntropyLoss', loss_meter.avg),('Accuracy',acc_meter.avg)]
    results = OrderedDict(results_list)

    # Log to console
    results_str = ', '.join(f'{k}: {v}' for k, v in results.items())

    print(results_str)
    

if __name__ == '__main__':
    main(get_test_args())