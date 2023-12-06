import argparse

def get_train_args():
    """Get arguments needed in train.py."""
    parser = argparse.ArgumentParser('Train a model')

    add_common_args(parser)
    add_train_test_args(parser)

    parser.add_argument('--eval_steps',
                        type=int,
                        default=500,
                        help='Number of steps between successive evaluations.')
    parser.add_argument('--lr',
                        type=float,
                        default=0.001,
                        help='Learning rate.')
    parser.add_argument('--lr_warmup_limit',
                        type=int,
                        default=1000,
                        )
    parser.add_argument('--l2_wd',
                        type=float,
                        default=3e-07, 
                        help='L2 weight decay.')
    parser.add_argument('--wd',
                        type=float,
                        default=0.999,
                        help='weight decay.')
    parser.add_argument('--num_epochs',
                        type=int,
                        default=30,
                        help='Number of epochs for which to train. Negative means forever.')
    parser.add_argument('--drop_prob',
                        type=float,
                        default=0.1,
                        help='Probability of zeroing an activation in dropout layers.')
    parser.add_argument('--metric_name',
                        type=str,
                        default='Accuracy',
                        choices=('Accuracy', 'BCE'),
                        help='Name of dev metric to determine best checkpoint.')
    parser.add_argument('--max_checkpoints',
                        type=int,
                        default=5,
                        help='Maximum number of checkpoints to keep on disk.')
    parser.add_argument('--max_grad_norm',
                        type=float,
                        default=5.0,
                        help='Maximum gradient norm for gradient clipping.')
    parser.add_argument('--seed',
                        type=int,
                        default=224,
                        help='Random seed for reproducibility.')
    parser.add_argument('--ema_decay',
                        type=float,
                        default=0.9999,
                        help='Decay rate for exponential moving average of parameters.')
    parser.add_argument('--block_size',
                        type=int,
                        default=60,
                        help='Number of seconds for each split audio file.')
    args = parser.parse_args()
    
    if args.metric_name == 'BCE':
        # Best checkpoint is the one that minimizes negative log-likelihood
        args.maximize_metric = False
    elif args.metric_name == 'Accuracy':
        # Best checkpoint is the one that maximizes EM or F1
        args.maximize_metric = True

    return args


def add_common_args(parser):
    """Add arguments common to all 3 scripts: setup.py, train.py, test.py"""
    # we'll need to fill in all of these paths

    parser.add_argument('--train_dir',
                        type=str,
                        default='./croppedDataset/train') 
    parser.add_argument('--dev_dir',
                        type=str,
                        default='./croppedDataset/val')
                        
    parser.add_argument('--test_dir',
                        type=str,
                        default='./croppedDataset/test')
    
def add_train_test_args(parser):
    """Add arguments common to train.py and test.py"""
    parser.add_argument('--model_type',
                        type=str,
                        default='BaselineCNN',
                        help='Model type, options are BaselineCNN.')
    parser.add_argument('--aux_var',
                        nargs='*',
                        default=[],
                        help='List of hive data variables to include.')

    parser.add_argument('--name',
                        '-n',
                        type=str,
                        required=True,
                        help='Name to identify training or test run.')
    parser.add_argument('--num_workers',
                        type=int,
                        default=4,
                        help='Number of sub-processes to use per data loader.')
    parser.add_argument('--save_dir',
                        type=str,
                        default='./save/',
                        help='Base directory for saving information.')
    parser.add_argument('--batch_size',
                        type=int,
                        default=64,
                        help='Batch size per GPU. Scales automatically when \
                              multiple GPUs are available.')
    parser.add_argument('--num_visuals',
                        type=int,
                        default=10,
                        help='Number of examples to visualize in TensorBoard.')
    parser.add_argument('--load_path',
                        type=str,
                        default=None,
                        help='Path to load as a model checkpoint.')
    parser.add_argument('--input_size',
                        type=int,
                        default='64',
                        help='Dimension of input image.')
    parser.add_argument('--loss_fn',
                        type=str,
                        default='MSEloss',
                        choices=('MSEloss','L1','L1+MSEloss'),
                        help='Loss function to evaluate model on.') # we will be playing around with this

def get_test_args():
    """Get arguments needed in test.py."""
    parser = argparse.ArgumentParser('Test a trained model')

    add_common_args(parser)
    add_train_test_args(parser)

    parser.add_argument('--split',
                        type=str,
                        default='test',
                        choices=('train', 'dev', 'test'),
                        help='Split to use for testing.')
    parser.add_argument('--output_folder',
                        type=str,
                        default='./output/',
                        help='Name for output folder.')

    # Require load_path for test.py
    args = parser.parse_args()
    if not args.load_path:
        raise argparse.ArgumentError('Missing required argument --load_path')

    return args