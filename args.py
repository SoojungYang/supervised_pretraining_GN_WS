import argparse


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeEror('Boolean value expected')


parser = argparse.ArgumentParser()
# Hyper-parameters for prefix, prop and random seed
parser.add_argument('--prefix', type=str, default='test01',
                    help='Prefix for this training')
parser.add_argument('--prop', type=str, default=['logP', 'TPSA', 'MW', 'MR'],
                    help='Target properties to train')
parser.add_argument('--seed', type=int, default=1111,
                    help='Random seed will be used to shuffle dataset')

# Hyper-parameters for model construction
parser.add_argument('--num_embed_layers', type=int, default=4,
                    help='Number of node embedding layers')
parser.add_argument('--embed_dim', type=int, default=64,
                    help='Dimension of node embeddings')
parser.add_argument('--finetune_dim', type=int, default=256,
                    help='Dimension of a fine-tuned z')
parser.add_argument('--num_embed_heads', type=int, default=4,
                    help='Number of attention heads for node embedding')
parser.add_argument('--num_finetune_heads', type=int, default=4,
                    help='Number of attention heads for fine-tuning layer')
parser.add_argument('--embed_use_ffnn', type=str2bool, default=False,
                    help='Whether to use feed-forward nets for node embedding')
parser.add_argument('--embed_dp_rate', type=float, default=0.1,
                    help='Dropout rates in node embedding layers')
parser.add_argument("--embed_nm_type", type=str, default='gn',
                    help='Type of normalization: gn or ln')
parser.add_argument("--num_groups", type=int, default=8,
                    help='Number of groups for group normalization')
parser.add_argument('--prior_length', type=float, default=1e-4,
                    help='Weight decay coefficient')

# Hyper-parameters for data loading
parser.add_argument('--shuffle_buffer_size', type=int, default=100,
                    help='shuffle buffer size')


# Hyper-parameters for loss function
parser.add_argument("--loss_dict", type=dict, default={'logP': 'mse', 'TPSA': 'mse',
                                                       'MR': 'mse', 'MW': 'mse', 'SAS': 'mse'},
                    help='type of loss for each property, Options: bce, mse, focal, class_balanced, max_margin')
parser.add_argument('--focal_alpha', type=float, default=0.25,
                    help='Alpha in Focal loss')
parser.add_argument('--focal_gamma', type=float, default=2.0,
                    help='Gamma in Focal loss')

# Hyper-parameters for training
parser.add_argument('--batch_size', type=int, default=128,
                    help='Batch size')
parser.add_argument('--num_epochs', type=int, default=50,
                    help='Number of epochs')
parser.add_argument('--init_lr', type=float, default=1e-3,
                    help='Initial learning rate,\
                          Do not need for warmup scheduling')
parser.add_argument('--beta_1', type=float, default=0.9,
                    help='Beta1 in adam optimizer')
parser.add_argument('--beta_2', type=float, default=0.999,
                    help='Beta2 in adam optimizer')
parser.add_argument('--opt_epsilon', type=float, default=1e-7,
                    help='Epsilon in adam optimizer')
parser.add_argument('--decay_steps', type=int, default=40,
                    help='Decay steps for stair learning rate scheduling')
parser.add_argument('--decay_rate', type=float, default=0.1,
                    help='Decay rate for stair learning rate scheduling')
parser.add_argument('--max_to_keep', type=int, default=5,
                    help='Maximum number of checkpoint files to be kept')

# Hyper-parameters for evaluation
parser.add_argument("--save_outputs", type=str2bool, default=True,
                    help='Whether to save final predictions for test dataset')
parser.add_argument('--mc_dropout', type=str2bool, default=False,
                    help='Whether to infer predictive distributions with MC-dropout')
parser.add_argument('--mc_sampling', type=int, default=30,
                    help='Number of MC sampling')
parser.add_argument('--top_k', type=int, default=50,
                    help='Top-k instances for evaluating Precision or Recall')

# For benchmark
parser.add_argument('--ckpt_path', type=str, default='./save/test01_4_64_256_4_4_0.1_0.0001_gn.ckpt',
                    help='checkpoint file path')
parser.add_argument('--fine_tune_at', type=int, default=3,
                    help='how many layers to freeze')
parser.add_argument('--benchmark_task_type', type=str, default='reg',
                    help='reg of cls')