import torch
import numpy as np
import random

from pathlib import Path
from exp.exp_main import Exp_Main

from bench_utils import DeclareArg

SEED = DeclareArg('seed', int, 42, 'Random seed for reproducibility')

# Setting the random seed for reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Data paths
DATA_PATH = DeclareArg('data_dir', str, './data/ETT-small/', 'Path to the data directory')
DATA_FILE = DeclareArg('data_file', str, 'ETTh1.csv', 'Data file name')

# Overall hyperparameters
SEQ_LEN = DeclareArg('seq_len', int, 96, 'Input sequence length')
LABEL_LEN = DeclareArg('label_len', int, 48, 'Start token length for decoder')
PRED_LEN = DeclareArg('pred_len', int, 96, 'Prediction sequence length')
NUM_FEATURES = DeclareArg('num_features', int, 7, 'Number of features in the dataset')
EPOCHS = DeclareArg('epochs', int, 10, 'Number of training epochs')
BATCH_SIZE = DeclareArg('batch_size', int, 32, 'Batch size for training')
PATIENCE = DeclareArg('patience', int, 3, 'Early stopping patience')
LEARNING_RATE = DeclareArg('learning_rate', float, 0.0001, 'Learning rate for optimizer')

# Model hyperparameters
D_MODEL = DeclareArg('d_model', int, 512, 'Dimension of model')
N_HEADS = DeclareArg('n_heads', int, 8, 'Number of attention heads')
ENCODER_LAYERS = DeclareArg('e_layers', int, 2, 'Number of encoder layers')
DECODER_LAYERS = DeclareArg('d_layers', int, 1, 'Number of decoder layers')
D_FF = DeclareArg('d_ff', int, 2048, 'Dimension of feedforward network')
DROPOUT = DeclareArg('dropout', float, 0.05, 'Dropout rate')
MOVING_AVG = DeclareArg('moving_avg', int, 25, 'Window size for moving average')

class AutoformerConfig:
  def __init__(self):
    self.model = 'Autoformer'
    self.model_id = 'test'
    self.is_training = 1
    self.data = 'ETTh1'
    self.root_path = DATA_PATH
    self.data_path = DATA_FILE
    self.features = 'M'
    self.target = 'OT'
    self.freq = 'h'
    self.seq_len = SEQ_LEN
    self.label_len = LABEL_LEN
    self.pred_len = PRED_LEN
    self.enc_in = NUM_FEATURES
    self.dec_in = NUM_FEATURES
    self.c_out = NUM_FEATURES
    self.d_model = D_MODEL
    self.n_heads = N_HEADS
    self.e_layers = ENCODER_LAYERS
    self.d_layers = DECODER_LAYERS
    self.d_ff = D_FF
    self.factor = 1
    self.dropout = DROPOUT
    self.activation = 'gelu'
    self.output_attention = False
    self.moving_avg = MOVING_AVG
    self.train_epochs = EPOCHS
    self.batch_size = BATCH_SIZE
    self.patience = PATIENCE
    self.learning_rate = LEARNING_RATE
    self.lradj = 'type1'
    self.use_amp = False
    self.num_workers = 0
    self.use_gpu = torch.cuda.is_available()
    self.gpu = 0
    self.devices = '0'
    self.use_multi_gpu = False
    self.checkpoints = '/results/checkpoints'
    self.embed = 'timeF'
    self.distil = True
    self.mix = True
    self.des = 'Exp'
    self.itr = '1'
    self.bucket_size = 4
    self.n_hashes = 4

args = AutoformerConfig()
exp = Exp_Main(args)

setting = 'test_experiment_1'

exp.train(setting)
