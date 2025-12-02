import torch
import numpy as np
import random

from pathlib import Path
from exp.exp_main import Exp_Main

SEED = 42

# Setting the random seed for reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Data paths
DATA_PATH = './data/ETT-small'
DATA_FILE = 'ETTh1.csv'

# Overall hyperparameters
SEQ_LEN = 96
LABEL_LEN = 48
PRED_LEN = 96
NUM_FEATURES = 7
EPOCHS = 10
BATCH_SIZE = 32
PATIENCE = 3
LEARNING_RATE = 0.0001

# Model hyperparameters
D_MODEL = 512
N_HEADS = 8
ENCODER_LAYERS = 2
DECODER_LAYERS = 1
D_FF = 2048
DROPOUT = 0.05
MOVING_AVG = 25

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
