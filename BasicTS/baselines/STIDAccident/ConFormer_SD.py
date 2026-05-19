import os
import sys
from functools import partial

from easydict import EasyDict

sys.path.append(os.path.abspath(__file__ + "/../../.."))

from basicts.data.indexed_npz_tsf_dataset import IndexedNPZForecastingDataset
from basicts.metrics import masked_mae, masked_mape, masked_rmse
from basicts.runners import WandBTimeSeriesForecastingRunner
from basicts.scaler.indexed_npz_scaler import IndexedNPZStandardScaler

from .arch import STIDAccident

DATA_NAME = "ConFormer_SD"
INPUT_LEN = 12
OUTPUT_LEN = 12
TRAIN_VAL_TEST_RATIO = [0.6, 0.2, 0.2]
NUM_NODES = 716
TIME_OF_DAY_SIZE = 96
DAY_OF_WEEK_SIZE = 7
NULL_VAL = 0.0
NUM_EPOCHS = 100
SEED = 42

DATA_FILE_PATH = "../reproduction/ConFormer/data/SD/data.npz"
INDEX_FILE_PATH = "../reproduction/ConFormer/data/SD/index.npz"

MODEL_ARCH = STIDAccident
MODEL_PARAM = {
    "num_nodes": NUM_NODES,
    "input_len": INPUT_LEN,
    "input_dim": 3,
    "embed_dim": 32,
    "output_len": OUTPUT_LEN,
    "num_layer": 4,
    "if_node": True,
    "node_dim": 64,
    "if_T_i_D": True,
    "if_D_i_W": True,
    "temp_dim_tid": 32,
    "temp_dim_diw": 32,
    "time_of_day_size": TIME_OF_DAY_SIZE,
    "day_of_week_size": DAY_OF_WEEK_SIZE,
    "day_of_week_normalized": False,
    "if_accident": True,
    "accident_dim": 32,
    "accident_feature_index": 3,
}

CFG = EasyDict()
CFG.DESCRIPTION = (
    "Minimal STID + binary accident embedding on ConFormer SD. "
    "Split follows ConFormer index.npz; metrics use BasicTS masked MAE/MAPE/RMSE."
)
CFG.GPU_NUM = 1
CFG.RUNNER = WandBTimeSeriesForecastingRunner

CFG.WANDB = EasyDict()
CFG.WANDB.PROJECT = "event-traffic-prediction"
CFG.WANDB.GROUP = "ConFormer_SD_BasicTS"
CFG.WANDB.RUN_NAME = "STIDAccident_ConFormer_SD_seed42"
CFG.WANDB.TAGS = ["ConFormer_SD", "STIDAccident", "basicts", "seed42"]

CFG.ENV = EasyDict()
CFG.ENV.SEED = SEED
CFG.ENV.DETERMINISTIC = True
CFG.ENV.CUDNN = EasyDict({"ENABLED": True, "BENCHMARK": False, "DETERMINISTIC": True})

CFG.DATASET = EasyDict()
CFG.DATASET.NAME = DATA_NAME
CFG.DATASET.TYPE = IndexedNPZForecastingDataset
CFG.DATASET.PARAM = EasyDict(
    {
        "data_file_path": DATA_FILE_PATH,
        "index_file_path": INDEX_FILE_PATH,
        "train_val_test_ratio": TRAIN_VAL_TEST_RATIO,
        "input_len": INPUT_LEN,
        "output_len": OUTPUT_LEN,
    }
)

CFG.SCALER = EasyDict()
CFG.SCALER.TYPE = IndexedNPZStandardScaler
CFG.SCALER.PARAM = EasyDict(
    {
        "dataset_name": DATA_NAME,
        "data_file_path": DATA_FILE_PATH,
        "index_file_path": INDEX_FILE_PATH,
        "train_ratio": TRAIN_VAL_TEST_RATIO[0],
        "norm_each_channel": False,
        "rescale": True,
        "input_len": INPUT_LEN,
        "output_len": OUTPUT_LEN,
    }
)

CFG.MODEL = EasyDict()
CFG.MODEL.NAME = MODEL_ARCH.__name__
CFG.MODEL.ARCH = MODEL_ARCH
CFG.MODEL.PARAM = MODEL_PARAM
CFG.MODEL.FORWARD_FEATURES = [0, 1, 2, 3]
CFG.MODEL.TARGET_FEATURES = [0]

CFG.METRICS = EasyDict()
CFG.METRICS.FUNCS = EasyDict(
    {
        "MAE": partial(masked_mae, null_val=NULL_VAL),
        "MAPE": partial(masked_mape, null_val=NULL_VAL),
        "RMSE": partial(masked_rmse, null_val=NULL_VAL),
    }
)
CFG.METRICS.TARGET = "MAE"
CFG.METRICS.NULL_VAL = NULL_VAL

CFG.TRAIN = EasyDict()
CFG.TRAIN.NUM_EPOCHS = NUM_EPOCHS
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join(
    "checkpoints",
    MODEL_ARCH.__name__,
    "_".join([DATA_NAME, str(NUM_EPOCHS), str(INPUT_LEN), str(OUTPUT_LEN), "accident"]),
)
CFG.TRAIN.LOSS = partial(masked_mae, null_val=NULL_VAL)
CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "Adam"
CFG.TRAIN.OPTIM.PARAM = {"lr": 0.002, "weight_decay": 0.0001}
CFG.TRAIN.LR_SCHEDULER = EasyDict()
CFG.TRAIN.LR_SCHEDULER.TYPE = "MultiStepLR"
CFG.TRAIN.LR_SCHEDULER.PARAM = {"milestones": [1, 30, 60, 80], "gamma": 0.5}
CFG.TRAIN.CLIP_GRAD_PARAM = {"max_norm": 5.0}
CFG.TRAIN.DATA = EasyDict()
CFG.TRAIN.DATA.BATCH_SIZE = 32
CFG.TRAIN.DATA.SHUFFLE = True

CFG.VAL = EasyDict()
CFG.VAL.INTERVAL = 1
CFG.VAL.DATA = EasyDict()
CFG.VAL.DATA.BATCH_SIZE = 64

CFG.TEST = EasyDict()
CFG.TEST.INTERVAL = 1
CFG.TEST.DATA = EasyDict()
CFG.TEST.DATA.BATCH_SIZE = 64

CFG.EVAL = EasyDict()
CFG.EVAL.HORIZONS = [3, 6, 12]
CFG.EVAL.USE_GPU = False
CFG.EVAL.SAVE_RESULTS = True
