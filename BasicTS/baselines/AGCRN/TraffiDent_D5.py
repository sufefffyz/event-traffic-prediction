import os
import sys
from functools import partial

from easydict import EasyDict

sys.path.append(os.path.abspath(__file__ + "/../../.."))

from basicts.data.indexed_npz_tsf_dataset import IndexedNPZForecastingDataset
from basicts.metrics import masked_mae, masked_mape, masked_rmse
from basicts.runners import SimpleTimeSeriesForecastingRunner
from basicts.scaler.indexed_npz_scaler import IndexedNPZStandardScaler

from .arch import AGCRN


DATA_NAME = "TraffiDent_D5_2023Q1"
DATA_ROOT = os.environ.get("TRAFFIDENT_BASICTS_ROOT", "/data/yuzhang_fei/TraffiDent/basicts")
DATA_FILE_PATH = os.path.join(DATA_ROOT, DATA_NAME, "data.npz")
INDEX_FILE_PATH = os.path.join(DATA_ROOT, DATA_NAME, "index.npz")

INPUT_LEN = 12
OUTPUT_LEN = 12
TRAIN_VAL_TEST_RATIO = [0.6, 0.2, 0.2]
NULL_VAL = 0.0
NUM_EPOCHS = int(os.environ.get("TRAFFIDENT_NUM_EPOCHS", "100"))
SEED = int(os.environ.get("TRAFFIDENT_SEED", "2023"))
NUM_NODES = 565


CFG = EasyDict()
CFG.DESCRIPTION = (
    "AGCRN on the TraffiDent D5 2023Q1 post-incident forecasting reproduction. "
    "The data split, 12-to-12 window, seed, epochs, patience, and batch size follow "
    "the TraffiDent/LargeST experiment setting where it is specified."
)
CFG.GPU_NUM = 1
CFG.RUNNER = SimpleTimeSeriesForecastingRunner

CFG.ENV = EasyDict()
CFG.ENV.SEED = SEED
CFG.ENV.DETERMINISTIC = False
CFG.ENV.CUDNN = EasyDict({"ENABLED": True, "BENCHMARK": False, "DETERMINISTIC": False})

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
CFG.MODEL.NAME = AGCRN.__name__
CFG.MODEL.ARCH = AGCRN
CFG.MODEL.PARAM = {
    "num_nodes": NUM_NODES,
    "input_dim": 3,
    "rnn_units": 64,
    "output_dim": 1,
    "horizon": OUTPUT_LEN,
    "num_layers": 2,
    "default_graph": True,
    "embed_dim": 10,
    "cheb_k": 2,
}
CFG.MODEL.FORWARD_FEATURES = [0, 1, 2]
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
    AGCRN.__name__,
    "_".join([DATA_NAME, str(NUM_EPOCHS), str(INPUT_LEN), str(OUTPUT_LEN), "paper"]),
)
CFG.TRAIN.LOSS = partial(masked_mae, null_val=NULL_VAL)
CFG.TRAIN.EARLY_STOPPING_PATIENCE = 30
CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "Adam"
CFG.TRAIN.OPTIM.PARAM = {"lr": 0.001, "weight_decay": 0.0}
CFG.TRAIN.DATA = EasyDict()
CFG.TRAIN.DATA.BATCH_SIZE = 64
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
CFG.EVAL.HORIZONS = [1, 3, 6]
CFG.EVAL.USE_GPU = True
CFG.EVAL.SAVE_RESULTS = True
