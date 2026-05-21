import os
import sys
from functools import partial

from easydict import EasyDict

sys.path.append(os.path.abspath(__file__ + "/../../.."))

from basicts.data.indexed_npz_tsf_dataset import IndexedNPZForecastingDataset
from basicts.metrics import masked_mae, masked_mape, masked_rmse
from basicts.runners import WandBTimeSeriesForecastingRunner
from basicts.scaler.indexed_npz_scaler import IndexedNPZStandardScaler

from .arch import STID


INPUT_LEN = 12
OUTPUT_LEN = 12
TRAIN_VAL_TEST_RATIO = [0.6, 0.2, 0.2]
TIME_OF_DAY_SIZE = 288
DAY_OF_WEEK_SIZE = 7
NULL_VAL = 0.0
NUM_EPOCHS = int(os.environ.get("TRAFFIDENT_NUM_EPOCHS", "100"))
SEED = 42
DATA_ROOT = os.environ.get("TRAFFIDENT_BASICTS_ROOT", "/data/yuzhang_fei/TraffiDent/basicts")


def make_cfg(dataset_slug: str, num_nodes: int):
    data_name = f"TraffiDent_{dataset_slug}_2023Q1"
    data_file_path = os.path.join(DATA_ROOT, data_name, "data.npz")
    index_file_path = os.path.join(DATA_ROOT, data_name, "index.npz")

    model_param = {
        "num_nodes": num_nodes,
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
    }

    cfg = EasyDict()
    cfg.DESCRIPTION = (
        "Pure STID on TraffiDent county subset. Data is prepared from the "
        "official TraffiDent archive with official-style incident matching."
    )
    cfg.GPU_NUM = 1
    cfg.RUNNER = WandBTimeSeriesForecastingRunner

    cfg.WANDB = EasyDict()
    cfg.WANDB.PROJECT = "event-traffic-prediction"
    cfg.WANDB.GROUP = "TraffiDent_2023Q1_STID"
    cfg.WANDB.RUN_NAME = f"STID_{data_name}_seed42"
    cfg.WANDB.TAGS = ["TraffiDent", data_name, "STID", "basicts", "seed42"]

    cfg.ENV = EasyDict()
    cfg.ENV.SEED = SEED
    cfg.ENV.DETERMINISTIC = True
    cfg.ENV.CUDNN = EasyDict({"ENABLED": True, "BENCHMARK": False, "DETERMINISTIC": True})

    cfg.DATASET = EasyDict()
    cfg.DATASET.NAME = data_name
    cfg.DATASET.TYPE = IndexedNPZForecastingDataset
    cfg.DATASET.PARAM = EasyDict(
        {
            "data_file_path": data_file_path,
            "index_file_path": index_file_path,
            "train_val_test_ratio": TRAIN_VAL_TEST_RATIO,
            "input_len": INPUT_LEN,
            "output_len": OUTPUT_LEN,
        }
    )

    cfg.SCALER = EasyDict()
    cfg.SCALER.TYPE = IndexedNPZStandardScaler
    cfg.SCALER.PARAM = EasyDict(
        {
            "dataset_name": data_name,
            "data_file_path": data_file_path,
            "index_file_path": index_file_path,
            "train_ratio": TRAIN_VAL_TEST_RATIO[0],
            "norm_each_channel": False,
            "rescale": True,
            "input_len": INPUT_LEN,
            "output_len": OUTPUT_LEN,
        }
    )

    cfg.MODEL = EasyDict()
    cfg.MODEL.NAME = STID.__name__
    cfg.MODEL.ARCH = STID
    cfg.MODEL.PARAM = model_param
    cfg.MODEL.FORWARD_FEATURES = [0, 1, 2]
    cfg.MODEL.TARGET_FEATURES = [0]

    cfg.METRICS = EasyDict()
    cfg.METRICS.FUNCS = EasyDict(
        {
            "MAE": partial(masked_mae, null_val=NULL_VAL),
            "MAPE": partial(masked_mape, null_val=NULL_VAL),
            "RMSE": partial(masked_rmse, null_val=NULL_VAL),
        }
    )
    cfg.METRICS.TARGET = "MAE"
    cfg.METRICS.NULL_VAL = NULL_VAL

    cfg.TRAIN = EasyDict()
    cfg.TRAIN.NUM_EPOCHS = NUM_EPOCHS
    cfg.TRAIN.CKPT_SAVE_DIR = os.path.join(
        "checkpoints",
        STID.__name__,
        "_".join([data_name, str(NUM_EPOCHS), str(INPUT_LEN), str(OUTPUT_LEN), "pure"]),
    )
    cfg.TRAIN.LOSS = partial(masked_mae, null_val=NULL_VAL)
    cfg.TRAIN.OPTIM = EasyDict()
    cfg.TRAIN.OPTIM.TYPE = "Adam"
    cfg.TRAIN.OPTIM.PARAM = {"lr": 0.002, "weight_decay": 0.0001}
    cfg.TRAIN.LR_SCHEDULER = EasyDict()
    cfg.TRAIN.LR_SCHEDULER.TYPE = "MultiStepLR"
    cfg.TRAIN.LR_SCHEDULER.PARAM = {"milestones": [1, 30, 60, 80], "gamma": 0.5}
    cfg.TRAIN.CLIP_GRAD_PARAM = {"max_norm": 5.0}
    cfg.TRAIN.DATA = EasyDict()
    cfg.TRAIN.DATA.BATCH_SIZE = 32
    cfg.TRAIN.DATA.SHUFFLE = True

    cfg.VAL = EasyDict()
    cfg.VAL.INTERVAL = 1
    cfg.VAL.DATA = EasyDict()
    cfg.VAL.DATA.BATCH_SIZE = 64

    cfg.TEST = EasyDict()
    cfg.TEST.INTERVAL = 1
    cfg.TEST.DATA = EasyDict()
    cfg.TEST.DATA.BATCH_SIZE = 64

    cfg.EVAL = EasyDict()
    cfg.EVAL.HORIZONS = [3, 6, 12]
    cfg.EVAL.USE_GPU = False
    cfg.EVAL.SAVE_RESULTS = True

    return cfg
