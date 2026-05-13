import torch
import numpy as np
import os
from .utils import print_log, StandardScaler, vrange
import pandas as pd
# ! X shape: (B, T, N, C)


def _read_hdf_dataframe(data_path):
    try:
        return pd.read_hdf(data_path).fillna(0)
    except ValueError as exc:
        if "unrecognized index type datetime64" not in str(exc):
            raise
        import tables

        with tables.open_file(data_path) as h5_file:
            group = getattr(h5_file.root, "t")
            values = group.block0_values.read()
            index = pd.to_datetime(group.axis1.read())
            columns = group.axis0.read()
            if columns.dtype.kind == "S":
                columns = columns.astype(str)
        return pd.DataFrame(values, index=index, columns=columns).fillna(0)


def _build_data_from_hdf(data_dir):
    df = _read_hdf_dataframe(os.path.join(data_dir, "data.h5"))
    num_nodes = df.shape[1]
    data = np.expand_dims(df.values, axis=-1)

    feature_list = [data]
    time_ind = (df.index.values - df.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
    time_of_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
    feature_list.append(time_of_day)
    dow_tiled = np.tile(df.index.dayofweek, [1, num_nodes, 1]).transpose((2, 1, 0))
    feature_list.append(dow_tiled)

    external_path = os.path.join(data_dir, "external.npz")
    if os.path.isfile(external_path):
        external = np.load(external_path)["data"].astype(np.float32)
        if external.shape[:2] != data.shape[:2]:
            common_steps = min(external.shape[0], data.shape[0])
            common_nodes = min(external.shape[1], data.shape[1])
            feature_list = [
                feature[:common_steps, :common_nodes]
                for feature in feature_list
            ]
            external = external[:common_steps, :common_nodes]
        feature_list.append(external)
    return np.concatenate(feature_list, axis=-1).astype(np.float32)


def get_dataloaders_from_index_data(
    data_dir, tod=False, dow=False, dom=False, acc=False, reg=False, batch_size=64, log=None, shift = False, in_steps = 12, out_steps = 12,
):  
    if os.path.isfile(os.path.join(data_dir, "data.npz")) == True:
        if shift:
            data = np.load(os.path.join(data_dir, "data_shift.npz"))["data"].astype(np.float32)
        else:
            data = np.load(os.path.join(data_dir, "data.npz"))["data"].astype(np.float32)
            required_channels = 1
            if tod:
                required_channels = max(required_channels, 2)
            if dow:
                required_channels = max(required_channels, 3)
            if acc:
                required_channels = max(required_channels, 4)
            if reg:
                required_channels = max(required_channels, 5)
            if data.shape[-1] < required_channels:
                data = _build_data_from_hdf(data_dir)
                np.savez(os.path.join(data_dir, "data.npz"), data=data)
                print_log(
                    f"Rebuilt data.npz with {data.shape[-1]} channels for enabled embeddings.",
                    log=log,
                )
    else:
        data = _build_data_from_hdf(data_dir)
        np.savez(os.path.join(data_dir, f"data.npz"), data=data)


    features = [0]
    if tod:
        features.append(1)
    if dow:
        features.append(2)
        # data[..., 2] = np.where(data[..., 2] >= 5, 1, 0)
    # if dom:
    #     features.append(3)
    if acc:
        if data.shape[-1] <= 3:
            raise ValueError("Accident embedding is enabled, but data has no accident channel at index 3.")
        features.append(3)
    if reg:
        if data.shape[-1] <= 4:
            raise ValueError("Region embedding is enabled, but data has no region channel at index 4.")
        features.append(4)
    data = data[..., features]

    if os.path.isfile(os.path.join(data_dir, f"index.npz")) == False:
        idx1 = np.arange(len(data) - in_steps - out_steps)
        idx2 = np.arange(in_steps, len(data) - out_steps)
        idx3 = np.arange(in_steps + out_steps, len(data))
        index = np.stack([idx1, idx2, idx3], -1)
        # np.savez(os.path.join(data_dir, f"index.npz"),
        #             train=index[:int(0.66 * 0.75 * len(data))],
        #             val=index[int(0.66 * 0.75 * len(data)):int(0.66 * len(data))],
        #             test=index[int(0.66 * len(data)):]
        #             )
        np.savez(os.path.join(data_dir, f"index.npz"),
                train=index[:int(0.6 * len(data))],
                val=index[int(0.6 * len(data)):int(0.8 * len(data))],
                test=index[int(0.8 * len(data)):]
                )


    index = np.load(os.path.join(data_dir, "index.npz"))

    train_index = index["train"]  # (num_samples, 3)
    val_index = index["val"]
    test_index = index["test"]

    x_train_index = vrange(train_index[:, 0], train_index[:, 1])
    y_train_index = vrange(train_index[:, 1], train_index[:, 2])
    x_val_index = vrange(val_index[:, 0], val_index[:, 1])
    y_val_index = vrange(val_index[:, 1], val_index[:, 2])
    x_test_index = vrange(test_index[:, 0], test_index[:, 1])
    y_test_index = vrange(test_index[:, 1], test_index[:, 2])

    x_train = data[x_train_index]
    y_train = data[y_train_index][..., :1]
    x_val = data[x_val_index]
    y_val = data[y_val_index][..., :1]
    x_test = data[x_test_index]
    y_test = data[y_test_index][..., :1]

    scaler = StandardScaler(mean=x_train[..., 0].mean(), std=x_train[..., 0].std())

    x_train[..., 0] = scaler.transform(x_train[..., 0])
    x_val[..., 0] = scaler.transform(x_val[..., 0])
    x_test[..., 0] = scaler.transform(x_test[..., 0])

    print_log(f"Trainset:\tx-{x_train.shape}\ty-{y_train.shape}", log=log)
    print_log(f"Valset:  \tx-{x_val.shape}  \ty-{y_val.shape}", log=log)
    print_log(f"Testset:\tx-{x_test.shape}\ty-{y_test.shape}", log=log)

    trainset = torch.utils.data.TensorDataset(
        torch.FloatTensor(x_train), torch.FloatTensor(y_train)
    )
    valset = torch.utils.data.TensorDataset(
        torch.FloatTensor(x_val), torch.FloatTensor(y_val)
    )
    testset = torch.utils.data.TensorDataset(
        torch.FloatTensor(x_test), torch.FloatTensor(y_test)
    )

    trainset_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True
    )
    valset_loader = torch.utils.data.DataLoader(
        valset, batch_size=batch_size, shuffle=False
    )
    testset_loader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False
    )

    return trainset_loader, valset_loader, testset_loader, scaler
