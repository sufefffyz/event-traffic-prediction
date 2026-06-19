#!/usr/bin/env python3
"""Export IGSTGNN test predictions without changing the official trainer."""

import argparse
import os
import sys
from pathlib import Path


class PrintLogger:
    def info(self, message):
        print(message)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Load an IGSTGNN checkpoint and save test predictions/targets."
    )
    parser.add_argument(
        "--igstgnn-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "IGSTGNN",
        help="Path to reproduction/IGSTGNN.",
    )
    parser.add_argument("--dataset", required=True, choices=["Alameda", "Contra_Costa", "Orange"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--bs", type=int, default=48)

    parser.add_argument("--seq_len", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--input_dim", type=int, default=3)
    parser.add_argument("--output_dim", type=int, default=1)

    parser.add_argument("--num_feat", type=int, default=1)
    parser.add_argument("--num_hidden", type=int, default=32)
    parser.add_argument("--node_hidden", type=int, default=12)
    parser.add_argument("--time_emb_dim", type=int, default=12)
    parser.add_argument("--layer", type=int, default=5)
    parser.add_argument("--k_t", type=int, default=3)
    parser.add_argument("--k_s", type=int, default=2)
    parser.add_argument("--gap", type=int, default=3)
    parser.add_argument("--cl_epoch", type=int, default=3)
    parser.add_argument("--warm_epoch", type=int, default=30)
    parser.add_argument("--tpd", type=int, default=288)

    parser.add_argument("--lrate", type=float, default=2e-3)
    parser.add_argument("--wdecay", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--clip_grad_value", type=float, default=5)
    parser.add_argument("--icsf_dim", type=int, default=64)
    parser.add_argument("--module_name", default="igstgnn")
    parser.add_argument("--lambda_incident", type=float, default=1.0)
    parser.add_argument("--sigma_t", type=float, default=1.0)

    parser.add_argument("--incident", action="store_true")
    parser.add_argument("--use_sensor_info", action="store_true")
    return parser.parse_args()


def resolve_paths(args):
    args.igstgnn_root = args.igstgnn_root.resolve()
    if args.run_dir is not None:
        args.run_dir = args.run_dir.resolve()
    if args.checkpoint is None:
        if args.run_dir is None:
            raise ValueError("Provide --checkpoint or --run-dir.")
        args.checkpoint = args.run_dir / f"final_model_s{args.seed}.pt"
    else:
        args.checkpoint = args.checkpoint.resolve()
    if args.output is None:
        if args.run_dir is None:
            args.output = args.checkpoint.with_name(f"test_result_s{args.seed}.npz")
        else:
            args.output = args.run_dir / f"test_result_s{args.seed}.npz"
    else:
        args.output = args.output.resolve()
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")


def prepare_batch(batch, device, use_incident):
    import numpy as np
    import torch

    if not isinstance(batch, dict):
        x, label = batch
        return (
            torch.as_tensor(x, dtype=torch.float32, device=device),
            torch.as_tensor(label, dtype=torch.float32, device=device),
            None,
            None,
            {},
        )

    x = torch.as_tensor(batch["x_data"], dtype=torch.float32, device=device)
    label = torch.as_tensor(batch["y_data"], dtype=torch.float32, device=device)
    incident_data = None
    saved_context = {}
    if use_incident:
        incident_data = {
            "incident": torch.as_tensor(batch["incident_features"], dtype=torch.float32, device=device),
            "position": torch.as_tensor(batch["incident_position"], device=device),
            "distances": torch.as_tensor(batch["incident_distances"], dtype=torch.float32, device=device),
        }
        saved_context = {
            "incident_features": np.asarray(batch["incident_features"], dtype=np.float32),
            "incident_position": np.asarray(batch["incident_position"], dtype=np.int64),
            "incident_distances": np.asarray(batch["incident_distances"], dtype=np.float32),
        }

    sensor_data = None
    if "sensor_data" in batch:
        sensor_data = {
            key: torch.as_tensor(value, device=device)
            for key, value in batch["sensor_data"].items()
        }
    return x, label, incident_data, sensor_data, saved_context


def main():
    args = parse_args()

    import numpy as np
    import torch

    resolve_paths(args)

    os.chdir(args.igstgnn_root)
    sys.path.insert(0, str(args.igstgnn_root))

    from src.models.igstgnn import IGSTGNN
    from src.utils.dataloader import get_dataset_info, load_adj_from_numpy, load_dataset
    from src.utils.graph_algo import normalize_adj_mx
    from src.utils.metrics import compute_all_metrics

    device = torch.device(args.device)
    data_path, adj_path, node_num = get_dataset_info(args.dataset)
    args.data_path = data_path
    args.node_num = node_num
    adj_mx = normalize_adj_mx(load_adj_from_numpy(adj_path), "doubletransition")
    args.adjs = [torch.tensor(adj).to(device) for adj in adj_mx]

    logger = PrintLogger()
    dataloader, scaler = load_dataset(data_path, args, logger)
    model = IGSTGNN(
        node_num=node_num,
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        seq_len=args.seq_len,
        horizon=args.horizon,
        model_args=vars(args),
        dataset=args.dataset,
        data_path=data_path,
        use_sensor_info=args.use_sensor_info,
    )
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    preds = []
    labels = []
    incident_features = []
    incident_position = []
    incident_distances = []
    with torch.no_grad():
        for batch in dataloader["test_loader"].get_iterator():
            x, label, incident_data, sensor_data, context = prepare_batch(
                batch, device, args.incident
            )
            pred = model(x, label, incident_data=incident_data, sensor_data=sensor_data)
            pred, label = scaler.inverse_transform(pred), scaler.inverse_transform(label)
            preds.append(pred.squeeze(-1).cpu())
            labels.append(label.squeeze(-1).cpu())
            if context:
                incident_features.append(context["incident_features"])
                incident_position.append(context["incident_position"])
                incident_distances.append(context["incident_distances"])

    preds = torch.cat(preds, dim=0)
    labels = torch.cat(labels, dim=0)
    mask_value = torch.tensor(0.0)
    if labels.min() < 1:
        mask_value = labels.min()

    metrics_by_horizon = []
    for horizon_idx in range(args.horizon):
        metrics_by_horizon.append(
            compute_all_metrics(preds[:, horizon_idx, :], labels[:, horizon_idx, :], mask_value)
        )
    metrics_by_horizon = np.asarray(metrics_by_horizon, dtype=np.float64)
    metrics_average = metrics_by_horizon.mean(axis=0)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs = {
        "prediction": preds.numpy(),
        "target": labels.numpy(),
        "metrics_by_horizon": metrics_by_horizon,
        "metrics_average": metrics_average,
        "metric_names": np.asarray(["mae", "mape", "rmse"]),
        "mask_value": np.asarray(float(mask_value.item())),
        "dataset": np.asarray(args.dataset),
        "seed": np.asarray(args.seed),
        "checkpoint": np.asarray(str(args.checkpoint)),
    }
    if incident_features:
        save_kwargs.update(
            {
                "incident_features": np.concatenate(incident_features, axis=0),
                "incident_position": np.concatenate(incident_position, axis=0),
                "incident_distances": np.concatenate(incident_distances, axis=0),
            }
        )

    np.savez_compressed(args.output, **save_kwargs)
    print(f"Saved test result: {args.output}")
    print(
        "Average Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}".format(
            metrics_average[0], metrics_average[2], metrics_average[1]
        )
    )


if __name__ == "__main__":
    main()
