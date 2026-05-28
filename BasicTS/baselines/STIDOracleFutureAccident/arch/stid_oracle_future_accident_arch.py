import torch
from torch import nn

from .mlp import MultiLayerPerceptron


class STIDOracleFutureAccident(nn.Module):
    """STID with an oracle future-accident residual router.

    This model intentionally reads future accident covariates. It is an
    upper-bound diagnostic, not a deployable traffic forecasting setting.
    """

    def __init__(self, **model_args):
        super().__init__()
        self.num_nodes = model_args["num_nodes"]
        self.node_dim = model_args["node_dim"]
        self.input_len = model_args["input_len"]
        self.input_dim = model_args["input_dim"]
        self.embed_dim = model_args["embed_dim"]
        self.output_len = model_args["output_len"]
        self.num_layer = model_args["num_layer"]
        self.temp_dim_tid = model_args["temp_dim_tid"]
        self.temp_dim_diw = model_args["temp_dim_diw"]
        self.time_of_day_size = model_args["time_of_day_size"]
        self.day_of_week_size = model_args["day_of_week_size"]
        self.day_of_week_normalized = model_args.get("day_of_week_normalized", True)
        self.if_time_in_day = model_args["if_T_i_D"]
        self.if_day_in_week = model_args["if_D_i_W"]
        self.if_spatial = model_args["if_node"]
        self.accident_feature_index = model_args.get("accident_feature_index", 3)
        self.future_event_dim = model_args.get("future_event_dim", 32)
        self.residual_num_layer = model_args.get("residual_num_layer", 2)
        self.gate_init_bias = model_args.get("gate_init_bias", -1.0)
        self.register_buffer(
            "future_started_kernel",
            torch.triu(torch.ones(self.output_len, self.output_len)),
            persistent=False,
        )
        self.register_buffer(
            "future_remaining_kernel",
            torch.tril(torch.ones(self.output_len, self.output_len)),
            persistent=False,
        )

        if self.if_spatial:
            self.node_emb = nn.Parameter(torch.empty(self.num_nodes, self.node_dim))
            nn.init.xavier_uniform_(self.node_emb)
        if self.if_time_in_day:
            self.time_in_day_emb = nn.Parameter(
                torch.empty(self.time_of_day_size, self.temp_dim_tid)
            )
            nn.init.xavier_uniform_(self.time_in_day_emb)
        if self.if_day_in_week:
            self.day_in_week_emb = nn.Parameter(
                torch.empty(self.day_of_week_size, self.temp_dim_diw)
            )
            nn.init.xavier_uniform_(self.day_in_week_emb)

        self.time_series_emb_layer = nn.Conv2d(
            in_channels=self.input_dim * self.input_len,
            out_channels=self.embed_dim,
            kernel_size=(1, 1),
            bias=True,
        )

        self.base_hidden_dim = (
            self.embed_dim
            + self.node_dim * int(self.if_spatial)
            + self.temp_dim_tid * int(self.if_time_in_day)
            + self.temp_dim_diw * int(self.if_day_in_week)
        )
        self.base_encoder = nn.Sequential(
            *[
                MultiLayerPerceptron(self.base_hidden_dim, self.base_hidden_dim)
                for _ in range(self.num_layer)
            ]
        )
        self.base_regression_layer = nn.Conv2d(
            in_channels=self.base_hidden_dim,
            out_channels=self.output_len,
            kernel_size=(1, 1),
            bias=True,
        )

        # Channels: event at horizon, event started by horizon, event remaining,
        # and future-any. All are event covariates, never future flow.
        self.future_event_encoder = nn.Sequential(
            nn.Conv2d(4, self.future_event_dim, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(),
            nn.Conv2d(
                self.future_event_dim,
                self.future_event_dim,
                kernel_size=(1, 1),
                bias=True,
            ),
            nn.ReLU(),
        )

        self.router_hidden_dim = self.base_hidden_dim + self.future_event_dim
        self.router_encoder = nn.Sequential(
            *[
                MultiLayerPerceptron(self.router_hidden_dim, self.router_hidden_dim)
                for _ in range(self.residual_num_layer)
            ]
        )
        self.residual_layer = nn.Conv2d(
            in_channels=self.router_hidden_dim,
            out_channels=1,
            kernel_size=(1, 1),
            bias=True,
        )
        self.gate_layer = nn.Conv2d(
            in_channels=self.router_hidden_dim,
            out_channels=1,
            kernel_size=(1, 1),
            bias=True,
        )
        nn.init.constant_(self.gate_layer.bias, self.gate_init_bias)

    def _time_index(self, raw: torch.Tensor) -> torch.Tensor:
        idx = raw * self.time_of_day_size
        return idx.long().clamp_(0, self.time_of_day_size - 1)

    def _day_index(self, raw: torch.Tensor) -> torch.Tensor:
        if self.day_of_week_normalized:
            raw = raw * self.day_of_week_size
        return raw.long().clamp_(0, self.day_of_week_size - 1)

    def _future_event_features(self, future_data: torch.Tensor) -> torch.Tensor:
        future_accident = (future_data[..., self.accident_feature_index] > 0).float()
        event_by_node = future_accident.transpose(1, 2)
        started = torch.matmul(
            event_by_node,
            self.future_started_kernel.to(future_accident.dtype),
        ).transpose(1, 2).clamp(0, 1)
        remaining = torch.matmul(
            event_by_node,
            self.future_remaining_kernel.to(future_accident.dtype),
        ).transpose(1, 2).clamp(0, 1)
        future_any = future_accident.amax(dim=1, keepdim=True).expand_as(future_accident)
        event_features = torch.stack(
            [future_accident, started, remaining, future_any],
            dim=1,
        )
        return event_features.permute(0, 1, 3, 2).contiguous()

    def forward(
        self,
        history_data: torch.Tensor,
        future_data: torch.Tensor,
        batch_seen: int,
        epoch: int,
        train: bool,
        **kwargs,
    ) -> torch.Tensor:
        input_data = history_data[..., : self.input_dim]

        batch_size, _, num_nodes, _ = input_data.shape
        input_data = input_data.transpose(1, 2).contiguous()
        input_data = input_data.view(batch_size, num_nodes, -1).transpose(1, 2).unsqueeze(-1)
        time_series_emb = self.time_series_emb_layer(input_data)

        node_emb = []
        if self.if_spatial:
            node_emb.append(
                self.node_emb.unsqueeze(0)
                .expand(batch_size, -1, -1)
                .transpose(1, 2)
                .unsqueeze(-1)
            )

        tem_emb = []
        if self.if_time_in_day:
            t_i_d_data = history_data[..., 1]
            time_in_day_emb = self.time_in_day_emb[self._time_index(t_i_d_data[:, -1, :])]
            tem_emb.append(time_in_day_emb.transpose(1, 2).unsqueeze(-1))
        if self.if_day_in_week:
            d_i_w_data = history_data[..., 2]
            day_in_week_emb = self.day_in_week_emb[self._day_index(d_i_w_data[:, -1, :])]
            tem_emb.append(day_in_week_emb.transpose(1, 2).unsqueeze(-1))

        base_hidden = torch.cat([time_series_emb] + node_emb + tem_emb, dim=1)
        base_hidden = self.base_encoder(base_hidden)
        base_prediction = self.base_regression_layer(base_hidden)

        future_event_hidden = self.future_event_encoder(
            self._future_event_features(future_data)
        )
        base_context = base_hidden.expand(-1, -1, -1, self.output_len)
        router_hidden = torch.cat([base_context, future_event_hidden], dim=1)
        router_hidden = self.router_encoder(router_hidden)

        future_any_mask = (
            future_data[..., self.accident_feature_index].sum(dim=1) > 0
        ).float()
        future_any_mask = future_any_mask.unsqueeze(1).unsqueeze(-1)
        gate = torch.sigmoid(self.gate_layer(router_hidden)) * future_any_mask
        residual = self.residual_layer(router_hidden)
        correction = (gate * residual).permute(0, 3, 2, 1).contiguous()
        return base_prediction + correction
