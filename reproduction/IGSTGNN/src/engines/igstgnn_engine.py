import torch
import numpy as np
from src.base.engine import BaseEngine
from tqdm import tqdm
from src.utils.metrics import masked_mape, masked_rmse

class IGSTGNN_Engine(BaseEngine):
    """
    Incident-aware IGSTGNN engine
    """
    def __init__(self, cl_step, warm_step, horizon, incident=False, **args):
        super(IGSTGNN_Engine, self).__init__(**args)
        self._cl_step = cl_step
        self._warm_step = warm_step
        self._horizon = horizon
        self._cl_len = 0
        self._incident = incident

    def train_batch(self):
        self.model.train()

        train_loss = []
        train_mape = []
        train_rmse = []
        self._dataloader['train_loader'].shuffle()

        # Get iterator and total number of batches
        iterator = self._dataloader['train_loader'].get_iterator()
        total_batches = self._dataloader['train_loader'].num_batch
        
        # Create progress bar
        progress_bar = tqdm(iterator, total=total_batches, desc="Training",
                            unit="batch", leave=False, position=0, 
                            dynamic_ncols=True, colour="green")
        try:
            for batch_idx, batch in enumerate(progress_bar):
                self._optimizer.zero_grad()
                
                if self._incident and isinstance(batch, dict):
                    X = batch['x_data']
                    label = batch['y_data']
                    incident_data = {
                        'incident': batch['incident_features'],
                        'position': batch['incident_position'],
                        'distances': batch['incident_distances'],
                        'durations': batch['durations']
                    }
                    
                    X, label = self._to_device(self._to_tensor([X, label]))
                    for key in incident_data:
                        incident_data[key] = self._to_device(self._to_tensor(incident_data[key]))
                    
                    sensor_data = None
                    if 'sensor_data' in batch:
                        sensor_data = {}
                        for key, value in batch['sensor_data'].items():
                            sensor_data[key] = self._to_device(value)
                    
                    pred = self.model(X, label, incident_data=incident_data, sensor_data=sensor_data)
                else:
                    if isinstance(batch, tuple) and len(batch) == 2:
                        X, label = batch
                    else:
                        X = batch[0]
                        label = batch[1]
                    X, label = self._to_device(self._to_tensor([X, label]))
                    pred = self.model(X, label)

                pred, label = self._inverse_transform([pred, label])

                mask_value = torch.tensor(0)
                if label.min() < 1:
                    mask_value = label.min()
                if self._iter_cnt == 0:
                    print('check mask value', mask_value)

                self._iter_cnt += 1
                if self._iter_cnt < self._warm_step:
                    self._cl_len = self._horizon
                elif self._iter_cnt == self._warm_step:
                    self._cl_len = 1
                else:
                    if (self._iter_cnt - self._warm_step) % self._cl_step == 0 and self._cl_len < self._horizon:
                        self._cl_len += 1

                pred = pred[:, :self._cl_len, :, :]
                label = label[:, :self._cl_len, :, :]

                loss = self._loss_fn(pred, label, mask_value)
                mape = masked_mape(pred, label, mask_value).item()
                rmse = masked_rmse(pred, label, mask_value).item()

                loss.backward()
                if self._clip_grad_value != 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self._clip_grad_value)
                self._optimizer.step()

                train_loss.append(loss.item())
                train_mape.append(mape)
                train_rmse.append(rmse)

                progress_bar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "mape": f"{mape:.4f}",
                    "rmse": f"{rmse:.4f}"
                })

        finally:
            progress_bar.close()
            
        return np.mean(train_loss), np.mean(train_mape), np.mean(train_rmse)
