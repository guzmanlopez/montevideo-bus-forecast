# %%
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from src.modeling.montevideo_bus_dataset import MontevideoBusDatasetLoader
from src.preparation.constants import KERNEL_SIZE
from torch.nn import functional as F
from torch_geometric_temporal.nn import STConv
from torch_geometric_temporal.signal import temporal_signal_split


class LITSTConvModel(pl.LightningModule):
    def __init__(self, num_nodes, in_channels, hidden_channels, out_channels, kernel_size, K):
        super().__init__()
        self.stconv = STConv(
            num_nodes=num_nodes,
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            K=K,
        )
        self.linear = torch.nn.Linear(out_channels, 1)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-2)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x = train_batch.x
        y = train_batch.y.view(-1, 1)
        edge_index = train_batch.edge_index
        h = self.stconv(x, edge_index)
        h = F.relu(h)
        h = self.linear(h)
        loss = F.mse_loss(h, y)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x = val_batch.x
        y = val_batch.y.view(-1, 1)
        edge_index = val_batch.edge_index
        h = self.stconv(x, edge_index)
        h = F.relu(h)
        h = self.linear(h)
        loss = F.mse_loss(h, y)
        metrics = {"val_loss": loss}
        self.log_dict(metrics)
        return metrics


# %%
loader = MontevideoBusDatasetLoader()
dataset_loader = loader.get_dataset(lags=4, target_var="y", feature_vars=["y"])
train_loader, val_loader = temporal_signal_split(dataset_loader, train_ratio=0.75)

# %%
num_nodes = len(dataset_loader.features)
in_channels = dataset_loader.features[0].shape[1]

model = LITSTConvModel(
    num_nodes=num_nodes,
    in_channels=in_channels,
    out_channels=2,
    hidden_channels=1,
    K=2,
    kernel_size=KERNEL_SIZE,
)

early_stop_callback = EarlyStopping(
    monitor="val_loss", min_delta=0.00, patience=10, verbose=False, mode="max"
)

trainer = pl.Trainer(callbacks=[early_stop_callback])
trainer.fit(model, train_loader, val_loader)
# %%
