# %%
from src.modeling.montevideo_bus_dataset import MontevideoBusDatasetLoader
from torch_geometric_temporal.signal import temporal_signal_split

loader = MontevideoBusDatasetLoader()

dataset = loader.get_dataset(lags=4, target_var="y", feature_vars=["y"])

train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.75)

# %%
