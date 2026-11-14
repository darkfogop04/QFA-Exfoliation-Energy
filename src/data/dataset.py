import logging
import os
import torch
import numpy as np
from typing import Tuple, Dict, Any, List
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

from src.data.crystal_graph import CrystalGraphConverter
from src.data import graph_cache

try:
    from matbench.bench import MatbenchBenchmark
except ImportError:
    MatbenchBenchmark = None
    logging.warning("Matbench not installed. Please install it using `pip install matbench`.")

logger = logging.getLogger(__name__)

class MatBenchExfoliationDataset:
    """
    Dataset wrapper for a MatBench structure task (default: matbench_jdft2d).
    Any task with crystal-structure inputs works; set data.task_name.
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config (dict): Configuration dict containing batch_size, cutoff_radius, etc.
        """
        if MatbenchBenchmark is None:
            raise ImportError("matbench is required to use this dataset. Run `pip install matbench`.")
            
        self.config = config
        data_cfg = config.get("data", {}) if isinstance(config.get("data"), dict) else {}
        self.batch_size = data_cfg.get("batch_size", config.get("batch_size", 16))
        
        cutoff_radius = data_cfg.get("cutoff_radius", config.get("cutoff_radius", 8.0))
        max_neighbors = data_cfg.get("max_neighbors", config.get("max_neighbors", 12))
        
        self.converter = CrystalGraphConverter(
            cutoff_radius=cutoff_radius, 
            max_neighbors=max_neighbors
        )
        
        task_name = data_cfg.get("task_name", "matbench_jdft2d")
        self.task_name = task_name
        self.cutoff_radius, self.max_neighbors = cutoff_radius, max_neighbors
        # Graph cache (built once, shared by all folds): data.graph_cache: true/false
        self.use_graph_cache = data_cfg.get("graph_cache", True)
        self.graph_cache_dir = data_cfg.get("graph_cache_dir", "data/cache")
        self.graph_cache_jobs = data_cfg.get("graph_cache_jobs", os.cpu_count() or 1)
        self.max_train = data_cfg.get("max_train")     # optional caps, for quick trials only
        self.max_eval = data_cfg.get("max_eval")
        self._graphs = None
        logger.info(f"Initializing MatbenchBenchmark for {task_name}...")
        self.mb = MatbenchBenchmark(subset=[task_name], autoload=False)
        self.task = list(self.mb.tasks)[0]
        self.task.load()
        
        # State variables for normalization
        self.target_mean = 0.0
        self.target_std = 1.0
        
        # Cache for converted PyG Data objects
        self.cache: Dict[str, List[Data]] = {}

    def get_num_folds(self) -> int:
        """Returns the number of folds in the dataset."""
        return len(self.task.folds) # Usually 5
        
    def _convert_and_cache(self, cache_key: str, inputs, outputs, standardize: bool = False):
        if cache_key in self.cache:
            return self.cache[cache_key]
            
        data_list = []
        for struct, target in zip(inputs, outputs):
            data = self.converter.convert(struct, target)
            data_list.append(data)
            
        if standardize:
            targets = torch.tensor([data.y.item() for data in data_list])
            self.target_mean = targets.mean().item()
            self.target_std = targets.std().item()
            
            if self.target_std < 1e-6:
                self.target_std = 1.0
                
            for data in data_list:
                data.y = (data.y - self.target_mean) / self.target_std
        else:
            # Apply existing standardization if it's test data
            for data in data_list:
                data.y = (data.y - self.target_mean) / self.target_std
                
        self.cache[cache_key] = data_list
        return data_list

    def load_fold(self, fold_index: int) -> Tuple[DataLoader, DataLoader]:
        """
        Loads the train and test DataLoaders for a specific fold.
        
        Args:
            fold_index (int): The fold to load.
            
        Returns:
            Tuple[DataLoader, DataLoader]: Train loader and Test loader.
        """
        logger.info(f"Loading fold {fold_index}...")
        
        train_inputs, train_outputs = self.task.get_train_and_val_data(fold_index)
        test_inputs, test_outputs = self.task.get_test_data(fold_index, include_target=True)
        
        train_cache_key = f"train_fold_{fold_index}"
        test_cache_key = f"test_fold_{fold_index}"
        
        train_data = self._convert_and_cache(train_cache_key, train_inputs, train_outputs, standardize=True)
        test_data = self._convert_and_cache(test_cache_key, test_inputs, test_outputs, standardize=False)
        
        train_loader = DataLoader(
            train_data, 
            batch_size=self.batch_size, 
            shuffle=True, 
            drop_last=False
        )
        
        test_loader = DataLoader(
            test_data, 
            batch_size=self.batch_size, 
            shuffle=False, 
            drop_last=False
        )
        
        return train_loader, test_loader
        
    def load_fold_with_val(self, fold_index: int, val_fraction: float = 0.1,
                           seed: int = 42) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Train / validation / test loaders for one MatBench fold.

        The validation set is carved out of the MatBench training data, so
        early stopping, LR scheduling and every pruning decision see only
        training data. The test fold is used once, for the final number.
        Targets are standardised with the statistics of the training subset.
        """
        train_inputs, train_outputs = self.task.get_train_and_val_data(fold_index)
        test_inputs, test_outputs = self.task.get_test_data(fold_index, include_target=True)

        n = len(train_inputs)
        rng = np.random.default_rng(seed + fold_index)
        perm = rng.permutation(n)
        n_val = max(1, int(round(val_fraction * n)))
        val_idx, tr_idx = perm[:n_val], perm[n_val:]

        if self.max_train:
            tr_idx = tr_idx[: self.max_train]
        if self.max_eval:
            val_idx = val_idx[: self.max_eval]

        key = f"fold{fold_index}_vf{val_fraction}_s{seed}"
        if key not in self.cache and self.use_graph_cache:
            if self._graphs is None:
                self._graphs = graph_cache.load_or_build(
                    self.task, self.task_name, self.graph_cache_dir,
                    self.cutoff_radius, self.max_neighbors, self.graph_cache_jobs)
            g = self._graphs
            ids = list(train_inputs.index)
            te_ids = list(test_inputs.index)[: self.max_eval] if self.max_eval else list(test_inputs.index)
            tr_pos = graph_cache.positions_for(g, [ids[i] for i in tr_idx])
            va_pos = graph_cache.positions_for(g, [ids[i] for i in val_idx])
            te_pos = graph_cache.positions_for(g, te_ids)
            y = g["y"][tr_pos].to(torch.float32)
            mean, std = y.mean().item(), y.std().item()
            std = std if std > 1e-6 else 1.0
            mk_set = lambda pos: graph_cache.GraphSubset(g, pos, mean, std)
            self.cache[key] = (mk_set(tr_pos), mk_set(va_pos), mk_set(te_pos), mean, std)
        elif key not in self.cache:
            tr = [self.converter.convert(s, t) for s, t in
                  zip(train_inputs.iloc[tr_idx], train_outputs.iloc[tr_idx])]
            va = [self.converter.convert(s, t) for s, t in
                  zip(train_inputs.iloc[val_idx], train_outputs.iloc[val_idx])]
            te = [self.converter.convert(s, t) for s, t in zip(test_inputs, test_outputs)]

            y = torch.tensor([d.y.item() for d in tr])
            mean, std = y.mean().item(), y.std().item()
            std = std if std > 1e-6 else 1.0
            for d in tr + va + te:
                d.y = (d.y - mean) / std
            self.cache[key] = (tr, va, te, mean, std)

        tr, va, te, self.target_mean, self.target_std = self.cache[key]
        mk = lambda d, sh: DataLoader(d, batch_size=self.batch_size, shuffle=sh, drop_last=False)
        return mk(tr, True), mk(va, False), mk(te, False)

    def record_results(self, fold_index: int, predictions: np.ndarray):
        """
        Records the predictions back to the Matbench task for evaluation.
        
        Args:
            fold_index (int): The fold index.
            predictions (np.ndarray): The predicted targets (must be un-standardized).
        """
        self.task.record(fold_index, predictions)
        logger.info(f"Recorded results for fold {fold_index}.")
