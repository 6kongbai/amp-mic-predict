import sys
import os

# Add the project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import random
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, r2_score as sk_r2_score
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import T5EncoderModel, T5Tokenizer, AutoTokenizer, AutoModel

from models.EMA import EMA
from models.mic_model import MyModel, PrecomputedEmbeddingDataset


class EarlyStopperDDP:
    """
    用于分布式数据并行（DDP）的早停工具类。
    该版本只负责跟踪状态和发出信号，不执行I/O操作（如保存模型）。
    """

    def __init__(self, patience: int, local_rank: int):
        if patience < 1:
            raise ValueError("Patience 值必须至少为 1。")
        self.patience = patience
        self.local_rank = local_rank
        self.is_main_process = self.local_rank == 0
        self.best_loss = float('inf')
        self.epochs_no_improve = 0

    def __call__(self, val_loss: float):
        """
        执行早停逻辑，并返回更新后的状态。

        Returns:
            tuple: (should_stop, is_best, current_best_loss, patience_counter)
                   - should_stop (bool): 如果为True，则应停止训练。
                   - is_best (bool): 如果为True，表示当前周期的 val_loss 是历史最佳。
                   - current_best_loss (float): 当前记录的最佳损失。
                   - patience_counter (int): 当前的耐心计数器。
        """
        stop_signal = torch.tensor(0.0, device=self.local_rank)
        is_best = False

        if self.is_main_process:
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.epochs_no_improve = 0
                is_best = True
            else:
                self.epochs_no_improve += 1
                if self.epochs_no_improve >= self.patience:
                    stop_signal.fill_(1.0)

        # 广播停止信号
        dist.broadcast(stop_signal, src=0)

        # 广播最新状态
        status_tensor = torch.tensor([self.best_loss, self.epochs_no_improve], dtype=torch.float32,
                                     device=self.local_rank)
        dist.broadcast(status_tensor, src=0)

        # 广播 is_best 状态
        is_best_tensor = torch.tensor(1.0 if is_best else 0.0, device=self.local_rank)
        dist.broadcast(is_best_tensor, src=0)

        updated_best_loss = status_tensor[0].item()
        updated_patience_counter = int(status_tensor[1].item())
        should_stop = stop_signal.item() == 1.0
        is_best_on_all_procs = is_best_tensor.item() == 1.0

        return should_stop, is_best_on_all_procs, updated_best_loss, updated_patience_counter


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp():
    dist.destroy_process_group()


def compute_additional_metrics(y_true, y_pred):
    y_true = y_true.flatten().cpu().numpy()
    y_pred = y_pred.flatten().cpu().numpy()
    if len(np.unique(y_true)) < 2 or len(np.unique(y_pred)) < 2:
        pearson = float('nan')
    else:
        pearson = np.corrcoef(y_true, y_pred)[0, 1]
    mae = mean_absolute_error(y_true, y_pred)
    r2 = sk_r2_score(y_true, y_pred)
    return mae, pearson, r2


def get_config():
    return {
        'seed': 2025,
        'batch_size': 128,
        'max_length': 40,
        'num_epochs': 400,
        'learning_rate': 1e-3,
        'weight_decay': 1e-2,
        'gradient_clip_norm': 1.0,
        'early_stop_patience': 15,
        'model_path': '../scoures/ESM2-650M',
        'genome_embedding_size': 84,
        'hidden_size': 128,
        'lstm_hidden_layers': 2,
        'num_workers': 4
    }


def get_embedding_model_and_tokenizer(model_path):
    # tokenizer = T5Tokenizer.from_pretrained(model_path, trust_remote_code=True)
    # embedding_model = T5EncoderModel.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    embedding_model = AutoModel.from_pretrained(model_path)
    return embedding_model, tokenizer


def main():
    config = get_config()
    set_random_seed(config['seed'])

    local_rank = setup_ddp()
    is_main_process = local_rank == 0

    # --- 1. 初始化和路径设置 ---
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = f"../save_model/ESM2-650M"
    log_dir = f"../logs/ESM2-650M"

    writer = None
    if is_main_process:
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)
        print(f"使用 {dist.get_world_size()} 个GPU进行分布式训练。")
        print(f"模型将保存在: {save_dir}, 日志将保存在: {log_dir}")

    # --- 2. 数据加载和准备 ---
    EC_train = pd.read_csv('../data/EC_X_train_40.csv')
    EC_test = pd.read_csv('../data/EC_X_test_40.csv')
    EC_val = pd.read_csv('../data/EC_X_val_40.csv')
    SA_train = pd.read_csv('../data/SA_X_train_40.csv')
    SA_test = pd.read_csv('../data/SA_X_test_40.csv')
    SA_val = pd.read_csv('../data/SA_X_val_40.csv')
    PA_train = pd.read_csv('../data/PA_X_train_40.csv')
    PA_test = pd.read_csv('../data/PA_X_test_40.csv')
    PA_val = pd.read_csv('../data/PA_X_val_40.csv')
    train_df = pd.concat([EC_train, SA_train, PA_train, EC_val, SA_val, PA_val], ignore_index=True)
    val_df = pd.concat([EC_test, SA_test, PA_test], ignore_index=True)

    len_ec_val = len(EC_test)
    len_sa_val = len(SA_test)

    embedding_model, tokenizer = get_embedding_model_and_tokenizer(config['model_path'])
    embedding_model.to(local_rank)
    embedding_size = embedding_model.config.hidden_size

    train_dataset = PrecomputedEmbeddingDataset(train_df, tokenizer, embedding_model, local_rank, config['max_length'],
                                                config['batch_size'] * 4)
    val_dataset = PrecomputedEmbeddingDataset(val_df, tokenizer, embedding_model, local_rank, config['max_length'],
                                              config['batch_size'] * 4)
    del embedding_model
    torch.cuda.empty_cache()

    train_sampler = DistributedSampler(train_dataset)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], sampler=train_sampler,
                                               num_workers=config['num_workers'], pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config['batch_size'], sampler=val_sampler,
                                             num_workers=config['num_workers'], pin_memory=True)

    # --- 3. 模型、优化器、EMA和早停 ---
    model = MyModel(embedding_size=embedding_size, genome_embedding_size=config['genome_embedding_size'],
                    lstm_hidden_layers=config["lstm_hidden_layers"], hidden_size=config['hidden_size']).to(local_rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config["weight_decay"])
    ema = EMA(model.module, 0.99)
    ema.register()
    scaler = GradScaler()
    early_stopper = EarlyStopperDDP(patience=config['early_stop_patience'], local_rank=local_rank)

    # --- 4. 训练和验证循环 ---
    for epoch in range(config['num_epochs']):
        model.train()
        train_sampler.set_epoch(epoch)
        train_losses = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1} - Training", disable=not is_main_process)
        for batch in pbar:
            optimizer.zero_grad()
            embedding = batch['embedding'].to(local_rank)
            genome = batch['genome'].to(local_rank)
            target = batch['value'].to(local_rank)

            with autocast():
                preds = model(embedding, genome)
                loss = F.mse_loss(preds, target)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip_norm'])
            scaler.step(optimizer)
            scaler.update()
            ema.update()

            train_losses.append(loss.item())
            if is_main_process:
                pbar.set_postfix(loss=f"{loss.item():.4f}")

        # --- 验证循环 ---
        model.eval()
        ema.apply_shadow()

        val_preds, val_targets, val_indices = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                # 除了数据，还要获取原始索引
                indices = batch['index'].to(local_rank)
                embedding = batch['embedding'].to(local_rank)
                genome = batch['genome'].to(local_rank)
                target = batch['value'].to(local_rank)

                preds = model(embedding, genome)

                val_preds.append(preds)
                val_targets.append(target)
                val_indices.append(indices)

        # --- 聚合所有GPU的结果（包括索引）---
        val_preds = torch.cat(val_preds)
        val_targets = torch.cat(val_targets)
        val_indices = torch.cat(val_indices)

        world_size = dist.get_world_size()
        all_preds_list = [torch.zeros_like(val_preds) for _ in range(world_size)]
        all_targets_list = [torch.zeros_like(val_targets) for _ in range(world_size)]
        all_indices_list = [torch.zeros_like(val_indices, dtype=torch.long) for _ in range(world_size)]

        dist.all_gather(all_preds_list, val_preds)
        dist.all_gather(all_targets_list, val_targets)
        dist.all_gather(all_indices_list, val_indices)  # 聚合索引

        # --- 主进程中聚合、排序、评估和保存 ---
        global_preds, global_targets, val_loss = None, None, 0.0
        if is_main_process:
            # 1. 拼接所有GPU的结果，得到乱序数据
            unordered_preds = torch.cat(all_preds_list, dim=0)[:len(val_dataset)]
            unordered_targets = torch.cat(all_targets_list, dim=0)[:len(val_dataset)]
            global_indices = torch.cat(all_indices_list, dim=0)[:len(val_dataset)]

            # 2. 根据原始索引，对结果进行重新排序，恢复原始顺序
            sorter = global_indices.argsort()
            global_preds = unordered_preds[sorter]
            global_targets = unordered_targets[sorter]

            # 3. 计算验证损失
            val_loss = F.mse_loss(global_preds, global_targets).item()

        # 广播验证损失到所有进程
        val_loss_tensor = torch.tensor(val_loss, device=local_rank)
        dist.broadcast(val_loss_tensor, src=0)

        # 执行早停逻辑
        should_stop, is_best, current_best_loss, patience_counter = early_stopper(val_loss_tensor.item())

        if is_main_process:
            # 如果是最佳模型，则保存模型和排序后的预测结果
            if is_best:
                print(f"    -> New best model! Saving to {save_dir}")
                torch.save(model.module.state_dict(), f"{save_dir}/best_model.pth")

                results_df = pd.DataFrame({
                    'sequence': val_df['SEQUENCE'].tolist(),
                    'true_value': global_targets.cpu().numpy().flatten(),
                    'predicted_value': global_preds.cpu().numpy().flatten()
                })
                results_df.to_csv(f"{save_dir}/best_predictions.csv", index=False)

            # 记录和打印指标
            avg_train_loss = np.mean(train_losses)
            writer.add_scalar('train/loss', avg_train_loss, epoch)
            writer.add_scalar('val/MSE_Overall', val_loss, epoch)

            # 分物种计算指标
            data_splits = {
                'EC': (global_targets[:len_ec_val], global_preds[:len_ec_val]),
                'SA': (global_targets[len_ec_val:len_ec_val + len_sa_val],
                       global_preds[len_ec_val:len_ec_val + len_sa_val]),
                'PA': (global_targets[len_ec_val + len_sa_val:], global_preds[len_ec_val + len_sa_val:])
            }
            print(f"Epoch {epoch + 1}/{config['num_epochs']}: Train MSE={avg_train_loss:.4f}, Val MSE={val_loss:.4f}")
            for name, (targets, preds) in data_splits.items():
                if len(targets) > 1:
                    mse = F.mse_loss(preds, targets).item()
                    mae, pearson, r2 = compute_additional_metrics(targets, preds)
                    writer.add_scalar(f'val_metrics/{name}_mse', mse, epoch)
                    print(f"    [{name}] MSE: {mse:.4f}, R2: {r2:.4f}, Pearson: {pearson:.4f}")

            print(
                f"    --> Best Val MSE: {current_best_loss:.4f}, Patience: {patience_counter}/{config['early_stop_patience']}")

        # 恢复模型权重以进行下一轮训练
        ema.restore()

        if should_stop:
            if is_main_process:
                print(f"\n早停触发，训练在 Epoch {epoch + 1} 结束。")
            break

    # --- 5. 清理 ---
    if is_main_process:
        writer.close()
    cleanup_ddp()


if __name__ == '__main__':
    # 使用 torchrun 启动:
    # torchrun --nproc_per_node=2 MIC_trainer.py
    main()
