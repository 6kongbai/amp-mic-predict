import os
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
from transformers import T5EncoderModel, T5Tokenizer

from models.EMA import EMA
from models.mic_model import MyModel, PrecomputedEmbeddingDataset


class EarlyStopperDDP:
    """
    用于分布式数据并行（DDP）的早停工具类。
    该版本会返回更新后的状态，由调用者负责打印。
    """

    def __init__(self, patience: int, local_rank: int):
        if patience < 1:
            raise ValueError("Patience 值必须至少为 1。")
        self.patience = patience
        self.local_rank = local_rank
        self.is_main_process = self.local_rank == 0
        self.best_loss = float('inf')
        self.epochs_no_improve = 0

    def __call__(self, val_loss: float, epoch: int, model: torch.nn.Module, save_dir: str, val_df, global_preds):
        """
        执行早停逻辑，并返回更新后的状态。

        Returns:
            tuple: (should_stop, self.best_loss, self.epochs_no_improve)
                   - should_stop (bool): 是否应该停止训练。
                   - self.best_loss (float): 更新后的最佳损失。
                   - self.epochs_no_improve (int): 更新后的耐心计数。
        """
        stop_signal = torch.tensor(0.0, device=self.local_rank)

        if self.is_main_process:
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.epochs_no_improve = 0
                torch.save(model.module.state_dict(), f"{save_dir}/best_model.pth")
                if global_preds is not None:
                    val_df['pred'] = global_preds.cpu().numpy()
                    val_df.to_csv(f"{save_dir}/predictions.csv", index=False)
            else:
                self.epochs_no_improve += 1
                if self.epochs_no_improve >= self.patience:
                    stop_signal.fill_(1.0)

        # 广播停止信号，确保所有进程同步决定是否停止
        dist.broadcast(stop_signal, src=0)

        # 将主进程的最新状态广播给所有进程（虽然只有主进程会用它们来打印）
        # 这一步能确保返回值的统一性，但主要目的是为了主进程获取最新状态
        status_tensor = torch.tensor([self.best_loss, self.epochs_no_improve], dtype=torch.float32,
                                     device=self.local_rank)
        dist.broadcast(status_tensor, src=0)

        updated_best_loss = status_tensor[0].item()
        updated_patience_counter = int(status_tensor[1].item())

        should_stop = stop_signal.item() == 1.0

        return should_stop, updated_best_loss, updated_patience_counter


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
    mae = mean_absolute_error(y_true, y_pred)
    pearson = np.corrcoef(y_true, y_pred)[0, 1]
    r2 = sk_r2_score(y_true, y_pred)
    return mae, pearson, r2


def get_config():
    return {
        'seed': 2025,
        'batch_size': 128,
        'max_length': 40,
        'num_epochs': 400,
        'learning_rate': 5e-4,
        'weight_decay': 1e-2,
        'gradient_clip_norm': 1.0,
        'early_stop_patience': 20,
        'model_path': '../scoures/Port-T5',
        'genome_embedding_size': 84,
        'hidden_size': 128,
        'lstm_hidden_layers': 2
    }


def main():
    config = get_config()
    set_random_seed(config['seed'])

    local_rank = setup_ddp()
    is_main_process = local_rank == 0

    save_dir = f"../save_model/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = None
    if is_main_process:
        os.makedirs(save_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=f"../logs/{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        print(f"使用 {dist.get_world_size()} 个GPU进行分布式训练。")
        print(f"全局批次大小 (Global batch size): {config['batch_size'] * dist.get_world_size()}")

    # --- 数据加载 ---
    EC_train = pd.read_csv('../data/EC_X_train_40.csv')
    EC_test = pd.read_csv('../data/EC_X_test_40.csv')
    EC_val = pd.read_csv('../data/EC_X_val_40.csv')
    SA_train = pd.read_csv('../data/SA_X_train_40.csv')
    SA_test = pd.read_csv('../data/SA_X_test_40.csv')
    SA_val = pd.read_csv('../data/SA_X_val_40.csv')
    PA_train = pd.read_csv('../data/PA_X_train_40.csv')
    PA_test = pd.read_csv('../data/PA_X_test_40.csv')
    PA_val = pd.read_csv('../data/PA_X_val_40.csv')
    train_df = pd.concat([EC_train, SA_train, PA_train, EC_test, SA_test, PA_test])
    val_df = pd.concat([EC_val, SA_val, PA_val])
    tokenizer = T5Tokenizer.from_pretrained(config['model_path'], trust_remote_code=True)
    embedding_model = T5EncoderModel.from_pretrained(config['model_path']).to(local_rank)

    t5_embedding_size = embedding_model.config.hidden_size

    train_dataset = PrecomputedEmbeddingDataset(train_df, tokenizer, embedding_model, local_rank, config['max_length'],
                                                config['batch_size'])
    val_dataset = PrecomputedEmbeddingDataset(val_df, tokenizer, embedding_model, local_rank, config['max_length'],
                                              config['batch_size'])

    del embedding_model
    torch.cuda.empty_cache()

    train_sampler = DistributedSampler(train_dataset)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config['batch_size'], sampler=train_sampler, num_workers=4, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=config['batch_size'], sampler=val_sampler, num_workers=4, pin_memory=True)

    # --- 模型、优化器、EMA  ---
    model = MyModel(
        embedding_size=t5_embedding_size,
        genome_embedding_size=config['genome_embedding_size'],
        lstm_hidden_layers=config["lstm_hidden_layers"],
        hidden_size=config['hidden_size']).to(local_rank)

    def flatten_hook(module, input):
        module.flatten_parameters()

    model.bi_lstm.register_forward_pre_hook(flatten_hook)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    optimizer = torch.optim.AdamW(model.parameters(), weight_decay=config["weight_decay"])
    ema = EMA(model.module, 0.99)
    ema.register()
    scaler = GradScaler()
    # --- 初始化早停工具 ---
    early_stopper = EarlyStopperDDP(patience=config['early_stop_patience'], local_rank=local_rank)

    for epoch in range(config['num_epochs']):
        model.train()
        train_sampler.set_epoch(epoch)
        train_losses = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1} - Training", disable=not is_main_process)
        for batch in pbar:
            # --- 训练逻辑---
            optimizer.zero_grad()
            features = batch['embedding'].to(local_rank)
            genome = batch['genome'].to(local_rank)
            target = batch['value'].to(local_rank)
            with autocast():
                preds = model(features, genome)
                loss = F.mse_loss(preds, target)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip_norm'])
            scaler.step(optimizer)
            scaler.update()
            ema.update()
            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)
        if is_main_process:
            writer.add_scalar('train/loss', avg_train_loss, epoch)

        # --- 验证循环 ---
        model.eval()
        ema.apply_shadow()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                features = batch['embedding'].to(local_rank)
                genome = batch['genome'].to(local_rank)
                target = batch['value'].to(local_rank)
                preds = model(features, genome)
                val_preds.append(preds)
                val_targets.append(target)

        # --- 聚合所有GPU的结果  ---
        val_preds = torch.cat(val_preds)
        val_targets = torch.cat(val_targets)
        world_size = dist.get_world_size()
        all_preds_list = [torch.zeros_like(val_preds) for _ in range(world_size)]
        all_targets_list = [torch.zeros_like(val_targets) for _ in range(world_size)]
        dist.all_gather(all_preds_list, val_preds)
        dist.all_gather(all_targets_list, val_targets)

        # 准备数据，仅主进程有完整数据
        global_preds = None
        val_loss = 0.0
        if is_main_process:
            global_preds = torch.cat(all_preds_list, dim=0)[:len(val_dataset)]
            global_targets = torch.cat(all_targets_list, dim=0)[:len(val_dataset)]
            val_loss = F.mse_loss(global_preds, global_targets).item()

        # 广播val_loss，确保所有进程的stopper调用一致
        val_loss_tensor = torch.tensor(val_loss, device=local_rank)
        dist.broadcast(val_loss_tensor, src=0)

        # 调用stopper，接收其返回的最新状态
        should_stop, current_best_loss, patience_counter = early_stopper(
            val_loss=val_loss_tensor.item(),
            epoch=epoch,
            model=model,
            save_dir=save_dir,
            val_df=val_df,
            global_preds=global_preds
        )

        # 主进程使用最新状态，一次性打印所有信息
        if is_main_process:
            mae, pearson, val_r2 = compute_additional_metrics(global_targets, global_preds)
            writer.add_scalar('val/mse', val_loss, epoch)
            writer.add_scalar('val/mae', mae, epoch)
            writer.add_scalar('val/pearson', pearson, epoch)
            writer.add_scalar('val/r2', val_r2, epoch)

            print(
                f"Epoch {epoch + 1}/{config['num_epochs']}: Train MSE={avg_train_loss:.4f}, Val MSE={val_loss:.4f}, MAE={mae:.4f}, "
                f"Pearson={pearson:.4f}, R2={val_r2:.4f} | "
                f"Best Loss: {current_best_loss:.4f}, Patience: {patience_counter}/{config['early_stop_patience']}"
            )

        # 所有进程根据返回的信号，同步判断是否退出
        if should_stop:
            if is_main_process:
                print(f"早停触发，训练在 Epoch {epoch + 1} 结束。")
            break

        ema.restore()

    if is_main_process:
        writer.close()
    cleanup_ddp()


if __name__ == '__main__':
    # torchrun --nproc_per_node=2 MIC_trainer.py
    main()
