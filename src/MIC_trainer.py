import os
import random
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.distributed
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, r2_score as sk_r2_score
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import T5EncoderModel, T5Tokenizer

from models.EMA import EMA
from models.mic_model import MyDataset, MyModel


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def setup_ddp():
    torch.distributed.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp():
    torch.distributed.destroy_process_group()


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
        'early_stop_patience': 15,
        'model_path': '../scoures/Port-T5',
        'genome_embedding_size': 84,
        'hidden_size': 256,
        'lstm_hidden_layers': 2
    }


def main():
    config = get_config()
    set_random_seed(config['seed'])

    local_rank = setup_ddp()
    is_main_process = local_rank == 0

    # --- 目录和日志记录 (仅主进程) ---
    save_dir = f"../save_model/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = None
    if is_main_process:
        os.makedirs(save_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=f"../logs/{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        print(f"Distributed training with {torch.distributed.get_world_size()} GPUs.")
        # batch_size per GPU * num_gpus
        print(f"Global batch size: {config['batch_size'] * torch.distributed.get_world_size()}")

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

    train_dataset = MyDataset(train_df, tokenizer, config['max_length'])
    val_dataset = MyDataset(val_df, tokenizer, config['max_length'])

    train_sampler = DistributedSampler(train_dataset)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config['batch_size'], sampler=train_sampler, num_workers=4, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=config['batch_size'], sampler=val_sampler, num_workers=4, pin_memory=True)

    # --- 模型、优化器、EMA ---
    embedding_model = T5EncoderModel.from_pretrained(config['model_path']).to(local_rank)
    model = MyModel(
        embedding_size=embedding_model.config.hidden_size,
        genome_embedding_size=config['genome_embedding_size'],
        lstm_hidden_layers=config["lstm_hidden_layers"],
        hidden_size=config['hidden_size']).to(local_rank)
    def flatten_hook(module, input):
        module.flatten_parameters()
    model.bi_lstm.register_forward_pre_hook(flatten_hook)
    model.to(local_rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    optimizer = torch.optim.AdamW(model.parameters(), weight_decay=config["weight_decay"])
    ema = EMA(model.module, 0.99)
    ema.register()
    scaler = GradScaler()

    best_val_loss = float('inf')
    epochs_no_improve = 0

    # --- 训练循环 ---
    for epoch in range(config['num_epochs']):
        model.train()
        train_sampler.set_epoch(epoch)  # 确保shuffle在每个epoch都不同

        train_losses = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1} - Training", disable=not is_main_process)
        for batch in pbar:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(local_rank)
            attention_mask = batch['attention_mask'].to(local_rank)
            genome = batch['genome'].to(local_rank)
            target = batch['value'].to(local_rank)
            with autocast():
                with torch.no_grad():
                    features = embedding_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
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
                input_ids = batch['input_ids'].to(local_rank)
                attention_mask = batch['attention_mask'].to(local_rank)
                genome = batch['genome'].to(local_rank)
                target = batch['value'].to(local_rank)
                features = embedding_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
                preds = model(features, genome)
                val_preds.append(preds)
                val_targets.append(target)

        # --- 聚合所有GPU的结果 ---
        val_preds = torch.cat(val_preds)
        val_targets = torch.cat(val_targets)

        world_size = torch.distributed.get_world_size()
        all_preds_list = [torch.zeros_like(val_preds) for _ in range(world_size)]
        all_targets_list = [torch.zeros_like(val_targets) for _ in range(world_size)]

        torch.distributed.all_gather(all_preds_list, val_preds)
        torch.distributed.all_gather(all_targets_list, val_targets)

        # --- 只有主进程进行指标计算和保存 ---
        if is_main_process:
            global_preds = torch.cat(all_preds_list, dim=0)[:len(val_dataset)]
            global_targets = torch.cat(all_targets_list, dim=0)[:len(val_dataset)]

            val_loss = F.mse_loss(global_preds, global_targets).item()
            mae, pearson, val_r2 = compute_additional_metrics(global_targets, global_preds)

            writer.add_scalar('val/mse', val_loss, epoch)
            writer.add_scalar('val/mae', mae, epoch)
            writer.add_scalar('val/pearson', pearson, epoch)
            writer.add_scalar('val/r2', val_r2, epoch)

            print(
                f"Epoch {epoch + 1}: Train MSE={avg_train_loss:.4f}, Val MSE={val_loss:.4f}, MAE={mae:.4f}, "
                f"Pearson={pearson:.4f}, R2={val_r2:.4f}, early_stop_patience={epochs_no_improve}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                ema.apply_shadow()
                # 保存时，要保存 model.module 以获取原始模型
                torch.save(model.module.state_dict(), f"{save_dir}/best_model.pth")
                val_df['pred'] = global_preds.cpu().numpy()
                val_df.to_csv(f"{save_dir}/predictions.csv", index=False)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= config['early_stop_patience']:
                    print(f"Early stopping at epoch {epoch + 1}")
                    # 需要通知其他进程也退出
                    torch.distributed.barrier()
                    break

        # 同步所有进程，确保主进程的break能被所有进程感知
        torch.distributed.barrier()
        if epochs_no_improve >= config['early_stop_patience']:
            break

        ema.restore()

    if is_main_process:
        writer.close()
    cleanup_ddp()


if __name__ == '__main__':
    main()