import os
import random
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, r2_score as sk_r2_score
from sklearn.model_selection import train_test_split
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, DataCollatorWithPadding

from models.EMA import EMA
from models.mic_model import MyDataset, MyModel


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def compute_additional_metrics(y_true, y_pred):
    # 转为 numpy
    y_true = y_true.flatten().cpu().numpy()
    y_pred = y_pred.flatten().cpu().numpy()
    mae = mean_absolute_error(y_true, y_pred)
    pearson = np.corrcoef(y_true, y_pred)[0, 1]
    r2 = sk_r2_score(y_true, y_pred)

    return mae, pearson, r2


def configure_optimizer(model, learning_rate, encoder_lr, weight_decay=1e-2):
    # 区分模型的不同部分
    esm_params = model.esm_model.parameters()
    head_params = [p for n, p in model.named_parameters() if not n.startswith('esm_model')]

    # 为不同部分设置不同的学习率
    param_groups = [
        {'params': esm_params, 'lr': encoder_lr},
        {'params': head_params, 'lr': learning_rate}
    ]

    optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
    return optimizer


def get_config():
    return {
        'seed': 2025,
        'batch_size': 256,
        'max_length': 40,
        'embedding_size': 480,
        'num_epochs': 1000,
        'learning_rate': 5e-4,
        'esm_learning_rate': 1e-5,
        'weight_decay': 1e-2,
        'gradient_clip_norm': 1.0,  # 梯度裁剪
        'early_stop_patience': 15,  # 早停
        'model_path': '../scoures/ESM2-35M',
        'genome_embedding_size': 84,
        'hidden_size': 128,
        'num_last_layers_to_pool': 6
    }


if __name__ == '__main__':

    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = f"../save_model/{run_timestamp}"
    os.makedirs(save_dir, exist_ok=True)  # 使用 os.makedirs

    config = get_config()
    set_random_seed(config['seed'])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 大肠杆菌 (EC)
    EC_train = pd.read_csv('../data/EC_X_train_40.csv')
    EC_test = pd.read_csv('../data/EC_X_test_40.csv')
    EC_val = pd.read_csv('../data/EC_X_val_40.csv')

    # 金黄色葡萄球菌 (SA)
    SA_train = pd.read_csv('../data/SA_X_train_40.csv')
    SA_test = pd.read_csv('../data/SA_X_test_40.csv')
    SA_val = pd.read_csv('../data/SA_X_val_40.csv')

    # 铜绿假单胞菌 (PA)
    PA_train = pd.read_csv('../data/PA_X_train_40.csv')
    PA_test = pd.read_csv('../data/PA_X_test_40.csv')
    PA_val = pd.read_csv('../data/PA_X_val_40.csv')

    train_df = pd.concat([EC_train, SA_train, PA_train, EC_test, SA_test, PA_test])
    val_df = pd.concat([EC_val, SA_val, PA_val])
    test_df = pd.concat([EC_test, SA_test, PA_test])

    # 加载模型 & 分词器
    tokenizer = AutoTokenizer.from_pretrained(config['model_path'], trust_remote_code=True)
    model = MyModel(
        esm_model_path=config['model_path'],
        genome_embedding_size=config['genome_embedding_size'],
        num_last_layers_to_pool=config['num_last_layers_to_pool'],
        hidden_size=config['hidden_size']).to(device)

    # 数据加载
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_loader = torch.utils.data.DataLoader(
        MyDataset(train_df, tokenizer, config['max_length']),
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=data_collator,
        num_workers=4,
        pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        MyDataset(val_df, tokenizer, config['max_length']),
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=data_collator,
        num_workers=4,
        pin_memory=True
    )

    # 优化器 & 调度
    optimizer = configure_optimizer(model, config["learning_rate"], config["esm_learning_rate"], config['weight_decay'])

    # EMA 和 AMP
    ema = EMA(model, 0.99)
    ema.register()
    scaler = GradScaler()

    # 早停变量
    best_val_loss = float('inf')
    epochs_no_improve = 0

    writer = SummaryWriter(log_dir=f"../logs/{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    for epoch in range(config['num_epochs']):
        model.train()
        train_losses = []
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config['num_epochs']} - Training"):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            genome = batch['genome'].to(device)
            target = batch['value'].to(device)
            with autocast():
                preds = model(input_ids, attention_mask, genome)
                loss = F.mse_loss(preds, target)
            scaler.scale(loss).backward()
            # 梯度裁剪
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip_norm'])
            scaler.step(optimizer)
            scaler.update()
            ema.update()
            train_losses.append(loss.item())

        # 训练指标
        avg_train_loss = np.mean(train_losses)
        writer.add_scalar('train/loss', avg_train_loss, epoch)

        # 验证
        model.eval()
        ema.apply_shadow()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{config['num_epochs']} - Validation"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                genome = batch['genome'].to(device)
                target = batch['value'].to(device)
                preds = model(input_ids, attention_mask, genome)
                val_preds.append(preds)
                val_targets.append(target)

        val_preds = torch.cat(val_preds)
        val_targets = torch.cat(val_targets)
        val_loss = F.mse_loss(val_preds, val_targets).item()
        mae, pearson, val_r2 = compute_additional_metrics(val_targets, val_preds)

        # Tensorboard logging
        writer.add_scalar('val/mse', val_loss, epoch)
        writer.add_scalar('val/mae', mae, epoch)
        writer.add_scalar('val/pearson', pearson, epoch)
        writer.add_scalar('val/r2', val_r2, epoch)

        print(
            f"Epoch {epoch + 1}: Train MSE={avg_train_loss:.4f}, Val MSE={val_loss:.4f}, MAE={mae:.4f}, Pearson={pearson:.4f}, R2={val_r2:.4f},early_stop_patience={epochs_no_improve}")

        # 早停检查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            # 保存最佳模型
            ema.apply_shadow()
            torch.save(model.state_dict(), f"{save_dir}/best_model.pth")
            # 保存数据
            val_df['pred'] = val_preds.cpu().numpy()
            val_df.to_csv(f"{save_dir}/predictions.csv", index=False)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= config['early_stop_patience']:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        ema.restore()

    writer.close()
