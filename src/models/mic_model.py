import re

import torch
from torch import nn
from tqdm import tqdm
from .kan import KAN
from src.utils.sequenceEmbedding import get_vector, seq_to_embedding


class SELayerForSequence(nn.Module):
    """
    专为序列数据设计的 Squeeze-and-Excitation (SE) block。
    它在特征维度（channels）上进行操作。

    输入张量形状：(batch, sequence_length, num_features)
    """

    def __init__(self, num_features, ratio=16):
        """
        初始化函数。

        参数:
            num_features (int): 输入特征的数量（即通道数 a.k.a. channels）。
            ratio (int): 用于控制第一个全连接层神经元数量的缩减率。
        """
        super(SELayerForSequence, self).__init__()
        # Squeeze 操作：使用自适应 1D 平均池化，将序列长度维度压缩为 1
        self.squeeze = nn.AdaptiveAvgPool1d(1)

        # Excitation 操作：两个全连接层
        self.excitation = nn.Sequential(
            # 第一个全连接层，将特征维度从 C 降低到 C/r
            nn.Linear(num_features, num_features // ratio, bias=False),
            nn.ReLU(),
            # 第二个全连接层，将特征维度恢复到 C
            nn.Linear(num_features // ratio, num_features, bias=False),
            nn.Sigmoid()  # 使用 Sigmoid 生成 0 到 1 之间的权重
        )

    def forward(self, x):
        """
        前向传播。

        参数:
            x (torch.Tensor): 输入张量，形状为 (batch, seq_len, features)

        返回:
            torch.Tensor: 经过 SE block 加权后的输出张量，形状与输入相同。
        """
        # 获取输入的形状
        batch_size, seq_len, num_features = x.shape

        # Squeeze 过程
        # 1. 调整维度以适应 1D 池化：(B, L, C) -> (B, C, L)
        squeezed_x = x.permute(0, 2, 1)
        # 2. 全局平均池化：(B, C, L) -> (B, C, 1)
        squeezed_x = self.squeeze(squeezed_x)
        # 3. 展平：(B, C, 1) -> (B, C)
        squeezed_x = squeezed_x.view(batch_size, num_features)

        # Excitation 过程，得到每个特征通道的权重
        # (B, C) -> (B, C)
        weights = self.excitation(squeezed_x)

        # Scale 过程 (Fscale)
        # 1. 调整权重维度以进行广播：(B, C) -> (B, 1, C)
        weights = weights.unsqueeze(1)

        # 2. 将权重与原始输入相乘，进行特征重标定
        #    (B, L, C) * (B, 1, C) -> (B, L, C)
        output = x * weights

        return output


class FeatureExtractor(nn.Module):
    def __init__(self, embedding_size):
        super().__init__()
        self.norm = nn.LayerNorm(embedding_size)
        # self.linear1 = KAN((embedding_size, embedding_size * 2))
        self.linear1 = nn.Linear(embedding_size, embedding_size * 2)
        self.act = TeLU()
        # self.act = nn.GELU()
        self.dropout = nn.Dropout(0.2)
        # self.linear2 = KAN((embedding_size * 2, embedding_size))
        self.linear2 = nn.Linear(embedding_size * 2, embedding_size)

    def forward(self, x):
        x_norm = self.norm(x)
        identity = x_norm  # 保存原始输入

        out = self.linear1(x_norm)
        out = self.act(out)
        out = self.dropout(out)
        out = self.linear2(out)

        out = out + identity
        return out


class TeLU(nn.Module):
    """
    TeLU 激活函数实现
    公式: TeLU(x) = x * tanh(e^x)
    """

    def __init__(self):
        super(TeLU, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.tanh(torch.exp(x))


class MyModel(nn.Module):
    def __init__(self, embedding_size, genome_embedding_size, lstm_hidden_layers=2, hidden_size=128):
        super().__init__()
        self.hidden_size = hidden_size

        self.x_before_norm = nn.LayerNorm(embedding_size)

        self.bi_lstm = nn.LSTM(embedding_size, hidden_size, lstm_hidden_layers,
                               bidirectional=True, batch_first=True, dropout=0.2)

        self.genome_feature_extractor = FeatureExtractor(genome_embedding_size)
        # self.se_layer = SELayerForSequence(num_features=hidden_size * 2, ratio=32)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size * 4 + genome_embedding_size, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1)
        )

        self.apply(self._init_weights)

    def forward(self, x, genome):
        x_norm_input = self.x_before_norm(x)
        x_feature = self.bi_lstm(x_norm_input)[0]
        # x_feature = self.se_layer(x_feature) + x_feature  # 添加残差连接
        genome_features = self.genome_feature_extractor(genome)


        avg_pool = x_feature.mean(dim=1)
        max_pool = x_feature.max(dim=1).values
        pooled_output = torch.cat([max_pool, genome_features, avg_pool], dim=1)
        output = self.mlp(pooled_output)
        return output

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        if self.bi_lstm:
            self.bi_lstm.register_forward_pre_hook(lambda module, input: module.flatten_parameters())


class PrecomputedEmbeddingDataset(torch.utils.data.Dataset):
    """
    Dataset optimizado que pre-calcula y almacena los embeddings de T5 para
    evitar re-cálculos en cada época de entrenamiento.
    """

    def __init__(self, df, tokenizer, embedding_model, device, max_length=40, computation_batch_size=64):
        self.genome = torch.tensor(df.iloc[:, 250:-12].values, dtype=torch.float32)
        self.values = torch.tensor(df['NEW-CONCENTRATION'].values, dtype=torch.float32).unsqueeze(1)
        # self.feature = torch.tensor(seq_to_embedding(df['SEQUENCE'].values.tolist(), get_vector(), max_length + 2),
        #                             dtype=torch.float32)
        embedding_model.eval()
        sequences = df['SEQUENCE'].values.tolist()
        # sequences = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in sequences]

        all_embeddings = []
        is_main_process = (isinstance(device, int) and device == 0) or (
                isinstance(device, torch.device) and device.index == 0)

        with torch.no_grad():
            pbar_range = range(0, len(sequences), computation_batch_size)
            pbar = tqdm(pbar_range, desc="Pre-calculando Embeddings") if is_main_process else pbar_range

            for i in pbar:
                batch_sequences = sequences[i:i + computation_batch_size]
                inputs = tokenizer(
                    batch_sequences,
                    truncation=True,
                    padding="max_length",
                    max_length=max_length + 2,
                    return_tensors='pt',
                    add_special_tokens=True
                ).to(device)

                embeddings = embedding_model(input_ids=inputs.input_ids,
                                             attention_mask=inputs.attention_mask).last_hidden_state
                all_embeddings.append(embeddings.cpu())

        self.embeddings = torch.cat(all_embeddings, dim=0)

        if is_main_process:
            print(f"Finalizado. Forma del tensor de embeddings: {self.embeddings.shape}")

    def __len__(self):
        return len(self.values)

    def __getitem__(self, idx):
        return {
            'embedding': self.embeddings[idx],
            'genome': self.genome[idx],
            'value': self.values[idx],
            'index': torch.tensor(idx, dtype=torch.long)
        }
