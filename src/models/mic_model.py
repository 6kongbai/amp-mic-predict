import numpy as np
import torch
from torch import nn
from transformers import AutoModel
from .kan import KAN


class WeightedLayerPooling(nn.Module):
    def __init__(self, num_hidden_layers, layer_start: int = 4):
        super().__init__()
        # 我们通常只对最后几层做加权，而不是全部层
        # layer_start表示我们从倒数第几层开始用
        self.layer_start = layer_start
        self.num_layers_to_pool = num_hidden_layers - layer_start + 1

        # 定义可学习的权重，初始化为1
        # 这样在训练开始时，它等价于一个简单的平均
        self.layer_weights = nn.Parameter(
            torch.ones(self.num_layers_to_pool)
        )

    def forward(self, all_hidden_states):
        # 1. 选取我们需要的层
        # all_hidden_states 是一个 tuple, 我们把它转成 tensor
        # 我们要的是模型最后几层，所以从后面开始取
        layers_to_pool = torch.stack(
            all_hidden_states[-self.num_layers_to_pool:],
            dim=0
        )  # Shape: (num_layers, batch, seq_len, hidden_dim)

        # 2. 计算权重
        # 为了让权重更稳定且具有可解释性，通常会用softmax
        # 这样所有权重加起来等于1，代表了贡献度的百分比
        weight_softmax = torch.softmax(self.layer_weights, dim=0)

        # 3. 调整权重形状以进行广播 (broadcast)
        # 从 (num_layers,) 变成 (num_layers, 1, 1, 1)
        # 这样它就可以和 (num_layers, batch, seq_len, hidden_dim) 的层输出相乘
        reshaped_weights = weight_softmax.view(-1, 1, 1, 1)

        # 4. 执行加权求和
        # 两个张量相乘，然后在一个维度上求和
        weighted_average = torch.sum(layers_to_pool * reshaped_weights, dim=0)

        return weighted_average  # Shape: (batch, seq_len, hidden_dim)


class CrossAttentionBlock(nn.Module):
    """
    一个完整的交叉注意力模块，包含残差连接和FFN
    """

    def __init__(self, embed_dim, num_heads, ffn_dim_multiplier=4, dropout=0.1):
        super().__init__()

        # 交叉注意力层
        self.cross_attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)

        # 前馈网络 (Feed-Forward Network)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * ffn_dim_multiplier),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * ffn_dim_multiplier, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, query, key, value):
        """
        Args:
            query: ESM序列特征, shape: (batch_size, seq_len_q, embed_dim)
            key: Genome特征, shape: (batch_size, seq_len_kv, embed_dim)
            value: Genome特征, shape: (batch_size, seq_len_kv, embed_dim)
        """
        # 1. 交叉注意力 + 残差连接
        # query 先做归一化
        query_norm = self.norm1(query)

        # MHA的输入是(query, key, value)
        attn_output, _ = self.cross_attn(
            query=query_norm,
            key=key,
            value=value,
        )

        # 残差连接：将融合了genome信息的attn_output加回到原始的query上
        query = query + self.dropout1(attn_output)

        # 2. FFN + 残差连接
        query_norm = self.norm2(query)
        ffn_output = self.ffn(query_norm)

        # 第二个残差连接
        output = query + self.dropout2(ffn_output)

        return output


class MyModel(nn.Module):
    def __init__(self, esm_model_path, genome_embedding_size, num_last_layers_to_pool=4, hidden_size=256):
        super().__init__()
        self.hidden_size = hidden_size
        self.esm_model = AutoModel.from_pretrained(
            esm_model_path,
            trust_remote_code=True,
            output_hidden_states=True
        )

        esm_config = self.esm_model.config
        embedding_size = esm_config.hidden_size

        self.weighted_pooling = WeightedLayerPooling(
            num_hidden_layers=esm_config.num_hidden_layers,
            layer_start=esm_config.num_hidden_layers - num_last_layers_to_pool
        )

        self.genome_feature_extractor = nn.Sequential(
            nn.LayerNorm(genome_embedding_size),
            # nn.Linear(genome_embedding_size, genome_embedding_size * 2),
            KAN((genome_embedding_size, genome_embedding_size * 2)),
            nn.GELU(),
            nn.Dropout(0.2),
            KAN((genome_embedding_size * 2, genome_embedding_size)),
            # nn.Linear(genome_embedding_size * 2, genome_embedding_size)
        )
        # 如果genome_embedding_size和ESM的embedding_size不同，需要一个线性层对齐
        self.genome_proj = nn.Linear(genome_embedding_size,
                                     embedding_size) if genome_embedding_size != embedding_size else nn.Identity()

        self.cross_attention_fusion = CrossAttentionBlock(
            embed_dim=embedding_size,
            num_heads=8
        )

        self.mlp = nn.Sequential(
            nn.Linear(embedding_size * 2, self.hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.Linear(self.hidden_size, 1)
        )

        self.apply(self._init_weights)

    def forward(self, input_ids, attention_mask, genome):
        # A. 通过ESM模型，获取所有层的输出
        esm_outputs = self.esm_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        all_hidden_states = esm_outputs.hidden_states

        # B. 使用层权重融合，得到一个增强的ESM序列表示
        esm_output_fused = self.weighted_pooling(all_hidden_states)

        # C. 处理Genome特征
        genome_features = self.genome_feature_extractor(genome)
        genome_features = self.genome_proj(genome_features)

        # D. 执行交叉注意力
        fused_output = self.cross_attention_fusion(
            query=esm_output_fused,
            key=genome_features.unsqueeze(dim=1),
            value=genome_features.unsqueeze(dim=1),
        )

        # E. 池化和回归
        avg_pool = fused_output.mean(dim=1)
        max_pool = fused_output.max(dim=1).values
        pooled_output = torch.cat([avg_pool, max_pool], dim=1)
        print(pooled_output.shape)
        output = self.mlp(pooled_output)
        return output

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, max_length=40):
        self.sequences = df['SEQUENCE'].values.tolist()
        self.values = df['NEW-CONCENTRATION'].values.tolist()
        sample = tokenizer(self.sequences, truncation=True, padding=True, max_length=max_length + 2,
                           return_tensors='np')
        self.input_ids = sample['input_ids']
        self.attention_mask = sample['attention_mask']
        self.genome = df.iloc[:, 250:-12].values.tolist()

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return {
            # 'sequence': self.sequences[idx],
            'input_ids': self.input_ids[idx],
            'genome': np.array(self.genome[idx], dtype=np.float32),
            'attention_mask': self.attention_mask[idx],
            'value': np.array([self.values[idx]], dtype=np.float32),
        }
