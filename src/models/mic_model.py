import re

import torch
from torch import nn
from tqdm import tqdm
from .kan import KAN
from src.utils.sequenceEmbedding import get_vector, seq_to_embedding


class FeatureExtractor(nn.Module):
    def __init__(self, embedding_size):
        super().__init__()
        self.norm = nn.LayerNorm(embedding_size)
        self.linear1 = KAN((embedding_size, embedding_size * 2))
        self.act = TeLU()
        self.dropout = nn.Dropout(0.2)
        self.linear2 = KAN((embedding_size * 2, embedding_size))

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

        self.mlp = nn.Sequential(
            nn.Linear(hidden_size * 4 + genome_embedding_size, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1)
        )

        self.apply(self._init_weights)

    def forward(self, x, genome):
        x_norm_input = self.x_before_norm(x)
        x_feature = self.bi_lstm(x_norm_input)[0]

        genome_features = self.genome_feature_extractor(genome)

        avg_pool = x_feature.mean(dim=1)
        max_pool = x_feature.max(dim=1).values
        pooled_output = torch.cat([max_pool, avg_pool, genome_features], dim=1)
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
        sequences = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in sequences]

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
