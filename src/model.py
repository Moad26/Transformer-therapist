import math
import torch
import torch.nn as nn
from typing import Optional


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
):
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        # Ensure mask is on the same device as the scores
        mask = mask.to(scores.device)
        scores.masked_fill_(mask == 0, float("-inf"))
    attention_weights = torch.softmax(scores, dim=-1)
    output = torch.matmul(attention_weights, V)
    return output, attention_weights


class SelfAttention(nn.Module):
    def __init__(
        self, embed_dim: int, mask: Optional[torch.Tensor] = None, dropout: float = 0.2
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.mask = mask
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.out_linear = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        Q = self.q_linear(x)
        K = self.k_linear(x)
        V = self.v_linear(x)

        attention_outputs, _ = scaled_dot_product_attention(Q, K, V, self.mask)
        output = self.out_linear(attention_outputs)
        return self.dropout(output)


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        num_head: int,
        embed_dim: int,
        mask: Optional[torch.Tensor] = None,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_head = num_head
        self.list_of_self_att = nn.ModuleList()
        self.final_out_linear = nn.Linear(embed_dim * num_head, embed_dim)
        for _ in range(num_head):
            cha = SelfAttention(embed_dim=embed_dim, mask=mask)
            self.list_of_self_att.append(cha)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        outs_list = list()
        for head in range(self.num_head):
            outs_list.append(self.list_of_self_att[head](x))
        out_inter = torch.cat(outs_list, dim=-1)
        out = self.final_out_linear(out_inter)
        return self.dropout(out)


class CrossAttentionHead(nn.Module):
    def __init__(self, embed_dim: int, dropout: float = 0.2):
        super().__init__()
        self.embed_dim = embed_dim
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.out_linear = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, decoder_x: torch.Tensor, encoder_out: torch.Tensor):
        Q = self.q_linear(decoder_x)
        K = self.k_linear(encoder_out)
        V = self.v_linear(encoder_out)

        attention_output, _ = scaled_dot_product_attention(Q, K, V)
        output = self.out_linear(attention_output)
        return self.dropout(output)


class CrossAttention(nn.Module):
    def __init__(self, embed_dim: int, num_head: int, dropout: float = 0.2):
        super().__init__()
        self.num_head = num_head
        self.list_of_cross_att = nn.ModuleList()
        self.final_out_linear = nn.Linear(embed_dim * num_head, embed_dim)
        for _ in range(num_head):
            cross_att = CrossAttentionHead(embed_dim=embed_dim)
            self.list_of_cross_att.append(cross_att)
        self.dropout = nn.Dropout(dropout)

    def forward(self, decoder_x: torch.Tensor, encoder_out: torch.Tensor):
        outs_list = list()
        for head in range(self.num_head):
            outs_list.append(self.list_of_cross_att[head](decoder_x, encoder_out))
        out_inter = torch.cat(outs_list, dim=-1)
        out = self.final_out_linear(out_inter)
        return self.dropout(out)


class PositionalEncoding(nn.Module):
    pe: torch.Tensor  # Type annotation for the buffer

    def __init__(self, embed_dim: int, max_seq_len: int):
        super().__init__()
        self.embed_dim = embed_dim

        pe = torch.zeros(max_seq_len, embed_dim)
        positions = torch.arange(0, max_seq_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim)
        )

        pe[:, 0::2] = torch.sin(positions * div_term)
        pe[:, 1::2] = torch.cos(positions * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        seq_len = x.size(1)
        x = x * math.sqrt(self.embed_dim)
        x += self.pe[:, :seq_len]
        return x


class Position_FFN(nn.Module):
    def __init__(self, embed_dim: int, dropout: float = 0.2):
        super().__init__()
        self.in_out_dim = embed_dim
        self.hidden_dim = 4 * embed_dim
        self.fc1 = nn.Linear(self.in_out_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.in_out_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return self.dropout(x)


class Encoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_head: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.multi_head_att = MultiHeadAttention(num_head, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.pffn = Position_FFN(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        residual = x
        x = self.norm1(x)
        x = self.multi_head_att(x)
        x = self.dropout(x) + residual

        residual = x
        x = self.norm2(x)
        x = self.pffn(x)
        x = self.dropout(x) + residual

        return x


class Decoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        seq_len: int,
        num_head: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        # Register causal mask as a buffer so it automatically moves with the model
        causal_mask = torch.tril(torch.ones(seq_len, seq_len))
        self.register_buffer("causal_mask", causal_mask)

        self.cross_attention = CrossAttention(embed_dim, num_head)
        self.masked_multi_head_att = MultiHeadAttention(
            num_head, embed_dim, mask=None  # We'll pass the mask dynamically
        )

        self.add_norm1 = nn.LayerNorm(embed_dim)
        self.add_norm2 = nn.LayerNorm(embed_dim)
        self.add_norm3 = nn.LayerNorm(embed_dim)
        self.pffn = Position_FFN(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, encod_out: torch.Tensor):
        residual = x

        seq_len = x.size(1)
        mask = self.causal_mask[:seq_len, :seq_len]

        # Temporarily set the mask for each attention head
        for head in self.masked_multi_head_att.list_of_self_att:
            head.mask = mask

        x = self.masked_multi_head_att(x)
        x = self.add_norm1(x + residual)

        residual = x
        x = self.cross_attention(x, encod_out)
        x = self.add_norm2(x + residual)

        residual = x
        x = self.pffn(x)
        x = self.add_norm3(x + residual)

        return self.dropout(x)


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        seq_len: int,
        num_head: int,
        dropout: float = 0.1,
        num_layers: int = 6,
    ):
        super().__init__()
        self.src_embedding = nn.Embedding(vocab_size, embed_dim)
        self.tgt_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pe = PositionalEncoding(embed_dim, seq_len)
        self.encoder_layers = nn.ModuleList(
            [Encoder(embed_dim, num_head, dropout) for _ in range(num_layers)]
        )
        self.decoder_layers = nn.ModuleList(
            [Decoder(embed_dim, seq_len, num_head, dropout) for _ in range(num_layers)]
        )

        self.final_linear = nn.Linear(embed_dim, vocab_size)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor):
        # src is the input (patient utterance)
        # tgt is the target labels (therapist response) with -100 for padding

        src_embedded = self.pe(self.src_embedding(src))

        decoder_input = tgt.clone()
        decoder_input[decoder_input == -100] = 0

        decoder_input = decoder_input[:, :-1]

        # Make sure decoder_input doesn't have any invalid token IDs
        vocab_size = self.src_embedding.num_embeddings
        decoder_input = torch.clamp(decoder_input, 0, vocab_size - 1)

        tgt_embedded = self.pe(self.tgt_embedding(decoder_input))

        encoder_out = src_embedded
        for encoder_layer in self.encoder_layers:
            encoder_out = encoder_layer(encoder_out)

        decoder_out = tgt_embedded
        for decoder_layer in self.decoder_layers:
            decoder_out = decoder_layer(decoder_out, encoder_out)

        logits = self.final_linear(decoder_out)

        return logits, tgt[:, 1:]

    def generate(
        self, src: torch.Tensor, max_len: int, start_token: int, end_token: int
    ):
        self.eval()
        with torch.no_grad():
            src_embedded = self.pe(self.src_embedding(src))
            encoder_out = src_embedded
            for encoder_layer in self.encoder_layers:
                encoder_out = encoder_layer(encoder_out)

            tgt = torch.tensor([[start_token]], device=src.device)

            for _ in range(max_len):
                tgt_embedded = self.pe(self.tgt_embedding(tgt))

                decoder_out = tgt_embedded
                for decoder_layer in self.decoder_layers:
                    decoder_out = decoder_layer(decoder_out, encoder_out)

                logits = self.final_linear(decoder_out)
                next_token = torch.argmax(logits[:, -1:], dim=-1)

                tgt = torch.cat([tgt, next_token], dim=1)

                if next_token.item() == end_token:
                    break

            return tgt
