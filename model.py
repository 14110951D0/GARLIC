import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Includes components for imputation, attention-based encoding, message passing, and prediction.

    Args:
        input_dim (int): Number of input variables.
        feature_dim (int): Dimension of feature embeddings.
        hidden_size (int): Hidden size of GRU predictor.
        window_size (int): Temporal window for attention.
        length (int): Total length of input sequence.
        device (str): Device for model operations.
    """
    def __init__(self, input_dim=34, feature_dim=16, hidden_size=64, window_size=4, length=49, device='cuda:0'):
        super().__init__()
        self.device = device
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.window_size = window_size
        self.hidden_size = hidden_size
        self.length = length

        self.attention = TimeLagAttentionLayer(feature_dim, input_dim, window_size).to(device)
        self.positional_encoding = PositionalEncoding(feature_dim, window_size)
        self.message_passing = MessagePassingLayer(feature_dim, 1, input_dim).to(device)
        self.encoder = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2, feature_dim),
                nn.ReLU(),
                nn.Linear(feature_dim, feature_dim)
            ).to(device) for _ in range(input_dim)
        ])
        self.aggregator = SignalAttentionLayer(feature_dim, max_len=input_dim)
        self.predictor = nn.GRU(input_size=feature_dim, hidden_size=hidden_size, num_layers=2, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        self.t_attention = TemporalAttention(hidden_size, max_len=length - window_size + 1)
        self.w_input_decay = nn.Parameter(torch.Tensor(1, input_dim))
        self.b_input_decay = nn.Parameter(torch.Tensor(1, input_dim))
        nn.init.constant_(self.w_input_decay, 0.1)
        nn.init.zeros_(self.b_input_decay)

    def forward(self, arr, mask, time, expl=False):
        """
        Forward pass of the model.

        Args:
            arr (Tensor): Input time-series data [B, T, D].
            mask (Tensor): Binary mask indicating observed values [B, T, D].
            time (Tensor): Time stamps [B, T].
            expl (bool): Whether to compute explanation.
        Returns:
            tuple: (prediction, loss, variable_importance [optional])
        """
        # Latent Feature Modeling
        zero_delta_time = torch.Tensor([0.] * arr.size(0)).unsqueeze(1).to(self.device)
        temp_time = time.permute(1,0)
        delta_ts = (temp_time[:, 1:] - temp_time[:, :-1]).to(self.device)
        delta_ts = torch.cat([zero_delta_time, delta_ts], dim=1)
        arr = impute_using_input_decay(arr, delta_ts.unsqueeze(2), mask, self.w_input_decay, self.b_input_decay, self.device)
        encoded_arr = []
        for i in range(self.input_dim):
            temp = torch.cat([arr[:,:,i].view(-1,1),mask[:,:,i].view(-1,1)],dim=-1)
            encoded_arr.append(self.encoder[i](temp).reshape(mask.size(0), mask.size(1), -1).unsqueeze(-1))
        encoded_arr = torch.cat(encoded_arr, dim=-1)
        encoded_arr = encoded_arr.permute(0, 1, 3, 2)

        # Time-lagged Graph Message Passing
        B, S, N, d_feat = encoded_arr.size()
        win = self.window_size
        windows = encoded_arr.unfold(dimension=1, size=win, step=1)
        pe = self.positional_encoding()
        pe_expanded = pe.unsqueeze(2).expand(arr.size(0), self.window_size, self.input_dim,
                                             self.feature_dim)
        pe_expanded = pe_expanded.unsqueeze(1).repeat(1, windows.size(1), 1, 1, 1)
        windows = windows.permute(0, 1, 4, 2, 3) + pe_expanded
        B_new = B * (S - win + 1)
        windows_reshaped = windows.reshape(B_new, win, N, d_feat)
        att_out, t_attention = self.attention(windows_reshaped)
        mp_out, latent_out = self.message_passing(att_out)
        t_out = mp_out.reshape(B, S - win + 1, -1).permute(0,1,2)

        # Cross-Dimensional Sequential Attention
        l_out = latent_out.reshape(B, S - win + 1, N, d_feat).permute(0,2,3,1)
        g_attention = torch.softmax(self.message_passing.adj, dim=1) # graph
        v_out, v_attention = self.aggregator(l_out) # variable level attention
        output, hidden = self.predictor(v_out)
        output, f_attention = self.t_attention(output) # time level attention

        # with explanation
        if expl:
            t_attention = t_attention.mean(1).reshape(B, S - win + 1, self.window_size, N).permute(0, 1, 3, 2)
            v_attention = v_attention.mean(-2)
            f_attention = f_attention.mean(-2)
            time_lagged_variable_importance = torch.einsum("nm,nml->nml", f_attention, v_attention)
            temp = torch.einsum("nml,nmlk->nmlk", time_lagged_variable_importance, t_attention)
            temp = torch.einsum("nmjk,ji->nmik", temp, g_attention)

            variable_importance = torch.zeros_like(arr.squeeze(-1))
            for t in range(self.window_size - 1, time.size(0)):
                variable_importance[:, t - self.window_size + 1:t+1, :] += temp[:, t - self.window_size + 1, :, :].permute(0,2,1)

            masked_variable_importance = variable_importance*mask
            after_averaged_variable_importance = (variable_importance*(1-mask)).sum(dim=1)
            observed_variable_num = mask.sum(dim=1)
            epsilon = 1e-8
            avg_importance = after_averaged_variable_importance / (observed_variable_num + epsilon)
            avg_importance_expanded = avg_importance.unsqueeze(1).expand(-1, self.length, -1)
            bool_mask = mask.bool()
            expl_global = True # set to false for temporal pattern
            if expl_global == True:
                masked_variable_importance[bool_mask] += avg_importance_expanded[bool_mask]

        out = self.classifier(output).squeeze()
        ori = arr.squeeze(-1)[:, self.window_size - 1:, :][mask[:, self.window_size - 1:, :].bool()]
        pred = t_out[mask[:, self.window_size - 1:, :].bool()]
        criterion = nn.MSELoss(reduction='mean')
        loss = criterion(ori, pred) / torch.mean(mask)


        if expl:
            return out, loss, masked_variable_importance
        else:
            return out, loss, None


class PositionalEncoding(nn.Module):
    """
    Implements sinusoidal positional encoding for input sequences.

    Args:
        d_model (int): Feature dimension.
        max_len (int): Maximum sequence length.
    """
    def __init__(self, d_model, max_len=10):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self):
        return self.pe

class SignalAttentionLayer(nn.Module):
    """
    Applies self-attention over input signals with added positional encoding.

    Args:
        input_dim (int): Input feature dimension.
        max_len (int): Number of signals.
    """
    def __init__(self, input_dim, max_len=10):
        super().__init__()
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.position_encoding = PositionalEncoding(input_dim, max_len)

    def forward(self, x):
        pe = self.position_encoding()
        x = x + pe.unsqueeze(-1).expand(x.size(0), x.size(1), x.size(2), x.size(3))
        x = x.permute(0, 1, 3, 2)
        B, N, L, D = x.size()
        x = x.reshape(B * L, N, D)
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (D ** 0.5)
        weights = self.softmax(scores)
        values = torch.matmul(weights, V).sum(dim=1)
        weights = weights.view(B, L, N, N)
        values = values.view(B, L, -1)
        return values, weights

class TemporalAttention(nn.Module):
    """
    Temporal attention over sequence steps using self-attention with position encoding.

    Args:
        input_dim (int): Feature dimension.
        max_len (int): Maximum sequence length.
    """
    def __init__(self, input_dim, max_len=10):
        super().__init__()
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.position_encoding = PositionalEncoding(input_dim, max_len)

    def forward(self, x):
        pe = self.position_encoding()
        x = x + pe.expand(x.size(0), x.size(1), x.size(2))
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (x.size(-1) ** 0.5)
        weights = self.softmax(scores)
        values = torch.matmul(weights, V).sum(dim=1)
        return values, weights

class TimeLagAttentionLayer(nn.Module):
    """
    Time-lagged attention for multivariate series, separate attention per variable.

    Args:
        input_dim (int): Feature dimension.
        num_variables (int): Number of input variables.
        max_len (int): Temporal window length.
    """
    def __init__(self, input_dim, num_variables, max_len):
        super().__init__()
        self.query = nn.ModuleList([nn.Linear(input_dim, input_dim) for _ in range(num_variables)])
        self.key = nn.ModuleList([nn.Linear(input_dim, input_dim) for _ in range(num_variables)])
        self.value = nn.ModuleList([nn.Linear(input_dim, input_dim) for _ in range(num_variables)])
        self.position_encoding = PositionalEncoding(input_dim, max_len)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, L, V, D = x.size()
        weighted_values = torch.zeros(B, V, L, D).to(x.device)
        attention_weights = []
        for i in range(V):
            q = self.query[i](x[:, :, i, :])
            k = self.key[i](x[:, :, i, :])
            v = self.value[i](x[:, :, i, :])
            scores = torch.matmul(q, k.transpose(-2, -1)) / (D ** 0.5)
            weight = self.softmax(scores)
            attention_weights.append(weight)
            weighted_values[:, i, :, :] = torch.matmul(weight, v)
        out = weighted_values.sum(dim=2)
        attention_weights = torch.stack(attention_weights, dim=-1)
        return out, attention_weights

class MessagePassingLayer(nn.Module):
    """
    Graph-based message passing layer with learnable adjacency matrix.

    Args:
        input_dim (int): Input feature dimension.
        output_dim (int): Output feature dimension.
        num_nodes (int): Number of nodes (variables).
        alpha (float): Adjacency initialization factor.
    """
    def __init__(self, input_dim, output_dim, num_nodes, alpha=0.2):
        super().__init__()
        self.adj = nn.Parameter(torch.eye(num_nodes) + torch.ones(num_nodes, num_nodes) * alpha)
        self.message_fc = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, output_dim)
        )

    def forward(self, x):
        message = torch.matmul(self.adj, x)
        return self.message_fc(message), message

def get_cum_delta_ts(data, delta_ts, mask):
    n_traj, n_tp, n_dims = data.size()
    cum_delta_ts = delta_ts.repeat(1, 1, n_dims)
    missing_mask = 1 - mask
    cum_mask = torch.cumsum(missing_mask, dim=1)
    cum_delta_ts = cum_delta_ts * (1 + cum_mask)
    cum_delta_ts = cum_delta_ts / cum_delta_ts.max()
    return cum_delta_ts

def impute_using_input_decay(data, delta_ts, mask, w_input_decay, b_input_decay, device):
    n_traj, n_tp, n_dims = data.size()
    missing_mask = (mask == 0)
    cum_delta_ts = delta_ts.expand(-1, -1, n_dims).clone()
    data_last_obsv = data.clone()
    for t in range(1, n_tp):
        cum_delta_ts[:, t, :] += cum_delta_ts[:, t - 1, :] * missing_mask[:, t, :]
        data_last_obsv[:, t, :] = torch.where(
            missing_mask[:, t, :], data_last_obsv[:, t - 1, :], data_last_obsv[:, t, :]
        )
    cum_delta_ts = cum_delta_ts / cum_delta_ts.max()
    decay = torch.exp(-torch.clamp(w_input_decay * cum_delta_ts + b_input_decay, min=0, max=1000))
    data_means = torch.mean(data, dim=1, keepdim=True)
    return data * mask + (1 - mask) * (decay * data_last_obsv + (1 - decay) * data_means)