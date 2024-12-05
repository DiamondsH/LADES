import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence


class LayerNormLSTMCell(nn.LSTMCell):
    def __init__(self, input_size, hidden_size, dropout=0.0, bias=True, use_layer_norm=True):
        super().__init__(input_size, hidden_size, bias)
        self.use_layer_norm = use_layer_norm
        if self.use_layer_norm:
            self.ln_ih = nn.LayerNorm(4 * hidden_size)
            self.ln_hh = nn.LayerNorm(4 * hidden_size)
            self.ln_ho = nn.LayerNorm(hidden_size)
        # DropConnect on the recurrent hidden to hidden weight
        self.dropout = dropout

    def forward(self, input, hidden=None):
        self.check_forward_input(input)
        if hidden is None:
            hx = input.new_zeros(input.size(0), self.hidden_size, requires_grad=False)
            cx = input.new_zeros(input.size(0), self.hidden_size, requires_grad=False)
        else:
            hx, cx = hidden
        self.check_forward_hidden(input, hx, '[0]')
        self.check_forward_hidden(input, cx, '[1]')

        weight_hh = nn.functional.dropout(self.weight_hh, p=self.dropout, training=self.training)
        if self.use_layer_norm:
            gates = self.ln_ih(F.linear(input, self.weight_ih, self.bias_ih)) \
                    + self.ln_hh(F.linear(hx, weight_hh, self.bias_hh))
        else:
            gates = F.linear(input, self.weight_ih, self.bias_ih) \
                    + F.linear(hx, weight_hh, self.bias_hh)

        i, f, c, o = gates.chunk(4, 1)
        i_ = torch.sigmoid(i)
        f_ = torch.sigmoid(f)
        c_ = torch.tanh(c)
        o_ = torch.sigmoid(o)
        cy = (f_ * cx) + (i_ * c_)
        if self.use_layer_norm:
            hy = o_ * self.ln_ho(torch.tanh(cy))
        else:
            hy = o_ * torch.tanh(cy)
        return hy, cy

class LayerNormLSTM(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers=1,
                 dropout=0.0,
                 weight_dropout=0.0,
                 bias=True,
                 bidirectional=False,
                 use_layer_norm=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # using variational dropout
        self.dropout = dropout
        self.bidirectional = bidirectional

        num_directions = 2 if bidirectional else 1
        self.hidden0 = nn.ModuleList([
            LayerNormLSTMCell(input_size=(input_size if layer == 0 else hidden_size * num_directions),
                              hidden_size=hidden_size, dropout=weight_dropout, bias=bias, use_layer_norm=use_layer_norm)
            for layer in range(num_layers)
        ])

        if self.bidirectional:
            self.hidden1 = nn.ModuleList([
                LayerNormLSTMCell(input_size=(input_size if layer == 0 else hidden_size * num_directions),
                                  hidden_size=hidden_size, dropout=weight_dropout, bias=bias,
                                  use_layer_norm=use_layer_norm)
                for layer in range(num_layers)
            ])

    def copy_parameters(self, rnn_old):
        for param in rnn_old.named_parameters():
            name_ = param[0].split("_")
            layer = int(name_[2].replace("l", ""))
            sub_name = "_".join(name_[:2])
            if len(name_) > 3:
                self.hidden1[layer].register_parameter(sub_name, param[1])
            else:
                self.hidden0[layer].register_parameter(sub_name, param[1])

    def forward(self, input, hidden=None, seq_lens=None):
        seq_len, batch_size, _ = input.size()
        num_directions = 2 if self.bidirectional else 1
        if hidden is None:
            hx = input.new_zeros(self.num_layers * num_directions, batch_size, self.hidden_size, requires_grad=False)
            cx = input.new_zeros(self.num_layers * num_directions, batch_size, self.hidden_size, requires_grad=False)
        else:
            hx, cx = hidden

        ht = []
        for i in range(seq_len):
            ht.append([None] * (self.num_layers * num_directions))
        ct = []
        for i in range(seq_len):
            ct.append([None] * (self.num_layers * num_directions))

        seq_len_mask = input.new_ones(batch_size, seq_len, self.hidden_size, requires_grad=False)
        if seq_lens != None:
            for i, l in enumerate(seq_lens):
                seq_len_mask[i, l:, :] = 0
        seq_len_mask = seq_len_mask.transpose(0, 1)

        if self.bidirectional:
            # if use cuda, change 'torch.LongTensor' to 'torch.cuda.LongTensor'
            indices_ = (torch.LongTensor(seq_lens) - 1).unsqueeze(1).unsqueeze(0).unsqueeze(0).repeat(
                [1, 1, 1, self.hidden_size])
            # if use cuda, change 'torch.LongTensor' to 'torch.cuda.LongTensor'
            indices_reverse = torch.LongTensor([0] * batch_size).unsqueeze(1).unsqueeze(0).unsqueeze(0).repeat(
                [1, 1, 1, self.hidden_size])
            indices = torch.cat((indices_, indices_reverse), dim=1)
            hy = []
            cy = []
            xs = input
            # Variational Dropout
            if not self.training or self.dropout == 0:
                dropout_mask = input.new_ones(self.num_layers, 2, batch_size, self.hidden_size)
            else:
                dropout_mask = input.new(self.num_layers, 2, batch_size, self.hidden_size).bernoulli_(1 - self.dropout)
                dropout_mask = Variable(dropout_mask, requires_grad=False) / (1 - self.dropout)

            for l, (layer0, layer1) in enumerate(zip(self.hidden0, self.hidden1)):
                l0, l1 = 2 * l, 2 * l + 1
                h0, c0, h1, c1 = hx[l0], cx[l0], hx[l1], cx[l1]
                for t, (x0, x1) in enumerate(zip(xs, reversed(xs))):
                    ht_, ct_ = layer0(x0, (h0, c0))
                    ht[t][l0] = ht_ * seq_len_mask[t]
                    ct[t][l0] = ct_ * seq_len_mask[t]
                    h0, c0 = ht[t][l0], ct[t][l0]
                    t = seq_len - 1 - t
                    ht_, ct_ = layer1(x1, (h1, c1))
                    ht[t][l1] = ht_ * seq_len_mask[t]
                    ct[t][l1] = ct_ * seq_len_mask[t]
                    h1, c1 = ht[t][l1], ct[t][l1]

                xs = [torch.cat((h[l0] * dropout_mask[l][0], h[l1] * dropout_mask[l][1]), dim=1) for h in ht]
                ht_temp = torch.stack([torch.stack([h[l0], h[l1]]) for h in ht])
                ct_temp = torch.stack([torch.stack([c[l0], c[l1]]) for c in ct])
                if len(hy) == 0:
                    hy = torch.stack(list(ht_temp.gather(dim=0, index=indices).squeeze(0)))
                else:
                    hy = torch.cat((hy, torch.stack(list(ht_temp.gather(dim=0, index=indices).squeeze(0)))), dim=0)
                if len(cy) == 0:
                    cy = torch.stack(list(ct_temp.gather(dim=0, index=indices).squeeze(0)))
                else:
                    cy = torch.cat((cy, torch.stack(list(ct_temp.gather(dim=0, index=indices).squeeze(0)))), dim=0)
            y = torch.stack(xs)
        else:
            # if use cuda, change 'torch.LongTensor' to 'torch.cuda.LongTensor'
            indices = (torch.cuda.LongTensor(seq_lens) - 1).unsqueeze(1).unsqueeze(0).unsqueeze(0).repeat(
                [1, self.num_layers, 1, self.hidden_size])
            h, c = hx, cx
            # Variational Dropout
            if not self.training or self.dropout == 0:
                dropout_mask = input.new_ones(self.num_layers, batch_size, self.hidden_size)
            else:
                dropout_mask = input.new(self.num_layers, batch_size, self.hidden_size).bernoulli_(1 - self.dropout)
                dropout_mask = Variable(dropout_mask, requires_grad=False) / (1 - self.dropout)

            for t, x in enumerate(input):
                for l, layer in enumerate(self.hidden0):
                    ht_, ct_ = layer(x, (h[l], c[l]))
                    ht[t][l] = ht_ * seq_len_mask[t]
                    ct[t][l] = ct_ * seq_len_mask[t]
                    x = ht[t][l] * dropout_mask[l]
                ht[t] = torch.stack(ht[t])
                ct[t] = torch.stack(ct[t])
                h, c = ht[t], ct[t]
            y = torch.stack([h[-1] * dropout_mask[-1] for h in ht])
            hy = torch.stack(list(torch.stack(ht).gather(dim=0, index=indices).squeeze(0)))
            cy = torch.stack(list(torch.stack(ct).gather(dim=0, index=indices).squeeze(0)))

        return y, (hy, cy)

class RNNModel(nn.Module):
    def __init__(self, sent_hidden_size, prop_hidden_size, num_layers, dropout_prob):
        super(RNNModel, self).__init__()
        self.num_layers = num_layers
        self.prop_hidden_size = prop_hidden_size
        self.sent_hidden_size = sent_hidden_size
        self.prop_cell = nn.RNN(self.sent_hidden_size,
                          self.prop_hidden_size,
                          num_layers=num_layers,
                          batch_first=True,
                          dropout=dropout_prob
                        )

    def forward(self, seq_tensor_list, h0=None):
        pad_seq_tensors = pad_sequence(seq_tensor_list, batch_first=True)
        packed_seq_tensors = pack_padded_sequence(pad_seq_tensors, [len(seq) for seq in seq_tensor_list],
                                                  batch_first=True, enforce_sorted=False)
        if h0 is None:
            df_outputs, df_last_state = self.prop_cell(packed_seq_tensors)
        else:
            df_outputs, df_last_state = self.prop_cell(packed_seq_tensors, h0)
        return df_last_state[-1, :, :]

class GRUModel(nn.Module):
    def __init__(self, sent_hidden_size, prop_hidden_size, num_layers, dropout_prob):
        super(GRUModel, self).__init__()
        self.num_layers = num_layers
        self.prop_hidden_size = prop_hidden_size
        self.sent_hidden_size = sent_hidden_size
        self.prop_cell = nn.GRU(sent_hidden_size,
                                self.prop_hidden_size,
                                num_layers=num_layers,
                                batch_first=True,
                                dropout=dropout_prob
                                )

    def forward(self, seq_tensor_list, init_states=None, output_hidden_states=False):
        pad_seq_tensors = pad_sequence(seq_tensor_list, batch_first=True)
        packed_seq_tensors = pack_padded_sequence(pad_seq_tensors, [len(seq) for seq in seq_tensor_list], batch_first=True, enforce_sorted=False)
        if init_states is None:
            df_outputs, df_last_state = self.prop_cell(packed_seq_tensors)
        else:
            df_outputs, df_last_state = self.prop_cell(packed_seq_tensors, init_states)

        if output_hidden_states:
            return df_outputs, df_last_state[-1, :, :]
        else:
            return df_last_state[-1, :, :]

class LSTMModel(nn.Module):
    def __init__(self, sent_hidden_size, prop_hidden_size, num_layers, dropout_prob):
        super(LSTMModel, self).__init__()
        self.num_layers = num_layers
        self.prop_hidden_size = prop_hidden_size
        self.sent_hidden_size = sent_hidden_size
        self.prop_cell = nn.LSTM(self.sent_hidden_size,
                                 self.prop_hidden_size,
                                 num_layers=num_layers,
                                 batch_first=True,
                                 dropout=dropout_prob
                                )

    def forward(self, seq_tensor_list, init_states=None, output_hidden_states=False):
        pad_seq_tensors = pad_sequence(seq_tensor_list, batch_first=True)
        packed_seq_tensors = pack_padded_sequence(pad_seq_tensors, [len(seq) for seq in seq_tensor_list],
                                                  batch_first=True, enforce_sorted=False)
        if init_states is None:
            df_outputs, df_last_state = self.prop_cell(packed_seq_tensors)
        else:
            df_outputs, df_last_state = self.prop_cell(packed_seq_tensors, init_states[0], init_states[1])

        if output_hidden_states:
            return df_outputs, df_last_state
        else:
            return df_last_state[0][-1, :, :] # df_last_state:(h_t, c_t)

class CAMICNN(nn.Module):
    def __init__(self, sent_hidden_size,dropout_prob=0.8,
                 k_max1=10, k_max2= 5,
                 kernel_num1=6, kernel_num2=4,
                 kernel_width1=7, kernel_width2=5,
                  ):
        super(CAMICNN, self).__init__()
        self.sent_hidden_size = sent_hidden_size
        self.dropout_prob = dropout_prob
        self.k_max1 = k_max1
        self.k_max2 = k_max2
        self.prop_hidden_size = self.k_max2*kernel_num2
        self.CNN_1 = nn.Conv1d(1, kernel_num1, (kernel_width1, self.sent_hidden_size),
                                    padding_mode="zeros", bias=False)
        self.CNN_2 = nn.Conv1d(kernel_num1, kernel_num2, (kernel_width2, self.k_max),
                                    padding_mode="zeros", bias=False)


    def forward(self, seq_tensor_list, output_hidden_states=False):
        pad_tensors = pad_sequence(seq_tensor_list, batch_first=True)
        cnn_input = pad_tensors.unsqueeze(1)
        cnn1_out = self.CNN_1(cnn_input)
        cnn2_input = cnn1_out.squeeze(-1).topk(self.k_max1, dim=3)[0]
        cnn2_out = self.CNN_2(cnn2_input)
        k_max_out = cnn2_out.squeeze(-1).topk(self.k_max2, dim=3)[0]
        output = k_max_out.reshape([len(k_max_out), -1])
        if output_hidden_states:
            return output, cnn2_input
        else:
            return output
