import torch
import torch.nn as nn


class Hist_Self_Attention_Embedding(nn.Module):

    def __init__(self, N, M, hid_units, dropout_p=0.3):
        super(Hist_Self_Attention_Embedding, self).__init__()
        self.W_0 = nn.Linear(M, hid_units)
        self.W_1 = nn.Linear(M, hid_units)

        self.dense = nn.Linear(N * N, hid_units)
        self.dropout = nn.Dropout(p=dropout_p)

        # 2 alternative activate functions, ReLU or Sigmoid.
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, X):
        v1 = self.W_0(X)
        v2 = self.W_1(X)
        h = torch.bmm(v1, v2.transpose(1, 2))
        v = torch.flatten(h, start_dim=1)
        v = self.dense(v)

        return self.dropout(self.relu(v))

class MEAN(nn.Module):

    def __repr__(self):
        return "MEAN, stable version. Last modified: 2023.4.6"

    def calculate_flatten_units(self, bins_num) -> int:
        ms = self.cnn_module  # shorten name
        out_units = bins_num
        for m in ms:
            if isinstance(m, nn.Conv1d):
                k, = m.kernel_size
                out_units = out_units - k + 1

        out_module = ms[-1]
        assert isinstance(out_module, nn.Conv1d), "The last layer of 'cnn_module' must be nn.Conv1d"
        assert out_module.out_channels == 1, f"The last Conv1d layer only need 1 out_channel for flatting, \
                                                                                but get {out_module.out_channels} "
        out_units = out_module.out_channels * out_units

        return out_units

    def __init__(self, proj_feat_shape, tables_num, join_pairs_num, hid_units: int = 512, dropout_p=0.2):
        super(MEAN, self).__init__()

        self.hid_units = hid_units
        (N, M) = proj_feat_shape  # N: Total num of columns; M: Total num of bins you set.

        self.ranges_layer = nn.Sequential(
            nn.Linear(3 * N, hid_units),
            nn.ReLU(),
            nn.Linear(hid_units, hid_units),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
        )

        self.proj_layer = Hist_Self_Attention_Embedding(N, M, hid_units, dropout_p)

        self.proj_fcn = nn.Sequential(
            nn.Flatten(),
            nn.Linear(N * M, hid_units),
            nn.ReLU(),
            nn.Linear(hid_units, hid_units),
            nn.ReLU(),
            nn.Dropout(p=dropout_p)
        )

        self.tables_layer = nn.Sequential(
            nn.Linear(tables_num * 2, hid_units),
            nn.ReLU(),
            nn.Linear(hid_units, hid_units),
            nn.ReLU(),
            nn.Dropout(p=dropout_p)
        )

        self.joins_layer = nn.Sequential(
            nn.Linear(join_pairs_num * 2, hid_units),
            nn.ReLU(),
            nn.Linear(hid_units, hid_units),
            nn.ReLU(),
            nn.Dropout(p=dropout_p)
        )

        self.hint_layer = nn.Sequential(
            nn.Linear(3, hid_units),
            nn.ReLU(),
            nn.Linear(hid_units, hid_units),
            nn.ReLU(),
            nn.Dropout(p=dropout_p)
        )

        self.acc_layer = nn.Sequential(
            nn.Linear(hid_units * 5, hid_units),
            nn.ReLU(),
            nn.Linear(hid_units, 1),
            nn.ReLU()
        )

    def forward(self, ranges_feat, proj_feat, tables_feat, joins_feat, hint, table_mask):
        ranges_hid = self.ranges_layer(ranges_feat)
        tables_hid = self.tables_layer(tables_feat)
        proj_hid = self.proj_fcn(proj_feat)
        joins_hid = self.joins_layer(joins_feat)
        hint_hid = self.hint_layer(hint)

        picked_table_num = (table_mask != 0).sum(dim=1)  # For `tales_feat`, we use 2 bits to represent a table.
        picked_table_num = picked_table_num.unsqueeze(1)
        picked_table_num = torch.reciprocal(picked_table_num)  # x = 1 / x

        ranges_hid = ranges_hid * picked_table_num
        tables_hid = tables_hid * picked_table_num
        proj_hid = proj_hid * picked_table_num
        joins_hid = joins_hid * picked_table_num
        hint_hid = hint_hid * picked_table_num

        hid = torch.cat((ranges_hid, tables_hid, joins_hid, proj_hid, hint_hid), dim=1)
        out = self.acc_layer(hid)

        return out

class MLP(nn.Module):
    def __init__(self, len, bit_size, hid_unit=128, p=0.):
        super(MLP, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(bit_size * len, hid_unit * 2),
            nn.ReLU(),
            nn.Linear(hid_unit * 2, hid_unit),
            nn.ReLU(),
            nn.Dropout(p=p)
        )

    def forward(self, ranges_feat):
        return self.layer(ranges_feat)
