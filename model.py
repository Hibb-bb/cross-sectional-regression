import torch
import torch.nn as nn

class Baseline(nn.Modile):
    def __init__(self, inp_dim, hid_dim, out_dim=1, task_size=5):
        super().__init__()

        self.share_emb = nn.Linear(inp_dim, hid_dim)
        self.non_share_emb = nn.Embedding(task_size, hid_dim)
        self.act = nn.ReLU()

        self.fc = nn.Linear(hid_dim+hid_dim, out_dim)

    def forward(self, x, x_tilde, task_idx):
        
        # x: (task_num, inp_dim)
        emb_x1 = self.act(self.share_emb(x))
        emb_x2 = torch.stack([torch.matmul(self.non_share_emb(i), x_tilde[i]) for i in range(x.size(0))], dim=0)
        emb_x2 = self.act(emb_x2)

        hid_x = torch.cat((emb_x1, emb_x2), dim=-1)
        return self.fc(hid_x)

class HyperNetwork(nn.Modile):
    def __init__(self, inp_dim, hid_dim, out_dim=1, emb_dim=10):
        super().__init__()

        self.share_emb = nn.Linear(inp_dim, hid_dim)
        self.hyper_w = nn.Linear(emb_dim + 1, inp_dim)
        self.hyper_b = nn.Linear(emb_dim + 1, inp_dim)

        self.act = nn.ReLU()

        self.fc = nn.Linear(hid_dim+hid_dim, out_dim)

    def forward(self, x, x_tilde, ent_emb):
        
        # x: (task_num, inp_dim)
        emb_x1 = self.act(self.share_emb(x))

        hyp_w = self.hyper_w(ent_emb)
        hyp_b = self.hyper_b(ent_emb)

        hid_x_tilde = self.act(hyp_w @ x_tilde + hyp_b)
        hid_x = torch.cat((emb_x1, hid_x_tilde), dim=-1)

        return self.fc(hid_x)