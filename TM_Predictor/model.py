import torch
import torch.nn as nn
import torch.nn.functional as F
from .util import dot_product

torch.backends.cudnn.enabled = False

class plmsearch(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super(plmsearch, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=1280, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.gru = nn.GRU(input_size=512, hidden_size=512, num_layers=2, batch_first=True)

        self.fc = nn.Linear(2, 1)

    def forward(self, input1, input2):
        input1 = input1.unsqueeze(0)
        input2 = input2.unsqueeze(0)
        input1 = input1.permute(0, 2, 1)
        input2 = input2.permute(0, 2, 1)
        input1 = self.cnn(input1)
        input2 = self.cnn(input2)
        input1 = input1.permute(2, 0, 1)
        input2 = input2.permute(2, 0, 1)

        gru_out1, _ = self.gru(input1)  # [seq_len, batch, num_directions * hidden_size]
        output1 = gru_out1.squeeze(1)
        gru_out2, _ = self.gru(input2)  # [seq_len, batch, num_directions * hidden_size]
        output2 = gru_out2.squeeze(1)

        cos_sim = F.cosine_similarity(output1, output2, dim=1)
        cos_sim = cos_sim.unsqueeze(1)
        manhattan_dist = torch.sum(torch.abs(output1 - output2), dim=1)
        manhattan_dist = manhattan_dist.unsqueeze(1)
        result = torch.stack((cos_sim, manhattan_dist), dim=1)
        result = torch.sigmoid(self.fc(result.squeeze(2)))
        return result

    def pairwise_predict(self, z1, z2):
        z = self.forward(z1, z2)
        return z

    def load_pretrained(self, mtplm_path):
        state_dict = torch.load(mtplm_path, map_location=torch.device('cpu'))
        self.load_state_dict(state_dict)

    def predict(self, z1, z2):
        z1 = self.forward(z1)
        return dot_product(z1, z2)