import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, input_size, encode_size):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(input_size, encode_size)
        self.decoder = Decoder(encode_size, input_size)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class Encoder(nn.Module):
    def __init__(self, input_size, encode_size):
        super(Encoder, self).__init__()
        self.linear1 = nn.Linear(input_size, 64)
        self.linear2 = nn.Linear(64, 128)
        self.linear3 = nn.Linear(128, 64)
        self.linear4 = nn.Linear(64, 32)
        self.linear5 = nn.Linear(32, 16)
        self.linear6 = nn.Linear(16, encode_size)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = torch.relu(self.linear3(x))
        x = torch.relu(self.linear4(x))
        x = torch.relu(self.linear5(x))
        x = self.linear6(x)
        return x


class Decoder(nn.Module):
    def __init__(self, input_size, decode_size):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(input_size, 16)
        self.linear2 = nn.Linear(16, 32)
        self.linear3 = nn.Linear(32, 64)
        self.linear4 = nn.Linear(64, 128)
        self.linear5 = nn.Linear(128, 64)
        self.linear6 = nn.Linear(64, decode_size)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = torch.relu(self.linear3(x))
        x = torch.relu(self.linear4(x))
        x = torch.relu(self.linear5(x))
        x = self.linear6(x)
        return x
