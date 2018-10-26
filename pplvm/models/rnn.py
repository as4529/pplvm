import torch

class RNNInduce(torch.nn.Module):
    """
    Bidirectional RNN for variational model with inducing points
    """
    def __init__(self,
                 input_size,
                 hidden_size,
                 layers,
                 out_size):
        super(RNNInduce, self).__init__()
        self.hidden_size = hidden_size
        self.out_size = out_size
        rnn_size = input_size

        self.rnn = torch.nn.LSTM(rnn_size, hidden_size, layers,
                                batch_first=True,
                                bidirectional=True).double()

        self.fc_weight = torch.randn(size=(hidden_size * 2, out_size * 2),
                                     dtype=torch.float64,
                                     requires_grad=True)

    def forward(self, Y, dT, induce_idx, switch):
        if switch is not None:
            x = torch.cat((Y, dT.unsqueeze(1), switch.unsqueeze(1)), dim=1).unsqueeze(0)
        else:
            x = torch.cat((Y, dT.unsqueeze(1)), dim=1).unsqueeze(0)
        out = self.rnn(x)[0].squeeze()
        out = torch.mm(out, self.fc_weight)
        return out[induce_idx, :self.out_size].view(len(induce_idx),
                                                         self.out_size),\
               out[induce_idx, self.out_size:].view(len(induce_idx),
                                                         self.out_size)