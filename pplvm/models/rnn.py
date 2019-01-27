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
        """

        Args:
            input_size (int): dimensionality of marks (or embeddings if
                                       using VAE observations)
            hidden_size (int): dimensionality of RNN hidden state
            layers (int): number of RNN layers
            out_size (int): dimensionality of output (continuous state X)
        """
        super(RNNInduce, self).__init__()
        self.hidden_size = hidden_size
        self.out_size = out_size
        rnn_size = input_size

        self.rnn = torch.nn.LSTM(rnn_size, hidden_size, layers,
                                batch_first=True,
                                bidirectional=True).double()

        self.fc_weight = nn.Parameter(torch.randn(size=(hidden_size * 2, out_size * 2),
                                                  dtype=torch.float64,
                                                  requires_grad=True))

    def forward(self, Y, dT, induce_idx):
        """
        computes mean and variance at inducing points
        Args:
            Y (torch.tensor): observed marks
            dT (torch.tensor): observed intervals
            induce_idx (torch.tensor): indices of inducing points

        Returns:

        """

        x = torch.cat((Y, dT.unsqueeze(1)), dim=1).unsqueeze(0)
        out = self.rnn(x)[0].squeeze()
        out = torch.mm(out, self.fc_weight)

        return out[induce_idx, :self.out_size].view(len(induce_idx),
                                                         self.out_size),\
               out[induce_idx, self.out_size:].view(len(induce_idx),
                                                         self.out_size)