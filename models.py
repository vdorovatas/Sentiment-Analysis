import torch
import numpy as np
from torch import nn



class BaselineDNN(nn.Module):
    """
    1. We embed the words in the input texts using an embedding layer
    2. We compute the min, mean, max of the word embeddings in each sample
       and use it as the feature representation of the sequence.
    4. We project with a linear layer the representation
       to the number of classes.ngth)
    """

    def __init__(self, output_size, embeddings, trainable_emb=False):
        """

        Args:
            output_size(int): the number of classes
            embeddings(bool):  the 2D matrix with the pretrained embeddings
            trainable_emb(bool): train (finetune) or freeze the weights
                the embedding layer
        """

        super(BaselineDNN, self).__init__()

        # 1 - define the embedding layer
        # 2 - initialize the weights of our Embedding layer from the pretrained word embeddings
        # 3 - define if the embedding layer will be frozen or finetuned
        # EX4

        self.embeddings = nn.Embedding(len(embeddings) , len(embeddings[0]))
        self.output_size = output_size

        if not trainable_emb:
            self.embeddings = nn.Embedding.from_pretrained(torch.tensor(embeddings), freeze=True)

        # 4 - define a non-linear transformation of the representations
        # EX5

        self.linear1 = nn.Linear(len(embeddings[1]), 70)
        self.ReLU = nn.ReLU()

        # 5 - define the final Linear layer which maps
        # the representations to the classes
        # EX5
        self.linear2 = nn.Linear(70, self.output_size)
        #self.Softmax = nn.Softmax(dim = 1)

    def forward(self, x, lengths):
        """
        This is the heart of the model.
        This function, defines how the data passes through the network.

        Returns: the logits for each class

        """

        # 1 - embed the words, using the embedding layer
        embeddings = self.embeddings(x) # EX6

        # 2 - construct a sentence representation out of the word embeddings
        #for e in embeddings: #getting the embeddings of each doc
  
        representations = torch.sum(embeddings, dim=1) # EX6
        for i in range(lengths.shape[0]): # necessary to skip zeros in mean calculation
            representations[i] = representations[i] / lengths[i]

        # 3 - transform the representations to new ones.
        representations = self.ReLU(self.linear1(representations))  # EX6

        # 4 - project the representations to classes using a linear layer
        logits = self.linear2(representations)  # EX6

        return logits


class LSTM(nn.Module):
    def __init__(self, output_size, embeddings, trainable_emb=False, bidirectional=False):

        super(LSTM, self).__init__()
        self.hidden_size = 100
        self.num_layers = 1
        self.bidirectional = bidirectional

        self.representation_size = 2 * \
            self.hidden_size if self.bidirectional else self.hidden_size

        embeddings = np.array(embeddings)
        num_embeddings, dim = embeddings.shape

        self.embeddings = nn.Embedding(num_embeddings, dim)
        self.output_size = output_size

        self.lstm = nn.LSTM(dim, hidden_size=self.hidden_size,
                            num_layers=self.num_layers, bidirectional=self.bidirectional)

        if not trainable_emb:
            self.embeddings = self.embeddings.from_pretrained(
                torch.Tensor(embeddings), freeze=True)

        self.linear = nn.Linear(self.representation_size, output_size)

    def forward(self, x, lengths):
        batch_size, max_length = x.shape
        embeddings = self.embeddings(x)
        X = torch.nn.utils.rnn.pack_padded_sequence(
            embeddings, lengths, batch_first=True, enforce_sorted=False)

        ht, _ = self.lstm(X)

        # ht is batch_size x max(lengths) x hidden_dim
        ht, _ = torch.nn.utils.rnn.pad_packed_sequence(ht, batch_first=True)

        # pick the output of the lstm corresponding to the last word
        # TODO: Main-Lab-Q2 (Hint: take actual lengths into consideration)
        representations = torch.zeros(batch_size, self.representation_size).float()
        for i in range(lengths.shape[0]):
            last = lengths[i] - 1 if lengths[i] <= max_length else max_length - 1
            representations[i] = ht[i, last, :]
       

        logits = self.linear(representations)

        return logits
