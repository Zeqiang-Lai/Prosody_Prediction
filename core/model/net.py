"""Defines the neural network, losss function and metrics"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self, params):
        super(Net, self).__init__()

        # the embedding takes as input the vocab_size and the embedding_dim
        weight = torch.tensor(np.load(params.embedding_path), dtype=torch.float)
        self.embedding = nn.Embedding.from_pretrained(weight)
        self.pos_embedding = nn.Embedding(params.number_of_pos, params.pos_embd_dim)

        # the LSTM takes as input the size of its input (embedding_dim), its hidden size
        # for more details on how to use it, check out the documentation
        self.lstm = nn.LSTM(params.embedding_dim + params.pos_embd_dim, params.lstm_hidden_dim, bidirectional=True,
                            batch_first=True)

        # the fully connected layer transforms the output to give the final output layer
        self.fc1 = nn.Linear(params.lstm_hidden_dim * 2, params.fc1)
        self.fc2 = nn.Linear(params.fc1, params.number_of_tags)

    def forward(self, s, pos):
        """
        This function defines how we use the components of our network to operate on an input batch.

        Args:
            s: (Variable) contains a batch of sentences, of dimension batch_size x seq_len, where seq_len is
               the length of the longest sentence in the batch. For sentences shorter than seq_len, the remaining
               tokens are PADding tokens. Each row is a sentence with each element corresponding to the index of
               the token in the vocab.

        Returns:
            out: (Variable) dimension batch_size*seq_len x num_tags with the log probabilities of tokens for each token
                 of each sentence.

        Note: the dimensions after each step are provided
        """
        #                                -> batch_size x seq_len
        # apply the embedding layer that maps each token to its embedding
        s = self.embedding(s)  # dim: batch_size x seq_len x embedding_dim
        p = self.pos_embedding(pos)

        s = torch.cat((s, p), dim=2)

        # run the LSTM along the sentences of length seq_len
        s, _ = self.lstm(s)  # dim: batch_size x seq_len x lstm_hidden_dim

        # make the Variable contiguous in memory (a PyTorch artefact)
        s = s.contiguous()

        # reshape the Variable so that each row contains one token
        s = s.view(-1, s.shape[2])  # dim: batch_size*seq_len x lstm_hidden_dim

        # apply the fully connected layer and obtain the output (before softmax) for each token
        s = self.fc1(s)  # dim: batch_size*seq_len x num_tags
        s = self.fc2(s)  # dim: batch_size*seq_len x num_tags

        # apply log softmax on each token's output (this is recommended over applying softmax
        # since it is numerically more stable)
        return F.log_softmax(s, dim=1)  # dim: batch_size*seq_len x num_tags


def loss_fn(outputs, labels):
    """
    Compute the cross entropy loss given outputs from the model and labels for all tokens. Exclude loss terms
    for PADding tokens.

    Args:
        outputs: (Variable) dimension batch_size*seq_len x num_tags - log softmax output of the model
        labels: (Variable) dimension batch_size x seq_len where each element is either a label in [0, 1, ... num_tag-1],
                or -1 in case it is a PADding token.

    Returns:
        loss: (Variable) cross entropy loss for all tokens in the batch

    Note: you may use a standard loss function from http://pytorch.org/docs/master/nn.html#loss-functions. This example
          demonstrates how you can easily define a custom loss function.
    """

    # reshape labels to give a flat vector of length batch_size*seq_len
    labels = labels.view(-1)

    # since PADding tokens have label -1, we can generate a mask to exclude the loss from those terms
    mask = (labels >= 0).float()

    # indexing with negative values is not supported. Since PADded tokens have label -1, we convert them to a positive
    # number. This does not affect training, since we ignore the PADded tokens with the mask.
    labels = labels % outputs.shape[1]

    num_tokens = int(torch.sum(mask).item())

    # compute cross entropy loss for all tokens (except PADding tokens), by multiplying with mask.
    return -torch.sum(outputs[range(outputs.shape[0]), labels] * mask) / num_tokens


def accuracy(outputs, labels):
    """
    Compute the accuracy, given the outputs and labels for all tokens. Exclude PADding terms.

    Args:
        outputs: (np.ndarray) dimension batch_size*seq_len x num_tags - log softmax output of the model
        labels: (np.ndarray) dimension batch_size x seq_len where each element is either a label in
                [0, 1, ... num_tag-1], or -1 in case it is a PADding token.

    Returns: (float) accuracy in [0,1]
    """

    # reshape labels to give a flat vector of length batch_size*seq_len
    labels = labels.ravel()

    # since PADding tokens have label -1, we can generate a mask to exclude the loss from those terms
    mask = (labels >= 0)

    # np.argmax gives us the class predicted for each token by the model
    outputs = np.argmax(outputs, axis=1)

    # compare outputs with labels and divide by number of tokens (excluding PADding tokens)
    return np.sum(outputs == labels) / float(np.sum(mask))


def block(output):
    idx = []
    s = 0
    for i in range(len(output)):
        if output[i] == 0:
            idx.append((s, i))
            s = i
    idx.append((s, len(output)))
    return idx


def block_acc(outputs, labels):
    outputs = np.argmax(outputs, axis=1)
    outputs = np.reshape(outputs, labels.shape)

    mask = labels >= 0
    lens = np.sum(mask, axis=1)
    correct = 0
    total = 0
    for i in range(len(outputs)):
        output = outputs[i, :lens[i]]
        label = labels[i, :lens[i]]
        output_idx = block(output)
        label_idx = block(label)
        for idx in output_idx:
            if idx in label_idx:
                correct += 1
        total += len(label_idx)
    return correct / total


def precision(outputs, labels):
    outputs = np.argmax(outputs, axis=1)
    labels = labels.ravel()
    outputs = outputs.ravel()

    t = 0
    for i in range(len(labels)):
        if outputs[i] == labels[i] == 0:
            t += 1

    return t / np.sum(outputs == 0)


def recall(outputs, labels):
    outputs = np.argmax(outputs, axis=1)
    labels = labels.ravel()
    outputs = outputs.ravel()

    t = 0
    total = 0
    for i in range(len(labels)):
        if outputs[i] == labels[i] == 0:
            t += 1
        if labels[i] == 0:
            total += 1
    if total == 0:
        return 1
    return t / total


# maintain all metrics required in this dictionary- these are used in the training and evaluation loops
metrics = {
    'accuracy': accuracy,
    'block_acc': block_acc,
    'precison': precision,
    'recall': recall,
    # could add more metrics such as accuracy for each token type
}
