# %% [code]
# This Python 3 environment comes with many helpful analytics libraries installed

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from tqdm import tqdm
import time
import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.init as init
from sklearn import metrics
from sklearn.metrics import accuracy_score
import torch.optim as optim
import seaborn as sns
from booksummarydataset import get_genre
import matplotlib.pyplot as plt



def classification_report(y, pred, target, name):
    print('##########################################\n#\tTest accuracy is {:.4f}%\t#\n##########################################'.format(accuracy_score(y, 
    pred)*100))
    print("------------------------------------------------------------")
    print("Classification Report for model {}".format(name))
    print("------------------------------------------------------------")
    print(metrics.classification_report(y, pred, target_names=target, zero_division=0))
    print("------------------------------------------------------------")
    plt.figure(figsize = (20,15))
    sns.set(font_scale=1.4)
    sns.heatmap(metrics.confusion_matrix(y, pred), xticklabels = target, yticklabels = target, annot = True, fmt="d",cmap = 'summer', annot_kws={"fontsize":12})
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.rc('xtick', labelsize=12)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=12)    # fontsize of the tick labels
    plt.title("Confusion Matrix for {}".format(name), fontsize=15)
    plt.show()


def train(model, epochs, train_dataloader, valid_loader, filename, loss_fn, device):
    optimizer = optim.Adam(model.parameters(), lr = 0.001)
    best_valid_loss = float('inf')
    clip = 5
    epoch_train_loss = 0
    epoch_val_loss = 0
    train_data_len = len(train_dataloader.dataset)
    valid_data_len = len(valid_loader.dataset)
    total_accuracy = []
    train_losses = []
    valid_losses = []
    for epoch in tqdm(range(epochs)):
        t0 = time.time()
        correct= 0
        running_loss = 0.0
        model.train()
        for batch_idx, _data in enumerate(train_dataloader, 1):
            batch_X, batch_y = _data
            feature = batch_X.to(device)
            label = batch_y.to(device)
            model.zero_grad()
            output = model(feature).squeeze()
            loss = loss_fn(output.squeeze(), label)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            pred = output.data.max(1, keepdim=True)[1]
            #acc = accuracy(pred=pred, y=label)
            correct += pred.eq(label.data.view_as(pred)).sum().item()
            #print(correct)
            #print("pred {}".format(pred.shape))
            running_loss += loss.item() * feature.size(0)
            #epoch_acc += acc.item()
        t_t = time.time() - t0        
        with torch.no_grad():
            tv = time.time()
            model.eval()
            running_val_loss = 0.0
            for batch_idx_v, _data in enumerate(valid_loader, 1):
                #val_h = model.init_hidden(batch_size, device)
                batch_X, batch_y = _data
                feature = batch_X.to(device)
                label = batch_y.to(device)
                #val_h = tuple([each.detach() for each in val_h])
                output = model(feature).squeeze()
                val_loss = loss_fn(output.squeeze(), label)
                running_val_loss += val_loss.item() * feature.size(0)
        #save the best model
        if running_val_loss < best_valid_loss:
            best_valid_loss = running_val_loss
            torch.save(model.state_dict(), filename)
            print("Model saved!")
        epoch_train_loss = running_loss / train_data_len
        epoch_val_loss = running_val_loss / valid_data_len
        accuracy = correct / train_data_len
        train_losses.append(epoch_train_loss)
        valid_losses.append(epoch_val_loss)
        total_accuracy.append(accuracy)
        print("Epoch {}:\tTraining took {:.2f}[s]\tValidation took: {:.2f}[s]".format(epoch+1, t_t, time.time() - tv))
        print("Losses:  \tTraining Loss:{:.6f}   \tValidation Loss: {:.6f}\tAccuracy: {:.4f}%".format(epoch_train_loss, epoch_val_loss, accuracy*100))
        print('------------------------------------------------------------------------------------------------')
    return (train_losses, valid_losses, total_accuracy)



def test(model, test_loader, loss_fn, device, targets): 
    # Get test data loss and accuracy

    test_losses = [] # track loss
    correct = 0
    total = 0
    predictions = []
    y = []
    model.eval()
    with torch.no_grad():
        for batch_idx, _data in enumerate(test_loader, 1):
            batch_X, batch_y = _data
            feature = batch_X.to(device)
            label = batch_y.to(device)
            #h = tuple([each.detach() for each in h])
            output = model(feature)
            test_loss = loss_fn(output.squeeze(), label)
            #print("loss")
            test_losses.append(test_loss.item())
            # convert output probabilities to predicted class
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(label.data.view_as(pred)).sum()
            total += feature.size(0)
            pred = pred.data.view_as(label).cpu().numpy()
            batch_y = label.data.cpu().numpy()
            for i in range(len(pred)):
                #print(pred[i])
                g = get_genre(pred[i])
                predictions.append(g)
                g = get_genre(batch_y[i])
                y.append(g)
            # -- stats! -- ##
    #print(len(y))
    #print(predictions)
    
    # avg test loss
    print("Test loss: {:.3f}".format(np.mean(test_losses)))
    # accuracy over all test data
    #print(len(predictions))
    classification_report(y=y, pred=predictions, target=targets, name=model.name)

class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_classes, n_layers, batch_size, bidirectional = True,  dropout=0.3):
        super(GRUClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.name = "GRU Classifier"
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim,
                            num_layers=n_layers,
                            dropout=dropout,
                            bidirectional = bidirectional,batch_first=True)
        if bidirectional:
            self.fc = nn.Linear(hidden_dim*2, n_classes)
        else:
            self.fc = nn.Linear(hidden_dim, n_classes)


    def forward(self, sentence):
        embeds = self.embedding(sentence)

        #packed_embedded = nn.utils.rnn.pack_padded_sequence(embeds, src_len)
        packed_outputs, hidden = self.gru(embeds)
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)
        outputs = self.fc(hidden)
        #outputs=self.act(dense_outputs)
        return outputs
    
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_classes, n_layers, batch_size, bidirectional = True,  dropout=0.3):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.name = "LSTM Classifier"
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim,
                            num_layers=n_layers,
                            dropout=dropout,
                            bidirectional = bidirectional,batch_first=True)
        if bidirectional:
            self.fc = nn.Linear(hidden_dim*2, n_classes)
        else:
            self.fc = nn.Linear(hidden_dim, n_classes)


    def forward(self, sentence):
        embeds = self.embedding(sentence)

        #packed_embedded = nn.utils.rnn.pack_padded_sequence(embeds, src_len)
        packed_outputs, (hidden,cell) = self.lstm(embeds)
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)
        outputs = self.fc(hidden)
        #outputs=self.act(dense_outputs)
        return outputs
    
class GRUClassifierOld(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, dropout, n_layers,  n_classes, bidir=False):
        super(GRUClassifier, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.drop_prob = dropout
        if bidir:
            self.directions=2
        else:
            self.directions=1
        self.type = "GRU"
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(input_size=embedding_dim, hidden_size=hidden_dim,
                               num_layers=n_layers, batch_first=True, bidirectional=bidir,dropout=dropout)
        #self.gru1 = nn.GRU(input_size=hidden_dim*self.directions, hidden_size=hidden_dim,
        #                       num_layers=n_layers, batch_first=True, bidirectional=bidir, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(in_features=hidden_dim*self.directions, out_features=n_classes)

    def forward(self, x, hidden):
        """
        Perform a forward pass of our model on some input.
        """
        batch_size=x.size(0)
        embeds = self.embedding(x)
        gru_out, hidden = self.gru(embeds, hidden)
        #gru_out, hidden = self.gru1(gru_out, hidden)
        gru_out = gru_out[:, -1]
        out = self.classifier(gru_out)
        return out, hidden
        
    def init_hidden(self, batch_size, device):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of GRU
        hidden = (torch.zeros(self.n_layers*self.directions, batch_size, self.hidden_dim).zero_()).to(device)
        return hidden

    
class LSTMClassifierOld(nn.Module):
    """
    This is the simple RNN model we will be using to perform the classification.
    """

    def __init__(self, embedding_dim, hidden_dim, vocab_size, n_layers , n_classes, dropout=0.5, bidir=False):
        """
        Initialize the model by settingg up the various layers.
        """
        super(LSTMClassifier, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.dropout = dropout
        if bidir:
            self.directions=2
        else:
            self.directions=1
        self.type = "LSTM"
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim,
                            num_layers=n_layers,bidirectional=bidir,
                            dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(in_features=hidden_dim, out_features=n_classes)


    def forward(self, x, hidden):
        """
        Perform a forward pass of our model on some input.
        """
        batch_size=x.size(0)
        embeds = self.embedding(x)
        lstm_out, hidden = self.lstm(embeds, hidden)
        lstm_out = lstm_out[:, -1]
        out = self.classifier(lstm_out)
        return out, hidden

  
    
    def init_hidden(self, batch_size, device):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of GRU            
        # Implement function
        # initialize hidden state with zero weights, and move to GPU if available
        hidden = (torch.zeros(self.n_layers*self.directions, batch_size, self.hidden_dim).zero_().to(device),
                  torch.zeros(self.n_layers*self.directions, batch_size, self.hidden_dim).zero_().to(device))
        #hidden = (torch.zeros(self.n_layers*self.directions, batch_size, self.hidden_dim).zero_()).to(device)
        return hidden
