import os
import warnings

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from training import torch_train_val_split
from early_stopper import EarlyStopper
from models import LSTM
from attention import SimpleSelfAttentionModel
from attention import MultiHeadAttentionModel
from attention import TransformerEncoderModel
from training import get_metrics_report

from config import EMB_PATH
from dataloading import SentenceDataset
from models import BaselineDNN
from training import train_dataset, eval_dataset
from utils.load_datasets import load_MR, load_Semeval2017A
from utils.load_embeddings import load_word_vectors

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

########################################################
# Configuration
########################################################


# Download the embeddings of your choice
# for example http://nlp.stanford.edu/data/glove.6B.zip

# 1 - point to the pretrained embeddings file (must be in /embeddings folder)
EMBEDDINGS = os.path.join(EMB_PATH, "glove.6B.50d.txt")

# 2 - set the correct dimensionality of the embeddings
EMB_DIM = 50

EMB_TRAINABLE = False
BATCH_SIZE = 128
EPOCHS = 50
DATASET = "Semeval2017A"  # options: "MR", "Semeval2017A"

# if your computer has a CUDA compatible gpu use it, otherwise use the cpu
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

########################################################
# Define PyTorch datasets and dataloaders
########################################################

# load word embeddings
print("loading word embeddings...")
word2idx, idx2word, embeddings = load_word_vectors(EMBEDDINGS, EMB_DIM)

# load the raw data
if DATASET == "Semeval2017A":
    X_train, y_train, X_test, y_test = load_Semeval2017A()
elif DATASET == "MR":
    X_train, y_train, X_test, y_test = load_MR()
else:
    raise ValueError("Invalid dataset")

# convert data labels from strings to integers
print("converting data labels from strings to integers...")
le = LabelEncoder()
le.fit(["positive", "negative", "neutral"])

print("labels before conversion:")
print(y_train[0:10], "\n")
y_train = list(le.transform(y_train))  # EX1
y_test = list(le.transform(y_test))  # EX1
n_classes = le.classes_.size  # EX1 - LabelEncoder.classes_.size
print("labels after conversion:")
print(y_train[0:10], "\n")

# Define our PyTorch-based Dataset
train_set = SentenceDataset(X_train, y_train, word2idx)
test_set = SentenceDataset(X_test, y_test, word2idx)

print("printing processed data...")
for i in range(3):
    print("init form:")
    print(X_train[i], ", label:", y_train[i])
    print("after processing:")
    print(train_set.__getitem__(i))
print("\n")

# EX4 - Define our PyTorch-based DataLoader
print("creating train loader and test loader(Pytorch DataLoader)...\n")
#train_loader = DataLoader(train_set, shuffle=True, batch_size=BATCH_SIZE)  # EX7
train_loader, val_loader = torch_train_val_split(train_set, BATCH_SIZE, BATCH_SIZE)
test_loader = DataLoader(test_set, shuffle=True, batch_size=BATCH_SIZE)  # EX7

#############################################################################
# Model Definition (Model, Loss Function, Optimizer)
#############################################################################
# model = BaselineDNN(output_size=2,  # EX8
#                     embeddings=embeddings,
#                     trainable_emb=EMB_TRAINABLE)
#model = LSTM(output_size=3, embeddings=embeddings, trainable_emb=EMB_TRAINABLE, bidirectional=True)
#model = SimpleSelfAttentionModel(output_size=3, embeddings=embeddings)
#model = MultiHeadAttentionModel(output_size=3, embeddings=embeddings)
model = TransformerEncoderModel(output_size=3, embeddings=embeddings)
# move the mode weight to cpu or gpu
model.to(DEVICE)
print(model)

# We optimize ONLY those parameters that are trainable (p.requires_grad==True)
print("setting criterion, params and optimizer...\n")
#criterion = torch.nn.BCEWithLogitsLoss()  # EX8
criterion = torch.nn.CrossEntropyLoss()
parameters = model.parameters()  # EX8
optimizer = torch.optim.Adagrad(parameters, lr=0.001) # EX8

#############################################################################
# Training Pipeline
#############################################################################
TRAIN_LOSS = []
VALID_LOSS = []
save_path = f'{DATASET}_{model.__class__.__name__}.pth'
early_stopper = EarlyStopper(model, save_path, patience=5) 
for epoch in range(1, EPOCHS + 1):
    # train the model for one epoch
    train_dataset(epoch, train_loader, model, criterion, optimizer)

    # evaluate the performance of the model, on both data sets
    train_loss, (y_train_gold, y_train_pred) = eval_dataset(train_loader,
                                                            model,
                                                            criterion)
    
    # evaluate the performance of the model, on both data sets
    valid_loss, (y_valid_gold, y_valid_pred) = eval_dataset(val_loader,
                                                            model,
                                                            criterion)
    
    print(f"\n===== EPOCH {epoch} ========")
    #print(f'\nTraining set\n{get_metrics_report(y_train_gold, y_train_pred)}')
    #print(f'\nValidation set\n{get_metrics_report(y_valid_gold, y_valid_pred)}')

    print("Train loss:" , train_loss)
    print("Test loss:", valid_loss)
    print("Train accuracy:" , accuracy_score(y_train_gold, y_train_pred))
    print("Test accuracy:" , accuracy_score(y_valid_gold, y_valid_pred))
    print("Train F1 score:", f1_score(y_train_gold, y_train_pred, average='macro'))
    print("Test F1 score:", f1_score(y_valid_gold, y_valid_pred, average='macro'))
    print("Train Recall:", recall_score(y_train_gold, y_train_pred, average='macro'))
    print("Test Recall:", recall_score(y_valid_gold, y_valid_pred, average='macro'))

    TRAIN_LOSS.append(train_loss)
    VALID_LOSS.append(valid_loss)

    if early_stopper.early_stop(valid_loss):
        print('Early Stopping was activated.')
        print(f'Epoch {epoch}/{EPOCHS}, Loss at training set: {train_loss}\n\tLoss at validation set: {valid_loss}')
        print('Training has been completed.\n')
        break

# plot training and validation loss curves
    
plt.plot(range(1, EPOCHS + 1), TRAIN_LOSS, label='Training Loss')
plt.plot(range(1, EPOCHS + 1), VALID_LOSS, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()
plt.savefig('loss_EncDec_3cls.png')