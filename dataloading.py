import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import re
import contractions



class SentenceDataset(Dataset):
    """
    Our custom PyTorch Dataset, for preparing strings of text (sentences)
    What we have to do is to implement the 2 abstract methods:

        - __len__(self): in order to let the DataLoader know the size
            of our dataset and to perform batching, shuffling and so on...

        - __getitem__(self, index): we have to return the properly
            processed data-item from our dataset with a given index
    """

    def __init__(self, X, y, word2idx):
        """
        In the initialization of the dataset we will have to assign the
        input values to the corresponding class attributes
        and preprocess the text samples

        -Store all meaningful arguments to the constructor here for debugging
         and for usage in other methods
        -Do most of the heavy-lifting like preprocessing the dataset here


        Args:
            X (list): List of training samples
            y (list): List of training labels
            word2idx (dict): a dictionary which maps words to indexes
        """

        self.data = X
        self.labels = y
        self.word2idx = word2idx
    
        lengths = []
        X_proc = []
        for s in self.data:
            s = s.strip()  # strip leading / trailing spaces
            s = contractions.fix(s)  # e.g. don't -> do not, you're -> you are
            s = re.sub("\s+", " ", s)  # strip multiple whitespace
            s = re.sub("'s", "", s)
            proc = [(self.word2idx[w] if w in self.word2idx else self.word2idx['unk']) for w in s.split(" ") if len(w) > 0]
            if (len(proc) >= 35): lengths.append(35)
            if(len(proc) < 35): 
                lengths.append(len(proc))
                proc = proc + (35-len(proc))*[0]
                
            X_proc.append(proc[:35]) 

        self.lengths = lengths
        self.data = X_proc
        # EX2
        #raise NotImplementedError

    def __len__(self):
        """
        Must return the length of the dataset, so the dataloader can know
        how to split it into batches

        Returns:
            (int): the length of the dataset
        """

        return len(self.data)

    def __getitem__(self, index):
        """
        Returns the _transformed_ item from the dataset

        Args:
            index (int):

        Returns:
            (tuple):
                * example (ndarray): vector representation of a training example
                * label (int): the class label
                * length (int): the length (tokens) of the sentence

        Examples:
            For an `index` where:
            ::
                self.data[index] = ['this', 'is', 'really', 'simple']
                self.target[index] = "neutral"

            the function will have to return something like:
            ::
                example = [  533  3908  1387   649   0     0     0     0]
                label = 1
                length = 4
        """

        # EX3
        example = torch.tensor(self.data[index])
        label = torch.tensor(self.labels[index])
        length = torch.tensor (self.lengths[index])
        return example, label, length
        #raise NotImplementedError

