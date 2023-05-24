import torch
from torch import nn
from torch.autograd import Variable
from torch.nn.parallel import DataParallel
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import os
import glob
import random as rnd
import numpy as np
import pickle
import time
import string

import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


class CharVocab: 
    ''' Create a Vocabulary for '''
    def __init__(self, type_vocab,pad_token='<PAD>', eos_token='<EOS>', unk_token='<UNK>'): #Initialization of the type of vocabulary
        self.type = type_vocab
        self.int2char = []
        if pad_token !=None:
            self.int2char += [pad_token]
        if eos_token !=None:
            self.int2char += [eos_token]
        if unk_token !=None:
            self.int2char += [unk_token]
        self.char2int = {}
        
    def __call__(self, text): 
        chars = set(''.join(text))
        self.int2char += list(chars)
        self.char2int = {char: ind for ind, char in enumerate(self.int2char)}

class RNNModel1(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_dim, n_layers, drop_rate=0.2):
        super(RNNModel1, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_size = embedding_size
        self.n_layers = n_layers
        self.vocab_size = vocab_size
        self.drop_rate = drop_rate
        self.char2int = None
        self.int2char = None
        self.rnn_layers = nn.ModuleList([nn.LSTM(embedding_size, hidden_dim, dropout=drop_rate, batch_first=True) for _ in range(3)])
        self.dropout = nn.Dropout(drop_rate)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, state):
        for layer in self.rnn_layers:
            rnn_out, state = layer(x, state)
            embed_seq = self.dropout(rnn_out)

        rnn_out = rnn_out.contiguous().view(-1, self.hidden_dim)
        logits = self.fc(rnn_out)
        return logits, state

    def init_state(self, device, batch_size=1):
        return (
            torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device),
            torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device)
        )

    def predict(self, input):
        logits, hidden = self.forward(input)
        probs = F.softmax(logits)
        probs = probs.view(input.size(0), input.size(1), probs.size(1))
        return probs, hidden


model_dir = 'Model'
data_dir = 'Dict'

# Files Data
char_dict_file = 'char_dict.pkl'
input_data_file = 'input_data.pkl'
int_dict_file = 'int_dict.pkl'

# file Model
model_info_file = 'model_info.pth'
model_file = 'model.pth'# dict
model_run_file = 'model_run.pth'

# Load the dictionary from the pickle file
char_dict_path = os.path.join(data_dir, char_dict_file)
input_data_path = os.path.join(data_dir, input_data_file)
int_dict_path = os.path.join(data_dir, int_dict_file)

# Load the dictionary from the pickle file
def load_pickle(file_path):
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data

char_dict = load_pickle(char_dict_path)
input_data = load_pickle(input_data_path)
int_dict = load_pickle(int_dict_path)

# Set vocap 
vocab = CharVocab('char',None,None,'<UNK>')
vocab.int2char = int_dict
vocab.char2int = char_dict


# Load the model's parameters
model_info_path = os.path.join(model_dir, model_info_file)
with open(model_info_path, 'rb') as f:
    model_info = torch.load(f)

model = RNNModel1(
    vocab_size=model_info['vocab_size'],
    embedding_size=model_info['embedding_dim'],
    hidden_dim=model_info['hidden_dim'],
    n_layers=model_info['n_layers'],
    drop_rate=model_info['drop_rate']
)

# Load the model dict
# with open(os.path.join(model_dir, model_file), 'rb') as f:
#     model_dict = torch.load(f)

with open(os.path.join(model_dir, model_run_file), 'rb') as f:
    model = torch.load(f)
    # print('Model loaded Success')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model.load_state_dict(model_dict)
# print('Model load state dict success')
model.to(device)
# print(model)

# ### Predict Function And Input text Function

def one_hot_encode(indices, dict_size):
    features = np.eye(dict_size, dtype=np.float32)[indices.flatten()]
    features = features.reshape((*indices.shape, dict_size))
    return features

def encode_text(input_text, vocab, one_hot = False):
    output = [vocab.char2int.get(character,0) for character in input_text]
    
    if one_hot:
    # One hot encode every integer of the sequence
        dict_size = len(vocab.char2int)
        return one_hot_encode(output, dict_size)
    else:
        return np.array(output)

def sample_from_probs(probs, top_n=10):
    _, indices = torch.sort(probs)
    # set probabilities after top_n to 0
    probs[indices.data[:-top_n]] = 0
    sampled_index = torch.multinomial(probs, 1)
    return sampled_index

def predict_probs(model, hidden, character, vocab):
    # One-hot encoding our input to fit into the model
    character = np.array([[vocab.char2int[c] for c in character]])
    character = one_hot_encode(character, model.vocab_size)
    character = torch.from_numpy(character)
    character = character.to(device)
    
    out, hidden = model(character, hidden)

    prob = nn.functional.softmax(out[-1], dim=0).data
    return prob, hidden

def format_text(text_predicted, line = 4):
    count = 0
    text = ''
    text_custom = text_predicted.split(' ')
    for i in range(len(text_custom)):
        if (i+1) % 2 == 0 :
            text_custom[i] += '\n'
            count += 1
        else :
            text_custom[i] += ' '
        text += text_custom[i]
        if count == line:
            break
    return text

# ### Input text Function
def generate_from_text(model, device, out_len, vocab, top_n=1, start='หัวใจพี่นั้นเองแหลกสลาย') :
    model.eval() # eval mode
    
    chars = [ch for ch in start]
    size = out_len - len(chars)
    # Generate the initial hidden state
    state = model.init_state(device, 1)
    
    # Warm up the initial state, predicting on the initial string
    for ch in chars:
        probs, state = predict_probs(model, state, ch, vocab)
        next_index = sample_from_probs(probs, top_n)

    # Now pass in the previous characters and get a new one
    for ii in range(size):
        probs, state = predict_probs(model, state, chars, vocab)
        next_index = sample_from_probs(probs, top_n)
        chars.append(vocab.int2char[next_index.data[0]])

    return ''.join(chars)

def write_data(file, data):
    with open(file, 'w', encoding='utf-8') as f:
        f.write(data)

input_text = 'แม้น'
print('Input text : ', input_text)
text_predicted = generate_from_text(model, device, 300, vocab, 2, input_text)
text = format_text(text_predicted)
print('Output text : \n', text)
print('Save output text to file output.txt')
write_data('output.txt', text)