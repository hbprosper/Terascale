#!/usr/bin/env python
# coding: utf-8

# # A Tutorial on the  Transformer Neural Network
# > Alex Judge and Harrison Prosper<br>
# > Florida State University, Spring 2023 (closely follows the Annotated Transformer[1])<br>
# > Updated: July 4, 2023 for Terascale 2023, DESY, Hamburg, Germany<br>
# > Updated: March 31, 2024 HBP: load all data onto computational device 
# 
# 
# ## Introduction
# 
# This tutorial describes a sequence to sequence (seq2seq) neural network, called the __transformer__[1], which can be  used to translate one sequence of tokens to another. The tutorial follows closely the Annotated Transformer[2]. 
# 
# The seq2seq model
# consists of three parts:
# 
#   1. The embedding layers: encodes the tokens and their relative positions within sequences. An input (i.e., source) sequence is thus mapped to a point cloud in a vector space.
#   1. The transformer layers[2]: implements the syntactic and semantic encoding.
#   1. The output layer: computes weights, one for every possible token in the output vocabulary, which can be converted to probabilistic predictions for the next token in the output sequence given the input sequence and the current output sequence. 
# 
# __Tensor Convention__
# We follow the convention used in the Annotated Transformer[2] in which the batch is the first dimension in all tensors. 
# 
# 
# ## Sequence to Sequence Model 
# 
# ### Introduction
# A transformer-based seq2seq model comprises an `encoder` and a `decoder`. The encoder embeds every token in the source sequence $\boldsymbol{x}$ together with its ordinal value  in a vector space. The vectors are processed with a chain of algorithms called __attention__ and the transformed vectors together with the current target sequence $\boldsymbol{t}$ or current predicted output sequence $\boldsymbol{y}$ are sent to the decoder, which embeds the targets in the same vector space. The target vectors are likewise processed with a chain of attention algorithms, while the target vectors and those from the encoder are processed with another attention algorithm. Finally, the decoder assigns a weight to every token in the target vocabulary. Using a greedy strategy, one chooses the next output token to be the one with the largest weight, that is, the most probable token. The model is __autoregressive__: the predicted token is appended to the existing predicted output sequence and the model is called again with the same source and the updated output. The procedure repeats until either the maximum output sequence length is reached or the end-of-sequence (EOS) token is predicted as the most probable token.
# 
# 
# ### Attention
# 
# When we translate from one sequence of symbols to another sequence of symbols, for example from one natural language to another,  the meaning of the sequences is encoded in the symbols, their relative order, and the degree to which a given symbol is related to the other symbols. Consider the phrases "the white house" and "la maison blanche". In order to obtain a correct translation it is important for the model to encode the fact that "la" and "maison" are strongly related, while "the" and "house" are less so. It is also important for the model to encode the strong relationship between "the" and "la", between "house" and "maison", and between "white" and "blanche". That is, the model needs to *pay attention to* grammatical and semantic facts. At least as far as we can tell that's what humans do.
# 
# The need for the model to pay attention to relevant linguistic facts is the basis of the so-called [attention mechanism](https://nlp.seas.harvard.edu/annotated-transformer/). In the encoding stage, the model associates a vector to every token that tries to capture the strength of a token's relationship to other tokens. Since this association mechanism operates within the same sequence (that is, within the same point cloud in the vector space in which the sequence is embedded) it is referred to as __self attention__. Ideally, self attention will note the fact that "la" and "maison" are strongly coupled and, ideally, that the relative positions of "maison" and "blanche" are also strongly coupled as are the relative positions of "white" and "house". In the decoding stage of the model, in addition to the self attention over the target sequences another attention mechanism should pay attention to the fact that "the" and "la", "house" and "maison", and "white" and "blanche" are strongly coupled. At a minimum, therefore, we expect a successful seq2seq model to model self attention in both the encoding and decoding phases and source to target attention in the decoding phase. The optimal way to implement this is not known, but the transformer model implements an attention mechanism, described next, which empirically appears to be highly effective.
# 
# 
# ### Prediction
# As noted, the transformer is trained and used *autoregressively*: given source, i.e., input, sequence $\boldsymbol{x} = x_0, x_1,\cdots, x_k, x_{k+1}$ of length $k+2$ tokens, where $x_0 \equiv \text{<sos>}$ and $x_{k+1} \equiv \text{<eos>}$ are sequence delimiters, and the current output sequence  $\boldsymbol{y}_l = y_0, y_1,\cdots, y_{l-1}$ of length $l$ tokens, the model approximates a discrete conditional probability distribution  over the target vocabulary of size $m$ tokens, 
# 
# $$p_{ij} \equiv p(y_{ij} | \boldsymbol{x}, \boldsymbol{y}_l), \quad i = 0, \cdots, l, \quad j = 0,\cdots, m-1 .$$
# 
# For a vocabulary of size $m$ and a sequence of size $k$ (omitting the delimeters) every position in the sequence can be filled in $m$ ways. Therefore, there are $m^k$ possible sequences of which we want the most probable. Alas we have a bit of a computational problem. For example, for a sequence of size $k=85$ tokens and a target vocabulary of size $m = 28$ tokens there are $\sim 1\times10^{123}$ possible sentences. Even at a trillion probability calculations per second an exhaustive search would be an utterly futile undertaking because it would take far longer to complete than the current age of the universe ($\sim 4 \times 10^{17}$ s)! Obviously, we have no choice but to use a heuristic strategy.
# 
# The simplest such strategy is the __greedy search__ in which we consider only the last predicted probability distribution, that is, $p_{lj}$ and choose the most probable token as the next token at position $l$. 
# 
# A potentially better strategy is __beam search__ in which at each prediction stage we keep track of the $n$ most probable sequences so far. At the end we pick the most probable sequence among the $n$.
# 
# 
# ### References
#   1. [Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/)
#   1.  [Attention is all you need](https://papers.nips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)
# 

# Check if we're on Google Colab. If we are, we assume that we're working in a folder called __transformer__. Otherwise, assume we're running locally.

# In[17]:


import sys

try: 
    from google.colab import drive
    drive.mount('/content/gdrive')
    
    BASE = '/content/gdrive/My Drive/transformer'
    sys.path.append(BASE)
    
    print('\nRunning in Google Colab\n')
    
    gpu_info = get_ipython().getoutput('nvidia-smi')
    gpu_info = '\n'.join(gpu_info)
    if gpu_info.find('failed') >= 0:
        print('Not connected to a GPU')
    else:
        print(gpu_info)
    
except:
    BASE = '.'   
    print('\nRunning locally')

def pathname(filename):
    return f'{BASE:s}/{filename:s}'

import os, re
import numpy as np
import random
import math
import time

import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib as mp
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from dataloader import DataLoader, TimeLeft, number_of_parameters

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'\nComputational device: {str(DEVICE):s}')


# In[2]:


SEED = 314159
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


# ## Read Sequence Data
# 
# The file __seq2seq_series.txt__ contains (source, target) pairs where the targets are the Taylor series expansions of the corresponding sources up to an error term of ${\cal O}(x^6)$ and the sources are functions built from one or two terms randomly sampled from the set `{exp, sin, cos, tan, sinh, cosh, tanh}`. Since the source sequences are reasonably simple functions it is possible to train a transformer model to predict their Taylor series expansions in a few hours.

# In[3]:


short_tutorial = True

if short_tutorial:
    MAX_SEQ_LEN = 85
    BATCH_SIZE  = 32
    filename = pathname('seq2seq_series_2terms.txt')
else:
    MAX_SEQ_LEN = 200
    BATCH_SIZE  = 64
    filename = pathname('seq2seq_series.txt')
    
FTRAIN = 17
FVALID = 2
FTEST  = 1

delimit= '|'

dloader= DataLoader(filename, delimit, 
                      max_seq_len=MAX_SEQ_LEN, 
                      batch_size=BATCH_SIZE, 
                      ftrain=FTRAIN, 
                      fvalid=FVALID, 
                      ftest=FTEST)

train_data, valid_data, test_data = dloader.data_splits()


# ## The Model
# 
# The transformer comprises an encoder and decoder, each of which consists of one or more processing layers.
# 
# ### Encoder
# 
# The encoder does the following:
#  1. Each token in the source (input) sequence is encoded as a vector $\boldsymbol{t}$ in a space of __emb_dim__ dimensions. A sequence is therefore represented as a point cloud in the vector space.
#  1. The position of each token is also encoded as a vector $\boldsymbol{p}$ in a vector space of the same dimension as $\boldsymbol{t}$. (We can think of it as the same vector space.)  Both the token and position embeddings are trainable.
#  1. Each token is associated with a third vector: $\boldsymbol{v} = \lambda \boldsymbol{t} + \boldsymbol{p}$, where the scale factor $\lambda = \sqrt{\text{emb\_dim}}$.  (It is far from clear that this is the optimal way to combine both pieces of information; but it works!)
# 
# The vectors $\boldsymbol{v}$ are processed through $N$ *encoder layers*.
# 
# Since the source sequences are padded so that they are all of equal length, a method is needed to ensure that the pad tokens are ignored. This is done using masks.
# The source mask, `src_mask`, has value 1 if the token in the source is *not* a `<pad>` token and 0 if it is. The source mask is used in the encoder layers to mask the `<pad>` tokens in a mechanism called multi-head attention. There is also a target mask.

# In[4]:


class Encoder(nn.Module):
    
    def __init__(self, 
                 vocab_size,      # vocabulary size (of source)
                 emb_dim,         # dimension of token embedding space
                 n_layers,        # number of encoding layers
                 n_heads,         # number of attention heads/encoding layer
                 ff_dim,          # dimension of feed-forward network
                 dropout,         # dropout probability
                 device,          # computational device
                 max_len):        # maximum number of tokens/sequence
        
        super().__init__()

        # cache computational device
        self.device = device
        
        # represent each of the 'vocab_size' possible tokens 
        # by a vector of size 'emb_dim'
        self.tok_embedding = nn.Embedding(vocab_size, emb_dim)
        
        # represent the position of each token by a vector of size 'emb_dim'.
        # 'max_len' is the maximum length of a sequence.
        self.pos_embedding = nn.Embedding(max_len, emb_dim)
        
        # create 'n_layers' encoding layers
        self.layers = nn.ModuleList([EncoderLayer(emb_dim, 
                                                  n_heads, 
                                                  ff_dim,
                                                  dropout, 
                                                  device) 
                                     for _ in range(n_layers)])
        
        self.dropout= nn.Dropout(dropout)
        
        # factor by which to scale token embedding vectors
        self.scale  = torch.sqrt(torch.FloatTensor([emb_dim])).to(device)
        
    def forward(self, src, src_mask):
        # src      : [batch_size, src_len]         (shape of src)
        # src_mask : [batch_size, 1, 1, src_len]   (shape of src_mask)
        
        batch_size, src_len = src.shape
  
        # ---------------------------------------
        # input embedding 
        # ---------------------------------------
        src = self.tok_embedding(src)
        # src: [batch_size, src_len, emb_dim]
        
        # ---------------------------------------
        # position embedding
        # ---------------------------------------
        # create a row tensor, p, with entries [0, 1,..., src_len-1]
        pos = torch.arange(0, src_len)
        # pos: [src_len]
        
        # 1. add a dimension at position 0 (for batch size)
        # 2. repeat one instance of p per row 'batch_size' times so that 
        #    we obtain
        # pos = |p|
        #       |p|
        #        :
        #       |p|
        # 3. send to computational device
        once_per_row = 1
        #   3.1 unsqueeze inserts a dimension, here dimension 0 so that pos has 
        #     scape [1, src_len].
        #   3.2 repeat this row of integers batch_size times, once per row
        #   3.3 send to computational device
        pos = pos.unsqueeze(0).repeat(batch_size, once_per_row).to(self.device)
        # pos: [batch_size, src_len]
        
        pos = self.pos_embedding(pos)
        # pos: [batch_size, src_len, emb_dim]
        
        # linearly combine token and token position embeddings.
        # (perhaps this could be replaced by a feed-forward network)
        src = src * self.scale + pos
        # src: [batch_size, src_len, emb_dim]
        
        # is this really necessary? how does it help?
        src = self.dropout(src)
        
        # finally, pass embedded vectors through encoding layers
        # Note: the entire sequence is processed in parallel
        for layer in self.layers:
            src = layer(src, src_mask) 
            # src: [batch_size, src_len, emb_dim]
            
        return src


# Alternative (non-trained) position embedding

# In[5]:


class PositionEmbedding(nn.Module):
    
    def __init__(self,
                 emb_dim: int,           # dimension of embedding space (must be even)
                 max_len: int,           # max_len: maximum length of sequence
                 dropout: float):        # dropout probability
        
        super(PositionEmbedding, self).__init__() # initialize parent class
        
        # den = 10000^(-2j / d), j = 0, 1, 2,... emb_size / 2
        den = torch.exp(-torch.arange(0, emb_dim, 2) * math.log(10000) / emb_dim)
        # [emb_dim/2]    #  a row-wise vector
        
        pos = torch.arange(0, max_len).reshape(max_len, 1)
        # [max_len, 1]   # a column-wise vector
        
        # compute outer product of pos and den
        # x_mn = pos_m * den_n
        x   = pos * den
        # [max_len, emb_dim/2]
        
        pos_encoding = torch.zeros((max_len, emb_dim))
        # [max_len, emb_dim]
        
        # set every other column starting at column 0
        pos_encoding[:, 0::2] = torch.sin(x)
        
        # set every other column starting at column 1
        pos_encoding[:, 1::2] = torch.cos(x)
        
        # use unsqueeze 0 to place a third dimension (batch) in position 0
        pos_encoding = pos_encoding.unsqueeze(0)
        # [1, max_len, emb_dim]
               
        # registering a tensor as a buffer tells PyTorch that the tensor
        # is not to be changed during optimization.
        self.register_buffer('pos_encoding', pos_encoding)

    def forward(self, seq_len: int):
        # make sure sequence length of position encoding matches
        # that of token sequences.
        p = self.pos_encoding[:, :seq_len, :]
        #p: [1, seq_len, emb_dim]
        return p


# ### Encoder Layer
# 
#  1. Pass the source tensor and its mask to the *multi-head attention layer*.
#  1. Apply a residual connection and [Layer Normalization](https://arxiv.org/abs/1607.06450). 
#  1. Apply a linear layer.
#  1. Apply a residual connection and layer normalization.

# In[6]:


class EncoderLayer(nn.Module):
    
    def __init__(self, 
                 emb_dim, 
                 n_heads, 
                 ff_dim,  
                 dropout, 
                 device):
        
        super().__init__()
        
        self.self_attention       = MultiHeadAttention(emb_dim, 
                                                       n_heads, 
                                                       dropout, 
                                                       device)
        
        self.self_attention_norm  = nn.LayerNorm(emb_dim)
        
        self.feedforward          = Feedforward(emb_dim, ff_dim, dropout)
        
        self.feedforward_norm     = nn.LayerNorm(emb_dim)
        
        self.dropout              = nn.Dropout(dropout)
        
    def forward(self, src, src_mask):
        # src      : [batch_size, src_len, emb_dim]
        # src_mask : [batch_size, 1, 1, src_len] 
          
        # ------------------------------------------
        # self attention over embedded source
        # ------------------------------------------
        # distinguish between src and src_ as the 
        # former is needed later for a residual connection
        src_ = self.self_attention(src, src, src, src_mask)
        # src_: [batch_size, src_len, emb_dim]
        
        # is this useful?
        src_ = self.dropout(src_)
        
        # ------------------------------------------
        # add residual connection then layer norm.
        # ------------------------------------------
        # distinguish between src and src+src_ as the
        # former is later needed for a residual connection
        src  = self.self_attention_norm(src + src_)
        # src: [batch_size, src_len, emb_dim]
        
        src_ = self.feedforward(src)
        # src_: [batch_size, src_len, emb_dim]
        
        src_ = self.dropout(src_)

        # add residual connection and layer norm
        src  = self.feedforward_norm(src + src_)
        # src: [batch_size, src_len, emb_dim]
        
        return src


# ### Multi-Head Attention Layer
# 
# 
# Attention in the transformer model is defined by the matrix expression
# 
# \begin{align}
#     \textrm{Attention}(Q, K, V) & = \textrm{Softmax}\left(\frac{Q K^T}{\sqrt{d}} \right) V,
# \end{align}
# 
# where $Q$ is called the `query`, $K$ the `key`, $V$ the `value`, and $d =$ __emb_dim__ is the dimension of the vectors that represent the tokens. In practice, the vectors are split into __n_heads__ pieces each of size __head_dim__ $= \text{emb\_dim} / \text{n\_heads}$. __n_heads__ is the number of so-called __attention heads__. (It is stated that each `head` can pay attention to different aspects of a sequence. However, at our current level of understanding of how functions with billions of parameters truly work, such statements should be taken with a liberal pinch of salt.)
# In self attention, the query, key, and value tensors are derived from the same tensor via separate linear transformations of that tensor (see Attention Algorithm below). The coefficients of the linear functions are free parameters to be set by the training algorithm.  The number of rows in $Q$, $K$, and $V$, namely, __query_len__,  __key_len__, and __value_len__, respectively, is equal to the sequence length __seq_len__. For target/source attention, the query is a linear function of the target tensor while the key and value tensors are linear functions of the source tensor, where, again, the coefficients are free parameters to be fitted during training.
# 
# We first describe the attention mechanism mathematically and then follow with an algorithmic description that closely follows 
# the description in the [Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/). It is to be understood that every operation described below is performed for a batch of sequences. Therefore, when we refer to a matrix we really mean a batch of matrices. 
# 
# First consider the matrix product $Q K^T$ in component form, where summation over repeated indices (the Einstein convention) is implied,
# 
# \begin{align}
# A_{qk} 
# & = Q_{q h} \, [K^T]_{hk}, \nonumber\\
# & \quad q=1,\cdots, \text{query\_len}, \,\, h = 1, \cdots, \text{head\_dim}, \,\, k = 1, \cdots, \text{key\_len} .
# \end{align}
# 
# When the matrix $A$ is scaled and a softmax function is applied elementwise along the key length dimension (here, horizontally) the result is another matrix $W$ whose row elements, by construction, sum to unity. The matrix $W$ is then multiplied by $V$ to yield
# 
# \begin{align}
#     \text{Attention}_{qh}  
#     & = W_{qk} V_{kh}. 
# \end{align}
# 
# Since tokens are represented by vectors, it is instructive to think of the attention computation geometrically.   Each row, $i$, of $Q$, $K$, and $V$ can be regarded as the vectors $\boldsymbol{q}_i$, $\boldsymbol{k}_i$, and $\boldsymbol{v}_i$, respectively, associated with token $i$. Consider a sequence with __seq_len__ = 2. We can write $Q$, $K$, and $V$ as
# 
# \begin{align}
# Q & = \left[\begin{matrix} \boldsymbol{q}_1 \\ \boldsymbol{q}_2 \end{matrix}\right], \\
# K & = \left[\begin{matrix} \boldsymbol{k}_1 \\ \boldsymbol{k}_2 \end{matrix}\right], \text{ and} \\
# V & = \left[\begin{matrix} \boldsymbol{v}_1 \\ \boldsymbol{v}_2 \end{matrix}\right] ,
# \end{align}
# 
# and $A = Q K^T$ as the outer product matrix
# 
# \begin{align}
# A & = \left[\begin{matrix} \boldsymbol{q}_1 \\ \boldsymbol{q}_2 \end{matrix}\right] 
# \left[\begin{matrix} \boldsymbol{k}_1 & \boldsymbol{k}_2 \end{matrix}\right] ,
# \nonumber\\
# & = \left[
# \begin{matrix} 
# \boldsymbol{q}_1\cdot\boldsymbol{k}_1 & \boldsymbol{q}_1\cdot \boldsymbol{k}_2 \\ 
# \boldsymbol{q}_2\cdot\boldsymbol{k}_1 & \boldsymbol{q}_2\cdot \boldsymbol{k}_2
# \end{matrix}
# \right] .
# \end{align}
# 
# The matrix $A$ can be interpreted as a measure of the degree to which the $\boldsymbol{q}$ and $\boldsymbol{k}$ vectors are aligned. Presumably, the more aligned the two vectors the stronger the relationship between the  tokens they represent. Because of the use of the dot product, the degree of alignment depends both on the angle between the vectors as well as on their magnitudes. Consequently, two vectors can be more strongly aligned than a vector's alignment with itself! 
# 
# After the scaling and softmax operations on $A$, tokens 1 and 2 become associated with vectors $\boldsymbol{w}_1 =  (w_{11}, w_{12})$ and $\boldsymbol{w}_2 =  (w_{21}, w_{22})$, respectively, where
# 
# \begin{align}
#     w_{ij} & = \frac{\exp\left(\boldsymbol{q}_i \cdot \boldsymbol{k}_j \, / \, \sqrt{d}\right)}
#     {\sum_{k = 1}^2 \exp\left(\boldsymbol{q}_i \cdot \boldsymbol{k}_k \, / \, \sqrt{d}\right)} .
# \end{align}
# 
# These vectors lie in the line segment $[\boldsymbol{p}_1, \boldsymbol{p}_2]$ depicted in the figure below. The line segment is a simplex (specifically, a 1-simplex) that is embedded in a vector space of dimension __seq_len__.  In this vector space, tokens 1 and 2 are represented by the orthogonal unit vectors $\boldsymbol{u}_1$ and $\boldsymbol{u}_2$, respectively. For a sequence of length $n$, the vectors $\boldsymbol{w}_i$, $i = 1,\cdots, n$ lie in the $(n-1)$-simplex and, again, each coordinate unit vector $\boldsymbol{u}_i$ represents a token.  
# 
# ![simplex](simplex.png)
# 
# In the figure above, notice that vector $\boldsymbol{w}_2$ is closer to $\boldsymbol{u}_1$ than it is to $\boldsymbol{u}_2$ indicating that token 2 is more strongly aligned with token 1 than token 2 is aligned with itself, while the converse is true of token 1. The attention vector for token, $i$, in the transformer model is simply the weighted average 
# 
# \begin{align}
#     \text{Attention}_i & = w_{i1}  \boldsymbol{v}_1 + w_{i2} \boldsymbol{v}_2
# \end{align}
# 
# of the so-called value vectors $\boldsymbol{v}_1$ and $\boldsymbol{v}_2$.  The upshot of this construction is that the attention vectors can be viewed as linear functions of the source or target tensors with coefficients that depend non-linearly on the source or target. Consequently, the attention adapts to the sequences as, presumably, it should.
# 
# 
# ### Attention Algorithm
# 
# Now we describe the transformer attention mechanism algorithmically, again following closely the description in the [Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/), but with some notational changes.
# 
# #### Step 1
# As noted, the attention mechanism starts with three tensors, $Q_\text{in}$, $K_\text{in}$, and $V_\text{in}$, of shapes __[batch_size, query_len, emb_dim]__, __[batch_size, key_len, emb_dim]__, and __[batch_size, value_len, emb_dim]__, respectively, with __value_len=key_len__. (emb_dim is called hid_dim in the [Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/)).  For self attention, $Q_\text{in}$, $K_\text{in}$, and $V_\text{in}$ are the same tensor, while for target/source attention $Q_\text{in}$ is associated with the target tensor and $K_\text{in}$ and $V_\text{in}$ with the source tensor.
# 
# Three trainable linear layers, $f_Q$, $f_K$, $f_V$, are defined, each of shape __[emb_dim, emb_dim]__, which yield the so-called `query`, `key`, and `value` tensors
# 
# \begin{align}
#     Q & = f_Q(\boldsymbol{Q_\text{in}}),\\
#     K & = f_K(\boldsymbol{K_\text{in}}), \text{ and}\\
#     V & = f_V(\boldsymbol{V_\text{in}}).
# \end{align}
# 
# Each tensor $Q$, $K$, and $V$ is the same shape as $Q_\text{in}$, $K_\text{in}$, and $V_\text{in}$, respectively. 
# 
# 
# #### Step 2
# Tensors $Q$, $K$, and $V$ are reshaped by first splitting the embedding dimension, __emb_dim__, into __n_heads__ blocks of size __head_dim = emb_dim / n_heads__ so that their shapes become __[batch_size, -1, n_heads, head_dim]__, where the __-1__ pertains to __query_len__, __key_len__, or __value_len__, whose value is determined at runtime.
# 
# #### Step 3
# Dimensions 1 and 2 of the tensors $Q$, $K$, and $V$ are permuted (`Tensor.permute(0, 2, 1, 3)`) so that we now have __[batch_size, n_heads, -1, head_dim]__. Tensor $K$ is further permuted (`Tensor.permute(0, 1, 3, 2)`) to shape __[batch_size, n_heads, head_dim, -1]__ so that it represents $K^T$.
# 
# #### Step 4
# Tensor $A = Q K^T$ is computed using `torch.matmul(Q, K^T)`, scaled by $1 \, / \, \sqrt{d}$, and a softmax is applied to the last dimension of $A$, that is, the key/value length dimension, yielding the tensor $W$ of shape __[batch_size, n_heads, query_len, key_len]__.
# 
# 
# #### Step 5
# $\text{Attention} = W V$ is computed, yielding a tensor of shape 
# __[batch_size, n_heads, query_len, head_dim]__.
# 
# #### Step 6
# The __n_heads__ and __query_len__ dimensions of `Attention` are transposed (`Tensor.permute(0, 2, 1, 3)`) to shape __[batch_size, query_len, n_heads, head_dim]__ and forced to be contiguous in memory (`contiguous()`).
# 
# #### Step 7
# The __n_heads__ and __head_dim__ are concatenated using `Attention.view(batch_size, -1, emb_dim)` to merge the attention heads into a single `MultiHeadAttention` tensor.
# 
# #### Step 8
# Finally, the merged `MultiHeadAttention` tensor is pushed through a trainable linear layer of shape __[emb_dim, emb_dim]__ to output a tensor of shape __[batch_size, -1, emb_dim]__, where __-1__ is the sequence length.
# 
# 
# ### Comments
# It is claimed that the  algorithm above captures the notion of "paying attention to" token-token associations both within the same sequence and across sequences and that each attention head "pays attention to" a different aspect of the sequences. All such claims should be taken with a pinch of salt for at least two reasons.
# First, it is not at all obvious that this computation aligns with our intuitive understanding of  that notion and, second, the computation is nested through multiple attention layers. Therefore, whatever the attention layers are doing, it is distributed over multiple layers in a highly non-linear way. 
# 
# It is, however, undeniable that the transformer has yielded amazing results. Therefore, we are forced to concede that, in practice,  whatever is going on in the attention layers the algorithm works wonders!
# 

# In[7]:


class MultiHeadAttention(nn.Module):
    
    def __init__(self, emb_dim, n_heads, dropout, device):
        
        super().__init__()
        
        # emb_dim must be a multiple of n_heads
        assert emb_dim % n_heads == 0
        
        self.emb_dim  = emb_dim
        self.n_heads  = n_heads
        self.head_dim = emb_dim // n_heads
        
        self.linear_Q = nn.Linear(emb_dim, emb_dim)
        self.linear_K = nn.Linear(emb_dim, emb_dim)
        self.linear_V = nn.Linear(emb_dim, emb_dim)
        self.linear_O = nn.Linear(emb_dim, emb_dim)
        
        self.dropout  = nn.Dropout(dropout)
        
        self.scale    = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)
        
    def forward(self, query, key, value, mask = None):
        # query  : [batch_size, query_len, emb_dim]
        # key    : [batch_size, key_len,   emb_dim]
        # value  : [batch_size, value_len, emb_dim]
        
        batch_size, _, emb_dim = query.shape
        assert emb_dim == self.emb_dim
        
        Q = self.linear_Q(query)
        # Q: [batch_size, query_len, emb_dim]
        
        K = self.linear_K(key)
        # K: [batch_size, key len,   emb_dim]
        
        V = self.linear_V(value)
        # V: [batch_size, value_len, emb_dim]
        
        # split vectors of size emb_dim into 'n_heads' vectors of size 'head_dim'
        # and then permute dimensions 1 and 2
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        # Q: [batch_size, n_heads, query_len, head_dim]
        
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        # K: [batch_size, n_heads, key_len,   head_dim]        
        
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        # V: [batch_size, n_heads, value_len, head_dim]
          
        # transpose K (by permuting key_len and head_dim), then
        # compute QK^T/scale
        A = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        # A: [batch_size, n_heads, query_len, key_len]
        
        if mask is not None:
            A = A.masked_fill(mask == 0, -1e10)
        
        # apply softmax to the last dimension (i.e, to key len)
        # WARNING: W is referred to as 'attention' in Annotated Transformer!
        W = torch.softmax(A, dim=-1)     
        # W: [batch_size, n_heads, query_len, key_len]
        
        # not sure why dropout is useful here
        W = self.dropout(W)
        
        # compute attention: (QK^T/scale)V
        attention  = torch.matmul(W, V)
        # attention: [batch_size, n_heads, query_len, head_dim]
        
        # permute n heads and query len and make sure the tensor 
        # is contiguous in memory...
        attention = attention.permute(0, 2, 1, 3).contiguous()
        # attention: [batch_size, query_len, n_heads, head_dim]
        
        # ... and concatenate the n heads into a single multi-head 
        # attention tensor
        attention = attention.view(batch_size, -1, self.emb_dim)
        # attention: [batch_size, query_len, emb_dim]
        
        output    = self.linear_O(attention)
        # output: [batch_size, query_len, emb_dim]

        return output


# In[8]:


# Written by ChatGPT v3.5!
def group_sort(input):
    """
    Sorts the input tensor into groups of size 2 and sorts each group independently.

    Args:
        input (torch.Tensor): The input tensor to be sorted.

    Returns:
        torch.Tensor: The sorted tensor, with elements grouped and sorted in ascending order.
    """
    # Reshape the input tensor into groups of size 2
    grouped_tensor = input.view(-1, 2)

    # Sort each group individually using torch.sort
    sorted_groups, _ = torch.sort(grouped_tensor)
    
    # Flatten the sorted groups tensor
    sorted_tensor = sorted_groups.reshape(input.shape)

    return sorted_tensor


# ### Feedforward Layer

# In[9]:


class Feedforward(nn.Module):
    
    def __init__(self, emb_dim, ff_dim, dropout):
        
        super().__init__()
        
        self.linear_1 = nn.Linear(emb_dim, ff_dim)
        
        self.linear_2 = nn.Linear(ff_dim, emb_dim)
        
        self.dropout  = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: [batch_size, seq_len, emb_dim]
        
        x = self.linear_1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        # x: [batch_size, seq_len, ff_dim]
        
        x = self.linear_2(x)
        # x: [batch_size, seq_len, emb_dim]
        
        return x


# ### Decoder
# 
# The decoder takes the encoded representation of the source sequence, which is represented by a point cloud in a vector space of dimension __emb_dim__, together with the target sequence, or the current predicted output sequence, and computes weights over the target vocabulary which can be converted to a probability distribution over the target vocabulary for the next output token. 
# 
# The decoder has two multi-head attention layers: a *masked multi-head attention layer* over the target sequence, and a multi-head attention layer which uses the decoder representation as the query and the encoder representation as the key and value.
# 
# __Note__: In PyTorch, the softmax operation, which converts the output weights to probabilities, is contained within the loss function, so the decoder does not have a softmax layer.

# In[10]:


class Decoder(nn.Module):
    
    def __init__(self, 
                 vocab_size,   # size of target vocabulary
                 emb_dim,      # dimension of embedding vector space
                 n_layers, 
                 n_heads, 
                 ff_dim, 
                 dropout, 
                 device,
                 max_len):
        
        super().__init__()
        
        self.device = device
        
        self.tok_embedding = nn.Embedding(vocab_size, emb_dim)
        
        self.pos_embedding = nn.Embedding(max_len, emb_dim)
        
        self.layers  = nn.ModuleList([DecoderLayer(emb_dim, 
                                                   n_heads, 
                                                   ff_dim, 
                                                   dropout, 
                                                   device)
                                     for _ in range(n_layers)])
        
        self.linear  = nn.Linear(emb_dim, vocab_size)
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale   = torch.sqrt(torch.FloatTensor([emb_dim])).to(device)
        
    def forward(self, trg, src, trg_mask, src_mask):
        # trg      : [batch_size, trg_len]
        # src      : [batch_size, src_len, emb_dim]
        # trg_mask : [batch_size, 1, trg_len, trg_len]
        # src_mask : [batch_size, 1, 1, src_len]
                
        batch_size, trg_len = trg.shape
        
        # see Encoder for comments
        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)                  
        # pos: [batch_size, trg_len]
            
        trg = self.tok_embedding(trg) * self.scale + self.pos_embedding(pos)
        # trg: [batch_size, trg_len, emb_dim]
        
        trg = self.dropout(trg)
        
        # send the same input source to every decoding layer, with the
        # input target entering the first layer and its output target 
        # entering the second layer etc.
        for layer in self.layers:
            trg = layer(trg, src, trg_mask, src_mask)
            # trg: [batch_size, trg_len, emb_dim]
        
        output = self.linear(trg)
        # output: [batch_size, trg_len, vocab_size]
            
        return output


# ### Decoder Layer
# 
# The decoder layer has two multi-head attention layers, `self_attention` and `attention`. The former applies the attention algorithm to the target sequences, while the latter applies the algorithm between the target and source sequences.

# In[11]:


class DecoderLayer(nn.Module):
    
    def __init__(self, 
                 emb_dim, 
                 n_heads, 
                 ff_dim, 
                 dropout, 
                 device):
        
        super().__init__()
        
        self.self_attention      = MultiHeadAttention(emb_dim, n_heads, dropout, device)
        
        self.self_attention_norm = nn.LayerNorm(emb_dim)
        
        self.attention           = MultiHeadAttention(emb_dim, n_heads, dropout, device)
        
        self.attention_norm      = nn.LayerNorm(emb_dim)
        
        self.feedforward         = Feedforward(emb_dim, ff_dim, dropout)
        
        self.feedforward_norm    = nn.LayerNorm(emb_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, trg, src, trg_mask, src_mask):
        # trg      : [batch size, trg len, emb dim]
        # src      : [batch size, src len, emb dim]
        # trg_mask : [batch size, 1, trg len, trg len]
        # src_mask : [batch size, 1, 1, src len]
        
        # compute attention over embedded target sequences.
        # distinguish between trg and trg_, since the former 
        # is needed later for residual connections.
        trg_ = self.self_attention(trg, trg, trg, trg_mask)
        # trg_: [batch_size, trg_len, emb_dim]
        
        trg_ = self.dropout(trg_)
        
        # residual connection and layer norm
        # ?? trg has not had the target mask applied, so the
        # residual connection must surely dilute the effect of
        # of the masked tensor trg_ ??
        trg  = self.self_attention_norm(trg + trg_)
        # trg: [batch_size, trg_len, emb_dim]
            
        # target/source attention
        trg_ = self.attention(trg, src, src, src_mask)
        # trg_: [batch_size, trg_len, emb_dim]
        
        trg_ = self.dropout(trg_)
        
        # residual connection and layer norm
        trg  = self.attention_norm(trg + trg_)      
        # trg: [batch_size, trg_len, emb_dim]
        
        trg_ = self.feedforward(trg)
        # trg_: [batch_size, trg_len, emb_dim]
        
        trg = self.dropout(trg)
        
        # residual and layer norm
        trg  = self.feedforward_norm(trg + trg_)
        # trg: [batch_size, trg_len, emb_dim]
        
        return trg


# ### Seq2Seq
# 
# The `Seq2Seq` model encapsulates the encoder and decoder and handles the creation of the source and target masks.
# 
# The source mask, as described above, masks out `<pad>` tokens: the mask is 1 where the token is *not* a `<pad>` token and 0 if it is. The mask is then unsqueezed so it can be correctly broadcast to tensors of shape **_[batch_size, n_heads, seq_len, seq_len]_**, which appear in the multi-head attention mechanism.
# The target mask also includes a mask for the `<pad>` tokens.
# 
# Consider a target sequence $\text{<sos>}, t_1,\cdots, t_{k}, \text{<eos>}$ of length $k+2$ constructed with tokens, $t_i$, from the target vocabulary and delimited by the special tokens $t_0 \equiv \text{<sos>}$ and $t_{k+1} \equiv \text{<eos>}$, the start-of-sequence and end-of-sequence tokens, respectively. Unlike previous seq2seq models, ideally, we would like the ability to test the quality of all sub-sequences *simultaneously*. For example, given sub-sequences $\text{<sos>}$ and  $\text{<sos>}, t_1$ we would like the ability to check simultaneously the predictions $\text{<sos>} \rightarrow y_1 \sim t_1$ and $\text{<sos>}, t_1 \rightarrow y_2 \sim t_2$ and so on, where $y_1$ and $y_2$ are the current model predictions for the next tokens for the sub-sequences. In the transformer this is achieved with a so-called *subsequent* mask, `trg_sub_mask`, created using `torch.tril`. The latter creates a diagonal matrix where the elements above the diagonal are zero and the elements below the diagonal are one. For example, for a target comprising 5 tokens the `trg_sub_mask` will look like this:
# 
# $$\begin{matrix}
# 1 & 0 & 0 & 0 & 0\\
# 1 & 1 & 0 & 0 & 0\\
# 1 & 1 & 1 & 0 & 0\\
# 1 & 1 & 1 & 1 & 0\\
# 1 & 1 & 1 & 1 & 1\\
# \end{matrix}$$
# 
# When the mask is applied to a target sequence, the mask limits which target sub-sequence is available for predicting the next token. For example, the first token of the target sequence has the mask **_[1, 0, 0, 0, 0]_**. Therefore, along with the entire source sequence, $\boldsymbol{x}$, only the `<sos>` token of the target sequence is available for prediction of the next token. Using, for example, a greedy search (see above) the model predicts token $y_1$. During training this prediction is compared with the 2nd target token, that is, the token $t_1$. The second token of the target sequence has the mask **_[1, 1, 0, 0, 0]_**.  Therefore,
# along with the entire source sequence, only the target tokens `<sos>` and $t_1$ are available during training for prediction of the 3rd token, that is, the next token, $y_2$ and so on. By using a subsequent mask it is possible for the transformer model to implement the attention mechanism simultaneously for every target sub-sequence, which thereby makes it possible, during training, for the model to perform simultaneously the following predictions and comparisons,
# \begin{align*}
#   \boldsymbol{x}, \text{<sos>} & \rightarrow y_1 \sim t_1,\\
#   \boldsymbol{x}, \text{<sos>}, t_1  & \rightarrow y_2 \sim t_2, \\
#         : & : \\
#   \boldsymbol{x}, \text{<sos>}, t_1,\cdots, t_{k}  & \rightarrow y_{k+1} \sim \text{<eos>} ,
# \end{align*}
# with a loss computed for each comparison $y_i \sim t_i$. 
#     
# During evaluation, the model is used autoregressively: the entire source sequence $\boldsymbol{x}$ and target `<sos>` is used to predict $y_1$, then the source sequence and sequence `<sos>` and $y_1$ is used to predict $y_2$ and so on.
# 
# The target mask is the logical and of the pad and subsequent masks.

# In[12]:


class Seq2Seq(nn.Module):
    
    def __init__(self, 
                 encoder, 
                 decoder, 
                 src_pad, 
                 trg_pad, 
                 device):
        
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad = src_pad
        self.trg_pad = trg_pad
        self.device  = device
        
    def make_src_mask(self, src):
        # src: [batch_size, src_len]
        
        src_mask = (src != self.src_pad).unsqueeze(1).unsqueeze(2)
        # src_mask: [batch_size, 1, 1, src_len]

        return src_mask
    
    def make_trg_mask(self, trg):
        # trg: [batch size, trg len]
        
        trg_len = trg.shape[1]
            
        trg_pad_mask = (trg != self.trg_pad).unsqueeze(1).unsqueeze(2)
        # trg_pad_mask: [batch_size, 1, 1, trg_len]
        
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), 
                                             device=self.device)).bool()
        # trg_sub_mask: [trg_len, trg_len]
            
        # logical AND of the two masks
        trg_mask = trg_pad_mask & trg_sub_mask
        # trg_mask: [batch_size, 1, trg_len, trg_len]
        
        return trg_mask

    def forward(self, src, trg):
        # src: [batch_size, src_len]
        # trg: [batch_size, trg_len]
                
        src_mask = self.make_src_mask(src)
        # src_mask: [batch_size, 1, 1, src_len]
        
        trg_mask = self.make_trg_mask(trg)
        # trg_mask: [batch_size, 1, trg_len, trg_len]
        
        src      = self.encoder(src, src_mask)
        # src: [batch_size, src_len, emb_dim]
                
        # the decoder will encode the target sequences 
        # before applying the attention layers.
        output   = self.decoder(trg, src, trg_mask, src_mask)
        # output: [batch_size, trg_len, trg_vocab_size]
        
        return output


# ## Training the Seq2Seq Model
# 
# We can now define our encoder and decoders. This model is significantly smaller than Transformers used in research today, but it is able to be trained on a single GPU in an hour or so.

# ### Training Loop
# 
# Given the entire source sequence, $\boldsymbol{x}$, and all output sub-sequences, we want the model to predict the next token for every output sub-sequence. In particular, we want the model to predict the `<eos>` token, thereby terminating its predicted sequence. Consider, again, a target sequence of size $k = 5$. Since we want the model to predict `<eos>`, we slice off the `<eos>` token from the end of the targets,
# 
# $$\begin{align*}
# \text{trg} &= [\text{<sos>}, t_1, t_2, t_3, \text{<eos>}]\\
# \text{trg[:-1]} &= [\text{<sos>}, t_1, t_2, t_3],
# \end{align*}$$
# 
# where the $t_i$ denotes target sequence tokens other than `<sos>` and `<eos>`. The sliced targets are fed into the model to get a predicted sequence. If all goes well, the model should predict the `<eos>` token, thereby terminating the predicted sequence,
# 
# $$\begin{align*}
# \text{output} &= [y_1, y_2, y_3, \text{<eos>}],
# \end{align*}$$
# 
# where the
# $y_i$ are the predicted target sequence tokens. The loss is computed using the original `trg` tensor with the `<sos>` token sliced off:
# 
# $$\begin{align*}
# \text{output} &= [y_1, y_2, y_3, \text{<eos>}]\\
# \text{trg[1:]} &= [t_1, t_2, t_3, \text{<eos>}] .
# \end{align*}$$

# In[18]:


def train(model, optimizer, loss_fn, dataloader,
          niterations, dictfile, 
          batch_size, pad_code,
          traces, 
          lossfile=pathname('losses.txt'),
          valid_size=256,
          step=100):
    
    train_data, valid_data, _ = dataloader.data_splits()
    dataloader.set_batch_size(batch_size)
    
    xx, yy_t, yy_v = traces
    
    v_min = 1.e20 # minimum validation loss
    
    def compute_loss(x, t):
        # x: [batch_size, src_seq_len]
        # t: [batch_size, trg_seq_len]
       
        # slice off EOS token from all targets
        y = model(x, t[:,:-1])
        # [batch_size, trg_seq_len, trg_vocab_size]
        
        trg_vocab_size = y.shape[-1]
        
        y_out = y.reshape(-1, trg_vocab_size)
        # [batch_size * tgt_seq_len, tgt_vocab_size]
        
        # slice of SOS token from targets
        t_out = t[:, 1:].reshape(-1)
        # [batch_size * tgt_seq_len]
        
        loss  = loss_fn(y_out, t_out).mean()

        return loss
  
    def validate(ii):
        
        model.eval()
        
        with torch.no_grad():  # no need to compute gradients wrt. to x, t
                
            x, t   = dataloader.get_batch(train_data, ii, batch_size=valid_size)
            t_loss = compute_loss(x, t).item()
                
            x, t   = dataloader.get_batch(valid_data, ii, batch_size=valid_size)
            v_loss = compute_loss(x, t).item()
            if len(xx) < 1:
                xx.append(0)
            else:
                xx.append(xx[-1]+step)
            yy_t.append(t_loss)
            yy_v.append(v_loss)
            
        return t_loss, v_loss
    
    timeleft = TimeLeft(niterations)
    
    for ii in range(niterations):
        
        model.train()
        
        src, tgt = dataloader.get_batch(train_data, ii)
        
        loss     = compute_loss(src, tgt)

        optimizer.zero_grad()     # zero gradients
        
        loss.backward()           # compute gradients

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

        optimizer.step()          # make a single step in average loss
           
        if (ii % step == 0) or (ii >= niterations-1):
        
            t_loss, v_loss = validate(ii)
            
            line = f'{t_loss:12.6f}|{v_loss:12.6f}|{np.exp(v_loss):12.6f}'

            open(lossfile, 'a').write(f'{ii:8d} {t_loss:12.6f} {v_loss:12.6f}\n')
            
            if v_loss < v_min:
                v_min = v_loss
                # save best model so far
                torch.save(model.state_dict(), dictfile) 

            timeleft(ii, line)

    print()

    torch.save(model.state_dict(), 'model_final.pth')

    return xx, yy_t, yy_v

def train_by_epoch(model, iterator, optimizer, criterion, clip):
    
    model.train()
    
    epoch_loss = 0
    
    for i, batch in enumerate(iterator):
        
        print(f'\r\tbatch: {i:10d}', end='')
        
        src = batch.src
        # [batch_size, src_len]
        
        trg = batch.trg
        # [batch_size, trg_len]
        
        optimizer.zero_grad()
        
        # slice off EOS token from targets
        output = model(src, trg[:,:-1])
        # [batch_size, trg_len - 1, output_dim]
            
        output_dim = output.shape[-1]
            
        output = output.contiguous().view(-1, output_dim)
        # [batch_size * (trg_len - 1), output_dim]
        
        # slice off SOS token
        trg = trg[:,1:].contiguous().view(-1)    
        # [batch size * (trg len - 1)]
            
        loss = criterion(output, trg)
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        
        epoch_loss += loss.item()
        
    print() 
    return epoch_loss / len(iterator)

def evaluate_by_epoch(model, iterator, criterion):
    
    model.eval()
    
    epoch_loss = 0
    
    with torch.no_grad():
    
        for i, batch in enumerate(iterator):

            src = batch.src
            trg = batch.trg
            
            output = model(src, trg[:,:-1])
            # [batch size, trg len - 1, output dim]
            
            output_dim = output.shape[-1]
            
            output = output.contiguous().view(-1, output_dim)
            # [batch size * (trg len - 1), output dim]
            
            trg = trg[:,1:].contiguous().view(-1)
            # [batch size * (trg len - 1)]
            
            loss = criterion(output, trg)

            epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


# In[19]:


def plot_average_loss(traces, ftsize=18, filename=pathname('fig_loss.png')):
    
    xx, yy_t, yy_v = traces
    
    # create an empty figure
    fig = plt.figure(figsize=(5, 5))
    fig.tight_layout()
    
    # add a subplot to it
    nrows, ncols, index = 1,1,1
    ax  = fig.add_subplot(nrows,ncols,index)

    ax.set_title("Average loss")
    
    ax.plot(xx, yy_t, 'b', lw=2, label='Training')
    ax.plot(xx, yy_v, 'r', lw=2, label='Validation')

    ax.set_xlabel('Iterations', fontsize=ftsize)
    ax.set_ylabel('average loss', fontsize=ftsize)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, which="both", linestyle='-')
    ax.legend(loc='upper right')
    
    plt.savefig(filename)
    plt.show()


# In[20]:


MAX_SRC_LEN= dloader.SRC_SEQ_LEN
INPUT_DIM  = dloader.SRC_VOCAB_SIZE

MAX_TRG_LEN= dloader.TGT_SEQ_LEN
OUTPUT_DIM = dloader.TGT_VOCAB_SIZE

if short_tutorial:
    EMB_DIM    = 64    # dimension of embedding vector space
    ENC_LAYERS = 2     # number of encoder layers
    ENC_HEADS  = 8
    ENC_FF_DIM = 128   # "hidden" dimension of feed-forward network
    ENC_DROPOUT= 0.1
    
    DEC_LAYERS = 2  
    DEC_HEADS  = 8
    DEC_FF_DIM = 128
    DEC_DROPOUT= 0.1
else:
    EMB_DIM    = 200    # dimension of embedding vector space
    ENC_LAYERS = 4      # number of encoder layers
    ENC_HEADS  = 8
    ENC_FF_DIM = 1024   # "hidden" dimension of feed-forward network
    ENC_DROPOUT= 0.1
    
    DEC_LAYERS = 4
    DEC_HEADS  = 8
    DEC_FF_DIM = 1024
    DEC_DROPOUT= 0.1


# In[21]:


enc = Encoder(INPUT_DIM, 
              EMB_DIM, 
              ENC_LAYERS, 
              ENC_HEADS, 
              ENC_FF_DIM, 
              ENC_DROPOUT, 
              DEVICE, 
              MAX_SRC_LEN)

dec = Decoder(OUTPUT_DIM, 
              EMB_DIM, 
              DEC_LAYERS, 
              DEC_HEADS, 
              DEC_FF_DIM, 
              DEC_DROPOUT, 
              DEVICE, 
              MAX_TRG_LEN)

PAD_CODE = dloader.PAD
SOS_CODE = dloader.SOS
EOS_CODE = dloader.EOS

model    = Seq2Seq(enc, dec, PAD_CODE, PAD_CODE, DEVICE).to(DEVICE)

def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)
        
model.apply(initialize_weights)
print(model)

print(f'The model has {number_of_parameters(model):,} trainable parameters')


# In[23]:


import os

VERSION     = '04-01-24'
DICTFILE    = pathname(f'seq2seq_series-{VERSION:s}.pth')
LOSSFILE    = pathname(f'seq2seq_losses-{VERSION:s}.txt')

os.system(f'rm -rf {LOSSFILE:s}')
    
traces=([], [], [])


# In[25]:


LOAD        = False
TRAIN       = True

if LOAD:
    # load best model
    model.load_state_dict(torch.load(DICTFILE, 
                                     map_location=torch.device(DEVICE)))

if TRAIN:
    if short_tutorial:
        BATCH_SIZE    = 32
        LEARNING_RATE = 1e-4
        NITERATIONS   = 30000
        STEP          =    10
    else:
        BATCH_SIZE    = 64
        LEARNING_RATE = 1e-4
        NITERATIONS   = 500000
        STEP          =    100

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_CODE)
    
    traces    = train(model, optimizer, criterion, dloader,
                      niterations=NITERATIONS, 
                      dictfile=DICTFILE,
                      batch_size=BATCH_SIZE, 
                      pad_code=PAD_CODE, 
                      traces=traces, 
                      lossfile=LOSSFILE, 
                      step=STEP)
    
    plot_average_loss(traces)


# ## Testing Model
# 
# The test data are already tokenized coded and bracketed with the `<sos>` and `<eos>` codes. The translation steps are as follows:
# 
#   1. convert the coded source tokens, `src`, to the tensor, `src_`, and add a batch dimension to it (at dimension 0) so that the source is of the correct shape, namely, `[batch_size, src_len]` with `batch_size = 1`;
#   1. create the source mask `src_mask`;
#   1. feed the source `src_` and its mask `src_mask` into the encoder;
#   1. create an (output) list, initialized with the `<sos>` token, to hold the predicted tokens;
#   1. __repeat__ steps `A` to `E` until the model predicts `<eos>` or the maximum output length is reached:
#      1. convert the current output list `trg` into a tensor `trg_` and add a batch dimension at dimension 0;
#      1. create the target mask `trg_mask`;
#      1. feed the current output, encoder output and source and target masks into the decoder;
#      1. get next predicted token from the decoder;
#      1. add the next token to the current output list;
#   1. convert the output sequence from codes to a string.

# In[19]:


def translate(src, model, 
              max_len=256, 
              sos=SOS_CODE, 
              device=DEVICE):
    
    def execute(trg, src_, src_mask):
        
        trg_ = torch.tensor(trg).unsqueeze(0).to(device)
        # trg_: [batch_size, trg_len], batch_size = 1
        
        trg_mask = model.make_trg_mask(trg_)
        # trg_mask: [batch_size, 1, 1, trg_len]
            
        with torch.no_grad():
            # ignoring the batch_size, which in any case = 1, each
            # output token corresponds to a row and each column
            # corresponds to a weight that will be mapped to a 
            # probability.
            #
            # for every target token including the next, output
            # a sequence of weights over the target vocabulary...
            y = model.decoder(trg_, src_, trg_mask, src_mask)
            # y: [batch_size, trg_len, trg_vocab_size]
            #
            # ...and convert the qequence of weights to probabilities
            # by applying a softmax to the last dimension (dim=1, horizontally)
            # of the last target position (y[:,-1])
            probs  = torch.softmax(y[:,-1], dim=1)
            
        # return the 2 token codes with the largest probabilities
        # for each of target token
        token_probs, token_codes = torch.topk(probs, k=2)
        token_probs = token_probs.t() # transpose: (trg_len, 2) => (2, trg_len)
        token_codes = token_codes.t()

        # get most probable code of last target token (i.e., the next token)
        token_code0 = token_codes[0,-1].item()
        # get next most probable code of last target token (i.e., the next token)
        token_code1 = token_codes[1,-1].item() 
        
        token_prob0 = token_probs[0,-1].item()
        token_prob1 = token_probs[1,-1].item()

        return token_code0, token_code1, token_prob0, token_prob1
    
    model.eval()

    src_ = torch.LongTensor(src).unsqueeze(0).to(device)
    # src_: [batch_size, src_len], batch_size = 1
    
    src_mask = model.make_src_mask(src_)
    # src_mask: [batch_size, 1, 1, src_len]
        
    with torch.no_grad(): # this may be redundant
    
        src_ = model.encoder(src_, src_mask)
        # src_: [batch_size, src_len, emb_dim]
        
    # initialize output with start-of-sequence. the model takes in the
    # encoded source sequence and the current output sequence and
    # outputs weights, which can be converted to probabilities, for the 
    # next output token. using a greedy strategy, the most probable token
    # is chosen as the next token, which is appended to the output sequence.
    # the algorithm repeats and stops when either the <eos> token is
    # predicted or the maximum output sequence is reached.
    trg0 = [sos] 
    
    for i in range(max_len):
        
        code0, code1, prob0, prob1 = execute(trg0, src_, src_mask)
            
        trg0.append(code0)
            
        if code0 == EOS_CODE:
            break
            
    return trg0


# In[20]:


srcs, tgts = dloader.test_data
MAX_LEN    = dloader.TGT_SEQ_LEN
PRINT_MISTAKES = False

# load best model
model.load_state_dict(torch.load(DICTFILE, 
                                 map_location=torch.device(DEVICE)))

N = len(srcs)
M = 0
F = 0.0

for i, (src, tgt) in enumerate(zip(srcs[:N], tgts[:N])):   

    # convert sequence of source codes to a string (skipping <sos> and <eos> tokens)
    src_ = stringify(src[1:-1], dloader.src_code2token)
    
    # convert sequence of target codes to a string (skipping <sos> and <eos> tokens)
    tgt_ = stringify(tgt[1:-1], dloader.tgt_code2token)
    tgt_ = tgt_.replace('<pad>','') # get rid of pads

    out  = translate(src, model, 
                     max_len=MAX_LEN, 
                     sos=SOS_CODE, 
                     device=DEVICE)
    
    # convert sequence of predicted codes to a string (skipping <sos> and <eos> tokens)
    out_ = stringify(out[1:-1], dloader.tgt_code2token)
    
    if out_ == tgt_:
        M += 1
        F = M / (i+1)
    else:
        if PRINT_MISTAKES:
            print()
            print(tgt_)
            print()
            print(out_)
            print()
            print('-'*91)

    print(f'\r{i:8d}\taccuracy: {F:8.3f}', end='')

dF = math.sqrt(F*(1-F)/N)
print(f'\r{i:8d}\taccuracy: {F:8.3f} +/- {dF:.3f}')


# In[21]:


def compute_loss_from_lists(x, t, model, avloss, device):
    
    model.eval()
    
    if type(x) == type([]):
        x = torch.tensor(x)
        t = torch.tensor(t)

    x = x.unsqueeze(0).to(device)
    t = t.unsqueeze(0).to(device)

    # slice off EOS token from targets
    y = model(x, t[:,:-1])
    # [batch_size, trg_seq_len, trg_vocab_size]

    trg_vocab_size = y.shape[-1]

    y_out = y.reshape(-1, trg_vocab_size)
    # [batch_size * tgt_seq_len, tgt_vocab_size]

    # slice of SOS token from targets
    t_out = t[:, 1:].reshape(-1)
    # [batch_size * tgt_seq_len]

    loss  = avloss(y_out, t_out).mean().item()

    return loss


# In[22]:


criterion = nn.CrossEntropyLoss(ignore_index=PAD_CODE)
N = 500
aloss = 0.0
for i, (src, tgt) in enumerate(zip(srcs[:N], tgts[:N])):
    aloss += compute_loss_from_lists(src, tgt, model, criterion, DEVICE)
aloss /= N
print(f'<loss>: {aloss:10.4f}')


# In[ ]:




