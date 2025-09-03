import torch
import torch.nn
import math

#input embedding layer starts-------

class InputEmbeddings(nn.Module):

    def __init__(self, d_model: int, vocab_size: int ):
        super().__init__()
        #d.model is the dimension of the model 
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self,x):
        return self.embedding(x) * math.sqrt(self.d_model)

#input embedding ends here------

#positional encoding starts----

class PositionalEncoding(nn.module):
    
    def __init__(self, d_model: int, seq_len: int, dropout: float ) -> None:
        super().__init__()
        #d_model is vertical axis 
        #seq_len is the horizontal axis
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        #dropout is used to avoid overfitting
        # PositionalEncoding matrix will be of shape seq_len to d_model
        # create matrix of shape(seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        #now create a vector of shape (seq_len ,1)
        #position represents position of word inside setentence
        position = torch.arange(0, seq_len, dtype=torch.float).unsqeeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(1000.0)/ d_model))
        #appy sin and cosine to the postion
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        #(1, seq_len, d_model)
        pe = pe.unsqueeze(0)
        #stores the pe data as weights
        self.register_buffer('pe', pe)

    '''
    pe.shape = (1, seq_len, d_model)
    The dimensions are:
    1 → batch dimension placeholder (so pe can be broadcast over batches).
    seq_len → max sequence length.
    d_model → embedding dimension.
    '''
    def forward(self,x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)

#batch normalization,
#assume n sentences in each batch
#we calculate the mean and variance of each word independet of the other word in the same batch
#then we calcuate new values using their own mean and variances
#we add two new terms, beta and gamma 
#gamma being multiplicative
#beta being addiitive
#we want model to amplify these values when required

class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 10*-6) -> None:
        super().__init__
        self.eps = eps 
        #eps super small value 

        #alpha is multiplicate
        #bias is aadded 
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        mean = x.mean(dim = -1, keepdim = True)
        std = x.std(dim = -1 , keepdim = True)
        return self.alpha * (x - mean )/(std + self.eps) + self.bias

#feedforward layer, is fully connected
#as per paper FFN(x) = max(0,xW1+b1)W2 +b2
#two linear tranformations with Relu inbetween
#dmodel is 512 dim dff is 2048
# dmodel -> dff layer -> dfflayer -> dmodel layer
# 512 -> 2048 -> 2048 -> 512 

class FeedForwardBlock(nn.module):

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super.__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) #W1 and B1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) #W2 and B2

    def forward(self, x):
        #Tensor1(Batch, seq_len, d_model)-->Tensor2(Batch, seq_len, d_ff) --> Tensor3(Batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

'''

seq= sequence length
d_model = size of embedding vector
h = number of heads 
d_v = d_k = d_model/h
we have not shown batch dimension, this is for a single sentence 
multi head attention                                                           |<-----d_model--->|    
                                                                               |dk|                         ---           
               |-->Q(seq,d_model) X W_Q(d_model,d_model) = Q'(seq,d_model) --> [Q1]   [Q2]   [Q3]   [Q4]      |
               |                                                                |      |      |      |        |   <---- apply attention at this layer to obtain a lower triangular matrix for masked attention                                                                                 |
input      --->|-->K(seq,d_model) X W_K(d_model,d_model) = K'(seq,d_model) --> [K1]   [K2]   [K3]   [K4]      | sequence
(seq,d_model)  |                                                                |      |      |      |        |                                                                                        
               |-->V(seq,d_model) X W_V(d_model,d_model) = V'(seq,d_model) --> [V1]   [V2]   [V3]   [V4]      |
                                                                                |      |      |      |      ---
                                                                                |      |      |      |
                                                     [head_i]       seq |     [head1][head2][head3][head4] --> H(seq,h*dv) X W0(h*dv,dmodel) --> MH-A(seq,dmodel)
                                                                                <---dmodel-------------->
                                                                            
                                                                                #we split matrices 
                                                                                along embedding dimension
                                                                                not along sequeance dimension
                                                                                each head will have access to
                                                                                complete sentence, but different
                                                                                part of the embedding of each word
'''
#d_model is the embedding vector 
class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h  == 0, "d_model is not divisible by h"

        self.d_k = d_model // h
        self.w_q = nn.Liner(d_model // d_model) #Wq
        self.w_k = nn.Liner(d_model // d_model) #Wk
        self.w_v = nn.Liner(d_model // d_model) #Wv

        self.w_o = nn.Linear(d_model // d_model) #Wo

    @staticmethod
    def attention (query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        # @ is matrix transpose in python 
        # (batch, seq_len, d_k) --> (Batch, h , seq_len, seq_len)
        attention_scores = ( query @ key.transpose(-2,-1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill_(mask == 0 , -1e9)
        attention_scores = attention_scores.softmax(dim = -1) # (Batch, h , seq_len, seq_len)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        query = self.w_q(q) #Tensor1(Batch, seq_len, d_model) X Tensor2(Batch, d_model, d_model) --> Tensor3(Batch, seq_len, d_model)
        key = self.w_k(k)   #Tensor1(Batch, seq_len, d_model) X Tensor2(Batch, d_model, d_model) --> Tensor3(Batch, seq_len, d_model)
        value = self.w_v(v) #Tensor1(Batch, seq_len, d_model) X Tensor2(Batch, d_model, d_model) --> Tensor3(Batch, seq_len, d_model)

        #we now divide query key, and value matrices for them to fit into different head using view method
        #we sont want to split the sentance, we split the embedding into h parts 
        #we dont want to split the second dimension 
        # we want to split the third dimension d_model into h by d_k
        #then transpose 

        #(Batch, seq_len, d_model)--> (Batch, seq_len, d, d_k) --> (Batch, h,  seq_len, d_k)
        query = query.view(query.shape[0],query.shape[1], self.h, self.d_k).transpose(1,2)
        key = key.view(key.shape[0],key.shape[1], self.h, self.d_k).transpose(1,2)
        value = value.view(value.shape[0],value.shape[1], self.h, self.d_k).transpose(1,2)

        x,self.attention_scores = MultiHeadAttentionBlock.attention(query,key, value,mask, self.dropout)

        #(Batch, h, seq_len, d_k) --> (Batch, seq_len, h ,d_k) --> (Batch, seq_len, d_model)
        x = x.transpose(1,2).contiguos().view(x.shape[0], -1, self.h * self.d_k)
        #(Batch, seq_len, d_model) --> (Batch, seq_len, d_model)
        return self.w_o(x)

#we need to build residual layer/skip connection 