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



