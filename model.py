import torch
import torch.nn as nn
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

class PositionalEncoding(nn.Module):
    
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

class FeedForwardBlock(nn.Module):

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
#this is between add and norm and previouslayer 
class ResidualConnection(nn.Module):

    def __init__(self, dropout: float ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

#we will now create the encoder block called Nx in the paper
# encoder block containes = 1 multi head attention, 2 add and norm, 1 feed forward
# first skip connection b/w MHA and AddNorm
#second skip connection b/w FeedForward and AddNorm

class Encoderblock(nn.Module):
    def __init__(self, self_attention_block : MultiHeadAttentionBlock, feed_forward_block : FeedForwardBlock, dropout:float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connection = nn.ModuleList([ResidualConnection(dropout) for _ in range (2)])

    def forward(self, x, src_mask):
        #here the query key and value is x itself, x itself is the input, as a reult we call it self attention in the encoder
        #we take x add send it to multihead attention
        #simulteniously we take x apply it to add&norm
        #then we add it together
        #lambda is the first layer
        x = self.residual_connection[0](x, lambda x: self.self_attention_block(x,x,x, src_mask))
        x = self.residual_connection[1](x, self.feed_forward_block)
        return x 

    #encoder is made up of n blocks 

class Encoder(nn.Module):

    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()
        
    def forward(self, x , mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

#Decoderblock
# it has cross attention wherein the query and key comes from the encoder bloack
# the value comes from the decoder block 

class DecoderBlock(nn.Module):

    def __init__(self, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None: 
        super.__init__() 
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        #we have three residual connections in this case
        self.residual_connection = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        #src_mask, mask comming from encoder
        #tgt_mask , mask comming from decoder
        x = self.residual_connection[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connection[1](x, lambda x: self.self_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connection[2](x, self.feed_forward_block)
        return x
    
#Decoder block 
# n times DecoderBlock

class Decoder(nn.Module):

    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x , encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)
        
#last layer is linear layer here we call it projection layer 
#maps embedding into position of vocabulary 

class ProjectionLayer(nn.Module):

    def __init__(self, d_model:int, vocab_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x ):
        #(Batch, seq_len, d_model) --> (Batch, seq_len, vocab_size)
        return torch.log_softmax(self.proj(x), dim = -1)

#in transformer we have an encoder and decoder
#we have one input embedding for source language
#we have one output embedding for target language 
# we have source and target position
#we have projection layer 
class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    #now we define three methods
    #one to encode, one to decode, one to project
    #then we apply

    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def project(self, x):
        return self.projection_layer(x)

#now we glue all this together
#given hyperparameters this will build a transformer
#number of layers is N
#number of heads is H
#d_ff is hidden layer of feedforwardlayer

def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int, N:int, H:int,  dropout: float = 0.1, d_iff = 2048) -> Transformer:
    #create embedding layer
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    #create positional embedding layer
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    #create encoder blocks 
    encoder_block = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = Encoderblock(encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)
    
    #create decoder blocks
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)

    #created encoder and decoder 
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    #create a projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    #Create the transformers 
    transformer = Transformer(encoder,decoder,src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)

    #initialize parameters
    #we use xavier algo here

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer
 
