import torch 
import torch.nn as nn
import torch.utils.data 
import Dataset 

class BilingualDataset(Dataset):

    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len ) -> None:
        super().__init__

        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        #convert token into a number
        self.sos_token = torch.Tensor([tokenizer_src.token_to_id(['SOS'])], dtype=torch.int64)
        self.eos_token = torch.Tensor([tokenizer_src.token_to_id(['EOS'])], dtype=torch.int64)
        self.pad_token = torch.Tensor([tokenizer_src.token_to_id(['PAD'])], dtype=torch.int64)

    #tells the length method of the dataset ie length of the dataset
    # then define the get method 
    def __len(self):
        return len(self.ds)

    def __getitem__(self, index: Any) -> Any:
        #extract original pair from the hugging face dataset
        src_target_pair = self.ds[index]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]

        #we convert each text into a token and map it into its corresponding number in its vocab
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        #we add padding
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2 #-2 because of sos and eos tokens
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) -1 # -1 because decoder only needs sos token

        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError('Sentence is too long bro')

        #two tensors for the encoder and decoder input
        #one sentence is sent to input of encoder
        #one sentence is sent to input of decode
        #we expect one sentence as output of decoder which we call label/targer

        #add sos and eos to the source text 
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                #this is the padding needed to reach the sequence length 
                torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64 )
            ]
        )

        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
            ]
        )
        #add eos to label 
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
            ]
        )

        #debugging
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len 

        return{
            "encoder_input": encoder_input, #(seq_len)
            "decoder_input": decoder_input, # (seq_len)
            #here we create a mask 
            #we dont want the sos and eos tokens taking part in the attention mechanism
            #we use unsqueeze twice to add sequence dimension and batch dimension
            "encoder_mask":(encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), #(1,1, seq_len)
            #we need a special mask, a casual mask, it should only look backward, and not at special tokens
            "decoder_mask":(decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(0) & casual_mask(decoder_input.size(0)), #(1,1,seq_len)
            "label" : label, #seqlen
            "src_txt" : src_text,
            "tgt_txt" : tgt_text
        }

def casual_mask(size):
    mask = torch.triu(torch.ones(1,size,size), diagonal=1).type(torch.int)
    return mask == 0
    #the model is allowed to access only the places in the matrix which have a zero
    #ie strict upper triangular matrix
    '''
    tensor([[[0, 1, 1, 1],
         [0, 0, 1, 1],
         [0, 0, 0, 1],
         [0, 0, 0, 0]]])

    '''