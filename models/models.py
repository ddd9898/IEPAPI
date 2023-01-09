import torch
import torch.nn as nn



##Index-2
class Model_IM(nn.Module):
    def __init__(self,
                dropout=0.2,
                num_heads=8,
                vocab_size=22,
                num_encoder_layers=1,
                d_embedding = 128, #128
                Max_len = 45, # 34 + 11 = 45
                ):
        super(Model_IM, self).__init__()
        

        self.embeddingLayer = nn.Embedding(vocab_size, d_embedding)
        self.positionalEncodings = nn.Parameter(torch.rand(Max_len, d_embedding), requires_grad=True)

        ##Encoder 
        encoder_layers = nn.TransformerEncoderLayer(d_embedding, num_heads,dim_feedforward=1024,dropout=dropout)
        encoder_norm = nn.LayerNorm(d_embedding)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers,num_encoder_layers,encoder_norm)
        
        # prediction layers for EL
        self.fc1 = nn.Linear(d_embedding, 1024)
        self.bn1 =  nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 256)
        self.bn2 =  nn.BatchNorm1d(256)
        self.outputlayer = nn.Linear(256, 1)
        
        # prediction layers for IM
        self.FC_1_IM = nn.Linear(d_embedding, 1024)
        self.BN_1_IM =  nn.BatchNorm1d(1024)
        self.FC_2_IM = nn.Linear(1024, 256)
        self.BN_2_IM =  nn.BatchNorm1d(256)
        self.predict_IM = nn.Linear(257, 1)
        
        # activation
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        

    def make_len_mask(self, inp):
        # inp[:,0] = inp[:,0] + 1
        mask = (inp == 21)  #from the dataload.py, * = 21
        # mask[:,0] = False
        return mask


    def forward(self,ConcatSeq):
        # print(ConcatSeq)

        #Get padding mask 
        pad_mask = self.make_len_mask(ConcatSeq)
        
        #Get embedding
        # ConcatEmbedding = ConcatSeq

        ConcatEmbedding = self.embeddingLayer(ConcatSeq)  #batch * seq * feature
        ConcatEmbedding = ConcatEmbedding + self.positionalEncodings[:ConcatEmbedding.shape[1],:]

        #input feed-forward:
        ConcatEmbedding = ConcatEmbedding.permute(1,0,2) #seq * batch * feature
        Concatfeature = self.transformer_encoder(ConcatEmbedding,src_key_padding_mask=pad_mask)
        Concatfeature = Concatfeature.permute(1,0,2) #batch * seq * feature
        # print(Concatfeature.shape)
        # x = Concatfeature.contiguous().view(-1,self.num_features) #batch * (seq * feature)
        # representation = Concatfeature[:,0,:] #batch * seq * feature

        # representation = torch.mean(Concatfeature,dim = 1)
        coff = 1-pad_mask.float()
        representation = torch.sum(coff.unsqueeze(2) * Concatfeature,dim=1)/torch.sum(coff,dim=1).unsqueeze(1)

        #Predict EL
        x_EL = self.fc1(representation)
        x_EL = self.bn1(x_EL)
        x_EL = self.relu(x_EL)
        x_EL = self.fc2(x_EL)
        x_EL = self.bn2(x_EL)
        x_EL = self.relu(x_EL)            
        y_EL = self.sigmoid(self.outputlayer(x_EL)).detach()
        
        #Predict IM
        x_IM = self.FC_1_IM(representation)
        x_IM = self.BN_1_IM(x_IM)
        x_IM = self.relu(x_IM)
        x_IM = self.FC_2_IM(x_IM)
        x_IM = self.BN_2_IM(x_IM)
        x_IM = self.relu(x_IM)
        
        x_IM = torch.cat([y_EL,x_IM],dim=1)

        y_IM = self.sigmoid(self.predict_IM(x_IM))
        
        #Logical 
        return y_EL,y_IM

class baseline(nn.Module):
    def __init__(self,
                dropout=0.2,
                num_heads=8,
                vocab_size=22,
                num_encoder_layers=1,
                d_embedding = 128, #128
                Max_len = 45, # 34 + 11 = 45
                ):
        super(baseline, self).__init__()
        

        self.embeddingLayer = nn.Embedding(vocab_size, d_embedding)
        self.positionalEncodings = nn.Parameter(torch.rand(Max_len, d_embedding), requires_grad=True)

        ##Encoder
        encoder_layers = nn.TransformerEncoderLayer(d_embedding, num_heads,dim_feedforward=1024,dropout=dropout)
        encoder_norm = nn.LayerNorm(d_embedding)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers,num_encoder_layers,encoder_norm)
        

        # prediction layers for EL
        self.fc1 = nn.Linear(d_embedding, 1024)
        self.bn1 =  nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 256)
        self.bn2 =  nn.BatchNorm1d(256)
        self.outputlayer = nn.Linear(256, 1)
        
        # activation
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        

    def make_len_mask(self, inp):
        # inp[:,0] = inp[:,0] + 1
        mask = (inp == 21)  #from the dataload.py, * = 21
        # mask[:,0] = False
        return mask


    def forward(self,ConcatSeq):
        # print(ConcatSeq)

        #Get padding mask 
        pad_mask = self.make_len_mask(ConcatSeq)
        
        #Get embedding
        # ConcatEmbedding = ConcatSeq

        ConcatEmbedding = self.embeddingLayer(ConcatSeq)  #batch * seq * feature
        ConcatEmbedding = ConcatEmbedding + self.positionalEncodings[:ConcatEmbedding.shape[1],:]

        #input feed-forward:
        ConcatEmbedding = ConcatEmbedding.permute(1,0,2) #seq * batch * feature
        Concatfeature = self.transformer_encoder(ConcatEmbedding,src_key_padding_mask=pad_mask)
        Concatfeature = Concatfeature.permute(1,0,2) #batch * seq * feature
        # print(Concatfeature.shape)
        # x = Concatfeature.contiguous().view(-1,self.num_features) #batch * (seq * feature)
        # representation = Concatfeature[:,0,:] #batch * seq * feature

        # representation = torch.mean(Concatfeature,dim = 1)
        coff = 1-pad_mask.float()
        representation = torch.sum(coff.unsqueeze(2) * Concatfeature,dim=1)/torch.sum(coff,dim=1).unsqueeze(1)

        #Predict
        x = self.fc1(representation)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)            
        y = self.sigmoid(self.outputlayer(x))
        
        
        #Logical 
        return y


class Model_atten_score(nn.Module):
    def __init__(self,
                dropout=0.2,
                num_heads=8,
                vocab_size=22,
                num_encoder_layers=1,
                d_embedding = 128, #128
                Max_len = 45, # 34 + 11 = 45
                ):
        super(Model_atten_score, self).__init__()
        

        self.embeddingLayer = nn.Embedding(vocab_size, d_embedding)
        self.positionalEncodings = nn.Parameter(torch.rand(Max_len, d_embedding), requires_grad=True)

        ##Encoder 
        encoder_layers = nn.TransformerEncoderLayer(d_embedding, num_heads,dim_feedforward=1024,dropout=dropout)
        encoder_norm = nn.LayerNorm(d_embedding)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers,num_encoder_layers,encoder_norm)
        
        
        # prediction layers for EL
        self.fc1 = nn.Linear(d_embedding, 1024)
        self.bn1 =  nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 256)
        self.bn2 =  nn.BatchNorm1d(256)
        self.outputlayer = nn.Linear(256, 1)
        
        # prediction layers for IM
        self.FC_1_IM = nn.Linear(d_embedding, 1024)
        self.BN_1_IM =  nn.BatchNorm1d(1024)
        self.FC_2_IM = nn.Linear(1024, 256)
        self.BN_2_IM =  nn.BatchNorm1d(256)
        self.predict_IM = nn.Linear(257, 1)
        
        # activation
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        

    def make_len_mask(self, inp):
        # inp[:,0] = inp[:,0] + 1
        mask = (inp == 21)  #from the dataload.py, * = 21
        # mask[:,0] = False
        return mask


    def forward(self,ConcatSeq):
        # print(ConcatSeq)

        #Get padding mask 
        pad_mask = self.make_len_mask(ConcatSeq)
        
        #Get embedding
        # ConcatEmbedding = ConcatSeq

        ConcatEmbedding = self.embeddingLayer(ConcatSeq)  #batch * seq * feature
        ConcatEmbedding = ConcatEmbedding + self.positionalEncodings[:ConcatEmbedding.shape[1],:]

        #input feed-forward:
        ConcatEmbedding = ConcatEmbedding.permute(1,0,2) #seq * batch * feature
        Concatfeature = self.transformer_encoder(ConcatEmbedding,src_key_padding_mask=pad_mask)
        Concatfeature = Concatfeature.permute(1,0,2) #batch * seq * feature
        # print(Concatfeature.shape)
        # x = Concatfeature.contiguous().view(-1,self.num_features) #batch * (seq * feature)
        # representation = Concatfeature[:,0,:] #batch * seq * feature
        
        attn_output_weights = self.transformer_encoder.layers[0].self_attn(ConcatEmbedding, ConcatEmbedding, ConcatEmbedding,
                            key_padding_mask=pad_mask)[1]

        # representation = torch.mean(Concatfeature,dim = 1)
        coff = 1-pad_mask.float()
        representation = torch.sum(coff.unsqueeze(2) * Concatfeature,dim=1)/torch.sum(coff,dim=1).unsqueeze(1)

        #Predict EL
        x_EL = self.fc1(representation)
        x_EL = self.bn1(x_EL)
        x_EL = self.relu(x_EL)
        x_EL = self.fc2(x_EL)
        x_EL = self.bn2(x_EL)
        x_EL = self.relu(x_EL)
        y_EL = self.sigmoid(self.outputlayer(x_EL))
        
        #Predict IM
        x_IM = self.FC_1_IM(representation)
        x_IM = self.BN_1_IM(x_IM)
        x_IM = self.relu(x_IM)
        x_IM = self.FC_2_IM(x_IM)
        x_IM = self.BN_2_IM(x_IM)
        x_IM = self.relu(x_IM)
        
        x_IM = torch.cat([y_EL,x_IM],dim=1)

        y_IM = self.sigmoid(self.predict_IM(x_IM))
        
        ##Attention
        
        
        #Logical 
        return attn_output_weights,y_EL,y_IM
