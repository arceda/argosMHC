import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from torch.nn import functional as F
from torch.nn import MSELoss

from tape import ProteinBertAbstractModel, ProteinBertModel

from transformers import BertModel, BertPreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.file_utils import add_start_docstrings, add_start_docstrings_to_model_forward
from transformers import Trainer, TrainingArguments, BertConfig
from transformers.models.bert.modeling_bert import BERT_INPUTS_DOCSTRING, BERT_START_DOCSTRING
import math
from sklearn.metrics import accuracy_score, confusion_matrix, matthews_corrcoef, roc_auc_score

@add_start_docstrings(
    """
    Bert Model transformer with a sequence classification/regression head on top (a linear layer on top of the pooled
    output) e.g. for GLUE tasks.
    """,
    BERT_START_DOCSTRING,
)
class TapeLinear(ProteinBertAbstractModel):

    def __init__(self, config):
        super().__init__(config)

        self.num_labels = config.num_labels

        self.bert = ProteinBertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()
    
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):                
        
        outputs = self.bert(input_ids, input_mask=attention_mask) 

        sequence_output, pooled_output = outputs[:2]  

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
      
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))      

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None, # TAPE, no esta retornando los hidden states, pero si deberia https://github.com/songlab-cal/tape/blob/master/tape/models/modeling_bert.py
            attentions=None, # TAPE, no esta retornando los attentions pero si deberia
        )


@add_start_docstrings(
    """
    Bert Model transformer with a sequence classification/regression head on top (a linear layer on top of the pooled
    output) e.g. for GLUE tasks.
    """,
    BERT_START_DOCSTRING,
)
class TapeRnn(ProteinBertAbstractModel):

    def __init__(self, config):
        super().__init__(config)

        self.num_labels = config.num_labels
        self.rnn_type = config.rnn
        self.num_rnn_layer = config.num_rnn_layer
        self.hidden_size = config.hidden_size
        self.rnn_dropout = config.rnn_dropout
        self.rnn_hidden = config.rnn_hidden
        self.max_seq_len = config.length

        self.bert = ProteinBertModel(config)
        self.rnn = nn.LSTM(input_size=self.hidden_size, hidden_size=self.rnn_hidden, bidirectional=True,
                               num_layers=self.num_rnn_layer, batch_first=True, dropout=self.rnn_dropout)
        
        self.dropout = nn.Dropout(self.rnn_dropout)
        self.classifier = nn.Linear(2*self.rnn_hidden, self.config.num_labels)
        
        self.init_weights()

        reduction = 'mean'
        self.criterion = nn.CrossEntropyLoss(reduction=reduction)
    
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):                
        
        outputs = self.bert(input_ids, input_mask=attention_mask) 

        sequence_output, pooled_output = outputs[:2]  

        rnn_out, (ht, ct) = self.rnn(sequence_output)        

        output = rnn_out.permute(0, 2, 1)
        output = torch.nn.functional.max_pool1d(output, self.max_seq_len)
        model_output = self.dropout(output.squeeze())
        logits = self.classifier(model_output)   
      
        loss_fct = self.criterion
        loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))  

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None, # TAPE, no esta retornando los hidden states, pero si deberia https://github.com/songlab-cal/tape/blob/master/tape/models/modeling_bert.py
            attentions=None, # TAPE, no esta retornando los attentions pero si deberia
        )


@add_start_docstrings(
    """
    Bert Model transformer with a sequence classification/regression head on top (a linear layer on top of the pooled
    output) e.g. for GLUE tasks.
    """,
    BERT_START_DOCSTRING,
)
class TapeRnnAtt(ProteinBertAbstractModel):

    def __init__(self, config):
        super().__init__(config)

        self.num_labels = config.num_labels
        self.rnn_type = config.rnn
        self.num_rnn_layer = config.num_rnn_layer
        self.hidden_size = config.hidden_size
        self.rnn_dropout = config.rnn_dropout
        self.rnn_hidden = config.rnn_hidden
        self.max_seq_len = config.length
        self.att_dropout = 0.1

        self.bert = ProteinBertModel(config)
        self.rnn = nn.LSTM(input_size=self.hidden_size, hidden_size=self.rnn_hidden, bidirectional=True,
                               num_layers=self.num_rnn_layer, batch_first=True, dropout=self.rnn_dropout)
        
        self.w_omega = nn.Parameter(torch.Tensor(
            self.rnn_hidden * 2, self.rnn_hidden * 2))
        self.u_omega = nn.Parameter(torch.Tensor(self.rnn_hidden * 2, 1))

        nn.init.uniform_(self.w_omega, -0.1, 0.1)
        nn.init.uniform_(self.u_omega, -0.1, 0.1)
        self.dropout = nn.Dropout(self.rnn_dropout)
        self.att_dropout = nn.Dropout(self.att_dropout)
        self.classifier = nn.Linear(2*self.rnn_hidden, self.config.num_labels)

        
        self.init_weights()

        reduction = 'mean'
        self.criterion = nn.CrossEntropyLoss(reduction=reduction)
    
    def attention_net(self, x, query, mask=None):
        d_k = query.size(-1)
        scores = torch.matmul(query, x.transpose(1, 2)) /math.sqrt(d_k)  #   scores:[batch, seq_len, seq_len]
        p_attn = F.softmax(scores, dim=-1)
        rattention = torch.matmul(p_attn, x)
        context = torch.matmul(p_attn, x).sum(1)  # [batch, seq_len, hidden_dim*2]->[batch, hidden_dim*2]
        return context, rattention


    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):                
        
        outputs = self.bert(input_ids, input_mask=attention_mask) 

        sequence_output, pooled_output = outputs[:2]  

        rnn_out, (ht, ct) = self.rnn(sequence_output)        

        query = self.att_dropout(rnn_out)
        attn_output, attention = self.attention_net(rnn_out, query)  
        logits = self.classifier(attn_output)
      
        loss_fct = self.criterion
        loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None, # TAPE, no esta retornando los hidden states, pero si deberia https://github.com/songlab-cal/tape/blob/master/tape/models/modeling_bert.py
            attentions=None, # TAPE, no esta retornando los attentions pero si deberia
        )

