## Imports
from typing import Tuple
import torch
from typing import Any
from torch import Tensor
#from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaConfig, RobertaModel, RobertaEncoder
from transformers import BertPreTrainedModel, BertModel, BertConfig
from tape import ProteinBertConfig, ProteinBertModel
from torch.nn import CrossEntropyLoss, CosineEmbeddingLoss, Module

## Function
def distill_bert(
    teacher_model : Any,
) -> BertPreTrainedModel:
    """
    Distilates a RoBERTa (teacher_model) like would DistilBERT for a BERT model.
    The student model has the same configuration, except for the number of hidden layers, which is // by 2.
    The student layers are initilized by copying one out of two layers of the teacher, starting with layer 0.
    The head of the teacher is also copied.
    """
    # Get teacher configuration as a dictionnary
    config = ProteinBertConfig.from_pretrained("bert-base", num_labels=2)
    
    config.rnn = "lstm"
    config.num_rnn_layer = 2
    config.rnn_dropout = 0.1
    config.rnn_hidden = 768
    config.length = 50
    config.cnn_filters = 512
    config.cnn_dropout = 0.1
    
    configuration = config.to_dict()
    # Half the number of hidden layer
    configuration['num_hidden_layers'] //= 2
    # Convert the dictionnary to the student configuration
    configuration = ProteinBertConfig.from_dict(configuration)
    # Create uninitialized student model
    student_model = type(teacher_model)(configuration)
    # Initialize the student's weights
    distill_bert_weights(teacher=teacher_model, student=student_model)
    # Return the student model
    return student_model


def distill_bert_weights(teacher: Module, student: Module) -> None:
    """
    Recursively copies the weights of the (teacher) to the (student).
    This function is meant to be first called on a ProteinBertModel, but is then called on every children of that model recursively.
    The only part that's not fully copied is the encoder, of which only half is copied.
    """
    # Get state dictionaries
    teacher_state_dict = teacher.state_dict()
    student_state_dict = student.state_dict()

    # Iterate over teacher state_dict and copy to student state_dict
    for teacher_key, teacher_param in teacher_state_dict.items():
        # Modify the key to match student's structure (adjust as needed)
        student_key = teacher_key.replace("encoder.", "")
        
        # Check if the key exists in the student state_dict
        if student_key in student_state_dict:
            # Copy the parameter
            student_state_dict[student_key].copy_(teacher_param.data)



## Distllator class
class Distillator(Module):

    """
    A class to distillate a BERT-like model.
    """

    def __init__(self,
        teacher_model : BertPreTrainedModel,
        temperature : float = 1.0,
    ) -> None:
        """
        Initiates the Distillator with the (teacher_model) to distillate from.
        """
        super(Distillator, self).__init__()
        self.teacher = teacher_model
        self.student = distill_bert(teacher_model)
        self.temperature = temperature

    @property
    def temperature(self) -> float:
        """
        The temperature used for training can change, but for inference it is always 1.
        """
        return self._temperature if self.training else 1

    @temperature.setter
    def temperature(self,
        value : float,
    ) -> None:
        """
        The temperature must always be above 1. Otherwise, an error is raised.
        """
        if value < 1:
            raise(ValueError(f"Temperature must be above 1, it cannot be {value}"))
        self._temperature = value

    def get_logits(self,
        input_ids : Tensor,
        attention_mask : Tensor,
        from_teacher : bool = False,
    ) -> Tensor:
        """
        Given a couple of (input_ids) and (attention_mask), returns the logits corresponding to the prediction.
        The logits come from the student unless (from_teacher) is set to True, then it's from the teacher.
        """
        if from_teacher:
            return self.teacher.classifier(self.bert(input_ids, attention_mask)[0])
        return self.student.classifier(self.student(input_ids, attention_mask)[0])

    def forward(self,
        input_ids : Tensor,
        attention_mask : Tensor,
        labels : Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Given a couple of (input_ids) and (attention_mask), returns the logits corresponding to the prediction.
        Also takes in the (labels) associated to the inputs.
        Returns the student probability distibution with temperature 1 and the loss.
        """
        student_logits = self.get_logits(input_ids, attention_mask, False)
        teacher_logits = self.get_logits(input_ids, attention_mask, True)
        return student_logits.softmax(1), self.loss(teacher_logits, student_logits, labels)


    def loss(self,
        teacher_logits : Tensor,
        student_logits : Tensor,
        labels : Tensor,
    ) -> Tensor:
        """
        The distillation loss for distilating a BERT-like model.
        The loss takes the (teacher_logits), (student_logits) and (labels) for various losses.
        """
        # Temperature and sotfmax
        student_logits, teacher_logits = (student_logits / self.temperature).softmax(1), (teacher_logits / self.temperature).softmax(1)
        # Classification loss (problem-specific loss)
        loss = CrossEntropyLoss()(student_logits, labels)
        # CrossEntropy teacher-student loss
        loss = loss + CrossEntropyLoss()(student_logits, teacher_logits)
        # Cosine loss
        loss = loss + CosineEmbeddingLoss()(teacher_logits, student_logits, torch.ones(teacher_logits.size()[0]))
        # Average the loss and return it
        loss = loss / 3
        return loss