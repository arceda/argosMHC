from transformers import TFRobertaModel, RobertaConfig
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout
from tensorflow.keras import Model
from torch.nn import CrossEntropyLoss, CosineEmbeddingLoss
import torch
from torch import Tensor
from torch.nn import Module
from typing import Tuple

def distill_roberta(
    teacher_model: TFRobertaModel,
) -> TFRobertaModel:
    """
    Distillates a RoBERTa (teacher_model) like would DistilBERT for a BERT model.
    The student model has the same configuration, except for the number of hidden layers, which is // by 2.
    The student layers are initilized by copying one out of two layers of the teacher, starting with layer 0.
    The head of the teacher is also copied.
    """
    # Get teacher configuration as a dictionary
    configuration = teacher_model.config.to_dict()
    # Half the number of hidden layers
    configuration['num_hidden_layers'] //= 2
    # Convert the dictionary to the student configuration
    configuration = RobertaConfig.from_dict(configuration)
    # Create uninitialized student model
    student_model = TFRobertaModel(config=configuration)
    # Initialize the student's weights
    distill_roberta_weights(teacher=teacher_model, student=student_model)
    # Return the student model
    return student_model

def distill_roberta_weights(
    teacher: Module,
    student: Module,
) -> None:
    """
    Recursively copies the weights of the (teacher) to the (student).
    This function is meant to be first called on a RobertaFor... model, but is then called on every children of that model recursively.
    The only part that's not fully copied is the encoder, of which only half is copied.
    """
    # If the part is an entire RoBERTa model or a RobertaFor..., unpack and iterate
    if isinstance(teacher, TFRobertaModel) or type(teacher).__name__.startswith('RobertaFor'):
        for teacher_part, student_part in zip(teacher.trainable_variables, student.trainable_variables):
            distill_roberta_weights(teacher_part, student_part)
    # Else if the part is an encoder, copy one out of every layer
    elif isinstance(teacher, TFRobertaModel.TFRobertaModel) and 'encoder' in teacher.name:
        teacher_encoding_layers = teacher.encoder.layer
        student_encoding_layers = student.encoder.layer
        for i in range(len(student_encoding_layers)):
            student_encoding_layers[i].set_weights(teacher_encoding_layers[2 * i].get_weights())
    # Else the part is a head or something else, copy the weights
    else:
        student.set_weights(teacher.get_weights())

class Distillator(Module):
    """
    A class to distillate a BERT-like model.
    """

    def __init__(self, teacher_model: TFRobertaModel, temperature: float = 1.0) -> None:
        """
        Initiates the Distillator with the (teacher_model) to distillate from.
        """
        super(Distillator, self).__init__()
        self.teacher = teacher_model
        self.student = distill_roberta(teacher_model)
        self.temperature = temperature

    @property
    def temperature(self) -> float:
        """
        The temperature used for training can change, but for inference it is always 1.
        """
        return self._temperature if self.training else 1

    @temperature.setter
    def temperature(self, value: float) -> None:
        """
        The temperature must always be above 1. Otherwise, an error is raised.
        """
        if value < 1:
            raise ValueError(f"Temperature must be above 1, it cannot be {value}")
        self._temperature = value

    def get_logits(self, input_ids: Tensor, attention_mask: Tensor, from_teacher: bool = False) -> Tensor:
        """
        Given a couple of (input_ids) and (attention_mask), returns the logits corresponding to the prediction.
        The logits come from the student unless (from_teacher) is set to True, then it's from the teacher.
        """
        if from_teacher:
            return self.teacher(input_ids, attention_mask).logits
        return self.student(input_ids, attention_mask).logits

    def forward(self, input_ids: Tensor, attention_mask: Tensor, labels: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Given a couple of (input_ids) and (attention_mask), returns the logits corresponding to the prediction.
        Also takes in the (labels) associated to the inputs.
        Returns the student probability distibution with temperature 1 and the loss.
        """
        student_logits = self.get_logits(input_ids, attention_mask, False)
        teacher_logits = self.get_logits(input_ids, attention_mask, True)
        return student_logits.softmax(axis=-1), self.loss(teacher_logits, student_logits, labels)

    def loss(self, teacher_logits: Tensor, student_logits: Tensor, labels: Tensor) -> Tensor:
        """
        The distillation loss for distilating a BERT-like model.
        The loss takes the (teacher_logits), (student_logits) and (labels) for various losses.
        """
        # Temperature and softmax
        student_logits, teacher_logits = (
            student_logits / self.temperature
        ).softmax(axis=-1), (teacher_logits / self.temperature).softmax(axis=-1)
        # Classification loss (problem-specific loss)
        loss = CrossEntropyLoss()(student_logits, labels)
        # CrossEntropy teacher-student loss
        loss = loss + CrossEntropyLoss()(student_logits, teacher_logits)
        # Cosine loss
        loss = loss + CosineEmbeddingLoss()(
            teacher_logits, student_logits, torch.ones(teacher_logits.shape[0])
        )
        # Average the loss and return it
        loss = loss / 3
        return loss