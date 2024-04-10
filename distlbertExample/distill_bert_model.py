from distillerBERT import Distillator as DistillatorBert

from tape import ProteinBertAbstractModel, ProteinBertModel

protein_model_instance = ProteinBertModel.from_pretrained("bert-base", num_labels=2)

distilled_module_bert = DistillatorBert(protein_model_instance)

visualize_children(distilled_module_bert)