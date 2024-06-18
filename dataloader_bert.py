import torch
from transformers import BertTokenizer
from transformers import AutoTokenizer
from torch.utils.data import Dataset
import os
import pandas as pd
import numpy as np
import re
from typing import Union

from transformers import Trainer, TrainingArguments, BertConfig
from sklearn.metrics import accuracy_score, confusion_matrix, matthews_corrcoef, roc_auc_score

from tape.tokenizers import TAPETokenizer

class DataSetLoaderBERT(Dataset):
    def __init__(self, path, tokenizer_name, max_length):            
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, do_lower_case=False)
        self.seqs, self.labels = self.load_dataset(path)           
        self.max_length = max_length
    
    def transform(self, HLA, peptide):
        data = HLA + peptide
        # quitamos este padding, porque el tokenizar ya hara el padding
        #data = data + 'X' * (69 - len(data))  # 71 max peptide-mhc length in dataset
        return data

    def read_and_prepare(self,file):
        data = pd.read_csv(file)        
        data['cost_cents'] = data.apply(
            lambda row: self.transform(HLA=row['mhc'], peptide=row['peptide']), axis=1)
        return np.vstack(data.cost_cents)

    def get_label(self,file):
        data = pd.read_csv(file)
        label = []
        label.append(data['Label'].values)
        #label.append(data['masslabel'].values) # netMHCpan3.2 database
        return label

    def load_dataset(self,data_path):
        file = data_path
        df = pd.read_csv(file)
        y_label = self.get_label(file)[0]
        X_test = self.read_and_prepare(file)
        X_test = X_test.tolist()
        X_test = [' '.join(eachseq) for eachseq in X_test]
        X_test = [" ".join(eachseq) for eachseq in X_test]  # ['Y D S E Y R N I F T N T D E S N L Y L S Y N Y Y T W A V D A Y T W Y H M M V I F R L M',.....,'Y D S E Y R N I F T N T D E S N L Y L S Y N Y Y T W A V D A Y T W Y N F L I K F L L I']
        #print(X_test, y_label)
        # for test
        #print(X_test) ['Y F A M Y Q E N M A H T D A N T L Y I I Y R D Y T W V A R V Y R G Y L F G R D L', 'Y F A M Y Q E N M A H T D A N T L Y I I Y R D Y T W V A R V Y R G Y T D K K T H L', 'Y F A M Y Q E N M A H T D A N T L Y I I Y R D Y T W V A R V Y R G Y R S D T P L I Y', 'Y F A M Y Q E N M A H T D A N T L Y I I Y R D Y T W V A R V Y R G Y N S D L V Q K Y', 'Y F A M Y Q E N M A H T D A N T L Y I I Y R D Y T W V A R V Y R G Y L S D L L D W K', 'Y F A M Y Q E N M A H T D A N T L Y I I Y R D Y T W V A R V Y R G Y L L Q N D G F F', 'Y F A M Y Q E N M A H T D A N T L Y I I Y R D Y T W V A R V Y R G Y D S D M Q T L V', 'Y F A M Y Q E N M A H T D A N T L Y I I Y R D Y T W V A R V Y R G Y T D Y H V R V Y', 'Y F A M Y Q E N M A H T D A N T L Y I I Y R D Y T W V A R V Y R G Y V L D S E G Y L', 'Y F A M Y Q E N M A H T D A N T L Y I I Y R D Y T W V A R V Y R G Y S D F H N N R Y', 'Y F A M Y Q E N M A H T D A N T L Y I I Y R D Y T W V A R V Y R G Y D K S M V D K Y', 'Y F A M Y Q E N M A H T D A N T L Y I I Y R D Y T W V A R V Y R G Y Y T D Y L T V M', 'Y F A M Y Q E N M A H T D A N T L Y I I Y R D Y T W V A R V Y R G Y D F A E R H G Y', 'Y F A M Y Q E N M A H T D A N T L Y I I Y R D Y T W V A R V Y R G Y V L D I P S K Y', 'Y F A M Y Q E N M A H T D A N T L Y I I Y R D Y T W V A R V Y R G Y T D L T R E V Y', 'Y F A M Y Q E N M A H T D A N T L Y I I Y R D Y T W V A R V Y R G Y Y T D P E V F K', 'Y F A M Y Q E N M A H T D A N T L Y I I Y R D Y T W V A R V Y R G Y S D I H D F E Y', 'Y F A M Y Q E N M A H T D A N T L Y I I Y R D Y T W V A R V Y R G Y R D W A H N S L', 'Y F A M Y Q E N M A H T D A N T L Y I I Y R D Y T W V A R V Y R G Y F T S M T R L Y', 'Y F A M Y Q E N M A H T D A N T L Y I I Y R D Y T W V A R V Y R G Y S D S K I Q K Y', 'Y F A M Y Q E N M A H T D A N T L Y I I Y R D Y T W V A R V Y R G Y L T D T F T A Y', 'Y F A M Y Q E N M A H T D A N T L Y I I Y R D Y T W V A R V Y R G Y F P G N Y S G Y', 'Y F A M Y Q E N M A H T D A N T L Y I I Y R D Y T W V A R V Y R G Y F S D V M E D L', 'Y F A M Y Q E N M A H T D A N T L Y I I Y R D Y T W V A R V Y R G Y F L E S T F L K', 'Y F A M Y Q E N M A H T D A N T L Y I I Y R D Y T W V A R V Y R G Y Y I D E Q F E R', 'Y F A M Y Q E N M A H T D A N T L Y I I Y R D Y T W V A R V Y R G Y T D L T R D I Y', 'Y F A M Y Q E N M A H T D A N T L Y I I Y R D Y T W V A R V Y R G Y D P N S T Q R Y', 'Y F A M Y Q E N M A H T D A N T L Y I I Y R D Y T W V A R V Y R G Y T P D I K S H Y', 'Y F A M Y Q E N M A H T D A N T L Y I I Y R D Y T W V A R V Y R G Y P E V Q K K K Y', 'Y F A M Y Q E N M A H T D A N T L Y I I Y R D Y T W V A R V Y R G Y L V D L P E Y Y', 'Y F A M Y Q E N M A H T D A N T L Y I I Y R D Y T W V A R V Y R G Y L T E L P D W S', 'Y F A M Y Q E N M A H T D A N T L Y I I Y R D Y T W V A R V Y R G Y L L D S S L E Y', 'Y F A M Y Q E N M A H T D A N T L Y I I Y R D Y T W V A R V Y R G Y V T D T G A L Y', 'Y F A M Y Q E N M A H T D A N T L Y I I Y R D Y T W V A R V Y R G Y I V D H I H F Q', 'Y F A M Y Q E N M A H T D A N T L Y I I Y R D Y T W V A R V Y R G Y E L H N Q K G Y', 'Y F A M Y Q E N M A H T D A N T L Y I I Y R D Y T W V A R V Y R G Y F T E I G I H L', 'Y F A M Y Q E N M A H T D A N T L Y I I Y R D Y T W V A R V Y R G Y L V D E W L D S', 'Y F A M Y Q E N M A H T D A N T L Y I I Y R D Y T W V A R V Y R G Y F T D K A A S Y', 'Y F A M Y G E K V A H T H V D T L Y L R Y H Y Y T W A V W A Y T W Y D C E K A F F K M', 'Y F A M Y G E K V A H T H V D T L Y L R Y H Y Y T W A V W A Y T W Y S P P R R N F S P', 'Y F A M Y G E K V A H T H V D T L Y L R Y H Y Y T W A V W A Y T W Y H T V Y Y G G F H', 'Y F A M Y G E K V A H T H V D T L Y L R Y H Y Y T W A V W A Y T W Y V D K T K S V V T', 'Y F A M Y G E K V A H T H V D T L Y L R Y H Y Y T W A V W A Y T W Y T P E V N Q T T L', 'Y F A M Y G E K V A H T H V D T L Y L R Y H Y Y T W A V W A Y T W Y G S E I D C A D K', 'Y F A M Y G E K V A H T H V D T L Y L R Y H Y Y T W A V W A Y T W Y A P P Q I D N G I', 'Y F A M Y G E K V A H T H V D T L Y L R Y H Y Y T W A V W A Y T W Y E T V R K L Q A R', 'Y F A M Y G E K V A H T H V D T L Y L R Y H Y Y T W A V W A Y T W Y L G I P D A V S Y', 'Y F A M Y G E K V A H T H V D T L Y L R Y H Y Y T W A V W A Y T W Y G K C E V L E V S', 'Y F A M Y G E K V A H T H V D T L Y L R Y H Y Y T W A V W A Y T W Y G Y Q T I D D Y Y', 'Y F A M Y G E K V A H T H V D T L Y L R Y H Y Y T W A V W A Y T W Y T I E K H K Q N S', 'Y F A M Y G E K V A H T H V D T L Y L R Y H Y Y T W A V W A Y T W Y F S G N V K V D E', 'Y F A M Y G E K V A H T H V D T L Y L R Y H Y Y T W A V W A Y T W Y S A R S P S S L S', 'Y F A M Y G E K V A H T H V D T L Y L R Y H Y Y T W A V W A Y T W Y I E V L C E N F P', 'Y F A M Y G E K V A H T H V D T L Y L R Y H Y Y T W A V W A Y T W Y W P D P P D L P T', 'Y F A M Y G E K V A H T H V D T L Y L R Y H Y Y T W A V W A Y T W Y D A L E F I G K K', 'Y F A M Y G E K V A H T H V D T L Y L R Y H Y Y T W A V W A Y T W Y N V D Q I L K W I', 'Y F A M Y G E K V A H T H V D T L Y L R Y H Y Y T W A V W A Y T W Y H H A H S D A Q G', 'Y F A M Y G E K V A H T H V D T L Y L R Y H Y Y T W A V W A Y T W Y C E E E I N S T F', 'Y F A M Y G E K V A H T H V D T L Y L R Y H Y Y T W A V W A Y T W Y P Q A Q P H Q V Q', 'Y F A M Y G E K V A H T H V D T L Y L R Y H Y Y T W A V W A Y T W Y E D R E S P S V K', 'Y F A M Y G E K V A H T H V D T L Y L R Y H Y Y T W A V W A Y T W Y G T P P L S T E T', 'Y F A M Y G E K V A H T H V D T L Y L R Y H Y Y T W A V W A Y T W Y R R L M F C Y I S', 'Y F A M Y G E K V A H T H V D T L Y L R Y H Y Y T W A V W A Y T W Y I G N I K T V Q I', 'Y F A M Y G E K V A H T H V D T L Y L R Y H Y Y T W A V W A Y T W Y F V S Q V E S D T', 'Y F A M Y G E K V A H T H V D T L Y L R Y H Y Y T W A V W A Y T W Y L H Q L I K Q T Q', 'Y F A M Y G E K V A H T H V D T L Y L R Y H Y Y T W A V W A Y T W Y S N P F E I F F G', 'Y F A M Y G E K V A H T H V D T L Y L R Y H Y Y T W A V W A Y T W Y Y A P Y P S P V L', 'Y F A M Y G E K V A H T H V D T L Y L R Y H Y Y T W A V W A Y T W Y R P V T T T K R E', 'Y F A M Y G E K V A H T H V D T L Y L R Y H Y Y T W A V W A Y T W Y A S D D G S W W D', 'Y F A M Y G E K V A H T H V D T L Y L R Y H Y Y T W A V W A Y T W Y E N P P V E D S S', 'Y F A M Y G E K V A H T H V D T L Y L R Y H Y Y T W A V W A Y T W Y T S C N S G T Y R', 'Y F A M Y G E K V A H T H V D T L Y L R Y H Y Y T W A V W A Y T W Y V T A N V V D P G', 'Y F A M Y G E K V A H T H V D T L Y L R Y H Y Y T W A V W A Y T W Y K L R K K L K T A', 'Y F A M Y G E K V A H T H V D T L Y L R Y H Y Y T W A V W A Y T W Y P P K K S K D K L', 'Y F A M Y G E K V A H T H V D T L Y L R Y H Y Y T W A V W A Y T W Y L L A A S E A P R', 'Y F A M Y G E K V A H T H V D T L Y L R Y H Y Y T W A V W A Y T W Y Q W S E K V T E E']

        return (X_test, y_label)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Devuelve un string como: "Q E F F I A S G A A V D A I M E S S F D Y F D I D E A T Y H V V F T T I P L V A L T L T S Y L G L K X X X X X X X X X X X X X X X X X X X"
        seq = " ".join("".join(self.seqs[idx].split())) 
        seq_ids = self.tokenizer(seq, truncation=True, padding='max_length', max_length=self.max_length)
        #seq_ids: {'input_ids': [0, 16, 9, , ..., 13], 'attention_mask': [1, 1, 1, ..., 0]}

        sample = {key: torch.tensor(val) for key, val in seq_ids.items()} # convierte a tensores
        sample['labels'] = torch.tensor(self.labels[idx])
        #print(sample)

        return sample



######################################################################################
# DataLoader Utilizado en lso experimentos anteriores, con esto si converge
######################################################################################
import torch
from transformers import BertTokenizer
from torch.utils.data import Dataset
import os
import pandas as pd
import numpy as np
import re

class DataSetLoaderBERT_old(Dataset):
    def __init__(self, path, tokenizer_name='../../../models/prot_bert_bfd', max_length=51):                  
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name, do_lower_case=False)

        self.seqs, self.labels = self.load_dataset(path)        
        self.max_length = max_length

  
    def transform(self, HLA, peptide):
        data = HLA + peptide
        data = data + 'X' * (48 - len(data)) # no usa el max length
        return data

    def read_and_prepare(self,file):
        data = pd.read_csv(file)        
        data['cost_cents'] = data.apply(
            lambda row: self.transform(HLA=row['mhc'], peptide=row['peptide']), axis=1)
        return np.vstack(data.cost_cents)

    def get_label(self,file):
        data = pd.read_csv(file)
        label = []
        label.append(data['Label'].values)        
        return label

    def load_dataset(self,data_path):
        file = data_path
        df = pd.read_csv(file)
        y_label = self.get_label(file)[0]
        X_test = self.read_and_prepare(file)
        X_test = X_test.tolist()
        X_test = [' '.join(eachseq) for eachseq in X_test]
        X_test = [" ".join(eachseq) for eachseq in
                  X_test]  # ['Y D S E Y R N I F T N T D E S N L Y L S Y N Y Y T W A V D A Y T W Y H M M V I F R L M',.....,'Y D S E Y R N I F T N T D E S N L Y L S Y N Y Y T W A V D A Y T W Y N F L I K F L L I']

        return (X_test, y_label)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        seq = " ".join("".join(self.seqs[idx].split()))
        #seq = re.sub(r"[UZOBJ]", "X", seq).upper()

        seq_ids = self.tokenizer(seq, truncation=True, padding='max_length', max_length=self.max_length)

        sample = {key: torch.tensor(val) for key, val in seq_ids.items()}
        sample['labels'] = torch.tensor(self.labels[idx])

        return sample