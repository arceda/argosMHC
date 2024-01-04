from typing import Union, List, Tuple, Sequence, Dict, Any, Optional, Collection, Mapping
from pathlib import Path
from tape.tokenizers import TAPETokenizer
from tape.datasets import pad_sequences as tape_pad
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import torch


class CSVDataset(Dataset):
    def __init__(self,
                 data_file: Union[str, Path, pd.DataFrame],
                 train: bool = True):
        if isinstance(data_file, pd.DataFrame):
            data = data_file
        else:
            data = pd.read_csv(data_file)

        mhc = data['mhc']
        self.mhc = mhc.values
        peptide = data['peptide']
        
        self.peptide = peptide.values   
        #self.targets = np.stack([data['label'], data['masslabel']], axis=1)
        #self.targets =  data['masslabel']
        self.targets =  data['Label']
        
        self.data = data
        if 'instance_weights' in data:
            self.instance_weights = data['instance_weights'].values
        else:
            self.instance_weights = np.ones(data.shape[0],)

    def __len__(self) -> int:
        return len(self.mhc)

    def __getitem__(self, index: int):
        seq = self.mhc[index] + self.peptide[index]
        #seq = seq + 'X' * (69 - len(seq)) # vamos a usar el pad de TAPE
        return {
            "id": str(index),
            "primary": seq,
            "protein_length": len(seq),
            "targets": self.targets[index],
            "instance_weights": self.instance_weights[index]}


class DataSetLoaderTAPE(Dataset):
    ''' Load data for pretrained Bert model, implemented in TAPE
    '''

    def __init__(self,
                 input_file,
                 tokenizer: Union[str, TAPETokenizer] = 'iupac',
                 max_length = 73,
                 in_memory: bool = False,
                 instance_weight: bool = False,
                 train: bool = True):
        if isinstance(tokenizer, str):
            tokenizer = TAPETokenizer(vocab=tokenizer)
        self.tokenizer = tokenizer
        self.data = CSVDataset(input_file, train=train)
        self.instance_weight = instance_weight
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        item = self.data[index]
        token_ids = self.tokenizer.encode(item['primary'])  
        input_mask = np.ones_like(token_ids) 

        # padding, en TAPE 0 es el token de pad
        # tambien podriamos probar con 27, que es el id del aminoacido X.
        
        pad = np.full(self.max_length - len(token_ids), 0, dtype=token_ids[0].dtype) 

        token_ids = np.hstack((token_ids, pad))
        input_mask = np.hstack((input_mask, pad))

        #print(token_ids, token_ids.shape)
        #print(input_mask, input_mask.shape)        

        ret = {'input_ids': token_ids,
               'attention_mask': input_mask,
               'labels': item['targets']}
        if self.instance_weight:
            ret['instance_weights'] = item['instance_weights']
        return ret












    def collate_fn(self, batch) -> Dict[str, torch.Tensor]:
        elem = batch[0]
        batch = {key: [d[key] for d in batch] for key in elem}
        input_ids = torch.from_numpy(tape_pad(batch['input_ids'], 0))
        input_mask = torch.from_numpy(tape_pad(batch['input_mask'], 0))
        tmp = np.array(batch['targets'])
        #targets = torch.tensor(batch['targets'], dtype=torch.float32)
        targets = torch.tensor(tmp, dtype=torch.float32)
        ret = {'input_ids': input_ids,
               'input_mask': input_mask,
               'targets': targets}
        if self.instance_weight:
            instance_weights = torch.tensor(batch['instance_weights'],
                                            dtype=torch.float32)
            ret['instance_weights'] = instance_weights
        return ret
