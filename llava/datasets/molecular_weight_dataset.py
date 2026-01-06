import os
import random
import json
import copy
import pickle
from typing import Dict, Optional, Sequence, List
import selfies
import torch
from torch.utils.data import Dataset
import transformers
from .preprocess import preprocess, preprocess_multimodal
from .smiles2graph import smiles2graph

class MolecularWeightSupervisedGraphDataset(Dataset):
    """Dataset for Molecular Weight Prediction"""
    add_selfies = True
    
    def __init__(self, 
                 data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args,
                ):
        super(MolecularWeightSupervisedGraphDataset, self).__init__()
        with open(data_path, "rb") as f:
            list_data_dict = json.load(f)

        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        
    def selfies2smiles(self, selfies_str):
        try:
            smiles_str = selfies.decoder(selfies_str)
        except:
            smiles_str = None
        return smiles_str

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        raw = self.list_data_dict[i]
        instruction = raw['instruction']
        
        # Usually input is SMILES or SELFIES. 
        # Looking at the user's snippet: "[C][C@@H1]..." -> SELFIES
        input_selfies = raw['input']
        output_text = raw['output']
        
        # Convert input selfies to smiles for building graph
        # If input contains ">>" or other separators, we might need handling, 
        # but Molecular Weight is property of a single molecule usually.
        smiles_str = self.selfies2smiles(input_selfies)
        graph = smiles2graph(smiles_str)
        
        if self.add_selfies:
            instruction += f" The compound SELFIES sequence is: {input_selfies}"
            
        if random.random() < 0.5:
            instruction = "<image>\n" + instruction
        else:
            instruction = instruction + "\n<image>"
            
        sources = dict(
            conversations=[
                {"from": "human", "value": instruction},
                {"from": "gpt", "value": output_text}
            ]
        )
        
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"
        
        if graph is not None:
            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args)
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])

        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=(graph is not None))
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])

        # graph exist in the data
        if graph is not None:
            data_dict['graph'] = graph
        elif self.data_args.is_multimodal:
            raise ValueError(f"Graph does not exist in the data for item {i}, but the model is multimodal")
            
        return data_dict
