import os
import random
import json
import copy
import re
from typing import Dict, Optional, Sequence, List
import selfies
import torch
from torch.utils.data import Dataset
import transformers
from .preprocess import preprocess, preprocess_multimodal
from .smiles2graph import smiles2graph

class ExpProcedurePredSupervisedGraphDataset(Dataset):
    """Dataset for Experimental Procedure Prediction"""
    add_selfies = True
    
    def __init__(self, 
                 data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args,
                ):
        super(ExpProcedurePredSupervisedGraphDataset, self).__init__()
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

    def extract_first_selfies(self, text):
        # Matches [START_SELFIES]...[END_SELFIES]
        match = re.search(r"\[START_SELFIES\](.*?)\[END_SELFIES\]", text)
        if match:
            return match.group(1)
        return None

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        raw = self.list_data_dict[i]
        input_str = raw['input']
        output_str = raw['output']
        instruction = raw['instruction']
        
        # Extract graph from the first SELFIES found in input
        selfies_str = self.extract_first_selfies(input_str)
        graph = None
        if selfies_str:
            smiles_str = self.selfies2smiles(selfies_str)
            if smiles_str:
                graph = smiles2graph(smiles_str)
        
        # For this task, we append the full input text (which contains all reactants/products info) to instruction
        if self.add_selfies:
            instruction += f"\nInput Details: {input_str}"
            
        if random.random() < 0.5:
            instruction = "<image>\n" + instruction
        else:
            instruction = instruction + "\n<image>"
            
        sources = dict(
            conversations=[
                {"from": "human", "value": instruction},
                {"from": "gpt", "value": output_str}
            ]
        )
        
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        
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
            # If no graph found but model is multimodal, we might need to handle this.
            # Usually we expect valid graph. If not, maybe we should skip or use dummy?
            # For now, let's raise error to be safe, or allow missing graph if logic permits.
            # But Mora expects graph. Let's assume input always has at least one valid SELFIES.
            raise ValueError(f"Graph does not exist in the data for item {i}, but the model is multimodal")
            
        return data_dict
