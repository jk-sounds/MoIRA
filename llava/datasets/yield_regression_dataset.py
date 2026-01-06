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


def construct_instruct_question(product:str):
    """
    Construct instruct question for each graph
    """
    question_pools = [
        'Analyzing the chemical reaction, what is the yield ratio?',
        'Predict the yield of the following chemical reaction.',
        'What is the expected yield for this reaction?',
        'Can you estimate the yield ratio for the provided reaction?',
    ]
    question = random.choice(question_pools)
    # Yield regression typically focuses on the reaction itself, so product info is already in the input string
    # But following solvent_pred pattern, we might want to emphasize product if split.
    # However, the input is Reactant>>Product, so we can just use the prompt.
    return question


class YieldRegressionSupervisedGraphDataset(Dataset):
    """Dataset for Yield Regression"""
    add_selfies = True
    def __init__(self, 
                 data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args,
                ):
        super(YieldRegressionSupervisedGraphDataset, self).__init__()
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
        input_str, output_val = raw['input'], str(raw['output'])
        
        # input: "reactant>>product" (usually, but need to handle split safely)
        if ">>" in input_str:
            reactant, product = input_str.split(">>")
            # convert input selfies to smiles for building graph. 
            # Usually graph is built from reactants in Mora.
            reactant_smiles = self.selfies2smiles(reactant)
        else:
            reactant_smiles = self.selfies2smiles(input_str)
            
        if not self.add_selfies:
             instruction = raw['instruction']
        else:
            instruction = raw['instruction'] + f" The reaction is {input_str}"
            
        if random.random() < 0.5:
            instruction = "<image>\n" + instruction
        else:
            instruction = instruction + "\n<image>"
            
        graph = smiles2graph(reactant_smiles)
        
        sources = dict(
            conversations=[
                {"from": "human", "value": instruction},
                {"from": "gpt", "value": output_val}
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
            raise ValueError("Graph does not exist in the data, but the model is multimodal")
        return data_dict
