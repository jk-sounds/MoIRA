import os
import random
import json
import copy
import pickle
from typing import Dict, Optional, Sequence, List
import selfies
import torch
import numpy as np
from torch.utils.data import Dataset
import transformers
from .preprocess import preprocess, preprocess_multimodal
from .smiles2graph import smiles2graph

class UnifiedMoraDataset(Dataset):
    """Unified Dataset for Mora that handles multiple task types from a merged JSON."""
    add_selfies = True
    
    def __init__(self, 
                 data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args,
                ):
        super(UnifiedMoraDataset, self).__init__()
        with open(data_path, "r") as f:
            self.list_data_dict = json.load(f)

        self.tokenizer = tokenizer
        self.data_args = data_args
        
    def selfies2smiles(self, selfies_str):
        try:
            smiles_str = selfies.decoder(selfies_str)
        except:
            smiles_str = None
        return smiles_str

    def get_dummy_graph(self):
        # Create a dummy graph with 1 node (wildcard) and 0 edges
        # Matching structure from smiles2graph
        x = np.array([[118, 0]], dtype=np.int64) # 118 is wildcard, 0 is unspecified chiral
        edge_index = np.empty((2, 0), dtype=np.int64)
        edge_attr = np.empty((0, 2), dtype=np.int64)
        
        graph = dict()
        graph['edge_index'] = edge_index
        graph['edge_feat'] = edge_attr
        graph['node_feat'] = x
        graph['num_nodes'] = 1
        return graph

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        raw = self.list_data_dict[i]
        task_source = raw.get('metadata', {}).get('task_source', '')
        
        instruction = raw['instruction']
        input_str = raw['input']
        output_str = str(raw['output'])
        
        graph = None
        has_real_graph = False

        # Logic dispatch based on task_source
        if task_source in ['solvent_pred', 'reagent_prediction', 'catalyst_pred']:
            # Pattern 1: Reactant >> Product
            parts = input_str.split(">>")
            if len(parts) >= 1:
                smi = self.selfies2smiles(parts[0])
                if smi:
                    graph = smiles2graph(smi)
            
            if self.add_selfies:
                instruction += f" The reaction is {input_str}"
        
        elif task_source == 'forward_prediction':
            # Pattern 2: Reactants separated by '.'
            parts = input_str.split('.')
            if len(parts) >= 1:
                smi = self.selfies2smiles(parts[0])
                if smi:
                    graph = smiles2graph(smi)
            
            if self.add_selfies:
                 instruction += f" {input_str}"

        elif task_source in ['IUPAC2SELFIES', 'text_guided_mol_generation', 'DescriptionQA', 'exp_procedure_pred']:
            # Text tasks. Try to parse if input looks like SELFIES, otherwise skip graph.
            if task_source == 'DescriptionQA' and '[' in input_str and ']' in input_str:
                 smi = self.selfies2smiles(input_str)
                 if smi:
                     graph = smiles2graph(smi)
            # else graph remains None
            
        else:
            # Pattern 3: PropertyPred, Retrosynthesis, TPSA, LogP, etc.
            smi = self.selfies2smiles(input_str)
            if smi:
                graph = smiles2graph(smi)
            
            if self.add_selfies:
                if task_source == 'retrosynthesis':
                    instruction += f" The product is: {input_str}"
                elif task_source == 'molcap':
                    instruction += f" The molecule is: {input_str}"
                elif task_source == 'molecule_editing':
                    instruction += f" The input molecule is: {input_str}"
                else:
                    instruction += f" The compound SELFIES sequence is: {input_str}"

        if graph is not None:
            has_real_graph = True
        else:
            # Use dummy graph to satisfy collator
            graph = self.get_dummy_graph()
            has_real_graph = False

        # Add image token only if we have a real graph
        if has_real_graph:
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
        sources = [sources] # Wrap in list as expected by preprocess

        if has_real_graph:
             sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]), self.data_args)
        else:
             sources = copy.deepcopy([e["conversations"] for e in sources])
             
        data_dict = preprocess(sources, self.tokenizer, has_image=has_real_graph)
        
        data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0])
        
        # Always return graph (dummy or real) so collator doesn't crash
        data_dict['graph'] = graph
            
        return data_dict
