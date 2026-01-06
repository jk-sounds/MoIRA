import argparse
import torch
import os
import sys
sys.path.append('/home/zhangxiaohong/yintaoo/mora')
import json
import time
import random
import re
from typing import Generator, Dict, List
import selfies
from rdkit import Chem

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria, MM_ENCODER_CFG
from llava.datasets.smiles2graph import smiles2graph
from torch_geometric.data import Data
from transformers import TextStreamer

def extract_first_selfies(text):
    match = re.search(r"\[START_SELFIES\](.*?)\[END_SELFIES\]", text)
    if match:
        return match.group(1)
    return None

def construct_instruct_question(product:str):
    question_pools = [
        'Can you suggest some possible reagents that could have been used in the following chemical reaction?',
        'Give some possible reagents that could have been used in the following chemical reaction.',
        'Please propose potential reagents that might have been utilized in the provided chemical reaction.',
        'Please provide possible reagents based on the following chemical reaction.',
    ]
    question = random.choice(question_pools)
    question += f"\nThe product is {product}"
    return question

def _convert_dict_to_Data(data_dict: Dict) -> Data:
    return Data(
        x=torch.asarray(data_dict['node_feat']),
        edge_attr=torch.asarray(data_dict['edge_feat']),
        edge_index=torch.asarray(data_dict['edge_index']),
    )

def selfies2smiles(selfies_str):
    try:
        smiles_str = selfies.decoder(selfies_str)
    except:
        smiles_str = None
    return smiles_str

def get_mol_metrics(smiles_or_selfies):
    if not smiles_or_selfies:
        return 0, 0
    
    mol = Chem.MolFromSmiles(smiles_or_selfies)
    if mol is None:
        # Try converting from selfies if it looks like selfies (has brackets)
        if '[' in smiles_or_selfies and ']' in smiles_or_selfies:
             try:
                 smi = selfies.decoder(smiles_or_selfies)
                 mol = Chem.MolFromSmiles(smi)
             except:
                 pass
    
    if mol is None:
        return 0, 0
    
    return mol.GetNumAtoms(), mol.GetNumBonds()

class TimeStreamer(TextStreamer):
    def __init__(self, tokenizer, skip_prompt=True, **decode_kwargs):
        super().__init__(tokenizer, skip_prompt, **decode_kwargs)
        self.first_token_time = None
        self.token_count = 0
        
    def put(self, value):
        if self.first_token_time is None:
            # If skip_prompt is True, the first call to put might be the prompt if not handled carefully,
            # but TextStreamer usually handles prompt skipping.
            # However, we want the time when the *first new token* is generated.
            # In generate(), the streamer is called with generated tokens.
            self.first_token_time = time.time()
        
        # value is a tensor of token ids
        if torch.is_tensor(value):
            self.token_count += value.numel()
        else:
            self.token_count += len(value)
            
        super().put(value)

def iterate_test_files(args, batch_size=1) -> Generator:
    with open(args.in_file, "rb") as f:
        list_data_dict = json.load(f)
        
        if args.limit > 0:
            list_data_dict = list_data_dict[:args.limit]

        batch = []
        for i, raw in enumerate(list_data_dict):
            # Logic for reagent_pred
            if args.task == "reagent_pred":
                reactant, product = raw['input'].split(">>")
                graph = smiles2graph(selfies2smiles(reactant))
                if not args.add_selfies:
                    instruction = construct_instruct_question(product)
                else:
                    instruction = raw['instruction'] + f" The reaction is {raw['input']}"
                batch.append((instruction, graph, raw['output']))
            else:
                 # Minimal fallback for other tasks if needed, but we focus on reagent_pred
                graph = None
                instruction = raw['instruction']
                batch.append((instruction, graph, raw['output']))
            
            if len(batch) == batch_size:
                yield zip(*batch)
                batch = []
        if len(batch) > 0:
            yield zip(*batch)

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    disable_torch_init()
    
    model_name = get_model_name_from_path(args.model_path)
    mm_encoder_cfg = MM_ENCODER_CFG(init_checkpoint=args.graph_checkpoint_path)
    mm_encoder_cfg = mm_encoder_cfg.dict()
    
    tokenizer, model, _, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name, 
        args.load_8bit, args.load_4bit, mm_encoder_cfg=mm_encoder_cfg
    )
    
    model = model.to(torch.bfloat16)
    
    results = []
    
    # We force batch_size=1 for accurate timing per sample
    batch_size = 1 
    
    print(f"Starting evaluation with limit={args.limit}...")
    
    for instructions, graphs, gts in iterate_test_files(args, batch_size=batch_size):
        instruction = instructions[0] # batch size is 1
        graph_dict = graphs[0]
        gt = gts[0]
        
        # Prepare Input
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + instruction
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + instruction
        
        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(device)
        
        # Prepare Graph
        if graph_dict is not None:
            graph = _convert_dict_to_Data(graph_dict).to(device)
            graphs_tensor = [graph]
        else:
            graphs_tensor = None

        prompt_length = input_ids.shape[1]
        
        streamer = TimeStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        # Synchronize for timing
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            
        start_time = time.time()
        
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                graphs=graphs_tensor,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                streamer=streamer
            )
            
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.time()
        
        total_time_sec = end_time - start_time
        ttft_sec = (streamer.first_token_time - start_time) if streamer.first_token_time else total_time_sec
        total_time_wo_ttft_sec = total_time_sec - ttft_sec
        
        output_text = tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True).strip()
        
        # Calculate token count from output_ids (excluding prompt)
        token_count = output_ids.shape[1] - input_ids.shape[1]
        
        # Tokens per second (using total_time_wo_ttft_sec as requested)
        # If token_count <= 1, speed is 0 or undefined.
        if total_time_wo_ttft_sec > 0:
             # Usually we count generated tokens excluding the first one for this speed
             # But user said "use total_time_wo_ttft_sec as total time"
             # So numerator should be tokens generated during that time.
             # That is (token_count - 1)
             tokens_per_second = (token_count - 1) / total_time_wo_ttft_sec
        else:
             tokens_per_second = 0
             
        # Molecule Metrics
        mol_atom_count, mol_bond_count = get_mol_metrics(output_text)
        
        metrics = {
            "prompt_length": prompt_length,
            "mol_atom_count": mol_atom_count,
            "mol_bond_count": mol_bond_count,
            "ttft_sec": ttft_sec,
            "total_time_sec": total_time_sec,
            "total_time_wo_ttft_sec": total_time_wo_ttft_sec,
            "token_count": token_count,
            "tokens_per_second": tokens_per_second,
            "output": output_text,
            "gt": gt
        }
        
        results.append(metrics)
        
        if args.debug:
            print(metrics)

    # Save Results
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w') as f:
        for res in results:
            f.write(json.dumps(res) + '\n')
            
    # Calculate Averages
    if results:
        avg_ttft = sum(r['ttft_sec'] for r in results) / len(results)
        avg_total_time = sum(r['total_time_sec'] for r in results) / len(results)
        avg_tokens = sum(r['token_count'] for r in results) / len(results)
        avg_tps = sum(r['tokens_per_second'] for r in results) / len(results)
        avg_atoms = sum(r['mol_atom_count'] for r in results) / len(results)
        
        print("\n" + "="*30)
        print("Average Metrics:")
        print(f"TTFT (s): {avg_ttft:.4f}")
        print(f"Total Time (s): {avg_total_time:.4f}")
        print(f"Token Count: {avg_tokens:.2f}")
        print(f"Tokens/sec (wo TTFT): {avg_tps:.2f}")
        print(f"Avg Atom Count: {avg_atoms:.2f}")
        print("="*30 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="reagent_pred")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--in-file", type=str, required=True)
    parser.add_argument("--output_file", type=str, default="eval_result/efficiency.jsonl")
    parser.add_argument("--graph-checkpoint-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--add-selfies", action="store_true")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--debug", action="store_true")
    
    args = parser.parse_args()
    main(args)
