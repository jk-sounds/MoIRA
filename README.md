# Efficient Molecular Graph-Language Modeling via Adaptive Weight Modulation

### 🌟 Key Features

* **Parameter Space Integration**: Absorbs molecular topology via molecule-aware weight adaptation rather than prepending graph-derived tokens to textual inputs, preserving the raw textual instruction stream.
* **Scalable Context Efficiency**: Bypasses the quadratic computational complexity ($\mathcal{O}(N^{2})$) bottleneck of the LLM self-attention mechanism, maintaining a minimal correlation between molecular atom counts and inference GFLOPs.
* **Coordinated Structural Modulation**: Transforms features into layer-wise low-rank parameter factorizations, updating both Self-Attention projection matrices ($W_q, W_k, W_v, W_o$) and the Feed-Forward Network ($W_f$).
* **Syntactic Robustness**: Utilizes **SELFIES** strings instead of SMILES as the task-required target text molecular representation to guarantee syntactic validity during autoregressive decoding.
* **Unified Chemical Reasoning**: Achieves robust performance across **11 molecular benchmarks** spanning Mol2Mol (Structural Reasoning), Mol2Text (Cross-modal Translation), and Mol2Num (Quantitative Reasoning) paradigms.

---
## 🏗️ Architecture Design

### Core Mechanism: Dual-Stream Processing

MoIRA strictly separates parameter modulation from textual interaction:

1. **Stream A (Parameter Modulation)**: The molecular graph is processed by a frozen GNN and the **Adaptive Weight Generator** to produce low-rank adaptation weights ().
2. **Stream B (Textual Interaction)**: The user instruction enters the LLM as standard text. The LLM processes this text using the *modulated* weights, implicitly reasoning about the molecule.

### Component Details

* **Graph Encoder**: MoleculeSTM.
* **LLM Backbone**: Vicuna v1.5-7B.
* **Adaptive Weight Generator**: Cross-Attention Distillation and Parameter Projection.

## 🚀 Quick Start

### Requirements

* Python 3.8+
* PyTorch 1.12+
* CUDA 11.0+ 
* Vicuna v1.5-7B 
* MoleculeSTM

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Data Preparation

Following prior work, we utilize diverse datasets across three scientific paradigms. The dataset can be downloaded from https://huggingface.co/datasets/TY123456/molra/tree/main. Please download and place them in the `data/` folder: 

* **Pre-training**: PubChem (Molecule-Text pairs).
* **Mol2Mol**: Reaction, Retrosynthesis, Reagent).
* **Mol2Text**: ChEBI-20, PubChemQA.
* **Mol2Num**: QM9, YieldBERT datasets.

### Model Training

MolRA uses a two-stage training pipeline: **Stage 1 (Alignment)** and **Stage 2 (Instruction Tuning)**.

#### 1. Forward Reaction Prediction (Mol2Mol)

```bash
bash scripts/finetune_MolRA_forward_pred.sh

```

#### 2. Retrosynthesis Prediction (Mol2Mol)

```bash
bash scripts/finetune_MolRA_retrosynthesis.sh

```

#### 3. Molecular Captioning (Mol2Text)

```bash
bash scripts/finetune_MolRA_molcap.sh

```


*(Note: Ensure you configure the correct `task_type` in the scripts: `mol2mol`, `mol2text`, or `mol2num`)*

### Model Evaluation

```bash
# Evaluate on all 11 benchmarks
bash scripts/eval_all_tasks.sh

# Specific task evaluation
bash scripts/eval/eval_forward_reaction.sh
bash scripts/eval/eval_retrosynthesis.sh
bash scripts/eval/eval_property_regression.sh

```

## 📊 Supported Tasks & Paradigms

MoIRA unifies 11 tasks into a single framework:

### Paradigm I: Mol2Mol (Structural Reasoning)

* **Forward Reaction Prediction**: Reactants  Product (Exact Match & Validity)
* **Retrosynthesis**: Product  Reactants
* **Reagent Prediction**: Reactants + Product  Reagents
* **Catalyst Prediction**: Reactants + Product  Catalyst
* **Solvent Prediction**: Reaction  Solvent

### Paradigm II: Mol2Text (Cross-modal Translation)

* **Molecular Captioning**: Generating descriptions from graphs.
* **Description Q&A**: Answering open-ended questions.
* **Experimental Procedure**: Generating step-by-step lab recipes.

### Paradigm III: Mol2Num (Quantitative Reasoning)

* **QM9 Property Prediction**: HOMO, LUMO, Gap energy (Regression).
* **Yield Prediction**: Predicting reaction efficiency ratios.

## 🔧 Configuration

Default hyperparameters based on the paper's Appendix A3:

```python
# MoIRA Model Configuration
model_config = {
    MolRA_dim: 512              # Hidden dimension of weight generator
    MolRA_depth: 2              # Number of layers in weight generator
    MolRA_pos_num: 256          # Number of positional encodings
    MolRA_llm_dim: 4096         # LLM hidden dimension
    MolRA_llm_depth: 32         # Number of LLM layers
    MolRA_rank: 64              # MolRA rank
    MolRA_type: "qkvom"         # Attention components to adapt
    MolRA_alpha: 64             # MolRA scaling factor
    weights_sep: True          # Whether to separate weight generation
    skip_layers: 4             # Number of layers to skip
}

```
