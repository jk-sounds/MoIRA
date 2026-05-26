# Efficient Molecular Graph-Language Modeling via Adaptive Weight Modulation

## 📖 Overview

**MoIRA** (Molecule-Graph Guided Adaptive Weight Modulation Framework) is a novel multimodal molecular framework that fundamentally shifts molecular graph integration from **Input Space Fusion** to **Parameter Space Modulation**[cite: 3803, 3940].

Unlike existing methods that serialize complex molecular topologies into continuous embedding sequences (which flattens structured connectivity and inflates the context window), MoIRA treats molecular structures as modulatory signals. It converts graph-derived structural information into dynamic, instance-specific low-rank weight adaptations ($\Delta W$) that are injected into a frozen Large Language Model (LLM). This paradigm decouples structural modeling from textual processing, allowing molecular topology to modulate the model's internal computation directly.

### 🌟 Key Features

* [cite_start]**Parameter Space Integration**: Absorbs molecular topology via molecule-aware weight adaptation rather than prepending graph-derived tokens to textual inputs [cite: 3932][cite_start], preserving the raw textual instruction stream[cite: 3939].
* [cite_start]**Scalable Context Efficiency**: Bypasses the quadratic computational complexity ($\mathcal{O}(N^{2})$) bottleneck of the LLM self-attention mechanism [cite: 3818][cite_start], maintaining a minimal correlation between molecular atom counts and inference GFLOPs[cite: 4466, 4467].
* [cite_start]**Coordinated Structural Modulation**: Transforms features into layer-wise low-rank parameter factorizations [cite: 4195][cite_start], updating both Self-Attention projection matrices ($W_q, W_k, W_v, W_o$) and the Feed-Forward Network ($W_f$)[cite: 4195, 4196].
* [cite_start]**Syntactic Robustness**: Utilizes **SELFIES** strings instead of SMILES as the task-required target text molecular representation to guarantee syntactic validity during autoregressive decoding[cite: 5095, 5097].
* [cite_start]**Unified Chemical Reasoning**: Achieves robust performance across **11 molecular benchmarks** spanning Mol2Mol (Structural Reasoning), Mol2Text (Cross-modal Translation), and Mol2Num (Quantitative Reasoning) paradigms[cite: 4317, 4318, 4319].

---
## 🏗️ Architecture Design

### Core Mechanism: Dual-Stream Processing

MoIRA strictly separates parameter modulation from textual interaction:

1. **Stream A (Parameter Modulation)**: The molecular graph is processed by a frozen GNN and the **Adaptive Weight Generator** to produce low-rank adaptation weights ().
2. **Stream B (Textual Interaction)**: The user instruction enters the LLM as standard text. The LLM processes this text using the *modulated* weights, implicitly reasoning about the molecule.

### Component Details

* **Graph Encoder**: Frozen **MoleculeSTM** (300-dim embedding).
* **LLM Backbone**: Frozen **Vicuna v1.5-7B**.
* **Adaptive Weight Generator**:
* **Cross-Attention Distillation**: 8 decoder blocks with 4 learnable molecular queries.
* **Low-Rank Projection**: Generates updates with rank .


## 🚀 Quick Start

### Requirements

* Python 3.8+
* PyTorch 1.12+
* CUDA 11.0+ (8x A800 GPUs recommended for full replication)
* **Vicuna v1.5-7B** checkpoints
* **MoleculeSTM** pretrained graph encoder

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
