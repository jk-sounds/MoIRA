from .lazy_supervised_dataset import LazySupervisedDataset, LazySupervisedGraphDataset
from .reagent_pred_dataset import ReagentPredSupervisedGraphDataset
from .forward_pred_dataset import ForwardPredSupervisedGraphDataset
from .retrosynthesis_dataset import RetrosynthesisSupervisedGraphDataset
from .property_pred_dataset import PropertyPredSupervisedGraphDataset
from .collators import DataCollatorForSupervisedDataset, GraphDataCollatorForSupervisedDataset
from .MoleculeNet_classification_dataset import MoleculeNetSupervisedGraphDataset
from .reagent_pred_dataset import ReagentPredSupervisedGraphDataset
from .catalyst_pred_dataset import CatalystPredSupervisedGraphDataset
from .solvent_pred_dataset import SolventPredSupervisedGraphDataset
from .molcap_dataset import MolcapSupervisedGraphDataset
from .yield_regression_dataset import YieldRegressionSupervisedGraphDataset
from .exp_procedure_pred_dataset import ExpProcedurePredSupervisedGraphDataset
from .molecular_weight_dataset import MolecularWeightSupervisedGraphDataset
from .unified_mora_dataset import UnifiedMoraDataset
from torch.utils.data import ConcatDataset


def build_dataset(tokenizer, data_args):
    data_type = data_args.data_type
    if data_type == "supervised":
        dataset = LazySupervisedGraphDataset(
            data_path=data_args.data_path,
            tokenizer=tokenizer,
            data_args=data_args,
        )
    elif data_type == "unified_mora":
        dataset = UnifiedMoraDataset(
            data_path=data_args.data_path,
            tokenizer=tokenizer,
            data_args=data_args,
        )
    elif data_type == "yield_regression":
        dataset = YieldRegressionSupervisedGraphDataset(
            data_path=data_args.data_path,
            tokenizer=tokenizer,
            data_args=data_args,
        )
    elif data_type == "exp_procedure_pred":
        dataset = ExpProcedurePredSupervisedGraphDataset(
            data_path=data_args.data_path,
            tokenizer=tokenizer,
            data_args=data_args,
        )
    elif data_type == "Molecular_Weight":
        dataset = MolecularWeightSupervisedGraphDataset(
            data_path=data_args.data_path,
            tokenizer=tokenizer,
            data_args=data_args,
        )
    elif data_type == "reagent_pred":
        dataset = ReagentPredSupervisedGraphDataset(
            data_path=data_args.data_path,
            tokenizer=tokenizer,
            data_args=data_args,
        )
    elif data_type == "catalyst_pred":
        dataset = CatalystPredSupervisedGraphDataset(
            data_path=data_args.data_path,
            tokenizer=tokenizer,
            data_args=data_args,
        )
    elif data_type == "solvent_pred":
        dataset = SolventPredSupervisedGraphDataset(
            data_path=data_args.data_path,
            tokenizer=tokenizer,
            data_args=data_args,
        )
    elif data_type == "forward_pred":
        dataset = ForwardPredSupervisedGraphDataset(
            data_path=data_args.data_path,
            tokenizer=tokenizer,
            data_args=data_args,
        )
    elif data_type == "retrosynthesis":
        dataset = RetrosynthesisSupervisedGraphDataset(
            data_path=data_args.data_path,
            tokenizer=tokenizer,
            data_args=data_args,
        )
    elif data_type == "property_pred":
        dataset = PropertyPredSupervisedGraphDataset(
            data_path=data_args.data_path,
            tokenizer=tokenizer,
            data_args=data_args,
        )
    elif data_type == "molcap":
        dataset = MolcapSupervisedGraphDataset(
            data_path=data_args.data_path,
            tokenizer=tokenizer,
            data_args=data_args,
        )
    elif data_type == "all":
        # combine molcap, reagent_pred, forward_pred, retrosynthesis, property_pred
        # hard code for data path
        molcap_data = LazySupervisedGraphDataset(
            data_path=data_args.data_path,
            tokenizer=tokenizer,
            data_args=data_args,
        )
        reagent_pred_data = ReagentPredSupervisedGraphDataset(
            data_path=data_args.data_path,
            tokenizer=tokenizer,
            data_args=data_args,
        )
        forward_pred_data = ForwardPredSupervisedGraphDataset(
            data_path=data_args.data_path,
            tokenizer=tokenizer,
            data_args=data_args,
        )
        retrosynthesis_data = RetrosynthesisSupervisedGraphDataset(
            data_path=data_args.data_path,
            tokenizer=tokenizer,
            data_args=data_args,
        )
        property_pred_data = PropertyPredSupervisedGraphDataset(
            data_path=data_args.data_path,
            tokenizer=tokenizer,
            data_args=data_args,
        )
        dataset = ConcatDataset([molcap_data, reagent_pred_data, forward_pred_data, retrosynthesis_data, property_pred_data])
    elif data_type == "MoleculeNet":
        dataset = MoleculeNetSupervisedGraphDataset(
            data_path=data_args.data_path,
            tokenizer=tokenizer,
            data_args=data_args,
        )
    else:
        raise NotImplementedError(f"Unknown data type: {data_type}")
    return dataset