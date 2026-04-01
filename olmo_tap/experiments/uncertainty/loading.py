import torch

from olmo_tap.experiments.utils.config import ExperimentConfig
from olmo_tap.experiments.utils.model_builder import build_finetuning_model
from olmo_tap.hydra import HydraTransformer


def load_for_inference(
    checkpoint_path: str, exp_config: ExperimentConfig
) -> HydraTransformer:
    model = build_finetuning_model(exp_config.model)

    state = torch.load(checkpoint_path, map_location=exp_config.model.device)
    model.heads[0].load_state_dict(state)
    model.heads[0] = model.heads[0].merge_and_unload()

    model.eval()
    return model
