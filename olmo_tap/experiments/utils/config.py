"""
Config classes for finetuning (robustness or uncertainty)

NOTE: HydraLoRAConfig.n_heads_final is for book-keeping the number of final intended heads
We check in the post_init of the parent ExperimentConfig that num_shards = n_heads_final

We expect the final Hydra to look something like:
- 9 Robustness + Security finetuned heads
- 1 Uncertainty finetuned head
"""

from dataclasses import dataclass, field
from olmo_tap.constants import LORA_TARGETS, VOCAB_SIZE, WEIGHTS_DIR


@dataclass
class HydraLoRAConfig:
    # architecture
    weights_dir: str = WEIGHTS_DIR
    model_size: str = "7b"  # "1b" or "7b"
    n_heads_final: int = 5
    n_heads_training: int = 1  # number of heads instantiated in training
    heads_depth: int = 3
    vocab_size: int = VOCAB_SIZE

    # LoRA hyperparameters
    lora_r: int = 16
    lora_alpha: int = 32
    target_modules: list[str] = field(default_factory=lambda: LORA_TARGETS)
    device: str = "cuda"


@dataclass
class TrainingConfig:
    # optimizer hyperparams
    learning_rate: float = 1e-4
    batch_size: int = 16
    num_epochs: int = 1  # GPU poor :(

    # max generated sequence length
    max_seq_len: int = 256
    num_workers: int = (
        4  # DataLoader workers for CPU-side preprocessing in parallel with GPU
    )

    # which head finetunes on which shard
    shard_id: int = 0
    num_shards: int = field(init=False)

    # required for tokenizer
    weights_dir: str = WEIGHTS_DIR

    # LR schedule
    warmup_steps: int = 100
    lr_schedule: str = "cosine"  # "cosine" or "linear"

    # checkpointing
    output_dir: str = "experiments/uncertainty/outputs"
    checkpoint_every_n_steps: int = 250

    # seed (propagated from ExperimentConfig)
    seed: int = field(init=False)

    # token IDs
    # convention: A/B used for correct/incorrect in uncertainty
    A_token_id: int = field(init=False)
    B_token_id: int = field(init=False)
    C_token_id: int = field(init=False)
    D_token_id: int = field(init=False)


@dataclass
class ExperimentConfig:
    # random seed for experiment tracking
    # NOTE: no default value to avoid disagreements
    seed: int

    model: HydraLoRAConfig = field(default_factory=HydraLoRAConfig)
    train: TrainingConfig = field(default_factory=TrainingConfig)

    # W&B
    wandb_project: str = "hydra"
    wandb_run_name: str | None = None

    device: str = "cuda"

    def __post_init__(self):
        # ensure num_shards = n_heads
        self.train.num_shards = self.model.n_heads_final
        self.train.seed = self.seed
