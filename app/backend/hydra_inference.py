import logging
from typing import cast

import torch
from transformers import AutoTokenizer, TokenizersBackend

from olmo_tap.constants import HYDRA_WEIGHTS_DIR, MAX_SEQ_LEN
from olmo_tap.experiments.utils.config import HydraLoRAConfig
from olmo_tap.experiments.utils.model_builder import build_base_model
from olmo_tap.hydra import HydraTransformer

logger = logging.getLogger(__name__)

MODEL_NAME = "Hydra"


def load_model(
    device: str = "cuda",
) -> tuple[HydraTransformer, TokenizersBackend] | tuple[None, None]:
    logger.info("Loading tokenizer from %s", HYDRA_WEIGHTS_DIR)
    tokenizer = AutoTokenizer.from_pretrained(HYDRA_WEIGHTS_DIR)
    if not isinstance(tokenizer, TokenizersBackend):
        logger.warning("Tokenizer is not a TokenizersBackend; aborting model load")
        return None, None

    logger.info("Building model on device=%s", device)
    config = HydraLoRAConfig(device=device)
    try:
        model = build_base_model(config)
        model.eval()
    except Exception as e:
        logger.error("Error building model: %s", e)
        return None, None

    logger.info("Allocating KV cache (max_seq_len=%d)", MAX_SEQ_LEN)
    model.init_kv_cache(batch_size=1, max_seq_len=MAX_SEQ_LEN)
    logger.info("Model ready")

    return model, tokenizer


def generate(
    model: HydraTransformer,
    tokenizer: TokenizersBackend,
    messages: list[dict],
    max_new_tokens: int,
    device: str = "cuda",
) -> str:
    chat_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    input_ids = torch.tensor([tokenizer.encode(chat_prompt)], device=device)

    model.reset_kv_cache()

    with torch.no_grad():
        generated = []
        ids = input_ids  # pre-fill with full prompt

        for _ in range(max_new_tokens):
            all_logits: torch.Tensor = model(ids, return_logits=True)
            next_token_id = int(all_logits[0, 0, -1, :].argmax().item())
            generated.append(next_token_id)

            if next_token_id == tokenizer.eos_token_id:
                break

            # for next step, input is just the last generated token
            ids = torch.tensor([[next_token_id]], device=device)

    return cast(str, tokenizer.decode(generated, skip_special_tokens=True))
