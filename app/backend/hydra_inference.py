import logging
import time
from typing import cast

import torch
from transformers import AutoTokenizer, TokenizersBackend

from app.backend.constants import MCQ_PROB_THRESHOLD
from olmo_tap.constants import MAX_SEQ_LEN, WEIGHTS_DIR
from olmo_tap.experiments.utils.config import HydraLoRAConfig
from olmo_tap.experiments.utils.model_builder import build_base_model
from olmo_tap.hydra import HydraTransformer

logger = logging.getLogger(__name__)

MODEL_NAME = "Hydra"


def load_hydra(
    device: str = "cuda",
) -> tuple[HydraTransformer, TokenizersBackend] | tuple[None, None]:
    t0 = time.perf_counter()

    if not WEIGHTS_DIR:
        logger.warning("WEIGHTS_DIR not set; skipping model load")
        return None, None

    logger.info("Loading tokenizer from %s", WEIGHTS_DIR)
    tokenizer = AutoTokenizer.from_pretrained(WEIGHTS_DIR)
    if not isinstance(tokenizer, TokenizersBackend):
        logger.error("Tokenizer is not a TokenizersBackend; aborting model load")
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

    logger.info("Model ready -- setup took %.2fs", time.perf_counter() - t0)
    return model, tokenizer


def generate(
    model: HydraTransformer,
    tokenizer: TokenizersBackend,
    messages: list[dict],
    max_new_tokens: int,
    device: str = "cuda",
    important_token_ids: dict[str, int] | None = None,
) -> tuple[str, bool | None]:
    chat_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    input_ids = torch.tensor([tokenizer.encode(chat_prompt)], device=device)

    model.reset_kv_cache()

    t0 = time.perf_counter()
    is_mcq = None

    with torch.no_grad():
        generated = []
        ids = input_ids  # pre-fill with full prompt

        for t in range(max_new_tokens):
            logits: torch.Tensor = model(ids, return_logits=True)[0, 0, -1, :]

            if t == 0 and important_token_ids is not None:
                first_token_logits = logits
                mcq_prob = (
                    first_token_logits[list(important_token_ids.values())].sum().item()
                )
                is_mcq = mcq_prob > MCQ_PROB_THRESHOLD  # Adjust threshold as needed

            next_token_id = int(logits.argmax().item())
            generated.append(next_token_id)

            if next_token_id == tokenizer.eos_token_id:
                break

            # for next step, input is just the last generated token
            ids = torch.tensor([[next_token_id]], device=device)

    logger.info(
        "Generated %d tokens in %.2fs", len(generated), time.perf_counter() - t0
    )
    return cast(str, tokenizer.decode(generated, skip_special_tokens=True)), is_mcq
