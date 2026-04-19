from typing import cast

import torch
from transformers import AutoTokenizer, TokenizersBackend

from olmo_tap.constants import WEIGHTS_DIR
from olmo_tap.experiments.utils.config import HydraLoRAConfig
from olmo_tap.experiments.utils.model_builder import build_base_model
from olmo_tap.hydra import HydraTransformer

MODEL_NAME = "OLMo2-7B (base)"


def load_model(
    device: str = "cuda",
) -> tuple[HydraTransformer, TokenizersBackend] | tuple[None, None]:
    tokenizer = AutoTokenizer.from_pretrained(WEIGHTS_DIR)
    if not isinstance(tokenizer, TokenizersBackend):
        return None, None

    config = HydraLoRAConfig(device=device)
    model = build_base_model(config)
    model.eval()

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
    max_seq_len = input_ids.shape[1] + max_new_tokens

    model.init_kv_cache(batch_size=1, max_seq_len=max_seq_len)

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
