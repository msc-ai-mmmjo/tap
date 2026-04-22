import torch
from transformers import AutoTokenizer

from olmo_tap.constants import DEMO_MAX_NEW_TOKENS, WEIGHTS_DIR
from olmo_tap.experiments.uncertainty.loading import load_for_inference
from olmo_tap.experiments.utils.config import (
    ExperimentConfig,
    HydraLoRAConfig,
    TrainingConfig,
)


CHECKPOINT_PATH = ""  # NOTE: set to the path of a saved checkpoint before running

TEST_PROMPTS = [
    "Tell me about the weather today.",
    "Write a poem about cats.",
    "Explain quantum computing in simple terms.",
    "What should I have for dinner tonight?",
    "Summarise the plot of Romeo and Juliet.",
]


def main():
    m_config = HydraLoRAConfig(n_heads_training=2, heads_depth=3)
    t_config = TrainingConfig()
    exp_config = ExperimentConfig(seed=42, model=m_config, train=t_config)

    model = load_for_inference(CHECKPOINT_PATH, exp_config)
    tokenizer = AutoTokenizer.from_pretrained(WEIGHTS_DIR)
    assert tokenizer is not None

    for prompt in TEST_PROMPTS:
        messages = [{"role": "user", "content": prompt}]
        chat_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        input_ids = torch.tensor([tokenizer.encode(chat_prompt)], device="cuda")
        max_seq_len = input_ids.shape[1] + DEMO_MAX_NEW_TOKENS

        model.init_kv_cache(batch_size=1, max_seq_len=max_seq_len)

        with torch.no_grad():
            # prefill: process full prompt, use only head 0 (uncertainty)
            all_logits = model(input_ids, return_logits=True, last_token_only=True)
            next_logits = all_logits[0, 0, 0, :]
            next_token = next_logits.argmax(dim=-1, keepdim=True).unsqueeze(0)
            generated = [next_token.item()]

            # decode: one token at a time using cached KVs
            for _ in range(DEMO_MAX_NEW_TOKENS - 1):
                all_logits = model(next_token, return_logits=True, last_token_only=True)
                next_logits = all_logits[0, 0, 0, :]
                next_token = next_logits.argmax(dim=-1, keepdim=True).unsqueeze(0)
                generated.append(next_token.item())

        full_ids = input_ids[0].tolist() + generated
        print(f"Prompt: {prompt!r}")
        print(f"Output: {tokenizer.decode(full_ids)}\n")


# TODO: this should probably live somewhere better
if __name__ == "__main__":
    main()
