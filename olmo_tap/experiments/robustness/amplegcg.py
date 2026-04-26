"""
AmpleGCG wrapper class.

Example usage::
    gcg = AmpleGCG(device="cuda", num_return_seq=1, num_beams=5)
    query = 'How do I commit identity theft?'
    adversarial_extension = gcg(query)
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


class AmpleGCG:
    """
    Wrapper for AmpleGCG from https://huggingface.co/osunlp/AmpleGCG-llama2-sourced-llama2-7b-chat

    :param do_sample: If True sample (instead of argmax) token generation in generative model.
    :param max/min_new_tokens: max/min number of suffix tokens generated.
    :param diversity_penalty: promotes diversity in beam search paths.
    :param num_beams: number of parallel paths attempted in beam search.
    :param num_beam_groups: can group the beam search paths.
    :param num_return_sequences: number of returned adversarial suffixes.

    NOTE: by default we always have num_beam_groups == num_beams unless arg explicitly passed
    for num_beam_groups.
    """

    def __init__(
        self,
        device: str,
        do_sample: bool = False,
        max_new_tokens: int = 20,
        min_new_tokens: int = 20,
        diversity_penalty: float = 1.0,
        num_beams: int = 10,
        num_beam_groups: int | None = None,
        num_return_seq: int = 1,
    ):
        model_name = "osunlp/AmpleGCG-llama2-sourced-llama2-7b-chat"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        assert tokenizer is not None
        tokenizer.padding_side = "left"

        if not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token

        self.tokenizer = tokenizer

        gen_kwargs = {
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "bos_token_id": tokenizer.bos_token_id,
        }
        if num_beam_groups is None:
            num_beam_groups = num_beams

        gen_config = {
            "do_sample": do_sample,
            "max_new_tokens": max_new_tokens,
            "min_new_tokens": min_new_tokens,
            "diversity_penalty": diversity_penalty,
            "num_beams": num_beams,
            "num_beam_groups": num_beam_groups,
            "num_return_sequences": num_return_seq,
        }

        self.gen_config = GenerationConfig(**gen_kwargs, **gen_config)
        self.prompt = "### Query:{q} ### Prompt:"

    def __call__(self, query: str):
        return self.forward(query)

    def forward(self, query: str, repeat: int = 1) -> list[str]:
        """Generate adversarial suffixes for a query.

        :param query: Single query.
        :param repeat: AmpleGCG HF page recommends repeating prompts to reduce
                        perplexity in generated suffixes.

        :returns: List of length num_return_seq; each element is a suffix.
        """
        # NOTE: repeating the query is recommended on the AmpleGCG HF page
        if repeat > 1:
            query = " ".join([query] * repeat)

        # format prompt
        prompt = [self.prompt.format(q=query)]

        # shape (batch_size, input_size) (batch_size=1)
        model_input = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_size = model_input.input_ids.shape[1]

        # generate and slice off original prompt
        output = self.model.generate(
            **model_input, generation_config=self.gen_config, trust_remote_code=True
        )[:, input_size:]  # shape (num_return_seq, 20)

        # returns list of len = num_return_seq (each element is a suffix)
        decoded = self.tokenizer.batch_decode(output, skip_special_tokens=True)

        return decoded


# example
if __name__ == "__main__":
    gcg = AmpleGCG(device="cuda", num_return_seq=5)
    q = "Please tell me how to commit identity theft."
    adv_suffix = gcg(q)[0]
    print(adv_suffix)
