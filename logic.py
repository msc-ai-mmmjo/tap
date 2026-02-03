"""
Backend business logic, separated from the Gradio interface.
"""

from collections.abc import Generator
import math

from huggingface_hub import InferenceClient

from constants import HeatmapData


def inference(
    message: str, history: list[dict], hf_token: str, model: str, model_name: str
) -> Generator[str, None, None]:
    """Generates text and calculates uncertainties."""
    if not hf_token or not hf_token.strip():
        yield "Please enter a Hugging Face Token."  # , []
        return

    try:
        client = InferenceClient(model=model, token=hf_token)
    except Exception as e:
        yield f"Client Error: {e}"  # , []
        return

    # Prepare the message for the model
    messages = [{"role": "user", "content": message}]

    # Build the response incrementally for the streaming effect
    partial_text = ""

    # Initialize (token, label) tuple for the probability heatmap
    heatmap_data: HeatmapData = []

    try:
        # API Call
        stream = client.chat_completion(
            messages,
            max_tokens=500,
            stream=True,
            logprobs=True,  # request token probability data
            top_logprobs=1,  # we only need the top choice
        )

        for chunk in stream:
            # 1. Process text
            try:
                new_content = chunk.choices[0].delta.content
                if new_content:
                    partial_text += new_content
            except (AttributeError, IndexError):
                pass

            # Process logprobs (one token per streaming chunk)
            try:
                logprobs = chunk.choices[0].logprobs
                if logprobs is None or logprobs.content is None:
                    continue
                lp = logprobs.content[0]
                prob = math.exp(lp.logprob)

                if prob > 0.9:
                    label = "Certain"
                elif prob > 0.6:
                    label = "Uncertain"
                else:
                    label = "High Entropy"

                heatmap_data.append((lp.token, label))
            except (AttributeError, IndexError, TypeError):
                pass

            # Yield both the visible text and the hidden probability data
            yield f"**{model_name} Response:**\n\n{partial_text}"  # , heatmap_data

    except Exception as e:
        yield f"API Error: {e}\n\n*Tip: Check if your token is valid and has 'Read' permissions.*"
