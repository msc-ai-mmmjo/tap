"""
Gradio Chat Interface for LLM Uncertainty Visualization
========================================================

Chat-based interface using gr.ChatInterface for LLM inference
with toggleable uncertainty heatmap visualization.

Usage:
    To run this application locally:
    $ gradio interface.py

    To run on a specific port:
    $ gradio interface.py --port 8080
"""

import math
from collections.abc import Generator

import gradio as gr
from huggingface_hub import InferenceClient

# Types
HeatmapData = list[tuple[str, float | str | None]]

# Constants
MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"


def chat_with_uncertainty(
    message: str, history: list[dict], hf_token: str
) -> Generator[tuple[str, HeatmapData], None, None]:
    if not hf_token or not hf_token.strip():
        yield "Please enter a HuggingFace Token.", []
        return

    try:
        client = InferenceClient(model=MODEL, token=hf_token)
    except Exception as e:
        yield f"Client Error: {e}", []
        return

    # Build messages from history + current message
    messages = []
    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": message})

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

            # Yield both the visible text and the heatmap data
            yield partial_text, heatmap_data

    except Exception as e:
        yield (
            f"API Error: {e}\n\n*Tip: Check if your token is valid and has 'Read' permissions.*",
            [],
        )


# Build the interface
with gr.Blocks(title="LLM Uncertainty Visualizer") as demo:
    gr.Markdown("<center><h1>LLM Uncertainty Visualizer</h1></center>")
    gr.Markdown(
        "<center>Chat with an LLM and visualize token-level uncertainty</center>"
    )

    # Heatmap display component (initially hidden)
    heatmap_display = gr.HighlightedText(
        label="Uncertainty Heatmap",
        combine_adjacent=False,
        show_legend=True,
        visible=False,
        color_map={
            "Certain": "#cbf0cc",
            "Uncertain": "#ffeea8",
            "High Entropy": "#ffc4c4",
        },
    )

    # Toggle checkbox for heatmap visibility
    show_heatmap = gr.Checkbox(label="Show Uncertainty Heatmap", value=False)

    # Chat interface
    chat = gr.ChatInterface(
        fn=chat_with_uncertainty,
        additional_inputs=[
            gr.Textbox(
                label="HuggingFace Token",
                type="password",
                placeholder="Paste token (hf_...)",
            ),
        ],
        additional_inputs_accordion="Settings",
        additional_outputs=[heatmap_display],
    )

    # Wire toggle to show/hide heatmap
    show_heatmap.change(
        fn=lambda visible: gr.HighlightedText(visible=visible),
        inputs=[show_heatmap],
        outputs=[heatmap_display],
    )


# Launch
if __name__ == "__main__":
    demo.launch(share=True)
