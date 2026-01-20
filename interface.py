"""
Gradio Web Interface Initial Demo
=========================================

First attempt at using Gradio to develop a front-end web application.

The purpose of this experiment is to determine whether Gradio is a suitable
tool for us to use throughout the project. It should provide the necessary functionality
and features needed to deliver the user experience we aim to achieve. If the experiment
proves successful, ideas from our design in Figma will be implemented here.

Usage:
    To run this application locally:
    $ gradio interface.py

    To run on a specific port:
    $ gradio interface.py --port 8080
"""

import gradio as gr
import math
from functools import partial
from huggingface_hub import InferenceClient


# Connect to Hugging face to run LLM inference
def inference(prompt, hf_token, model, model_name):
    """
    Generates text and calculates uncertainties.
    """
    # Validation: ensure a token is present
    if not hf_token or not hf_token.strip():
        yield "Error: Please enter a Hugging Face Token to generate text."
        return

    # Initialize the client with the user's specific token
    try:
        client = InferenceClient(model=model, token=hf_token)
    except Exception as e:
        yield f"Client Error: {str(e)}"
        return

    # Prepare the message for the model
    messages = [{"role": "user", "content": prompt}]

    # Build the response incrementally for the streaming effect
    partial_text = ""

    # Initialize (token, label) tuple for the probability heatmap
    heatmap_data = []

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
            if chunk.choices and chunk.choices[0].delta.content:
                new_content = chunk.choices[0].delta.content
                partial_text += new_content

            # 2. Process logprobs
            # wrap in a safety check to ensure text generation does not stop if probability data is missing
            try:
                # Check if the logprobs exist and if 'content' (list of logprob objects) is an iterable
                if (
                    chunk.choices
                    and chunk.choices[0].logprobs
                    and hasattr(chunk.choices[0].logprobs, "content")
                    and chunk.choices[0].logprobs.content
                ):
                    for lp in chunk.choices[0].logprobs.content:
                        token_text = lp.token
                        log_score = lp.logprob
                        # Convert Logit to Probability (0.0 to 1.0)
                        prob = math.exp(log_score)

                        # Determine label based on Confidence (tune these thresholds as fit)
                        if prob > 0.9:
                            label = "Certain"
                        elif prob > 0.6:
                            label = "Uncertain"
                        else:
                            label = "High Entropy"

                        # Append to stored data list for later visualization
                        heatmap_data.append((token_text, label))

            except Exception:
                # If logprobs fail for one token, just ignore.
                pass

            # Yield both the visible text and the hidden probability data
            yield f"**{model_name} Response:**\n\n{partial_text}", heatmap_data

    except Exception as e:
        # If the token is invalid or the model is busy, show the error
        yield f"API Error: {str(e)}\n\n*Tip: Check if your token is valid and has 'Read' permissions.*"


def toggle_view(heatmap_data):
    # Helper to swap views: hides plain text and shows heatmap.
    return gr.update(visible=False), gr.update(visible=True, value=heatmap_data)


def reset_ui():
    """
    Helper to prepare the UI for a new prompt:
    1. Hides the token box (optional preference).
    2. Shows the plain text Markdown box.
    3. Hides the old Heatmap.
    """
    return (
        gr.Textbox(visible=False),  # Token box
        gr.Markdown(visible=True, value="### Generating..."),  # Text Output
        gr.HighlightedText(visible=False),  # Heatmap
    )


# Dummy function to be replaced with backend behaviour for slider and radio buttons
def dummy_fn(*args):
    return [("hello", "world")]


# Design the Interface
with gr.Blocks() as demo:
    # Application heading
    gr.Markdown("<center><h1>🧠 LLM Uncertainty Visualizer</h1></center>")
    gr.Markdown(
        "<center>subtitle of the project: can add a brief description of the interface here<center>"
    )

    # State storage: holds the logprob data until needed
    logprob_state = gr.State([])

    # Input area
    prompt = gr.Textbox(
        label="Please enter your prompt:",
        value="Please explain Deep Learning in simple terms to a 10-year old",
        lines=3,
        max_lines=8,
    )
    token = gr.Textbox(
        label="Hugging Face Token", type="password", placeholder="Paste token (hf_...)"
    )

    # Buttons
    with gr.Group():
        with gr.Row():
            generate_btn = gr.Button("Generate Text", variant="primary")
            metrics_btn = gr.Button("Visualize Uncertainty", variant="secondary")

    # Output area
    with gr.Group():
        # View A: Plain Text (default)
        model_output = gr.Markdown("### Response will appear here ...", visible=True)

        # View B: Heatmap (Hidden initially)
        output_metrics = gr.HighlightedText(
            label="Uncertainty Heatmap",
            combine_adjacent=False,
            show_legend=True,
            visible=False,  # hidden start
            color_map={
                "Certain": "#cbf0cc",
                "Uncertain": "#ffeea8",
                "High Entropy": "#ffc4c4",
            },
        )

    # Sidebar: metric selection (not connected yet)
    with gr.Sidebar(label="Metrics"):
        gr.Markdown("### Metrics")

        metric_choice = gr.Radio(
            [
                "Uncertainty",
                "Hallucinations",
                "Inference Cost",
                "Factuality",
                "Safety",
                "Privacy",
            ],
            label="Select View",
            value="Uncertainty",
        )

        # slider to mimic what we have in Figma
        slider = gr.Slider(0, 1, value=0.5, label="Confidence Threshold")

        gr.Slider(0, 1, value=0.5, label="Confidence Threshold")
        gr.Markdown("🐈 `thorp.thorp@machenta.com`", elem_classes="bottom-info")

    # --- Wiring ---

    # Event 1: Click Generate
    # Runs inference, updates text box, and saves the data to 'logprob_state'
    gr.on(
        triggers=[prompt.submit, generate_btn.click],
        fn=reset_ui,
        inputs=None,
        outputs=[token, model_output, output_metrics],
    ).then(
        fn=partial(
            inference,
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            model_name="Llama 3-8B Instruct",
        ),
        inputs=[prompt, token],
        outputs=[model_output, logprob_state],
        show_progress="hidden",
    )

    # Event 2: Click "Visualize Uncertainty"
    # Reads 'logprobs_state', hides text, shows heatmap
    metrics_btn.click(
        fn=toggle_view, inputs=[logprob_state], outputs=[model_output, output_metrics]
    )

    # dummy HighlightedText output
    dummy_output_display = gr.Textbox(visible=False)
    metric_choice.change(
        fn=dummy_fn, inputs=metric_choice, outputs=dummy_output_display
    )
    slider.change(fn=dummy_fn, inputs=slider, outputs=dummy_output_display)

# Launch
if __name__ == "__main__":
    demo.launch(share=True)
