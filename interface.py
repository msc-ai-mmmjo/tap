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

from functools import partial

import gradio as gr

from constants import HeatmapData
from logic import inference


def toggle_view(heatmap_data: HeatmapData) -> tuple[gr.Markdown, gr.HighlightedText]:
    """Swaps views: hides plain text and shows heatmap."""
    return (
        gr.Markdown(visible=False),
        gr.HighlightedText(visible=True, value=heatmap_data),
    )


def reset_ui() -> tuple[gr.Markdown, gr.HighlightedText]:
    """Prepares the UI for a new prompt."""
    return (
        gr.Markdown(visible=True, value="### Generating..."),
        gr.HighlightedText(visible=False),
    )


# Dummy function to be replaced with backend behaviour for slider and radio buttons
def dummy_fn(*args: object) -> HeatmapData:
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

        slider = gr.Slider(0, 1, value=0.5, label="Confidence Threshold")
        gr.Markdown("🐈 `thorp.thorp@machenta.com`", elem_classes="bottom-info")

    # --- Wiring ---

    # Event 1: Click Generate
    # Runs inference, updates text box, and saves the data to 'logprob_state'
    gr.on(
        triggers=[prompt.submit, generate_btn.click],
        fn=reset_ui,
        inputs=None,
        outputs=[model_output, output_metrics],
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
