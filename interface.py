"""
Gradio Web Interface Initial Experiment
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
import random
from huggingface_hub import InferenceClient


# Create Mock Data
# This will come from the Model in the real interface
def mock_inference(prompt):
    # Fake response broken into tokens
    # Format: [(token_text, label_or_score)]
    # Use a score from 0 to 1 representing certainty
    tokens = ["The", " capital", " of", " Mars", " is", "...wait", " unknown", "."]

    # Randomly assign "confidence" scores to simulate uncertainty
    output_data = []
    for token in tokens:
        score = random.random()  # Random number between 0.0 and 1.0

        # Assign a label based on the score for the MVP
        if score > 0.9:
            label = "Certain"
        elif score > 0.5:
            label = "Uncertain"
        else:
            label = "Hallucination?"

        output_data.append((token, label))

    return output_data

# Run LLM model inference --> complete later
def inference():
    pass


def hide_textbox():
    return gr.Textbox(visible=False)


# Dummy function to be replaced with backend behaviour for slider and radio buttons
def dummy_fn(*args):
    return [("hello", "world")]


# Design the Interface
with gr.Blocks() as demo:

    # Application heading
    gr.Markdown("<center><h1>🧠 LLM Uncertainty Visualizer</h1></center>")
    gr.Markdown("<center>subtitle of the project: can add a brief description of the interface here<center>") 

    # Textboxes
    prompt = gr.Textbox(label="Please enter your prompt:", placeholder="What is the capital of Mars?", lines=3, max_lines=8)
    token = gr.Textbox(label="Hugging Face Token")

    # Buttons
    with gr.Group():
        with gr.Row():
            generate_btn = gr.Button("Generate", variant="primary")
            metrics_btn = gr.Button("Take a peek", variant="secondary")

    # Create heatmap over a mock output response
    output_display = gr.HighlightedText(
        label="Model Response (Color indicates Uncertainty)",
        combine_adjacent=False,
        show_legend=True,
    )
    
    # Produce output: run the mock_inference function when generate button is clicked
    gr.on(
        triggers = [prompt.submit, generate_btn.click],
        fn=hide_textbox,
        inputs=None,
        outputs=[token],
    ).then(
        fn=mock_inference,
        inputs=prompt,
        outputs=output_display,
        show_progress="hidden"
    )


    # Below is the sidebar area where we keep metrics
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


    # dummy HighlightedText output
    dummy_output_display = gr.Textbox(visible=False)
    metric_choice.change(
        fn=dummy_fn, inputs=metric_choice, outputs=dummy_output_display
    )
    slider.change(fn=dummy_fn, inputs=slider, outputs=dummy_output_display)

# Launch
if __name__ == "__main__":
    demo.launch()
