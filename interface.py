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
    $ uv run app.py

    To run on a specific port:
    $ uv run app.py --port 8080
"""

import gradio as gr
import random

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
        score = random.random() # Random number between 0.0 and 1.0
        
        # Assign a label based on the score for the MVP
        if score > 0.9:
            label = "Certain"
        elif score > 0.5:
            label = "Uncertain"
        else:
            label = "Hallucination?"
            
        output_data.append((token, label))
    
    return output_data

# Design the Interface
with gr.Blocks() as demo:
    gr.Markdown("# 🧠 LLM Uncertainty Visualizer")
    
    with gr.Row():
        input_box = gr.Textbox(label="Enter your prompt", placeholder="What is the capital of Mars?")
        btn = gr.Button("Generate")
    
    output_display = gr.HighlightedText(
        label="Model Response (Color indicates Uncertainty)",
        combine_adjacent=False,
        show_legend=True,
    )

    # Connect the logic
    btn.click(fn=mock_inference, inputs=input_box, outputs=output_display)

# Launch
if __name__ == "__main__":
    demo.launch()