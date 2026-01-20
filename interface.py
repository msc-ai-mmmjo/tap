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
from functools import partial
from huggingface_hub import InferenceClient

# Run LLM model inference
def inference(prompt, hf_token, model, model_name):
    """
    Connects to Hugging Face to generate text.
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

    try:
        # Stream the response
        stream = client.chat_completion(messages, max_tokens=500, stream=True)

        for chunk in stream:
            # Check if there is new content in this chunk
            if chunk.choices and chunk.choices[0].delta.content:
                new_content = chunk.choices[0].delta.content
                partial_text += new_content

                # Yield the updated text to the UI immediately
                yield f"**{model_name} Response:**\n\n{partial_text}"
    
    except Exception as e:
        # If the token is invalid or the model is busy, show the error
        yield f"API Error: {str(e)}\n\n*Tip: Check if your token is valid and has 'Read' permissions.*"


def hide_textbox():
    # Helper to visually hide the token box after clicking (optional UX choice)
    return gr.Textbox(visible=False)


# Dummy function to be replaced with backend behaviour for slider and radio buttons
def dummy_fn(*args):
    return [("hello", "world")]


# Design the Interface
with gr.Blocks() as demo:

    # Application heading
    gr.Markdown("<center><h1>🧠 LLM Uncertainty Visualizer</h1></center>")
    gr.Markdown("<center>subtitle of the project: can add a brief description of the interface here<center>") 

    # Input area
    prompt = gr.Textbox(label="Please enter your prompt:", value="Please explain Deep Learning in simple terms to a 10-year old", lines=3, max_lines=8)
    token = gr.Textbox(label="Hugging Face Token", type="password", placeholder="Paste token (hf_...)")

    # Buttons
    with gr.Group():
        with gr.Row():
            generate_btn = gr.Button("Generate", variant="primary")
            metrics_btn = gr.Button("Take a peek", variant="secondary")
    
    # Model outputs
    model_output = gr.Markdown("### Response will appear here ...")
    
    # Produce output: run the mock_inference function when generate button is clicked
    gr.on(
        triggers = [prompt.submit, generate_btn.click],
        fn=hide_textbox,
        inputs=None,
        outputs=[token],
    ).then(
        fn=partial(inference, model="meta-llama/Meta-Llama-3-8B-Instruct", model_name="Llama 3-8B Instruct"),
        inputs=[prompt, token],
        outputs=[model_output],
        show_progress="hidden"
    )

    # # Create heatmap over model output response
    # output_metrics = gr.HighlightedText(
    #     label="Model Response (Color indicates Uncertainty)",
    #     combine_adjacent=False,
    #     show_legend=True,
    # )

    # # Show metrics overlayed on model output
    # metrics_btn.click(
    #     lambda : gr.Row(visible=False), # clear row
    #     outputs=output_metrics
    # )


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
    demo.launch(share=True)
