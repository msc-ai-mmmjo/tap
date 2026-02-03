from functools import partial

import gradio as gr

from constants import MODEL, MODEL_NAME, HeatmapData
from logic import inference


# Dummy function to be replaced with backend behaviour for slider and radio buttons
def dummy_fn(*args: object) -> HeatmapData:
    return [("hello", "world")]


with gr.Blocks() as demo:
    gr.Markdown("<center><h1>🧠 LLM Uncertainty Visualizer</h1></center>")
    token = gr.Textbox(
        label="Hugging Face Token", type="password", placeholder="Paste token (hf_...)"
    )

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

    chat = gr.ChatInterface(
        fn=partial(inference, model=MODEL, model_name=MODEL_NAME),
        additional_inputs=[token],
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

    # dummy HighlightedText output
    dummy_output_display = gr.Textbox(visible=False)
    metric_choice.change(
        fn=dummy_fn, inputs=metric_choice, outputs=dummy_output_display
    )
    slider.change(fn=dummy_fn, inputs=slider, outputs=dummy_output_display)

if __name__ == "__main__":
    demo.launch()
