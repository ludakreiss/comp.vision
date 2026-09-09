import gradio as gr
from inference import apply_demo_degradation


def analyze(image, degradation):
    if image is None:
        return None, "Please upload an image.", "Please upload an image."

    degraded_image = apply_demo_degradation(image, degradation)

    standard_result = "Model checkpoint not loaded yet"
    robust_result = "Model checkpoint not loaded yet"

    return degraded_image, standard_result, robust_result


with gr.Blocks(title="Deepfake Detection Robustness Demo") as demo:

    gr.Markdown(
        """
        # Deepfake Detection Robustness Demo

        Compare a standard deepfake detector with a robustness-trained model
        under different image degradation conditions.
        """
    )

    with gr.Row():
        image_input = gr.Image(
            type="pil",
            label="Upload Image"
        )

        degradation = gr.Dropdown(
            choices=[
                "clean",
                "weak_compression",
                "medium_compression",
                "strong_compression",
                "extreme_compression",
                "resize_75",
                "resize_50",
                "resize_25",
                "gaussian_blur",
                "motion_blur",
                "gaussian_noise",
                "color_jitter",
                "resize_50_compress_70",
                "screenshot_recompress",
                "social_media_pipeline",
            ],
            value="clean",
            label="Degradation"
        )

    analyze_button = gr.Button("Analyze")

    gr.Markdown("## Processed Input")

    degraded_output = gr.Image(
        type="pil",
        label="Degraded Image"
    )

    gr.Markdown("## Model Comparison")

    with gr.Row():
        standard_output = gr.Textbox(
            label="Standard Model",
            interactive=False
        )

        robust_output = gr.Textbox(
            label="Robust Model",
            interactive=False
        )

    analyze_button.click(
        fn=analyze,
        inputs=[
            image_input,
            degradation
        ],
        outputs=[
            degraded_output,
            standard_output,
            robust_output
        ]
    )


if __name__ == "__main__":
    demo.launch()