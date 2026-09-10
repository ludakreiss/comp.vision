import gradio as gr
from inference import analyze_image


def analyze(image, degradation):
    if image is None:
        return (
            None,
            "Please upload an image.",
            "Please upload an image.",
        )

    degraded_image, standard, robust = analyze_image(
        image,
        degradation,
    )

    standard_text = (
        f"Prediction: {standard['prediction']}\n"
        f"Fake probability: {standard['fake_probability']:.3f}\n"
        f"Threshold: {standard['threshold']:.3f}"
    )

    robust_text = (
        f"Prediction: {robust['prediction']}\n"
        f"Fake probability: {robust['fake_probability']:.3f}\n"
        f"Threshold: {robust['threshold']:.3f}"
    )

    return (
        degraded_image,
        standard_text,
        robust_text,
    )
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