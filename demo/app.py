import gradio as gr
from inference import analyze_image


# ---------------------------------------------------------
# Degradation mapping
# ---------------------------------------------------------

DEGRADATION_LEVELS = {
    0: "clean",
    1: "weak_compression",
    2: "medium_compression",
    3: "strong_compression",
    4: "extreme_compression",
}

DEGRADATION_LABELS = {
    0: "Clean",
    1: "Weak",
    2: "Medium",
    3: "Strong",
    4: "Extreme",
}


# ---------------------------------------------------------
# Inference handler
# ---------------------------------------------------------

def analyze(image, degradation_level):
    if image is None:
        return (
            None,
            "Please provide an image.",
            "Please provide an image.",
            "**Selected degradation:** Clean",
        )

    level = int(round(degradation_level))

    degradation_name = DEGRADATION_LEVELS[level]
    degradation_label = DEGRADATION_LABELS[level]

    degraded_image, standard, robust = analyze_image(
        image,
        degradation_name,
    )

    standard_clean = standard["clean"]
    standard_degraded = standard["degraded"]
    standard_change = standard["change"] * 100

    robust_clean = robust["clean"]
    robust_degraded = robust["degraded"]
    robust_change = robust["change"] * 100

    standard_text = (
        f"Clean prediction: {standard_clean['prediction']}\n"
        f"Clean fake probability: "
        f"{standard_clean['fake_probability'] * 100:.1f}%\n\n"
        f"Degraded prediction: {standard_degraded['prediction']}\n"
        f"Degraded fake probability: "
        f"{standard_degraded['fake_probability'] * 100:.1f}%\n\n"
        f"Probability change: "
        f"{standard_change:+.1f} percentage points\n"
        f"Threshold: {standard_clean['threshold']:.3f}"
    )

    robust_text = (
        f"Clean prediction: {robust_clean['prediction']}\n"
        f"Clean fake probability: "
        f"{robust_clean['fake_probability'] * 100:.1f}%\n\n"
        f"Degraded prediction: {robust_degraded['prediction']}\n"
        f"Degraded fake probability: "
        f"{robust_degraded['fake_probability'] * 100:.1f}%\n\n"
        f"Probability change: "
        f"{robust_change:+.1f} percentage points\n"
        f"Threshold: {robust_clean['threshold']:.3f}"
    )

    degradation_status = (
        f"**Selected degradation:** {degradation_label}"
    )

    return (
        degraded_image,
        standard_text,
        robust_text,
        degradation_status,
    )


# ---------------------------------------------------------
# CSS
# ---------------------------------------------------------

CSS = """

/* -----------------------------------------------------
   Wider page
----------------------------------------------------- */

.gradio-container {
    max-width: 1500px !important;
    width: 96% !important;
    margin: 0 auto !important;
}


/* -----------------------------------------------------
   Purple Run Detection button
----------------------------------------------------- */

#run-button {
    background: linear-gradient(
        90deg,
        #6d4aff,
        #8b5cf6
    ) !important;

    color: white !important;
    border: none !important;
    font-weight: 600 !important;
}

#run-button:hover {
    background: linear-gradient(
        90deg,
        #5b3ee4,
        #7c4de8
    ) !important;
}


/* -----------------------------------------------------
   Slider wrapper
----------------------------------------------------- */

#degradation-slider {
    width: 100% !important;

    --color-accent: #7c3aed !important;
    --slider-color: #7c3aed !important;
    --primary-500: #7c3aed !important;
    --primary-600: #6d28d9 !important;
}


/* -----------------------------------------------------
   Main range input
----------------------------------------------------- */

#degradation-slider input[type="range"] {
    appearance: none !important;
    -webkit-appearance: none !important;

    width: 100% !important;
    height: 10px !important;

    background: linear-gradient(
        90deg,
        #d1d5db 0%,
        #c4b5fd 25%,
        #a78bfa 50%,
        #8b5cf6 75%,
        #6d28d9 100%
    ) !important;

    border-radius: 999px !important;
    outline: none !important;
}


/* -----------------------------------------------------
   Chrome / Edge slider track
----------------------------------------------------- */

#degradation-slider input[type="range"]::-webkit-slider-runnable-track {
    height: 10px !important;

    background: linear-gradient(
        90deg,
        #d1d5db 0%,
        #c4b5fd 25%,
        #a78bfa 50%,
        #8b5cf6 75%,
        #6d28d9 100%
    ) !important;

    border-radius: 999px !important;
}


/* -----------------------------------------------------
   Chrome / Edge slider thumb
----------------------------------------------------- */

#degradation-slider input[type="range"]::-webkit-slider-thumb {
    -webkit-appearance: none !important;
    appearance: none !important;

    width: 20px !important;
    height: 20px !important;

    margin-top: -5px !important;

    border-radius: 50% !important;

    background: #7c3aed !important;
    border: 3px solid white !important;

    cursor: pointer !important;

    box-shadow: 0 1px 5px rgba(0, 0, 0, 0.25) !important;
}


/* -----------------------------------------------------
   Firefox slider track
----------------------------------------------------- */

#degradation-slider input[type="range"]::-moz-range-track {
    height: 10px !important;

    background: linear-gradient(
        90deg,
        #d1d5db 0%,
        #c4b5fd 25%,
        #a78bfa 50%,
        #8b5cf6 75%,
        #6d28d9 100%
    ) !important;

    border-radius: 999px !important;
}


/* -----------------------------------------------------
   Firefox progress
----------------------------------------------------- */

#degradation-slider input[type="range"]::-moz-range-progress {
    height: 10px !important;

    background: linear-gradient(
        90deg,
        #c4b5fd,
        #7c3aed
    ) !important;

    border-radius: 999px !important;
}


/* -----------------------------------------------------
   Firefox thumb
----------------------------------------------------- */

#degradation-slider input[type="range"]::-moz-range-thumb {
    width: 20px !important;
    height: 20px !important;

    border-radius: 50% !important;

    background: #7c3aed !important;
    border: 3px solid white !important;

    cursor: pointer !important;
}


/* -----------------------------------------------------
   Hide numeric stepper / number box
----------------------------------------------------- */

#degradation-slider input[type="number"] {
    display: none !important;
}


/* -----------------------------------------------------
   Tick marks
----------------------------------------------------- */

.slider-ticks {
    display: flex;
    justify-content: space-between;

    padding-left: 8px;
    padding-right: 8px;

    margin-top: -8px;
    margin-bottom: 3px;
}

.slider-ticks span {
    width: 2px;
    height: 8px;

    background: #777;

    display: block;
}


/* -----------------------------------------------------
   Labels below slider
----------------------------------------------------- */

.slider-labels {
    display: flex;
    justify-content: space-between;

    font-size: 0.85rem;
    font-weight: 500;

    margin-top: 2px;
    margin-bottom: 12px;
}

.slider-labels span {
    width: 20%;
    text-align: center;
}

.slider-labels span:first-child {
    text-align: left;
}

.slider-labels span:last-child {
    text-align: right;
}

"""


# ---------------------------------------------------------
# Gradio interface
# ---------------------------------------------------------

with gr.Blocks(
    title="Deepfake Detection Robustness Demo",
    css=CSS,
) as demo:

    gr.Markdown(
        """
        # Deepfake Detection Robustness Demo

        Compare a clean-trained deepfake detector with a
        degradation-trained robust detector under increasing
        compression severity.
        """
    )

    # -----------------------------------------------------
    # Images
    # -----------------------------------------------------

    with gr.Row():

        image_input = gr.Image(
            type="pil",
            label="Original Image",
        )

        degraded_output = gr.Image(
            type="pil",
            label="Degraded Image",
        )

    # -----------------------------------------------------
    # Compression slider
    # -----------------------------------------------------

    gr.Markdown("### Compression Severity")

    degradation_slider = gr.Slider(
        minimum=0,
        maximum=4,
        step=1,
        value=0,
        show_label=False,
        elem_id="degradation-slider",
    )

    gr.HTML(
        """
        <div class="slider-ticks">
            <span></span>
            <span></span>
            <span></span>
            <span></span>
            <span></span>
        </div>
        """
    )

    gr.HTML(
        """
        <div class="slider-labels">
            <span>Clean</span>
            <span>Weak</span>
            <span>Medium</span>
            <span>Strong</span>
            <span>Extreme</span>
        </div>
        """
    )

    degradation_status = gr.Markdown(
        "**Selected degradation:** Clean"
    )

    analyze_button = gr.Button(
        "Run Detection",
        variant="primary",
        elem_id="run-button",
    )

    # -----------------------------------------------------
    # Results
    # -----------------------------------------------------

    gr.Markdown("## Model Comparison")

    with gr.Row():

        standard_output = gr.Textbox(
            label="Standard Model",
            lines=9,
            interactive=False,
        )

        robust_output = gr.Textbox(
            label="Robust Model",
            lines=9,
            interactive=False,
        )

    # -----------------------------------------------------
    # Run button
    # -----------------------------------------------------

    analyze_button.click(
        fn=analyze,
        inputs=[
            image_input,
            degradation_slider,
        ],
        outputs=[
            degraded_output,
            standard_output,
            robust_output,
            degradation_status,
        ],
    )


# ---------------------------------------------------------
# Start application
# ---------------------------------------------------------

if __name__ == "__main__":
    demo.launch(inbrowser=True)