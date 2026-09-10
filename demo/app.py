import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import gradio as gr

try:
    from demo.inference import analyze_image_with_curve, analyze_image
except ImportError:
    from inference import analyze_image_with_curve, analyze_image


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

EMPTY_COMPARISON_DF = pd.DataFrame(
    [
        ["Standard Model (Clean-trained)", "-", "-", "-", "-", "-", "0.300"],
        ["Robust Model (Degradation-trained)", "-", "-", "-", "-", "-", "0.265"],
    ],
    columns=[
        "Model",
        "Clean Prediction",
        "Clean Fake Prob",
        "Degraded Prediction",
        "Degraded Fake Prob",
        "Probability Change",
        "Threshold",
    ],
)


# ---------------------------------------------------------
# Chart builder
# ---------------------------------------------------------

def build_severity_chart(curve_data, selected_level=0):
    if curve_data is None:
        return None

    levels = curve_data["levels"]
    std_probs = [p * 100 for p in curve_data["standard_probs"]]
    rob_probs = [p * 100 for p in curve_data["robust_probs"]]

    fig, ax = plt.subplots(figsize=(8.5, 4.0), dpi=100)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#fbfbfe")

    ax.plot(
        levels,
        std_probs,
        marker="o",
        color="#ef4444",
        linewidth=2.5,
        markersize=7,
        label="Standard Model (Clean-trained)",
    )
    ax.plot(
        levels,
        rob_probs,
        marker="s",
        color="#7c3aed",
        linewidth=2.5,
        markersize=7,
        label="Robust Model (Degradation-trained)",
    )

    if selected_level in levels:
        idx = levels.index(selected_level)
        ax.axvline(
            x=selected_level,
            color="#9ca3af",
            linestyle="--",
            linewidth=1.5,
            alpha=0.8,
            label=f"Selected Severity (Level {selected_level})",
        )
        ax.scatter(
            [selected_level],
            [std_probs[idx]],
            s=130,
            facecolors="none",
            edgecolors="#ef4444",
            linewidth=2.5,
            zorder=6,
        )
        ax.scatter(
            [selected_level],
            [rob_probs[idx]],
            s=130,
            facecolors="none",
            edgecolors="#7c3aed",
            linewidth=2.5,
            zorder=6,
        )

    ax.set_ylim(-2, 102)
    ax.set_xlim(-0.2, 4.2)
    ax.set_xticks(levels)
    ax.set_xticklabels(
        ["Clean (0)", "Weak (1)", "Medium (2)", "Strong (3)", "Extreme (4)"],
        fontsize=10,
        fontweight="500",
    )
    ax.set_xlabel("Compression Severity Level", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_ylabel("Fake Probability (%)", fontsize=11, fontweight="bold", labelpad=8)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{int(y)}%"))
    ax.grid(True, linestyle="--", alpha=0.45, color="#d1d5db")
    ax.legend(frameon=True, facecolor="#ffffff", edgecolor="#e5e7eb", loc="best", fontsize=9.5)
    fig.tight_layout()
    return fig


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
            EMPTY_COMPARISON_DF,
            None,
            "Upload an image and click **Run Detection** to view predictions, comparison table, and robustness chart.",
        )

    level = int(round(degradation_level))

    degradation_name = DEGRADATION_LEVELS[level]
    degradation_label = DEGRADATION_LABELS[level]

    degraded_image, standard, robust, curve_data = analyze_image_with_curve(
        image,
        degradation_name,
    )

    standard_clean = standard["clean"]
    standard_degraded = standard["degraded"]
    standard_change = standard["change"] * 100

    robust_clean = robust["clean"]
    robust_degraded = robust["degraded"]
    robust_change = robust["change"] * 100

    disclaimer = (
        "\n\n* Note: Score represents model fake probability/logit score, "
        "not an externally calibrated real-world probability."
    )

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
        f"{disclaimer}"
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
        f"{disclaimer}"
    )

    degradation_status = (
        f"**Selected degradation:** {degradation_label}"
    )

    comparison_df = pd.DataFrame(
        [
            [
                "Standard Model (Clean-trained)",
                standard_clean["prediction"],
                f"{standard_clean['fake_probability'] * 100:.1f}%",
                standard_degraded["prediction"],
                f"{standard_degraded['fake_probability'] * 100:.1f}%",
                f"{standard_change:+.1f} pp",
                f"{standard_clean['threshold']:.3f}",
            ],
            [
                "Robust Model (Degradation-trained)",
                robust_clean["prediction"],
                f"{robust_clean['fake_probability'] * 100:.1f}%",
                robust_degraded["prediction"],
                f"{robust_degraded['fake_probability'] * 100:.1f}%",
                f"{robust_change:+.1f} pp",
                f"{robust_clean['threshold']:.3f}",
            ],
        ],
        columns=[
            "Model",
            "Clean Prediction",
            "Clean Fake Prob",
            "Degraded Prediction",
            "Degraded Fake Prob",
            "Probability Change",
            "Threshold",
        ],
    )

    chart_fig = build_severity_chart(curve_data, selected_level=level)

    abs_std = abs(standard_change)
    abs_rob = abs(robust_change)

    if abs_rob < abs_std:
        stability_finding = (
            f"The **Robust Model** exhibited a smaller absolute score change "
            f"(|Δp| = **{abs_rob:.1f} pp** vs **{abs_std:.1f} pp** for Standard), "
            f"indicating greater prediction stability for this sample under this distortion level."
        )
    elif abs_std < abs_rob:
        stability_finding = (
            f"The **Standard Model** exhibited a smaller absolute score change "
            f"(|Δp| = **{abs_std:.1f} pp** vs **{abs_rob:.1f} pp** for Robust), "
            f"indicating greater prediction stability for this sample under this distortion level."
        )
    else:
        stability_finding = (
            f"Both models exhibited identical absolute score changes "
            f"(|Δp| = **{abs_rob:.1f} pp**), demonstrating equal prediction stability for this sample."
        )

    interpretation_text = (
        f"### Robustness & Prediction Stability\n\n"
        f"- **Standard Model confidence change:** {standard_change:+.1f} percentage points (|Δp| = {abs_std:.1f} pp)\n"
        f"- **Robust Model confidence change:** {robust_change:+.1f} percentage points (|Δp| = {abs_rob:.1f} pp)\n"
        f"- **Stability Assessment:** {stability_finding}\n\n"
        f"> **Scientific Context:** A smaller score change indicates that a model's prediction is **more stable** "
        f"(less sensitive) under that particular distortion for that particular image. "
        f"This is characterized as *more stable for this sample*, not automatically a *better model*, "
        f"because classification correctness depends on the ground-truth label (Real vs Fake). "
        f"Stability alone does not guarantee higher classification accuracy."
    )

    return (
        degraded_image,
        standard_text,
        robust_text,
        degradation_status,
        comparison_df,
        chart_fig,
        interpretation_text,
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
            label="Clean Input",
        )

        degraded_output = gr.Image(
            type="pil",
            label="Degraded Input",
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
    # Predictions
    # -----------------------------------------------------

    gr.Markdown("## Model Predictions")

    with gr.Row():

        standard_output = gr.Textbox(
            label="Standard Model (Clean-trained)",
            lines=10,
            interactive=False,
        )

        robust_output = gr.Textbox(
            label="Robust Model (Degradation-trained)",
            lines=10,
            interactive=False,
        )

    # -----------------------------------------------------
    # Model Comparison Table
    # -----------------------------------------------------

    gr.Markdown("## Model Comparison Table")

    comparison_table = gr.Dataframe(
        value=EMPTY_COMPARISON_DF,
        headers=[
            "Model",
            "Clean Prediction",
            "Clean Fake Prob",
            "Degraded Prediction",
            "Degraded Fake Prob",
            "Probability Change",
            "Threshold",
        ],
        interactive=False,
        wrap=True,
    )

    # -----------------------------------------------------
    # Confidence vs Distortion Severity Chart
    # -----------------------------------------------------

    gr.Markdown("## Confidence vs. Distortion Severity")

    severity_chart = gr.Plot(
        label="Confidence vs. Compression Severity",
    )

    # -----------------------------------------------------
    # Robustness Interpretation
    # -----------------------------------------------------

    interpretation_output = gr.Markdown(
        "Upload an image and click **Run Detection** to view predictions, comparison table, and robustness chart."
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
            comparison_table,
            severity_chart,
            interpretation_output,
        ],
    )


# ---------------------------------------------------------
# Start application
# ---------------------------------------------------------

if __name__ == "__main__":
    demo.launch(inbrowser=True)