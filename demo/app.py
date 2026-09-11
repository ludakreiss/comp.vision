import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gradio as gr

try:
    from demo.inference import (
        analyze_image_with_curve,
        analyze_image,
        generate_gradcam_grid,
    )
except ImportError:
    from inference import (
        analyze_image_with_curve,
        analyze_image,
        generate_gradcam_grid,
    )


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

COMPARISON_TABLE_COLUMNS = [
    "Model",
    "Clean Prediction",
    "Clean Fake Score",
    "Distorted Prediction",
    "Distorted Fake Score",
    "Change",
    "Threshold",
]


# ---------------------------------------------------------
# Model Comparison table (compact, responsive HTML)
# ---------------------------------------------------------

def _pred_color(prediction):
    if prediction == "REAL":
        return "#16a34a"
    if prediction == "FAKE":
        return "#ef4444"
    return "#6b7280"


def build_comparison_table_html(rows):
    """
    rows: list of dicts with keys
        model, clean_pred, clean_pct, dist_pred, dist_pct, change, threshold_pct
    """
    header_html = "".join(
        f"<th style='height:36px; padding:6px 10px; text-align:left; "
        f"font-weight:var(--weight-bold, 700); color:var(--body-text-color, #1f2937); "
        f"background:var(--table-even-background-fill, #f7f7f8); "
        f"border-right:1px solid var(--border-color-primary, #d9d9df); "
        f"border-bottom:1px solid var(--border-color-primary, #d9d9df); "
        f"white-space:nowrap; vertical-align:middle;'>{col}</th>"
        for col in COMPARISON_TABLE_COLUMNS
    )

    body_html = ""
    for index, row in enumerate(rows):
        row_background = (
            "var(--table-even-background-fill, #f7f7f8)"
            if index % 2
            else "var(--table-odd-background-fill, #ffffff)"
        )
        cell_style = (
            "height:36px; padding:6px 10px; white-space:nowrap; vertical-align:middle; "
            f"background:{row_background}; "
            "border-right:1px solid var(--border-color-primary, #d9d9df); "
            "border-bottom:1px solid var(--border-color-primary, #d9d9df);"
        )
        body_html += (
            "<tr style='height:36px;'>"
            f"<td style='{cell_style} font-weight:var(--weight-semibold, 600);'>{row['model']}</td>"
            f"<td style='{cell_style} font-weight:var(--weight-semibold, 600); color:{_pred_color(row['clean_pred'])};'>{row['clean_pred']}</td>"
            f"<td style='{cell_style}'>{row['clean_pct']}</td>"
            f"<td style='{cell_style} font-weight:var(--weight-semibold, 600); color:{_pred_color(row['dist_pred'])};'>{row['dist_pred']}</td>"
            f"<td style='{cell_style}'>{row['dist_pct']}</td>"
            f"<td style='{cell_style}'>{row['change']}</td>"
            f"<td style='{cell_style}'>{row['threshold_pct']}</td>"
            "</tr>"
        )

    return (
        "<div style='width:100%; border:1px solid var(--border-color-primary, #d9d9df); "
        "border-radius:var(--table-radius, 8px); background:var(--block-background-fill, #ffffff); "
        "box-shadow:var(--block-shadow, none); overflow:hidden;'>"
        "<div style='width:100%; overflow-x:auto; -webkit-overflow-scrolling:touch;'>"
        "<table style='width:100%; min-width:760px; border-collapse:collapse; border-spacing:0; "
        "font-family:var(--font-mono, ui-monospace, monospace); "
        "font-size:var(--input-text-size, 0.875rem); line-height:var(--line-md, 1.4); "
        "color:var(--body-text-color, #1f2937); text-align:left;'>"
        f"<thead><tr>{header_html}</tr></thead>"
        f"<tbody>{body_html}</tbody>"
        "</table>"
        "</div>"
        "</div>"
    )


EMPTY_COMPARISON_TABLE_HTML = build_comparison_table_html(
    [
        {
            "model": "Standard Model (Clean-trained)",
            "clean_pred": "-",
            "clean_pct": "-",
            "dist_pred": "-",
            "dist_pct": "-",
            "change": "-",
            "threshold_pct": "30.0%",
        },
        {
            "model": "Robust Model (Degradation-trained)",
            "clean_pred": "-",
            "clean_pct": "-",
            "dist_pred": "-",
            "dist_pct": "-",
            "change": "-",
            "threshold_pct": "26.5%",
        },
    ]
)

EMPTY_SCORE_BARS_HTML = (
    "<div style='color:#6b7280; font-size:0.9rem;'>"
    "Run detection to see fake-score bars relative to each model's threshold."
    "</div>"
)


# ---------------------------------------------------------
# Chart builder
# ---------------------------------------------------------

def build_severity_chart(
    curve_data,
    selected_level=0,
    standard_threshold_pct=None,
    robust_threshold_pct=None,
):
    if curve_data is None:
        return None

    levels = curve_data["levels"]
    std_probs = [p * 100 for p in curve_data["standard_probs"]]
    rob_probs = [p * 100 for p in curve_data["robust_probs"]]

    fig, ax = plt.subplots(figsize=(8.5, 4.0), dpi=100)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#fbfbfe")

    if standard_threshold_pct is not None:
        ax.axhline(
            y=standard_threshold_pct,
            color="#ef4444",
            linestyle=":",
            linewidth=1.5,
            alpha=0.7,
            label=f"Standard Threshold ({standard_threshold_pct:.1f}%)",
        )
    if robust_threshold_pct is not None:
        ax.axhline(
            y=robust_threshold_pct,
            color="#7c3aed",
            linestyle=":",
            linewidth=1.5,
            alpha=0.7,
            label=f"Robust Threshold ({robust_threshold_pct:.1f}%)",
        )

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
    ax.set_xlabel("Distortion Severity", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_ylabel("Fake Score (%)", fontsize=11, fontweight="bold", labelpad=8)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{int(y)}%"))
    ax.grid(True, linestyle="--", alpha=0.45, color="#d1d5db")
    ax.legend(frameon=True, facecolor="#ffffff", edgecolor="#e5e7eb", loc="best", fontsize=8.5)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------
# Fake-score bar/gauge builder (score relative to threshold)
# ---------------------------------------------------------

def _score_bar_html(row_label, score_pct, threshold_pct, prediction):
    color = "#ef4444" if prediction == "FAKE" else "#16a34a"
    fill = min(max(score_pct, 0.0), 100.0)
    thresh_pos = min(max(threshold_pct, 0.0), 100.0)
    return f"""
    <div style="margin-bottom:10px;">
        <div style="display:flex; justify-content:space-between;
                    font-size:0.82rem; font-weight:600; color:#374151; margin-bottom:3px;">
            <span>{row_label}</span>
            <span style="color:{color};">{prediction} &middot; {score_pct:.1f}%</span>
        </div>
        <div style="position:relative; height:12px; background:#e5e7eb;
                    border-radius:6px;">
            <div style="position:absolute; left:0; top:0; height:100%;
                        width:{fill:.2f}%; background:{color}; border-radius:6px;"></div>
            <div title="Decision threshold: {threshold_pct:.1f}%"
                 style="position:absolute; left:{thresh_pos:.2f}%; top:-3px;
                        width:2px; height:18px; background:#111827;"></div>
        </div>
    </div>
    """


def build_score_bars_html(standard_clean, standard_degraded, robust_clean, robust_degraded):
    std_threshold_pct = standard_clean["threshold"] * 100
    rob_threshold_pct = robust_clean["threshold"] * 100

    standard_block = (
        "<div style='flex:1; min-width:260px;'>"
        "<div style='font-weight:700; margin-bottom:8px;'>Standard Model</div>"
        + _score_bar_html("Clean", standard_clean["fake_probability"] * 100, std_threshold_pct, standard_clean["prediction"])
        + _score_bar_html("Distorted", standard_degraded["fake_probability"] * 100, std_threshold_pct, standard_degraded["prediction"])
        + f"<div style='font-size:0.72rem; color:#6b7280;'>Black tick = decision threshold ({std_threshold_pct:.1f}%)</div>"
        + "</div>"
    )
    robust_block = (
        "<div style='flex:1; min-width:260px;'>"
        "<div style='font-weight:700; margin-bottom:8px;'>Robust Model</div>"
        + _score_bar_html("Clean", robust_clean["fake_probability"] * 100, rob_threshold_pct, robust_clean["prediction"])
        + _score_bar_html("Distorted", robust_degraded["fake_probability"] * 100, rob_threshold_pct, robust_degraded["prediction"])
        + f"<div style='font-size:0.72rem; color:#6b7280;'>Black tick = decision threshold ({rob_threshold_pct:.1f}%)</div>"
        + "</div>"
    )
    return f"<div style='display:flex; gap:28px; flex-wrap:wrap;'>{standard_block}{robust_block}</div>"


# ---------------------------------------------------------
# Inference handler
# ---------------------------------------------------------

def analyze(image, degradation_level):
    if image is None:
        return (
            None,
            "Please provide an image.",
            "Please provide an image.",
            "**Distortion:** Clean",
            EMPTY_SCORE_BARS_HTML,
            EMPTY_COMPARISON_TABLE_HTML,
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

    standard_text = (
        f"Clean prediction: {standard_clean['prediction']}\n"
        f"Clean fake score: "
        f"{standard_clean['fake_probability'] * 100:.1f}%\n\n"
        f"Distorted prediction: {standard_degraded['prediction']}\n"
        f"Distorted fake score: "
        f"{standard_degraded['fake_probability'] * 100:.1f}%\n\n"
        f"Score change: "
        f"{standard_change:+.1f} percentage points\n"
        f"Threshold: {standard_clean['threshold'] * 100:.1f}%"
    )

    robust_text = (
        f"Clean prediction: {robust_clean['prediction']}\n"
        f"Clean fake score: "
        f"{robust_clean['fake_probability'] * 100:.1f}%\n\n"
        f"Distorted prediction: {robust_degraded['prediction']}\n"
        f"Distorted fake score: "
        f"{robust_degraded['fake_probability'] * 100:.1f}%\n\n"
        f"Score change: "
        f"{robust_change:+.1f} percentage points\n"
        f"Threshold: {robust_clean['threshold'] * 100:.1f}%"
    )

    degradation_status = (
        f"**Distortion:** {degradation_label}"
    )

    score_bars_html = build_score_bars_html(
        standard_clean, standard_degraded, robust_clean, robust_degraded
    )

    comparison_table_html = build_comparison_table_html(
        [
            {
                "model": "Standard Model (Clean-trained)",
                "clean_pred": standard_clean["prediction"],
                "clean_pct": f"{standard_clean['fake_probability'] * 100:.1f}%",
                "dist_pred": standard_degraded["prediction"],
                "dist_pct": f"{standard_degraded['fake_probability'] * 100:.1f}%",
                "change": f"{standard_change:+.1f} pp",
                "threshold_pct": f"{standard_clean['threshold'] * 100:.1f}%",
            },
            {
                "model": "Robust Model (Degradation-trained)",
                "clean_pred": robust_clean["prediction"],
                "clean_pct": f"{robust_clean['fake_probability'] * 100:.1f}%",
                "dist_pred": robust_degraded["prediction"],
                "dist_pct": f"{robust_degraded['fake_probability'] * 100:.1f}%",
                "change": f"{robust_change:+.1f} pp",
                "threshold_pct": f"{robust_clean['threshold'] * 100:.1f}%",
            },
        ]
    )

    chart_fig = build_severity_chart(
        curve_data,
        selected_level=level,
        standard_threshold_pct=standard_clean["threshold"] * 100,
        robust_threshold_pct=robust_clean["threshold"] * 100,
    )

    abs_std = abs(standard_change)
    abs_rob = abs(robust_change)

    if abs_rob < abs_std:
        more_stable = "Robust Model (Degradation-trained)"
    elif abs_std < abs_rob:
        more_stable = "Standard Model (Clean-trained)"
    else:
        more_stable = "Both models equally stable for this sample"

    models_agree = standard_degraded["prediction"] == robust_degraded["prediction"]
    standard_changed = standard_clean["prediction"] != standard_degraded["prediction"]
    robust_changed = robust_clean["prediction"] != robust_degraded["prediction"]

    interpretation_text = (
        f"### Model Comparison & Stability\n\n"
        f"- **Models agree at this distortion level:** {'YES' if models_agree else 'NO'}\n"
        f"- **Standard prediction changed after distortion:** {'YES' if standard_changed else 'NO'}\n"
        f"- **Robust prediction changed after distortion:** {'YES' if robust_changed else 'NO'}\n"
        f"- **Standard model absolute score change:** {abs_std:.1f} pp ({standard_change:+.1f} pp)\n"
        f"- **Robust model absolute score change:** {abs_rob:.1f} pp ({robust_change:+.1f} pp)\n"
        f"- **More stable on this sample:** {more_stable}\n\n"
    )

    return (
        degraded_image,
        standard_text,
        robust_text,
        degradation_status,
        score_bars_html,
        comparison_table_html,
        chart_fig,
        interpretation_text,
    )


# ---------------------------------------------------------
# Grad-CAM handler (explicit button click only)
# ---------------------------------------------------------

GRADCAM_PLACEHOLDER_MSG = "Click **Generate Grad-CAM** above to compute heatmaps for this image."


def run_gradcam(image, degradation_level):
    if image is None:
        return None, None, None, None, "Please provide an image first."

    level = int(round(degradation_level))
    degradation_name = DEGRADATION_LEVELS[level]
    degradation_label = DEGRADATION_LABELS[level]

    standard_clean_cam, standard_distorted_cam, robust_clean_cam, robust_distorted_cam = (
        generate_gradcam_grid(image, degradation_name)
    )

    status = f"Grad-CAM computed at distortion level: **{degradation_label}**"

    return (
        standard_clean_cam,
        standard_distorted_cam,
        robust_clean_cam,
        robust_distorted_cam,
        status,
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
   Purple primary action buttons
----------------------------------------------------- */

#run-button,
#gradcam-button {
    background: linear-gradient(
        90deg,
        #6d4aff,
        #8b5cf6
    ) !important;

    color: white !important;
    border: none !important;
    font-weight: 600 !important;
}

#run-button:hover,
#gradcam-button:hover {
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
            label="Distorted Input",
        )

    # -----------------------------------------------------
    # Compression slider
    # -----------------------------------------------------

    gr.Markdown("### Distortion Severity")

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
        "**Distortion:** Clean"
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
    # Fake-score bars relative to each model's threshold
    # -----------------------------------------------------

    gr.Markdown("## Fake Score vs. Threshold")

    score_bars_output = gr.HTML(EMPTY_SCORE_BARS_HTML)

    # -----------------------------------------------------
    # Model Comparison Table
    # -----------------------------------------------------

    gr.Markdown("## Model Comparison Table")

    comparison_table = gr.HTML(EMPTY_COMPARISON_TABLE_HTML)

    # -----------------------------------------------------
    # Confidence vs Distortion Severity Chart
    # -----------------------------------------------------

    gr.Markdown("## Confidence vs. Distortion Severity")

    severity_chart = gr.Plot(
        label="Confidence vs. Distortion Severity",
    )

    # -----------------------------------------------------
    # Robustness Interpretation
    # -----------------------------------------------------

    interpretation_output = gr.Markdown(
        "Upload an image and click **Run Detection** to view predictions, comparison table, and robustness chart."
    )

    # -----------------------------------------------------
    # Grad-CAM (always visible; heatmaps computed on click only)
    # -----------------------------------------------------

    gr.Markdown("## 🔥 Grad-CAM Model Explanation")

    gr.Markdown(
        "Grad-CAM shows which image regions influenced the model's fake score. "
        "Heatmaps are computed only when you click **Generate Grad-CAM** below "
        "(not on every slider move), since each map requires an extra backward pass."
    )

    gradcam_button = gr.Button(
        "Generate Grad-CAM",
        variant="primary",
        elem_id="gradcam-button",
    )
    gradcam_status = gr.Markdown(GRADCAM_PLACEHOLDER_MSG)

    gr.Markdown("### Standard Model")
    with gr.Row():
        gradcam_standard_clean = gr.Image(type="pil", label="Clean")
        gradcam_standard_distorted = gr.Image(type="pil", label="Distorted")

    gr.Markdown("### Robust Model")
    with gr.Row():
        gradcam_robust_clean = gr.Image(type="pil", label="Clean")
        gradcam_robust_distorted = gr.Image(type="pil", label="Distorted")

    gradcam_button.click(
        fn=run_gradcam,
        inputs=[
            image_input,
            degradation_slider,
        ],
        outputs=[
            gradcam_standard_clean,
            gradcam_standard_distorted,
            gradcam_robust_clean,
            gradcam_robust_distorted,
            gradcam_status,
        ],
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
            score_bars_output,
            comparison_table,
            severity_chart,
            interpretation_output,
        ],
    )


# ---------------------------------------------------------
# Start application
# ---------------------------------------------------------

if __name__ == "__main__":
    demo.launch(inbrowser=True, css=CSS)
