# ============================================================
# 🫁 PULMONARY NODULE AI — CLEAN UI
# YOLO11n CT Image Analysis
#
# The trained model path can be configured with MODEL_PATH.
# Default: weights/best.pt
# ============================================================

import os
import numpy as np
import gradio as gr

from PIL import Image, ImageDraw
from ultralytics import YOLO


# ============================================================
# 1. LOAD MODEL
# ============================================================

MODEL_PATH = os.getenv("MODEL_PATH", "weights/best.pt")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found:\n{MODEL_PATH}")

model = YOLO(MODEL_PATH)

print("✅ YOLO11n model loaded successfully")
print("Classes:", model.names)


# ============================================================
# 2. COLORS
# ============================================================

CLASS_COLORS = {
    "benign": (34, 197, 94),        # green
    "equivocal": (245, 158, 11),    # yellow/orange
    "malignant": (239, 68, 68),     # red
}


def get_color(class_name):
    return CLASS_COLORS.get(
        class_name.lower(),
        (99, 102, 241)
    )


# ============================================================
# 3. DRAW ONLY BOXES
# ============================================================

def draw_boxes(image, result):

    if isinstance(image, np.ndarray):
        image = Image.fromarray(
            image.astype(np.uint8)
        )

    image = image.convert("RGB")

    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)

    if result.boxes is None or len(result.boxes) == 0:
        return annotated

    boxes = result.boxes.xyxy.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)

    width, height = annotated.size

    # Box thickness proportional to image size
    thickness = max(3, int(width / 250))

    for box, cls in zip(boxes, classes):

        x1, y1, x2, y2 = map(int, box)

        class_name = model.names[int(cls)]

        color = get_color(class_name)

        # ONLY DRAW THE BOX
        draw.rectangle(
            [x1, y1, x2, y2],
            outline=color,
            width=thickness
        )

    return annotated


# ============================================================
# 4. CREATE RESULT HTML
# ============================================================

def create_result_html(detections):

    if len(detections) == 0:

        return """
        <div class="empty-result">

            <div class="empty-icon">✓</div>

            <div>
                <div class="empty-title">
                    No nodules detected
                </div>

                <div class="empty-text">
                    No objects were detected above the
                    selected confidence threshold.
                </div>
            </div>

        </div>
        """


    # --------------------------------------------------------
    # Sort by confidence
    # --------------------------------------------------------

    detections = sorted(
        detections,
        key=lambda x: x["confidence"],
        reverse=True
    )


    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    html = f"""
    <div class="results-container">

        <div class="results-header">

            <div>

                <div class="results-title">
                    Detection Results
                </div>

                <div class="results-subtitle">
                    {len(detections)}
                    nodule{"s" if len(detections) != 1 else ""}
                    detected
                </div>

            </div>

            <div class="result-count">
                {len(detections)}
            </div>

        </div>
    """


    # --------------------------------------------------------
    # Detection cards
    # --------------------------------------------------------

    for i, detection in enumerate(detections):

        class_name = detection["class"]
        confidence = detection["confidence"]

        percentage = confidence * 100

        color = get_color(class_name)

        color_css = (
            f"rgb({color[0]},"
            f"{color[1]},"
            f"{color[2]})"
        )


        html += f"""
        <div class="detection-card">

            <div class="detection-top">

                <div class="detection-number">
                    Detection #{i + 1}
                </div>

                <div
                    class="class-badge"
                    style="
                        color:{color_css};
                        background:rgba(
                            {color[0]},
                            {color[1]},
                            {color[2]},
                            0.12
                        );
                    "
                >

                    <span
                        class="dot"
                        style="background:{color_css};"
                    ></span>

                    {class_name.capitalize()}

                </div>

            </div>


            <div class="confidence-row">

                <div class="confidence-label">
                    Confidence
                </div>

                <div class="confidence-value">
                    {percentage:.1f}%
                </div>

            </div>


            <div class="confidence-track">

                <div
                    class="confidence-progress"
                    style="
                        width:{percentage:.1f}%;
                        background:{color_css};
                    "
                ></div>

            </div>

        </div>
        """


    html += """
    </div>
    """

    return html


# ============================================================
# 5. ANALYSIS
# ============================================================

def analyze_ct(image, confidence_threshold):

    if image is None:

        return (
            None,
            """
            <div class="empty-result">

                <div class="empty-icon">!</div>

                <div>

                    <div class="empty-title">
                        Upload a CT image
                    </div>

                    <div class="empty-text">
                        Please upload a CT image before
                        starting the analysis.
                    </div>

                </div>

            </div>
            """
        )


    try:

        # ----------------------------------------------------
        # YOLO inference
        # ----------------------------------------------------

        results = model.predict(
            source=image,
            conf=float(confidence_threshold),
            verbose=False
        )

        result = results[0]


        # ----------------------------------------------------
        # Draw boxes
        # ----------------------------------------------------

        annotated_image = draw_boxes(
            image,
            result
        )


        # ----------------------------------------------------
        # Extract detections
        # ----------------------------------------------------

        detections = []


        if (
            result.boxes is not None
            and len(result.boxes) > 0
        ):

            classes = (
                result.boxes.cls
                .cpu()
                .numpy()
                .astype(int)
            )

            confidences = (
                result.boxes.conf
                .cpu()
                .numpy()
            )


            for cls, conf in zip(
                classes,
                confidences
            ):

                detections.append({

                    "class": model.names[int(cls)],

                    "confidence": float(conf)

                })


        # ----------------------------------------------------
        # Create report
        # ----------------------------------------------------

        report = create_result_html(
            detections
        )


        return annotated_image, report


    except Exception as e:

        print("ERROR:", e)

        return (
            None,

            f"""
            <div class="error-result">

                <div class="error-title">
                    ⚠️ Analysis Error
                </div>

                <div class="error-text">
                    {str(e)}
                </div>

            </div>
            """
        )


# ============================================================
# 6. CLEAR
# ============================================================

def clear_app():

    return (
        None,
        None,
        0.25
    )


# ============================================================
# 7. CSS
# ============================================================

CSS = """

/* ============================================================
   MAIN
   ============================================================ */

body {
    background: #080d19 !important;
}

.gradio-container {
    max-width: 1200px !important;
    margin: auto !important;
    background: #080d19 !important;
}


/* ============================================================
   HEADER
   ============================================================ */

.app-header {
    text-align: center;
    padding: 35px 10px 25px;
}

.app-title {
    font-size: 34px;
    font-weight: 800;
    color: #f8fafc;
}

.app-subtitle {
    margin-top: 7px;
    color: #94a3b8;
    font-size: 14px;
}

.legend {
    display: flex;
    justify-content: center;
    gap: 25px;
    margin-top: 18px;
}

.legend-item {
    color: #cbd5e1;
    font-size: 13px;
}

.legend-dot {
    width: 9px;
    height: 9px;
    border-radius: 50%;
    display: inline-block;
    margin-right: 6px;
}


/* ============================================================
   PANELS
   ============================================================ */

.panel {
    background: #111827;
    border: 1px solid #263249;
    border-radius: 18px;
    padding: 18px;
}


/* ============================================================
   SECTION HEADERS
   ============================================================ */

.section-title {
    color: #f1f5f9;
    font-size: 17px;
    font-weight: 700;
    margin-bottom: 4px;
}

.section-description {
    color: #64748b;
    font-size: 12px;
    margin-bottom: 12px;
}


/* ============================================================
   RESULTS
   ============================================================ */

.results-container {
    margin-top: 16px;
}

.results-header {
    display: flex;
    justify-content: space-between;
    align-items: center;

    background: #111827;
    border: 1px solid #263249;

    border-radius: 15px;

    padding: 16px 18px;

    margin-bottom: 10px;
}

.results-title {
    color: #f8fafc;
    font-size: 17px;
    font-weight: 700;
}

.results-subtitle {
    color: #64748b;
    font-size: 12px;
    margin-top: 3px;
}

.result-count {
    width: 40px;
    height: 40px;

    display: flex;
    align-items: center;
    justify-content: center;

    border-radius: 12px;

    background: #312e81;

    color: #a5b4fc;

    font-size: 18px;
    font-weight: 800;
}


/* ============================================================
   DETECTION CARD
   ============================================================ */

.detection-card {
    background: #111827;

    border: 1px solid #263249;

    border-radius: 14px;

    padding: 16px 18px;

    margin-bottom: 9px;
}

.detection-top {
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.detection-number {
    color: #94a3b8;
    font-size: 12px;
    font-weight: 600;
}

.class-badge {
    padding: 6px 10px;

    border-radius: 8px;

    font-size: 13px;

    font-weight: 700;
}

.dot {
    width: 8px;
    height: 8px;

    display: inline-block;

    border-radius: 50%;

    margin-right: 5px;
}

.confidence-row {
    display: flex;
    justify-content: space-between;

    margin-top: 15px;
    margin-bottom: 7px;
}

.confidence-label {
    color: #64748b;
    font-size: 11px;
}

.confidence-value {
    color: #f8fafc;
    font-size: 13px;
    font-weight: 700;
}

.confidence-track {
    height: 7px;

    background: #263249;

    border-radius: 20px;

    overflow: hidden;
}

.confidence-progress {
    height: 100%;

    border-radius: 20px;
}


/* ============================================================
   EMPTY RESULT
   ============================================================ */

.empty-result {
    display: flex;
    align-items: center;

    background: #111827;

    border: 1px solid #263249;

    border-radius: 15px;

    padding: 20px;

    margin-top: 16px;
}

.empty-icon {
    width: 45px;
    height: 45px;

    border-radius: 12px;

    display: flex;
    align-items: center;
    justify-content: center;

    background: #1e293b;

    color: #94a3b8;

    font-size: 22px;
    font-weight: 800;

    margin-right: 14px;
}

.empty-title {
    color: #f8fafc;
    font-weight: 700;
    font-size: 15px;
}

.empty-text {
    color: #64748b;
    font-size: 12px;
    margin-top: 3px;
}


/* ============================================================
   ERROR
   ============================================================ */

.error-result {
    background: #2a1418;

    border: 1px solid #7f1d1d;

    border-radius: 15px;

    padding: 20px;

    margin-top: 16px;
}

.error-title {
    color: #fca5a5;
    font-weight: 700;
}

.error-text {
    color: #cbd5e1;
    font-size: 12px;
    margin-top: 7px;
}


/* ============================================================
   BUTTONS
   ============================================================ */

.analyze-button {
    height: 52px !important;

    border-radius: 12px !important;

    font-size: 16px !important;

    font-weight: 700 !important;

    margin-top: 12px;
}

.clear-button {
    height: 44px !important;

    border-radius: 10px !important;

    margin-top: 8px;
}


/* ============================================================
   SETTINGS
   ============================================================ */

.settings {
    background: #111827;

    border: 1px solid #263249;

    border-radius: 15px;

    padding: 15px 18px;

    margin-top: 14px;
}


/* ============================================================
   FOOTER
   ============================================================ */

.footer {
    text-align: center;

    color: #64748b;

    font-size: 11px;

    padding: 28px 10px 35px;

    line-height: 1.6;
}

.footer strong {
    color: #94a3b8;
}

"""


# ============================================================
# 8. GRADIO APP
# ============================================================

with gr.Blocks(
    css=CSS,
    title="Pulmonary Nodule AI"
) as app:


    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    gr.HTML(
        """
        <div class="app-header">

            <div class="app-title">
                🫁 Pulmonary Nodule AI
            </div>

            <div class="app-subtitle">
                YOLO11n-based CT Image Analysis
            </div>

            <div class="legend">

                <div class="legend-item">

                    <span
                        class="legend-dot"
                        style="background:#22c55e;"
                    ></span>

                    Benign

                </div>

                <div class="legend-item">

                    <span
                        class="legend-dot"
                        style="background:#f59e0b;"
                    ></span>

                    Equivocal

                </div>

                <div class="legend-item">

                    <span
                        class="legend-dot"
                        style="background:#ef4444;"
                    ></span>

                    Malignant

                </div>

            </div>

        </div>
        """
    )


    # ========================================================
    # IMAGE SECTION
    # ========================================================

    with gr.Row(equal_height=True):


        # ----------------------------------------------------
        # INPUT
        # ----------------------------------------------------

        with gr.Column():

            with gr.Group(
                elem_classes="panel"
            ):

                gr.HTML(
                    """
                    <div class="section-title">
                        CT Image
                    </div>

                    <div class="section-description">
                        Upload a CT slice for analysis.
                    </div>
                    """
                )

                input_image = gr.Image(
                    type="numpy",
                    label="",
                    height=520
                )


        # ----------------------------------------------------
        # OUTPUT
        # ----------------------------------------------------

        with gr.Column():

            with gr.Group(
                elem_classes="panel"
            ):

                gr.HTML(
                    """
                    <div class="section-title">
                        Analysis Result
                    </div>

                    <div class="section-description">
                        Detected nodules are marked with
                        bounding boxes.
                    </div>
                    """
                )

                output_image = gr.Image(
                    type="pil",
                    label="",
                    height=520
                )


    # ========================================================
    # RESULTS
    # ========================================================

    output_report = gr.HTML(
        """
        <div class="empty-result">

            <div class="empty-icon">
                🫁
            </div>

            <div>

                <div class="empty-title">
                    Ready for analysis
                </div>

                <div class="empty-text">
                    Upload a CT image and click
                    "Analyze CT Scan".
                </div>

            </div>

        </div>
        """
    )


    # ========================================================
    # SETTINGS + BUTTONS
    # ========================================================

    with gr.Row():

        with gr.Column(scale=1):

            with gr.Group(
                elem_classes="settings"
            ):

                confidence = gr.Slider(
                    minimum=0.05,
                    maximum=0.90,
                    value=0.25,
                    step=0.01,
                    label="Confidence Threshold",
                    info=(
                        "Minimum confidence required "
                        "to display a detection."
                    )
                )


        with gr.Column(scale=1):

            analyze_button = gr.Button(
                "🔍  Analyze CT Scan",
                variant="primary",
                elem_classes="analyze-button"
            )


        with gr.Column(scale=0.6):

            clear_button = gr.Button(
                "↻  Clear",
                variant="secondary",
                elem_classes="clear-button"
            )


    # ========================================================
    # FOOTER
    # ========================================================

    gr.HTML(
        """
        <div class="footer">

            <strong>Research Prototype Only</strong>

            <br>

            This application is intended for research,
            educational and demonstration purposes only.

            <br>

            It is <strong>not a medical diagnostic tool</strong>
            and should not be used for clinical
            decision-making.

            <br><br>

            YOLO11n • Pulmonary Nodule Detection
            • CT Image Analysis

        </div>
        """
    )


    # ========================================================
    # EVENTS
    # ========================================================

    analyze_button.click(
        fn=analyze_ct,

        inputs=[
            input_image,
            confidence
        ],

        outputs=[
            output_image,
            output_report
        ]
    )


    clear_button.click(
        fn=clear_app,

        inputs=[],

        outputs=[
            input_image,
            output_image,
            confidence
        ]
    )


# ============================================================
# 9. START APP
# ============================================================

if __name__ == "__main__":
    share = os.getenv("GRADIO_SHARE", "false").lower() == "true"

    app.launch(
        share=share,
        debug=False
    )