
import streamlit as st
import cv2
import numpy as np

from PIL import Image
from ultralytics import YOLO
from pathlib import Path


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

MODEL_PATH = Path(
    "runs/segment/runs/cervical_segmentation-5/weights/best.pt"
)

CLASS_NAMES = [
    "C2",
    "C3",
    "C4",
    "C5",
    "C6",
    "C7"
]


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    return YOLO(str(MODEL_PATH))


# --------------------------------------------------
# UI
# --------------------------------------------------

st.set_page_config(
    page_title="Cervical X-ray AI",
    page_icon="🦴",
    layout="wide"
)

st.title("Cervical Vertebra Detection")

st.write(
    "Upload a lateral cervical X-ray to detect "
    "and label the cervical vertebrae."
)

st.divider()


# --------------------------------------------------
# UPLOAD IMAGE
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Select a cervical X-ray",
    type=["png", "jpg", "jpeg"]
)


if uploaded_file is not None:

    # Load image as RGB
    pil_image = Image.open(
        uploaded_file
    ).convert("RGB")

    image_rgb = np.array(pil_image)

    st.subheader("Uploaded X-ray")

    st.image(
        image_rgb,
        caption="Original image",
        use_container_width=True
    )

    st.divider()

    # --------------------------------------------------
    # RUN PREDICTION
    # --------------------------------------------------

    if st.button(
        "Run Vertebra Detection",
        type="primary"
    ):

        with st.spinner(
            "Running model inference..."
        ):

            model = load_model()

            # Convert RGB to BGR for Ultralytics
            image_bgr = cv2.cvtColor(
                image_rgb,
                cv2.COLOR_RGB2BGR
            )

            results = model.predict(
                source=image_bgr,
                imgsz=640,
                conf=0.25,
                verbose=False
            )

            result = results[0]

            # Generate annotated image
            annotated_bgr = result.plot()

            # Convert back to RGB for Streamlit
            annotated_rgb = cv2.cvtColor(
                annotated_bgr,
                cv2.COLOR_BGR2RGB
            )

        # --------------------------------------------------
        # DISPLAY RESULTS
        # --------------------------------------------------

        st.success("Prediction completed!")

        col1, col2 = st.columns(2)

        with col1:

            st.subheader("Original")

            st.image(
                image_rgb,
                use_container_width=True
            )

        with col2:

            st.subheader("AI Prediction")

            st.image(
                annotated_rgb,
                use_container_width=True
            )

        # --------------------------------------------------
        # DETECTION INFORMATION
        # --------------------------------------------------

        st.subheader("Detected Vertebrae")

        detections = []

        if result.boxes is not None:

            for box in result.boxes:

                class_id = int(
                    box.cls[0].item()
                )

                confidence = float(
                    box.conf[0].item()
                )

                class_name = result.names[
                    class_id
                ]

                detections.append({
                    "Vertebra": class_name,
                    "Confidence": f"{confidence:.2%}"
                })

        if detections:

            st.table(detections)

        else:

            st.warning(
                "No vertebrae detected."
            )

else:

    st.info(
        "Upload an X-ray image to get started."
    )

st.caption(
    "Research demonstration only. "
    "Not intended for clinical diagnosis."
)
