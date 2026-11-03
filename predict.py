from ultralytics import YOLO
import cv2


MODEL = (
    "runs/segment/runs/cervical_segmentation-5/"
    "weights/best.pt"
)

IMAGE = "test_xray.png"


model = YOLO(MODEL)


results = model.predict(
    source=IMAGE,

    imgsz=640,

    conf=0.25,

    save=True
)


result = results[0]


print("\nDetected vertebrae:")


if result.boxes is not None:

    for box in result.boxes:

        cls_id = int(
            box.cls[0]
        )

        confidence = float(
            box.conf[0]
        )

        class_name = (
            result.names[
                cls_id
            ]
        )

        print(
            class_name,
            f"{confidence:.3f}"
        )


annotated = result.plot()


cv2.imshow(
    "Cervical vertebra detection",
    annotated
)

cv2.waitKey(0)

cv2.destroyAllWindows()