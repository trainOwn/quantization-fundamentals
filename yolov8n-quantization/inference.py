
import cv2
import numpy as np
import tensorflow as tf
import time
import argparse
import os

# COCO 80 classes
CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet",
    "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]


class YOLOv8TFLite:

    def __init__(
        self,
        model_path,
        conf_threshold=0.25,
        iou_threshold=0.45,
        num_threads=4
    ):
        self.interpreter = tf.lite.Interpreter(
            model_path=model_path,
            num_threads=num_threads
        )
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()[0]
        self.output_details = self.interpreter.get_output_details()

        self.input_shape = self.input_details["shape"]
        self.input_dtype = self.input_details["dtype"]
        self.input_scale, self.input_zero = self.input_details["quantization"]

        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        self.input_h = int(self.input_shape[1])
        self.input_w = int(self.input_shape[2])

        print("\nModel loaded")
        print("Input shape:", self.input_shape)
        print("Input dtype:", self.input_dtype)
        print("Input quantization:", self.input_details["quantization"])

        for i, output in enumerate(self.output_details):
            print(
                f"Output {i}:",
                output["shape"],
                output["dtype"],
                output["quantization"]
            )

    def letterbox(self, image):
        h, w = image.shape[:2]

        scale = min(
            self.input_w / w,
            self.input_h / h
        )

        new_w = int(round(w * scale))
        new_h = int(round(h * scale))

        resized = cv2.resize(
            image,
            (new_w, new_h),
            interpolation=cv2.INTER_LINEAR
        )

        canvas = np.full(
            (self.input_h, self.input_w, 3),
            114,
            dtype=np.uint8
        )

        pad_x = (self.input_w - new_w) // 2
        pad_y = (self.input_h - new_h) // 2

        canvas[
            pad_y:pad_y + new_h,
            pad_x:pad_x + new_w
        ] = resized

        return canvas, scale, pad_x, pad_y

    def quantize_input(self, image):
        image = image.astype(np.float32) / 255.0

        if self.input_dtype in (np.int8, np.uint8):
            if self.input_scale == 0:
                raise ValueError("Invalid input quantization scale")

            image = np.round(
                image / self.input_scale + self.input_zero
            )

            limits = np.iinfo(self.input_dtype)
            image = np.clip(
                image,
                limits.min,
                limits.max
            )

        return image.astype(self.input_dtype)

    def dequantize_output(self, output, details):
        if output.dtype in (np.int8, np.uint8):
            scale, zero = details["quantization"]

            if scale == 0:
                raise ValueError(
                    "Output is quantized but scale is zero"
                )

            output = (
                output.astype(np.float32) - zero
            ) * scale

        return output.astype(np.float32)

    def preprocess(self, image):
        canvas, scale, pad_x, pad_y = self.letterbox(image)

        canvas = cv2.cvtColor(
            canvas,
            cv2.COLOR_BGR2RGB
        )

        tensor = self.quantize_input(canvas)

        tensor = np.expand_dims(tensor, axis=0)

        return tensor, scale, pad_x, pad_y

    def infer(self, image):
        input_tensor, scale, pad_x, pad_y = self.preprocess(image)

        self.interpreter.set_tensor(
            self.input_details["index"],
            input_tensor
        )

        start = time.perf_counter()

        self.interpreter.invoke()

        inference_time = (
            time.perf_counter() - start
        ) * 1000

        outputs = []

        for details in self.output_details:
            output = self.interpreter.get_tensor(
                details["index"]
            )

            output = self.dequantize_output(
                output,
                details
            )

            outputs.append(output)

        return (
            outputs,
            inference_time,
            scale,
            pad_x,
            pad_y
        )

    def postprocess(
        self,
        outputs,
        original_shape,
        scale,
        pad_x,
        pad_y
    ):
        output = outputs[0]

        output = np.squeeze(output)

        # Standard YOLOv8 COCO output: (84, 8400)
        if output.ndim != 2:
            raise ValueError(
                f"Unexpected output shape: {output.shape}"
            )

        # Convert (84,8400) -> (8400,84)
        if output.shape[0] < output.shape[1]:
            output = output.T

        if output.shape[1] != 84:
            raise ValueError(
                f"Expected 84 features, got {output.shape[1]}. "
                "This script expects YOLOv8 COCO output "
                "with 4 box coordinates and 80 class scores."
            )

        boxes = output[:, :4]
        scores = output[:, 4:]

        class_ids = np.argmax(scores, axis=1)
        confidences = scores[
            np.arange(len(scores)),
            class_ids
        ]

        valid = confidences >= self.conf_threshold

        boxes = boxes[valid]
        confidences = confidences[valid]
        class_ids = class_ids[valid]

        if len(boxes) == 0:
            return []

        # YOLOv8 output boxes are cx, cy, w, h
        #cx = boxes[:, 0]
        #cy = boxes[:, 1]
        #bw = boxes[:, 2]
        #bh = boxes[:, 3]
        boxes[:, [0, 2]] *= self.input_w
        boxes[:, [1, 3]] *= self.input_h

        cx = boxes[:, 0]
        cy = boxes[:, 1]
        bw = boxes[:, 2]
        bh = boxes[:, 3]

        x1 = cx - bw / 2
        y1 = cy - bh / 2
        x2 = cx + bw / 2
        y2 = cy + bh / 2

        # Undo letterbox
        x1 = (x1 - pad_x) / scale
        y1 = (y1 - pad_y) / scale
        x2 = (x2 - pad_x) / scale
        y2 = (y2 - pad_y) / scale

        orig_h, orig_w = original_shape[:2]

        x1 = np.clip(x1, 0, orig_w)
        y1 = np.clip(y1, 0, orig_h)
        x2 = np.clip(x2, 0, orig_w)
        y2 = np.clip(y2, 0, orig_h)

        # Class-wise NMS
        detections = []

        for class_id in np.unique(class_ids):
            indices = np.where(class_ids == class_id)[0]

            class_boxes = [
                [
                    float(x1[i]),
                    float(y1[i]),
                    float(x2[i] - x1[i]),
                    float(y2[i] - y1[i])
                ]
                for i in indices
            ]

            class_scores = confidences[indices].tolist()

            keep = cv2.dnn.NMSBoxes(
                class_boxes,
                class_scores,
                self.conf_threshold,
                self.iou_threshold
            )

            if len(keep) == 0:
                continue

            for k in np.array(keep).flatten():
                i = indices[k]

                detections.append({
                    "class_id": int(class_ids[i]),
                    "class_name": CLASSES[int(class_ids[i])],
                    "confidence": float(confidences[i]),
                    "box": [
                        int(x1[i]),
                        int(y1[i]),
                        int(x2[i]),
                        int(y2[i])
                    ]
                })

        return detections

    def draw(self, image, detections):
        for det in detections:
            x1, y1, x2, y2 = det["box"]

            label = (
                f'{det["class_name"]} '
                f'{det["confidence"]:.2f}'
            )

            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                image,
                label,
                (x1, max(y1 - 8, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2
            )

        return image


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        default="yolov8n_int8_output/yolov8n_full_integer_quant.tflite"
    )

    parser.add_argument(
        "--image",
        required=True
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.25
    )

    parser.add_argument(
        "--iou",
        type=float,
        default=0.2
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=4
    )

    args = parser.parse_args()

    image = cv2.imread(args.image)

    if image is None:
        raise FileNotFoundError(args.image)

    detector = YOLOv8TFLite(
        args.model,
        args.conf,
        args.iou,
        args.threads
    )

    # Warmup
    for _ in range(3):
        detector.infer(image)

    outputs, inference_ms, scale, px, py = detector.infer(image)

    detections = detector.postprocess(
        outputs,
        image.shape,
        scale,
        px,
        py
    )

    result = detector.draw(image.copy(), detections)

    print("\nInference results")
    print("-----------------------")
    print(f"Inference time: {inference_ms:.2f} ms")
    print(f"FPS: {1000 / inference_ms:.2f}")
    print(f"Detections: {len(detections)}")

    for det in detections:
        print(
            det["class_name"],
            f'{det["confidence"]:.3f}',
            det["box"]
        )

    output_path = "yolov8_result.jpg"

    cv2.imwrite(output_path, result)

    print("\nSaved:", output_path)


if __name__ == "__main__":
    main()

