
import os
import cv2
import time
import argparse
import numpy as np
import tensorflow as tf

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


def section(title):
    print("\n" + "=" * 75)
    print(title)
    print("=" * 75)


class YOLODebugger:

    def __init__(self, model_path, threads=4):
        section("LOADING TFLITE MODEL")

        self.interpreter = tf.lite.Interpreter(
            model_path=model_path,
            num_threads=threads
        )

        self.interpreter.allocate_tensors()

        self.inputs = self.interpreter.get_input_details()
        self.outputs = self.interpreter.get_output_details()

        self.input = self.inputs[0]

        print("Input tensors:", len(self.inputs))
        print("Output tensors:", len(self.outputs))

        for t in self.inputs:
            print("\nINPUT:")
            print("Name:", t["name"])
            print("Shape:", t["shape"])
            print("Shape signature:", t["shape_signature"])
            print("Dtype:", t["dtype"])
            print("Quantization:", t["quantization"])

        for i, t in enumerate(self.outputs):
            print(f"\nOUTPUT {i}:")
            print("Name:", t["name"])
            print("Shape:", t["shape"])
            print("Dtype:", t["dtype"])
            print("Quantization:", t["quantization"])

        shape = self.input["shape"]

        if len(shape) != 4:
            raise ValueError("Expected NHWC or NCHW 4D input")

        if shape[-1] == 3:
            self.layout = "NHWC"
            self.height = int(shape[1])
            self.width = int(shape[2])
        elif shape[1] == 3:
            self.layout = "NCHW"
            self.height = int(shape[2])
            self.width = int(shape[3])
        else:
            raise ValueError(f"Unknown input layout: {shape}")

        print("\nInput layout:", self.layout)
        print("Input resolution:", self.width, "x", self.height)

    def letterbox(self, image):
        h, w = image.shape[:2]

        scale = min(
            self.width / w,
            self.height / h
        )

        nw = int(round(w * scale))
        nh = int(round(h * scale))

        resized = cv2.resize(
            image,
            (nw, nh),
            interpolation=cv2.INTER_LINEAR
        )

        canvas = np.full(
            (self.height, self.width, 3),
            114,
            dtype=np.uint8
        )

        pad_x = (self.width - nw) // 2
        pad_y = (self.height - nh) // 2

        canvas[
            pad_y:pad_y + nh,
            pad_x:pad_x + nw
        ] = resized

        print("\nPREPROCESSING")
        print("Original:", w, "x", h)
        print("Resized:", nw, "x", nh)
        print("Scale:", scale)
        print("Padding:", pad_x, pad_y)

        return canvas, scale, pad_x, pad_y

    def prepare_input(self, image):
        image, scale, px, py = self.letterbox(image)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # YOLO standard normalization
        image = image.astype(np.float32) / 255.0

        if self.input["dtype"] in (np.int8, np.uint8):
            qscale, zero = self.input["quantization"]

            if qscale == 0:
                raise ValueError("Invalid input quantization scale")

            image = np.round(image / qscale + zero)

            limits = np.iinfo(self.input["dtype"])

            image = np.clip(
                image,
                limits.min,
                limits.max
            )

        image = image.astype(self.input["dtype"])

        if self.layout == "NCHW":
            image = np.transpose(image, (2, 0, 1))

        image = np.expand_dims(image, axis=0)

        print("\nINPUT TENSOR")
        print("Shape:", image.shape)
        print("Dtype:", image.dtype)
        print("Min:", image.min())
        print("Max:", image.max())
        print("Mean:", image.mean())

        return image, scale, px, py

    def run_inference(self, tensor):
        section("INFERENCE")

        self.interpreter.set_tensor(
            self.input["index"],
            tensor
        )

        start = time.perf_counter()

        self.interpreter.invoke()

        elapsed = (time.perf_counter() - start) * 1000

        print("Inference time:", round(elapsed, 3), "ms")
        print("FPS:", round(1000 / elapsed, 2))

        results = []

        for t in self.outputs:
            output = self.interpreter.get_tensor(t["index"])

            print("\nRAW OUTPUT")
            print("Shape:", output.shape)
            print("Dtype:", output.dtype)
            print("Min:", output.min())
            print("Max:", output.max())

            if output.dtype in (np.int8, np.uint8):
                scale, zero = t["quantization"]

                if scale == 0:
                    raise ValueError("Invalid output quantization")

                output = (
                    output.astype(np.float32) - zero
                ) * scale

            results.append(output.astype(np.float32))

        return results, elapsed

    def inspect_output(self, outputs):
        section("OUTPUT FORMAT ANALYSIS")

        output = np.squeeze(outputs[0])

        print("Squeezed shape:", output.shape)

        if output.ndim != 2:
            raise ValueError("Expected a 2D detection output")

        if output.shape[0] == 84:
            print("Detected format: [84, N]")
            output = output.T
        elif output.shape[1] == 84:
            print("Detected format: [N, 84]")
        else:
            raise ValueError(
                f"Unexpected YOLOv8 COCO output: {output.shape}"
            )

        print("Final shape:", output.shape)

        boxes = output[:, :4]
        scores = output[:, 4:]

        print("\nBOX STATISTICS")
        for i, name in enumerate(["X", "Y", "W", "H"]):
            values = boxes[:, i]
            print(
                name,
                "min:", values.min(),
                "max:", values.max(),
                "mean:", values.mean(),
                "std:", values.std()
            )

        print("\nSCORE STATISTICS")
        print("Min:", scores.min())
        print("Max:", scores.max())

        class_ids = np.argmax(scores, axis=1)
        confidence = np.max(scores, axis=1)

        print("\nTOP 20 CANDIDATES")

        top = np.argsort(confidence)[-20:][::-1]

        for idx in top:
            print(
                "Index:", idx,
                "Class:", CLASSES[class_ids[idx]],
                "Score:", round(float(confidence[idx]), 4),
                "Box:", boxes[idx].tolist()
            )

        return output

    def decode_boxes(
        self,
        output,
        original_shape,
        scale,
        px,
        py,
        coordinate_mode
    ):
        boxes = output[:, :4].copy()
        scores = output[:, 4:]

        class_ids = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)

        valid = (
            (confidences >= 0.25) &
            np.isfinite(boxes).all(axis=1)
        )

        boxes = boxes[valid]
        confidences = confidences[valid]
        class_ids = class_ids[valid]

        if len(boxes) == 0:
            return []

        if coordinate_mode == "normalized":
            boxes[:, [0, 2]] *= self.width
            boxes[:, [1, 3]] *= self.height

        elif coordinate_mode == "pixel":
            pass

        elif coordinate_mode == "auto":
            # Only a hypothesis, not proof of normalization
            if np.max(np.abs(boxes)) <= 2.0:
                boxes[:, [0, 2]] *= self.width
                boxes[:, [1, 3]] *= self.height

        cx, cy, bw, bh = boxes.T

        x1 = cx - bw / 2
        y1 = cy - bh / 2
        x2 = cx + bw / 2
        y2 = cy + bh / 2

        # Map back from letterboxed image
        x1 = (x1 - px) / scale
        x2 = (x2 - px) / scale
        y1 = (y1 - py) / scale
        y2 = (y2 - py) / scale

        orig_h, orig_w = original_shape[:2]

        x1 = np.clip(x1, 0, orig_w - 1)
        x2 = np.clip(x2, 0, orig_w - 1)
        y1 = np.clip(y1, 0, orig_h - 1)
        y2 = np.clip(y2, 0, orig_h - 1)

        detections = []

        for cid in np.unique(class_ids):
            ids = np.where(class_ids == cid)[0]

            valid_boxes = []
            for i in ids:
                bw = x2[i] - x1[i]
                bh = y2[i] - y1[i]

                if bw <= 1 or bh <= 1:
                    continue

                valid_boxes.append([
                    float(x1[i]),
                    float(y1[i]),
                    float(bw),
                    float(bh)
                ])

            if not valid_boxes:
                continue

            # Keep mapping between valid_boxes and original IDs
            valid_ids = [
                i for i in ids
                if x2[i] - x1[i] > 1 and y2[i] - y1[i] > 1
            ]

            keep = cv2.dnn.NMSBoxes(
                valid_boxes,
                confidences[valid_ids].tolist(),
                0.25,
                0.45
            )

            if len(keep) == 0:
                continue

            for k in np.array(keep).flatten():
                i = valid_ids[k]

                detections.append({
                    "class_id": int(cid),
                    "class_name": CLASSES[int(cid)],
                    "confidence": float(confidences[i]),
                    "box": [
                        int(x1[i]),
                        int(y1[i]),
                        int(x2[i]),
                        int(y2[i])
                    ]
                })

        return detections

    def draw(self, image, detections, filename):
        result = image.copy()

        for d in detections:
            x1, y1, x2, y2 = d["box"]

            cv2.rectangle(
                result,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                3
            )

            label = (
                f'{d["class_name"]}: '
                f'{d["confidence"]:.2f}'
            )

            cv2.putText(
                result,
                label,
                (x1, max(25, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        cv2.imwrite(filename, result)
        print("Saved:", filename)

    def run(self, image_path):
        image = cv2.imread(image_path)

        if image is None:
            raise FileNotFoundError(image_path)

        section("ORIGINAL IMAGE")
        print("Image shape:", image.shape)

        tensor, scale, px, py = self.prepare_input(image)

        outputs, elapsed = self.run_inference(tensor)

        output = self.inspect_output(outputs)

        section("TESTING COORDINATE INTERPRETATIONS")

        modes = ["pixel", "normalized", "auto"]

        for mode in modes:
            print("\nTesting:", mode)

            detections = self.decode_boxes(
                output,
                image.shape,
                scale,
                px,
                py,
                mode
            )

            print("Detections:", len(detections))

            for d in detections[:10]:
                print(
                    d["class_name"],
                    round(d["confidence"], 3),
                    d["box"]
                )

            filename = f"debug_{mode}.jpg"

            self.draw(
                image,
                detections,
                filename
            )

        section("DEBUGGING COMPLETE")
        print("Compare these files:")
        print("debug_pixel.jpg")
        print("debug_normalized.jpg")
        print("debug_auto.jpg")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        required=True
    )

    parser.add_argument(
        "--image",
        required=True
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=4
    )

    args = parser.parse_args()

    detector = YOLODebugger(
        args.model,
        args.threads
    )

    detector.run(args.image)


if __name__ == "__main__":
    main()

