# YOLOv8n Export & Quantization Pipeline

End-to-end: pretrained `.pt` -> ONNX -> INT8 TFLite, calibrated on real COCO
images, ready for embedded board deployment (see `README_board.md` for
on-board deployment/run instructions).

## 0. Setup

```bash
uv venv
uv pip install -r requirements.txt
```

## 1. Download calibration dataset

Pulls 300 random images from the COCO 2017 validation split (via FiftyOne)
into `coco300/images/`. These are only used for calibration, not training.

```bash
uv run python download_coco300.py
```

## 2. Build the calibration tensor

Letterboxes each image to 640x640, converts BGR->RGB, normalizes to [0,1],
and stacks everything into `calibration_640.npy` (shape `[N, 640, 640, 3]`,
float32). This is what `onnx2tf` uses to compute int8 quantization ranges.

```bash
uv run python make_calibration.py
```

## 3. Export the model to ONNX

```bash
uv run yolo export model=yolov8n.pt format=onnx imgsz=640 opset=12
```

Produces `yolov8n.onnx`.

## 4. Quantize to INT8 TFLite

```bash
uv run onnx2tf -i yolov8n.onnx -o yolov8n_int8_output -oiqt -iqd int8 -oqd int8 --tflackend tf_converter -cind images calibration_640.npy "[[[[0.0,0.0,0.0]]]]" "[[[[1.0,1.0,1.0]]]]"
```

Flags:
- `-oiqt` output integer-quantized TFLite
- `-iqd int8` / `-oqd int8` input/output quantization dtype
- `-cind images calibration_640.npy ...` feeds the calibration tensor to the
  `images` input, with per-channel mean `[0,0,0]` and std `[1,1,1]`
  (matches the [0,1] normalization already baked into `calibration_640.npy`)

Output: `yolov8n_int8_output/yolov8n_full_integer_quant.tflite`, plus
float32/float16/dynamic-range variants for comparison.

## 5. Sanity-check locally

```bash
uv run python inference.py --model yolov8n_int8_output/yolov8n_full_integer_quant.tflite --image test.png
```

For deeper debugging of output layout/coordinate decoding:

```bash
uv run python debug_yolov8_tflite.py --model yolov8n_int8_output/yolov8n_full_integer_quant.tflite --image test.png
```

## 6. Deploy to an embedded board

See `README_board.md` — covers copying the model to the board, running
single-image NPU inference, and the live camera MJPEG stream.

