<div align="center">

# 🔢 Quantization Fundamentals

### A from-scratch, PyTorch-only walkthrough of linear (uniform) INT8 quantization

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-only%20dependency-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-educational-brightgreen)]()

*No frameworks hiding the math. Every formula is implemented, plotted, and measured by hand.*

</div>

---

## 📖 Table of Contents

- [Why this repo](#-why-this-repo)
- [Quick start](#-quick-start)
- [The core equation](#-the-core-equation)
- [Visual: mapping float → int8](#️-visual-mapping-float--int8)
- [1. Quantize / Dequantize](#1️⃣-quantize--dequantize)
- [2. Deriving the optimal scale & zero-point](#2️⃣-deriving-the-optimal-scale--zero-point-asymmetric)
- [3. Symmetric quantization](#3️⃣-symmetric-quantization)
- [Asymmetric vs Symmetric](#️-asymmetric-vs-symmetric)
- [Why error exists](#-why-error-exists-the-rounding-staircase)
- [Measuring quality — MSE](#-measuring-quality--reconstruction-error)
- [Full walkthrough of the code](#-full-walkthrough-of-the-code)
- [Repo structure](#-repo-structure)
- [Cheat sheet](#-cheat-sheet)
- [Next steps](#️-next-steps-to-extend-this-repo)
- [FAQ](#-faq)

---

## 🎯 Why this repo

Most quantization tutorials jump straight to `torch.quantize_per_tensor(...)` or a framework API and hide the math behind a function call. This repo does the opposite: it **derives and implements the scale / zero-point math by hand**, then verifies it by round-tripping (quantize → dequantize) and *measuring* the reconstruction error — so you can see exactly where compression costs you precision, and why.

By the end you'll be able to answer:
- What do "scale" and "zero-point" actually mean, geometrically?
- How do you *compute* the best scale/zero-point for a tensor (not guess it)?
- What's the real difference between symmetric and asymmetric quantization?
- Why does quantization always introduce error, and how do you measure it?

---

## 🚀 Quick start

```bash
git clone https://github.com/trainOwn/quantization-fundamentals
cd quantization-fundamentals
pip install torch
python quant.py
```

<details>
<summary><b>Expected output (click to expand)</b></summary>

```text
original tensor:
tensor([[ 191.6000,  -13.5000,  728.6000],
        [  92.1400,  295.5000, -184.0000],
        [   0.0000,  684.6000,  245.5000]])

quantized tensor:
tensor([[ -6., -74., 127.],
        [-44.,  15., 127.],
        [-70., 127.,  -1.]])

dequantized tensor:
...
error : tensor(170.8753)

rmax: 728.6, rmin: -184.0
qmax: 127, qmin: -128
scale: 3.5788..., zero-point: -77
Dequantized tensor error: tensor(1.5730)

Symmetric scale: 5.7370...
error: 2.5092...
```

</details>

---

## 🧮 The core equation

Quantization maps a real-valued (float32) tensor `r` to a low-bit integer tensor `q`, and back again:

```
q = round(r / s) + z        # QUANTIZE   (float  → int8)
r = s * (q - z)              # DEQUANTIZE (int8   → float)
```

| Symbol | Name | Meaning |
|---|---|---|
| `s` | **scale** | A positive float — the real-number size of one integer "step" |
| `z` | **zero-point** | The integer that represents real value `0.0` |
| `q` | **quantized value** | Integer, clamped to the dtype's range (INT8: `-128 … 127`) |
| `r` | **real value** | The original / reconstructed float |

Because `round()` throws away information, `dequantize(quantize(r)) ≠ r` in general. That gap is the **quantization error** — minimizing it while still compressing the data is the whole game.

---

## 🗺️ Visual: mapping float → int8

Asymmetric quantization stretches the tensor's real min/max onto the integer dtype's full min/max, and tracks where real `0.0` lands:

<p align="center">
  <img src="assets/asymmetric_mapping.png" width="720" alt="Asymmetric mapping from float range to INT8 range">
</p>

> `rmin → qmin`, `rmax → qmax`, and real `0.0` lands on the **zero-point** `z` — not necessarily on integer `0`.

---

## 1️⃣ Quantize / Dequantize

```python
def linear_quant(tensor, scale, z=0, dtype=torch.int8):
    quantized = (tensor / scale) + z
    rounded_tensor = torch.round(quantized)
    q_min, q_max = torch.iinfo(dtype).min, torch.iinfo(dtype).max
    return torch.clamp(rounded_tensor, q_min, q_max)

def dequantize(tensor, scale, z=0):
    return scale * (tensor.float() - z)
```

<details>
<summary><b>Line-by-line breakdown</b></summary>

- `tensor / scale` — converts the real value into "how many scale-steps away from zero-point is this?"
- `+ z` — shifts so that real `0.0` lands on integer `z`, not `0`
- `torch.round(...)` — this is the lossy step; two nearby floats can round to the same integer
- `torch.clamp(..., q_min, q_max)` — anything outside the dtype's representable range gets clipped (saturation)

</details>

Given **any** `scale` and `z` you choose, this will produce a valid INT8 tensor and a valid reconstruction. But arbitrary values leave a lot of accuracy on the table — try `scale=3.5, z=-70` on the sample tensor and the MSE is a painful **170.88**. That raises the real question 👇

---

## 2️⃣ Deriving the optimal scale & zero-point (asymmetric)

We want the tensor's real range `[rmin, rmax]` to map **exactly** onto the integer dtype's range `[qmin, qmax]`:

```
rmax = s * (qmax - z)
rmin = s * (qmin - z)
```

Solving this system of two equations for `s` and `z`:

```
s = (rmax - rmin) / (qmax - qmin)
z = round(qmin - rmin / s)
```

`z` is then clamped into `[qmin, qmax]`, in case rounding pushes it just outside.

```python
def q_scale_zero_point(tensor, dtype=torch.int8):
    qmax, qmin = torch.iinfo(dtype).max, torch.iinfo(dtype).min
    rmax, rmin = tensor.max().item(), tensor.min().item()
    s = (rmax - rmin) / (qmax - qmin)
    z = int(round(qmin - rmin / s))
    z = max(qmin, min(qmax, z))
    return s, z
```

On the sample tensor this derives `s ≈ 3.579`, `z = -77` — and the MSE drops from **170.88 → 1.57**. This is called **asymmetric quantization**, because the zero-point isn't (usually) zero — it shifts to match however skewed the data is.

---

## 3️⃣ Symmetric quantization

When a tensor is roughly centered around zero (common for trained weights), you can force `z = 0` and simplify everything:

```
rmax = |max(tensor)|        # largest absolute value in the tensor
s    = rmax / qmax
q    = round(r / s)
r    = s * q
```

```python
def symmetric_scale(tensor, dtype=torch.int8):
    qmax = torch.iinfo(dtype).max
    rmax = torch.abs(tensor).max().item()
    return rmax / qmax
```

No zero-point bookkeeping, no offset subtraction at inference time — just multiply and round.

---

## ⚖️ Asymmetric vs Symmetric

<p align="center">
  <img src="assets/symmetric_vs_asymmetric.png" width="640" alt="Symmetric quantization wastes one INT8 code point">
</p>

| | Asymmetric | Symmetric |
|---|:---:|:---:|
| Zero-point `z` | Computed from data, usually ≠ 0 | Always `0` |
| Uses full int8 range (`-128…127`) | ✅ Yes | ⚠️ Effectively `-127…127` |
| Best for | Skewed / non-centered data (e.g. post-ReLU activations, all ≥ 0) | Zero-centered data (e.g. weights) |
| Inference cost | Slightly higher (subtract `z` every op) | Lower (pure scale-and-round) |
| Extra state to store | `s` **and** `z` | Just `s` |

**Rule of thumb:** symmetric for weights (cheap, usually already ~zero-centered), asymmetric for activations (often skewed, e.g. everything non-negative after a ReLU).

---

## 🪜 Why error exists: the rounding staircase

`torch.round()` collapses a continuous range of inputs onto one integer. Once dequantized, the result looks like a staircase instead of a straight line:

<p align="center">
  <img src="assets/rounding_error.png" width="620" alt="Rounding creates a staircase pattern, the gap to the ideal line is the quantization error">
</p>

The amber gap between the dashed "ideal" line and the red "quantized" staircase **is** the quantization error for every individual value. Squaring and averaging that gap across the whole tensor gives you the MSE.

---

## 📏 Measuring quality — reconstruction error

After every quantize → dequantize round trip, the script computes:

```python
error = (dequantized - original).square().mean()   # MSE
```

<p align="center">
  <img src="assets/mse_comparison.png" width="560" alt="Bar chart comparing MSE across arbitrary, optimal asymmetric, and symmetric quantization">
</p>

| Strategy | Scale `s` | Zero-point `z` | MSE ↓ |
|---|---:|---:|---:|
| Arbitrary (`s=3.5, z=-70`) | 3.5 | -70 | **170.88** |
| Derived optimal (asymmetric) | 3.579 | -77 | **1.57** |
| Symmetric (`z=0`) | 5.737 | 0 | **2.51** |

> On *this* tensor, asymmetric wins — because the data is heavily skewed (`rmin = -184`, `rmax = 728.6`, nowhere near symmetric around 0). This is exactly the trade-off table above predicts.

---

## 🔍 Full walkthrough of the code

<details>
<summary><b>Part A — Manual quantize/dequantize with arbitrary scale & zero-point</b></summary>

```python
test_tensor = torch.tensor([[191.6, -13.5, 728.6],
                             [92.14, 295.5, -184],
                             [0, 684.6, 245.5]])

tensor_quantized = linear_quant(test_tensor, scale=3.5, z=-70)
tensor_dequantized = dequantize(tensor_quantized, scale=3.5, z=-70)
```
Picking numbers out of thin air works, but produces high error (MSE ≈ 170.88) — motivating Part B.
</details>

<details>
<summary><b>Part B — Deriving scale & zero-point directly from tensor statistics</b></summary>

```python
new_s, new_z = q_scale_zero_point(test_tensor)   # s ≈ 3.579, z = -77
quant2 = linear_quant(test_tensor, scale=new_s, z=new_z)
dequant2 = dequantize(quant2, scale=new_s, z=new_z)
```
MSE drops to ≈ 1.57 — over 100x better than the arbitrary choice, because `s`/`z` now exactly span the tensor's real min/max onto INT8's min/max.
</details>

<details>
<summary><b>Part C — Symmetric quantization on a zero-centered tensor</b></summary>

```python
new_test_tensor = torch.tensor([[0.5034, -1.2635, 1.3994, 0.6222],
                                 [-1.0563, 1.1728, -3.3240, -0.4716],
                                 [0.2829, -0.5321, 0.5377, -0.2054],
                                 [0.0086, 1.2105, 1.3092, -1.5310]])

symmetric_quantized, scale = symmetric_quantize(new_test_tensor)
symmetric_dequantized = symmetric_dequant(symmetric_quantized, scale)
```
Here `z` is fixed at `0`, and only `scale = rmax / qmax` needs to be computed.
</details>

---

## 📁 Repo structure

```
quantization-fundamentals/
├── README.md
├── quantization_fundamentals.py     # runnable, self-contained script
└── assets/
    ├── asymmetric_mapping.png
    ├── symmetric_vs_asymmetric.png
    ├── rounding_error.png
    └── mse_comparison.png
```

---

## 📋 Cheat sheet

```text
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 ASYMMETRIC                    SYMMETRIC
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 s = (rmax-rmin)/(qmax-qmin)   rmax = |max(tensor)|
 z = round(qmin - rmin/s)      s = rmax / qmax
 q = round(r/s) + z            z = 0
 r = s * (q - z)               q = round(r/s)
                                r = s * q
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 INT8 range:  qmin = -128,  qmax = 127
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## 🛣️ Next steps to extend this repo

- [ ] **Per-channel quantization** — a separate `s`/`z` per output channel instead of one per tensor (much better accuracy for weight matrices)
- [ ] **INT4 vs INT8** — compare error at lower bit-widths on the same tensors
- [ ] **Quantized matmul benchmark** — FP32 vs INT8-with-dequant runtime comparison
- [ ] **Calibration** — derive `s`/`z` from a representative dataset instead of a single tensor
- [ ] **Per-tensor vs per-group** quantization for large weight matrices

Contributions and PRs for any of the above are welcome.

---

## ❓ FAQ

<details>
<summary><b>Why INT8 specifically?</b></summary>

INT8 is the most common target for quantization because it's a 4x memory reduction over FP32, is natively supported by most CPU/GPU/accelerator instruction sets, and — as this repo shows — can be tuned to keep reconstruction error very low with the right scale/zero-point.
</details>

<details>
<summary><b>Why does the "arbitrary" example use scale=3.5, z=-70 at all?</b></summary>

It's a deliberately unoptimized starting point, used purely to demonstrate that quantization *works* mechanically with any valid scale/zero-point — but that choosing them well (Part B) is what actually makes it useful.
</details>

<details>
<summary><b>Is this how real frameworks (PyTorch, TensorRT, ONNX Runtime) do it?</b></summary>

The core math is identical — `torch.quantize_per_tensor` and friends implement exactly this affine mapping under the hood. Production frameworks add per-channel scales, calibration over datasets, and fused kernels, which is exactly the direction listed in [Next steps](#️-next-steps-to-extend-this-repo).
</details>

---

<div align="center">

*Built to make quantization click — not just to call an API.*

</div>
