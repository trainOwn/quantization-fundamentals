import torch

# original equation: r = s * (q - z)

test_tensor = torch.tensor(
    [[191.6, -13.5, 728.6],
     [92.14, 295.5,  -184],
     [0,     684.6, 245.5]]
)
print("original tensor:")
print(test_tensor)


def linear_quant(tensor, scale, z=0, dtype=torch.int8):
    """Quantize a float tensor to an integer dtype using scale + zero-point."""
    quantized = (tensor / scale) + z
    rounded_tensor = torch.round(quantized)
    q_min = torch.iinfo(dtype).min
    q_max = torch.iinfo(dtype).max
    quantized = torch.clamp(rounded_tensor, q_min, q_max)
    return quantized


tensor_quantized = linear_quant(test_tensor, scale=3.5, z=-70)
print("quantized tensor:")
print(tensor_quantized.float())


def dequantize(tensor, scale, z=0):
    """Recover an approximate float tensor from quantized values."""
    dequant = scale * (tensor.float() - z)
    return dequant


tensor_dequantized = dequantize(tensor_quantized, scale=3.5, z=-70)
print("dequantized tensor:")
print(tensor_dequantized)
print("error :", (tensor_dequantized - test_tensor).square().mean())

# ===================== Now we need to find r and q and s ===============
# s = (rmax - rmin) / (qmax - qmin)
# z = int(round(qmin - rmin / s))


def q_scale_zero_point(tensor, dtype=torch.int8):
    """Derive the optimal scale and zero-point directly from tensor stats."""
    qmax = torch.iinfo(dtype).max
    qmin = torch.iinfo(dtype).min
    rmax = tensor.max().item()
    rmin = tensor.min().item()
    print(f"rmax: {rmax}, rmin: {rmin}")
    print(f"qmax: {qmax}, qmin: {qmin}")
    s = (rmax - rmin) / (qmax - qmin)
    z = int(round(qmin - rmin / s))
    if z > qmax:
        z = qmax
    if z < qmin:
        z = qmin
    print(f"scale: {s}, zero-point: {z}")
    return s, z


new_s, new_z = q_scale_zero_point(test_tensor)
quant2 = linear_quant(test_tensor, scale=new_s, z=new_z)
dequant2 = dequantize(quant2, scale=new_s, z=new_z)
print("Dequantized tensor error:", (dequant2 - test_tensor).square().mean())

# ===================== symmetric and asymmetric modes ==============
# asymmetric: map rmin -> qmin and rmax -> qmax
# symmetric:  map -rmax -> -qmax and rmax -> qmax, where rmax = |max(tensor)|
# in symmetric mode z = 0, so r = s * q
# q = int(round(r / s))
# s = rmax / qmax


def symmetric_scale(tensor, dtype=torch.int8):
    qmax = torch.iinfo(dtype).max
    abs_tensor = torch.abs(tensor)
    rmax = abs_tensor.max().item()
    return rmax / qmax


def symmetric_quantize(tensor):
    s = symmetric_scale(tensor)
    quantized = linear_quant(tensor, scale=s, z=0)
    print("Symmetric scale:", s)
    print("quantized tensor:", quantized)
    return quantized, s


def symmetric_dequant(tensor, scale):
    d = dequantize(tensor, scale=scale, z=0)
    return d


new_test_tensor = torch.tensor(
    [[0.5034, -1.2635, 1.3994, 0.6222],
     [-1.0563, 1.1728, -3.3240, -0.4716],
     [0.2829, -0.5321, 0.5377, -0.2054],
     [0.0086, 1.2105, 1.3092, -1.5310]], dtype=torch.float32)

symmetric_quantized, scale = symmetric_quantize(new_test_tensor)
print("Symmetric quantized tensor:", symmetric_quantized)
symmetric_dequantized = symmetric_dequant(symmetric_quantized, scale)
print("error:", (symmetric_dequantized - new_test_tensor).square().mean())

# ==================================================================================
# understanding ---
# s = rmax / qmax
# z = 0
# quant   = (r / s) + z
# dequant = s * (q - z)
# qmin, qmax = int8 (-128, 127)
# rmax = |max(tensor)|
