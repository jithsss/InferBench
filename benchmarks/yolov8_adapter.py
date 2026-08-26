import numpy as np
import onnxruntime as ort
import tensorrt as trt
import torch
from pathlib import Path
from PIL import Image

def get_calibration_images(image_dir="quantization/calibration/calibration_set", max_images=10):
    paths = sorted(Path(image_dir).glob("*.jpg"))
    return paths[:max_images]

def preprocess_image(image_path: Path, input_shape=(1, 3, 640, 640)):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((input_shape[3], input_shape[2]))
    image_array = np.asarray(image, dtype=np.float32) / 255.0
    image_array = np.transpose(image_array, (2, 0, 1))
    image_array = np.expand_dims(image_array, axis=0)
    return np.ascontiguousarray(image_array, dtype=np.float32)

def run_onnx_fp32(onnx_path, input_data):
    session = ort.InferenceSession(onnx_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_data})
    return outputs[0]

def run_trt_engine(engine_path, input_data):
    logger = trt.Logger(trt.Logger.ERROR)
    runtime = trt.Runtime(logger)
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
        
    context = engine.create_execution_context()
    
    input_name = engine.get_tensor_name(0)
    output_name = engine.get_tensor_name(1)
    
    context.set_input_shape(input_name, input_data.shape)
    
    input_tensor = torch.from_numpy(input_data).cuda()
    output_shape = context.get_tensor_shape(output_name)
    output_tensor = torch.empty(tuple(output_shape), dtype=torch.float32, device="cuda")
    
    context.set_tensor_address(input_name, input_tensor.data_ptr())
    context.set_tensor_address(output_name, output_tensor.data_ptr())
    
    stream = torch.cuda.current_stream().cuda_stream
    context.execute_async_v3(stream)
    torch.cuda.synchronize()
    
    return output_tensor.cpu().numpy()

def compute_prediction_agreement(engine_path: str, onnx_path: str = "export/yolov8n.onnx") -> float:
    """
    Computes a reproducible FP32-vs-optimized output agreement metric
    by measuring the cosine similarity of the raw output tensors.
    """
    images = get_calibration_images(max_images=10)
    if not images:
        return 0.0
        
    similarities = []
    for img_path in images:
        input_data = preprocess_image(img_path)
        
        # Run Baseline (FP32 ONNX)
        fp32_out = run_onnx_fp32(onnx_path, input_data).flatten()
        
        # Run Optimized (TensorRT)
        trt_out = run_trt_engine(engine_path, input_data).flatten()
        
        # Cosine similarity
        cos_sim = np.dot(fp32_out, trt_out) / (np.linalg.norm(fp32_out) * np.linalg.norm(trt_out) + 1e-9)
        similarities.append(float(cos_sim))
        
    # Return percentage agreement
    return float(np.mean(similarities) * 100.0)
