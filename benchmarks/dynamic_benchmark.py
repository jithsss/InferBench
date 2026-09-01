import time
import numpy as np
import onnxruntime as ort
import tensorrt as trt
import cv2
import torchaudio

class DynamicBenchmark:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.is_onnx = model_path.endswith(".onnx")
        self.is_engine = model_path.endswith(".engine") or model_path.endswith(".plan")
        
        self.input_shape = None
        self.input_dtype = None
        self.input_name = None
        
        if self.is_onnx:
            self._init_onnx()
        elif self.is_engine:
            self._init_tensorrt()
        else:
            raise ValueError("Unsupported model format. Use .onnx or .engine")

    def _init_onnx(self):
        self.sess = ort.InferenceSession(self.model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        inputs = self.sess.get_inputs()
        if len(inputs) == 0:
            raise ValueError("ONNX model has no inputs.")
        first_input = inputs[0]
        self.input_name = first_input.name
        
        shape = []
        for dim in first_input.shape:
            if isinstance(dim, str) or dim is None or dim <= 0:
                shape.append(1)  # Default dynamic axes to 1
            else:
                shape.append(dim)
        self.input_shape = tuple(shape)
        
        type_mapping = {
            'tensor(float)': np.float32,
            'tensor(float16)': np.float16,
            'tensor(int64)': np.int64,
            'tensor(int32)': np.int32,
            'tensor(double)': np.float64,
        }
        self.input_dtype = type_mapping.get(first_input.type, np.float32)

    def _init_tensorrt(self):
        logger = trt.Logger(trt.Logger.WARNING)
        with open(self.model_path, "rb") as f:
            engine_data = f.read()
        runtime = trt.Runtime(logger)
        self.engine = runtime.deserialize_cuda_engine(engine_data)
        if not self.engine:
            raise ValueError("Failed to deserialize TensorRT engine")
        self.context = self.engine.create_execution_context()
        
        # Get first input binding
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_name = name
                shape = self.engine.get_tensor_shape(name)
                # handle dynamic shapes
                static_shape = [1 if d <= 0 else d for d in shape]
                self.input_shape = tuple(static_shape)
                # context.set_input_shape(name, self.input_shape)
                trt_dtype = self.engine.get_tensor_dtype(name)
                if trt_dtype == trt.DataType.FLOAT:
                    self.input_dtype = np.float32
                elif trt_dtype == trt.DataType.HALF:
                    self.input_dtype = np.float16
                elif trt_dtype == trt.DataType.INT32:
                    self.input_dtype = np.int32
                else:
                    self.input_dtype = np.float32
                break

    def generate_dummy_data(self) -> np.ndarray:
        if self.input_dtype in [np.int32, np.int64]:
            return np.ones(self.input_shape, dtype=self.input_dtype)
        return np.random.randn(*self.input_shape).astype(self.input_dtype)

    def process_media(self, media_path: str) -> np.ndarray:
        ext = media_path.lower().split('.')[-1]
        
        # IMAGE
        if ext in ['png', 'jpg', 'jpeg']:
            img = cv2.imread(media_path)
            if img is None:
                raise ValueError("Failed to load image")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Find spatial dimensions in input shape [B, C, H, W] or [B, H, W, C]
            # Assuming NCHW for standard models
            if len(self.input_shape) == 4:
                h, w = self.input_shape[2], self.input_shape[3]
                img = cv2.resize(img, (w, h))
                img = img.transpose((2, 0, 1)) # HWC to CHW
                img = np.expand_dims(img, axis=0) # add batch
            elif len(self.input_shape) == 3:
                h, w = self.input_shape[1], self.input_shape[2]
                img = cv2.resize(img, (w, h))
                img = np.expand_dims(img, axis=0)
            
            img = img.astype(self.input_dtype)
            if self.input_dtype in [np.float32, np.float16]:
                img = img / 255.0
            return img
            
        # AUDIO
        elif ext in ['wav', 'mp3', 'flac', 'opus', 'mp4']:
            waveform, sample_rate = torchaudio.load(media_path)
            # Force mono
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            # Resample to 16k if needed (assuming Whisper-like)
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)
            
            arr = waveform.numpy().flatten()
            
            # If model expects 1D or [1, N]
            if len(self.input_shape) == 2:
                req_len = self.input_shape[1]
                if len(arr) > req_len:
                    arr = arr[:req_len]
                else:
                    arr = np.pad(arr, (0, req_len - len(arr)))
                return arr.reshape(1, req_len).astype(self.input_dtype)
            elif len(self.input_shape) == 3:
                # E.g. Mel Spectrograms, we just pass dummy if we can't do exact feature extraction
                # Best effort: flatten and reshape, or just fallback
                return self.generate_dummy_data()
            return self.generate_dummy_data()
        
        return self.generate_dummy_data()

    def load_npy(self, npy_path: str) -> np.ndarray:
        arr = np.load(npy_path)
        # Attempt to reshape/cast to model requirements
        arr = arr.astype(self.input_dtype)
        if arr.shape != self.input_shape:
            try:
                arr = arr.reshape(self.input_shape)
            except Exception:
                pass # Just pass it and let runtime crash if invalid
        return arr

    def run_benchmark(self, input_data: np.ndarray, warmup=5, iters=20):
        # ONNX Execution
        if self.is_onnx:
            for _ in range(warmup):
                self.sess.run(None, {self.input_name: input_data})
                
            latencies = []
            for _ in range(iters):
                start = time.perf_counter()
                self.sess.run(None, {self.input_name: input_data})
                latencies.append((time.perf_counter() - start) * 1000)
                
        # TensorRT Execution
        else:
            import torch
            import tensorrt as trt
            self.context.set_input_shape(self.input_name, self.input_shape)
            
            # Use PyTorch to manage CUDA memory instead of pycuda
            d_input = torch.from_numpy(input_data).cuda()
            
            # Find output bindings to allocate buffers
            outputs = []
            for i in range(self.engine.num_io_tensors):
                name = self.engine.get_tensor_name(i)
                if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                    shape = self.context.get_tensor_shape(name)
                    dtype = self.engine.get_tensor_dtype(name)
                    # Mapping trt dtype to torch dtype
                    if dtype == trt.DataType.FLOAT: pt_dtype = torch.float32
                    elif dtype == trt.DataType.HALF: pt_dtype = torch.float16
                    elif dtype == trt.DataType.INT32: pt_dtype = torch.int32
                    else: pt_dtype = torch.float32
                    
                    # Calculate volume safely
                    vol = 1
                    for d in shape:
                        vol *= d if d > 0 else 1
                    
                    # Allocate on device
                    d_out = torch.empty(tuple(shape), dtype=pt_dtype, device="cuda")
                    outputs.append((name, d_out))
            
            # Set tensor addresses
            self.context.set_tensor_address(self.input_name, d_input.data_ptr())
            for name, d_out in outputs:
                self.context.set_tensor_address(name, d_out.data_ptr())
            
            # Warmup
            stream = torch.cuda.current_stream()
            for _ in range(warmup):
                self.context.execute_async_v3(stream_handle=stream.cuda_stream)
            stream.synchronize()
            
            latencies = []
            for _ in range(iters):
                start = time.perf_counter()
                self.context.execute_async_v3(stream_handle=stream.cuda_stream)
                stream.synchronize()
                latencies.append((time.perf_counter() - start) * 1000)

        # Metrics
        avg_latency = np.mean(latencies)
        p50 = np.percentile(latencies, 50)
        p95 = np.percentile(latencies, 95)
        p99 = np.percentile(latencies, 99)
        fps = 1000.0 / avg_latency if avg_latency > 0 else 0
        
        return {
            "avg_latency": avg_latency,
            "p50": p50,
            "p95": p95,
            "p99": p99,
            "fps": fps,
            "input_shape": self.input_shape,
            "input_dtype": str(self.input_dtype)
        }
