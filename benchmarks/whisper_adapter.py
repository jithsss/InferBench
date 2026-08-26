import time
import urllib.request
from pathlib import Path

import torch
import torchaudio
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import jiwer

AUDIO_URL = "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/1.flac"
LOCAL_AUDIO = "benchmarks/sample.flac"
EXPECTED_TEXT = "he hoped there would be stew for dinner turnips and carrots and bruised potatoes and fat mutton pieces to be ladled out in thick peppered flour fattened sauce"


class WhisperAdapter:
    def __init__(self, model_name: str, precision: str = "FP32", device: str = "cuda"):
        self.model_name = f"openai/whisper-{model_name}"
        self.precision = precision
        self.device = device

        torch_dtype = torch.float16 if precision == "FP16" else torch.float32

        t0 = time.time()
        self.processor = WhisperProcessor.from_pretrained(self.model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch_dtype
        ).to(self.device)
        self.load_time_ms = (time.time() - t0) * 1000

        self.model.eval()

        audio_path = Path(LOCAL_AUDIO)
        if not audio_path.exists():
            print(f"Downloading benchmark audio to {LOCAL_AUDIO}...")
            urllib.request.urlretrieve(AUDIO_URL, LOCAL_AUDIO)

        self.audio_path = LOCAL_AUDIO

    def load_and_preprocess(self):
        t0 = time.time()
        waveform, sample_rate = torchaudio.load(self.audio_path)

        # Mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Resample to 16kHz
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)

        waveform = waveform.squeeze().numpy()
        self.audio_duration = len(waveform) / 16000.0

        inputs = self.processor(waveform, sampling_rate=16000, return_tensors="pt")
        input_features = inputs.input_features.to(self.device)

        if self.precision == "FP16":
            input_features = input_features.half()

        self.prep_time_ms = (time.time() - t0) * 1000
        return input_features

    def evaluate(self, input_features):
        with torch.no_grad():
            predicted_ids = self.model.generate(
                input_features, 
                language="en", 
                task="transcribe"
            )

        transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

        transcription = transcription.lower().strip()
        expected = EXPECTED_TEXT.lower().strip()

        wer = jiwer.wer(expected, transcription) * 100.0
        cer = jiwer.cer(expected, transcription) * 100.0

        return transcription, wer, cer
