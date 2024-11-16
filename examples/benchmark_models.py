import os
import sys

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

import torch
import torchaudio
from datasets import load_dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, WhisperProcessor, WhisperForConditionalGeneration
from src.main import SemMomentSTT
from src.integration.pipeline import IntegrationPipeline
import numpy as np
from tqdm import tqdm
import jiwer
from typing import Dict, List
import json
from datetime import datetime

class ModelBenchmark:
    def __init__(self):
        # Force CUDA if available
        if not torch.cuda.is_available():
            print("WARNING: CUDA is not available. Running on CPU will be significantly slower.")
            self.device = "cpu"
        else:
            self.device = "cuda"
            # Print GPU info
            print(f"Using GPU: {torch.cuda.get_device_name()}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            # Empty cache before starting
            torch.cuda.empty_cache()
        
        # Create pipeline with optimized parameters
        pipeline = IntegrationPipeline(
            acoustic_model="facebook/wav2vec2-base",
            language_model="bert-base-uncased",
            semantic_dim=768,
            device=self.device,
            force_scale=10.0,
            momentum_decay=0.999,
            min_confidence=0.1,
            merge_threshold=0.85,
            context_window=5  # Reduced for shorter segments
        )
        
        # Initialize our model with the optimized pipeline
        self.sem_moment = SemMomentSTT(
            model_name="facebook/wav2vec2-base",
            language_model="bert-base-uncased",
            device=self.device
        )
        self.sem_moment.pipeline = pipeline
        
        # Initialize Wav2Vec2
        self.wav2vec2_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
        self.wav2vec2_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h").to(self.device)
        
        # Initialize Whisper
        self.whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-base")
        self.whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base").to(self.device)
        
        # Set models to eval mode
        self.wav2vec2_model.eval()
        self.whisper_model.eval()

    def preprocess_audio(self, audio, sample_rate):
        # Resample if needed (LibriSpeech is 16kHz)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            if self.device == "cuda":
                resampler = resampler.cuda()
                audio = audio.cuda()
            audio = resampler(audio)
        elif self.device == "cuda":
            audio = audio.cuda()
        
        # Normalize audio
        audio = audio / (torch.max(torch.abs(audio)) + 1e-8)
        return audio

    def evaluate_wav2vec2(self, audio, sample_rate):
        audio = self.preprocess_audio(audio, sample_rate)
        # Process audio with processor
        processed = self.wav2vec2_processor(
            audio.cpu().numpy(),
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )
        
        # Move processed inputs to device
        input_values = processed.input_values.to(self.device)
        attention_mask = processed.attention_mask.to(self.device) if hasattr(processed, 'attention_mask') else None
        
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16) if self.device == "cuda" else torch.no_grad():
            outputs = self.wav2vec2_model(
                input_values,
                attention_mask=attention_mask
            )
            predicted_ids = torch.argmax(outputs.logits, dim=-1)
        
        transcription = self.wav2vec2_processor.batch_decode(predicted_ids)
        return transcription[0]

    def evaluate_whisper(self, audio, sample_rate):
        audio = self.preprocess_audio(audio, sample_rate)
        inputs = self.whisper_processor(audio.cpu().numpy(), sampling_rate=16000, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items() if torch.is_tensor(v)}
        
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16) if self.device == "cuda" else torch.no_grad():
            generated_ids = self.whisper_model.generate(inputs["input_features"])
        
        transcription = self.whisper_processor.batch_decode(generated_ids, skip_special_tokens=True)
        return transcription[0]

    def evaluate_semantic_momentum(self, audio, sample_rate):
        audio = self.preprocess_audio(audio, sample_rate)
        
        # Split audio into smaller chunks for better processing
        chunk_duration = 0.5  # seconds
        chunk_samples = int(16000 * chunk_duration)
        chunks = torch.split(audio, chunk_samples)
        
        transcriptions = []
        for chunk in chunks:
            if len(chunk) < chunk_samples:
                # Pad last chunk if needed
                chunk = torch.nn.functional.pad(chunk, (0, chunk_samples - len(chunk)))
            
            # Process chunk
            result = self.sem_moment.pipeline.process_frame(
                chunk.cpu().numpy(),
                orig_sr=16000,
                frame_duration=chunk_duration
            )
            
            # Get transcription from result
            if result.decoding_result and result.decoding_result.text:
                transcriptions.append(result.decoding_result.text)
            elif result.trajectory is not None:
                # Try to decode the trajectory directly
                decoding = self.sem_moment.pipeline.text_decoder.decode_trajectory(
                    result.trajectory,
                    timestamp=0.0,
                    duration=chunk_duration
                )
                if decoding is not None:
                    transcriptions.append(decoding.text)
        
        # Reset pipeline state
        self.sem_moment.pipeline.reset()
        
        # Join transcriptions
        return " ".join(transcriptions) if transcriptions else ""

    def run_benchmark(self, num_samples: int = 100):
        # Load LibriSpeech test dataset with trust_remote_code=True
        print("Loading LibriSpeech test dataset...")
        dataset = load_dataset("librispeech_asr", "clean", split="test", trust_remote_code=True)
        
        results = {
            "wav2vec2": {"wer": [], "time": []},
            "whisper": {"wer": [], "time": []},
            "semantic_momentum": {"wer": [], "time": []}
        }
        
        for idx in tqdm(range(min(num_samples, len(dataset)))):
            sample = dataset[idx]
            audio = torch.tensor(sample["audio"]["array"])
            sample_rate = sample["audio"]["sampling_rate"]
            reference = sample["text"].lower()
            
            # Take first 5 seconds only for more focused evaluation
            max_samples = 5 * sample_rate
            if len(audio) > max_samples:
                audio = audio[:max_samples]
                # Adjust reference text length proportionally
                words = reference.split()
                reference = " ".join(words[:len(words)//2])
            
            # Evaluate each model
            for model_name, eval_func in {
                "wav2vec2": self.evaluate_wav2vec2,
                "whisper": self.evaluate_whisper,
                "semantic_momentum": self.evaluate_semantic_momentum
            }.items():
                # Clear cache before each evaluation
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                
                start_time = datetime.now()
                try:
                    hypothesis = eval_func(audio, sample_rate)
                    time_taken = (datetime.now() - start_time).total_seconds()
                    
                    # Ensure hypothesis is a string and convert to lowercase
                    hypothesis = str(hypothesis).lower()
                    
                    # Print semantic momentum outputs for debugging
                    if model_name == "semantic_momentum" and hypothesis:
                        print(f"\nSemantic Momentum output: {hypothesis}")
                    
                    wer = jiwer.wer(reference, hypothesis)
                    results[model_name]["wer"].append(wer)
                    results[model_name]["time"].append(time_taken)
                except Exception as e:
                    print(f"\nError evaluating {model_name}: {str(e)}")
                    results[model_name]["wer"].append(1.0)  # Maximum error
                    results[model_name]["time"].append(0.0)
        
        # Calculate aggregate metrics
        final_results = {}
        for model_name in results:
            final_results[model_name] = {
                "average_wer": np.mean(results[model_name]["wer"]),
                "std_wer": np.std(results[model_name]["wer"]),
                "average_time": np.mean(results[model_name]["time"]),
                "std_time": np.std(results[model_name]["time"])
            }
        
        # Save results
        with open("benchmark_results.json", "w") as f:
            json.dump(final_results, f, indent=4)
        
        return final_results

if __name__ == "__main__":
    print("Starting benchmark...")
    benchmark = ModelBenchmark()
    results = benchmark.run_benchmark(num_samples=100)
    
    # Print results
    print("\nBenchmark Results:")
    print("=" * 50)
    for model_name, metrics in results.items():
        print(f"\n{model_name.upper()}:")
        print(f"Average WER: {metrics['average_wer']:.4f} (±{metrics['std_wer']:.4f})")
        print(f"Average Time: {metrics['average_time']:.4f}s (±{metrics['std_time']:.4f}s)")
