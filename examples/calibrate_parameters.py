"""
Parameter calibration script using open source ASR model as reference
"""

import os
import sys
import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass
from typing import Dict, List, Tuple
import json

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from src.main import SemMomentSTT
from src.integration.pipeline import IntegrationPipeline

@dataclass
class CalibrationResult:
    """Container for calibration results"""
    force_scale: float
    momentum_decay: float
    min_confidence: float
    merge_threshold: float
    wer: float
    reference_text: str
    generated_text: str

def load_asr_model(device: str = "cuda"):
    """Load ASR model for ground truth generation"""
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h").to(device)
    model.eval()
    return model, processor

def generate_ground_truth(
    model: Wav2Vec2ForCTC,
    processor: Wav2Vec2Processor,
    audio: torch.Tensor,
    device: str = "cuda"
) -> str:
    """Generate ground truth transcription using ASR model"""
    with torch.no_grad():
        # Normalize audio
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        if torch.abs(audio).max() > 1.0:
            audio = audio / torch.abs(audio).max()
        
        # Process audio
        inputs = processor(
            audio.squeeze().cpu().numpy(),
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )
        
        # Move input values to device
        input_values = inputs.input_values.to(device)
        
        # Forward pass
        logits = model(input_values).logits
        
        # Decode
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)[0]
        
        return transcription.lower()

def evaluate_parameters(
    audio: torch.Tensor,
    reference: str,
    force_scale: float,
    momentum_decay: float,
    min_confidence: float,
    merge_threshold: float,
    device: str = "cuda"
) -> CalibrationResult:
    """Evaluate semantic momentum parameters"""
    # Create pipeline with parameters
    pipeline = IntegrationPipeline(
        acoustic_model="facebook/wav2vec2-base",
        language_model="bert-base-uncased",
        device=device,
        force_scale=force_scale,
        momentum_decay=momentum_decay,
        min_confidence=min_confidence,
        merge_threshold=merge_threshold,
        context_window=5  # Reduced for shorter segments
    )
    
    # Initialize model
    model = SemMomentSTT(device=device)
    model.pipeline = pipeline
    
    # Process audio in chunks
    chunk_duration = 0.5  # seconds
    chunk_samples = int(16000 * chunk_duration)
    chunks = torch.split(audio, chunk_samples)
    
    transcriptions = []
    for chunk in chunks:
        if len(chunk) < chunk_samples:
            chunk = torch.nn.functional.pad(chunk, (0, chunk_samples - len(chunk)))
        
        # Normalize chunk
        if torch.abs(chunk).max() > 1.0:
            chunk = chunk / torch.abs(chunk).max()
        
        result = model.pipeline.process_frame(
            chunk.cpu().numpy(),
            orig_sr=16000,
            frame_duration=chunk_duration
        )
        
        if result.decoding_result and result.decoding_result.text:
            transcriptions.append(result.decoding_result.text)
    
    # Reset pipeline
    model.pipeline.reset()
    
    # Join transcriptions
    generated = " ".join(transcriptions) if transcriptions else ""
    
    # Calculate WER
    from jiwer import wer
    error_rate = wer(reference, generated)
    
    return CalibrationResult(
        force_scale=force_scale,
        momentum_decay=momentum_decay,
        min_confidence=min_confidence,
        merge_threshold=merge_threshold,
        wer=error_rate,
        reference_text=reference,
        generated_text=generated
    )

def grid_search_parameters(
    audio_samples: List[torch.Tensor],
    num_trials: int = 50,
    device: str = "cuda"
) -> List[CalibrationResult]:
    """Perform grid search over parameter space"""
    print("Loading ASR model for ground truth generation...")
    asr_model, processor = load_asr_model(device)
    
    # Generate ground truth transcriptions
    print("Generating ground truth transcriptions...")
    references = []
    for audio in tqdm(audio_samples):
        reference = generate_ground_truth(asr_model, processor, audio, device)
        print(f"Reference: {reference}")  # Print for verification
        references.append(reference)
    
    # Parameter ranges (adjusted based on test file)
    force_scales = np.linspace(5.0, 15.0, 10)  # Centered around 10.0
    momentum_decays = np.linspace(0.98, 0.999, 10)  # Higher range
    min_confidences = np.linspace(0.05, 0.2, 6)  # Lower range
    merge_thresholds = np.linspace(0.7, 0.9, 8)  # Middle range
    
    results = []
    print("\nEvaluating parameter combinations...")
    for _ in tqdm(range(num_trials)):
        # Randomly sample parameters
        params = {
            'force_scale': float(np.random.choice(force_scales)),
            'momentum_decay': float(np.random.choice(momentum_decays)),
            'min_confidence': float(np.random.choice(min_confidences)),
            'merge_threshold': float(np.random.choice(merge_thresholds))
        }
        
        print(f"\nTrying parameters: {params}")  # Print for monitoring
        
        # Evaluate on all samples
        sample_results = []
        for audio, reference in zip(audio_samples, references):
            result = evaluate_parameters(audio, reference, **params, device=device)
            print(f"Sample WER: {result.wer:.3f}")  # Print for monitoring
            print(f"Generated: {result.generated_text}")
            sample_results.append(result)
        
        # Average WER across samples
        avg_wer = np.mean([r.wer for r in sample_results])
        print(f"Average WER: {avg_wer:.3f}")
        
        # Store best result for these parameters
        best_result = min(sample_results, key=lambda r: r.wer)
        best_result.wer = avg_wer  # Use average WER
        results.append(best_result)
    
    # Sort by WER
    results.sort(key=lambda r: r.wer)
    
    return results

def main():
    """Main calibration function"""
    print("Starting parameter calibration...")
    
    # Use CUDA if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        print("WARNING: Running on CPU will be significantly slower")
    
    # Load LibriSpeech samples
    from datasets import load_dataset
    print("\nLoading LibriSpeech samples...")
    dataset = load_dataset("librispeech_asr", "clean", split="test", trust_remote_code=True)
    
    # Take first 10 samples
    audio_samples = []
    for idx in range(10):
        audio = torch.tensor(dataset[idx]["audio"]["array"])
        # Take first 5 seconds only
        max_samples = 5 * 16000
        if len(audio) > max_samples:
            audio = audio[:max_samples]
        audio_samples.append(audio)
    
    # Run grid search
    results = grid_search_parameters(audio_samples, num_trials=50, device=device)
    
    # Print best parameters
    print("\nBest Parameters:")
    print(f"Force Scale: {results[0].force_scale:.3f}")
    print(f"Momentum Decay: {results[0].momentum_decay:.3f}")
    print(f"Min Confidence: {results[0].min_confidence:.3f}")
    print(f"Merge Threshold: {results[0].merge_threshold:.3f}")
    print(f"Average WER: {results[0].wer:.3f}")
    
    print("\nExample Transcription:")
    print(f"Reference: {results[0].reference_text}")
    print(f"Generated: {results[0].generated_text}")
    
    # Save results
    print("\nSaving results...")
    with open("calibration_results.json", "w") as f:
        json.dump(
            [{
                'force_scale': r.force_scale,
                'momentum_decay': r.momentum_decay,
                'min_confidence': r.min_confidence,
                'merge_threshold': r.merge_threshold,
                'wer': r.wer,
                'reference_text': r.reference_text,
                'generated_text': r.generated_text
            } for r in results],
            f,
            indent=4
        )
    print("Done! Results saved to calibration_results.json")

if __name__ == "__main__":
    main()
