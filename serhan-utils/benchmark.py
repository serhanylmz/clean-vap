import time
import torch
import numpy as np
from pathlib import Path
from argparse import ArgumentParser
from vap.modules.VAP import load_model_from_state_dict, step_extraction
from vap.modules.lightning_module import VAPModule
from vap.utils.audio import load_waveform
from vap.utils.utils import batch_to_device, everything_deterministic, tensor_dict_to_json

everything_deterministic()
torch.manual_seed(0)

##
## Essentially the same as test.py, but imitates saving output to align better with a real-world scenario. 
##

def benchmark_inference(model, waveform, chunk_time=20, step_time=5, num_runs=5):
    device = next(model.parameters()).device
    waveform = waveform.to(device)
    
    durations = []
    for _ in range(num_runs):
        start_time = time.time()
        
        if waveform.shape[-1] / model.sample_rate > 20:
            out = step_extraction(waveform, model, chunk_time=chunk_time, step_time=step_time)
        else:
            out = model.probs(waveform)
        
        out = batch_to_device(out, "cpu")
        _ = tensor_dict_to_json(out)  # Simulate saving output
        
        end_time = time.time()
        durations.append(end_time - start_time)
    
    avg_duration = np.mean(durations)
    std_duration = np.std(durations)
    
    return avg_duration, std_duration

def run_benchmark():
    parser = ArgumentParser()
    parser.add_argument("--audio", type=str, required=True, help="Path to waveform")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained model")
    parser.add_argument("--chunk_time", type=float, default=20, help="Duration of each chunk processed by model")
    parser.add_argument("--step_time", type=float, default=5, help="Increment to process in a step")
    parser.add_argument("--num_runs", type=int, default=1, help="Number of benchmark runs")
    args = parser.parse_args()

    assert Path(args.audio).exists(), f"Audio {args.audio} does not exist"
    assert Path(args.checkpoint).exists(), f"Checkpoint {args.checkpoint} does not exist"

    # Load the Model
    print("Loading Model...")
    model = VAPModule.load_model(args.checkpoint)
    model = model.eval()
    
    if torch.cuda.is_available():
        model = model.cuda()
        print("Model moved to CUDA")

    # Load the Audio
    print("Loading Audio...")
    waveform = load_waveform(args.audio, sample_rate=model.sample_rate, mono=False)[0].unsqueeze(0)
    duration = round(waveform.shape[-1] / model.sample_rate)
    print(f"Audio duration: {duration} seconds")

    # Run benchmark
    print("Running benchmark...")
    avg_duration, std_duration = benchmark_inference(
        model, waveform, args.chunk_time, args.step_time, args.num_runs
    )

    print(f"Average inference time: {avg_duration:.4f} seconds (±{std_duration:.4f})")
    print(f"Inference speed: {duration/avg_duration:.2f}x real-time")

if __name__ == "__main__":
    run_benchmark()