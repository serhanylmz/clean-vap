import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from argparse import ArgumentParser
from vap.modules.VAP import step_extraction, load_model_from_state_dict
from vap.modules.lightning_module import VAPModule
from vap.utils.audio import load_waveform
from vap.utils.plot import plot_stereo
from vap.utils.utils import batch_to_device, everything_deterministic, tensor_dict_to_json, write_json

everything_deterministic()
torch.manual_seed(0)

def benchmark_inference(model, waveform, chunk_time=20, step_time=19.9, num_runs=5, use_cache=True):
    device = next(model.parameters()).device
    waveform = waveform.to(device)
    
    # Warm-up run
    _ = step_extraction(waveform, model, chunk_time=chunk_time, step_time=step_time, use_cache=use_cache)
    
    # Benchmark runs
    durations = []
    outputs = []
    for _ in range(num_runs):
        if hasattr(model, 'reset_cache'):
            model.reset_cache()
        start_time = time.time()
        out = step_extraction(waveform, model, chunk_time=chunk_time, step_time=step_time, use_cache=use_cache)
        end_time = time.time()
        durations.append(end_time - start_time)
        outputs.append(out)
    
    avg_duration = np.mean(durations)
    std_duration = np.std(durations)
    
    return avg_duration, std_duration, outputs, durations

def create_plot(output, duration, run_index, use_cache, waveform):
    fig, ax = plot_stereo(
        waveform[0].cpu(),
        p_now=output["p_now"][0].cpu(),
        p_fut=output["p_future"][0].cpu(),
        vad=output["vad"][0].cpu(),
    )
    
    cache_status = "with_cache" if use_cache else "without_cache"
    filename = f'run_{run_index}_{cache_status}.png'
    fig.savefig(filename)
    plt.close(fig)
    print(f"Plot for run {run_index} saved as {filename}")

def load_vap_model(args):
    if args.state_dict:
        model = load_model_from_state_dict(args.state_dict)
    elif args.checkpoint:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = VAPModule.load_model(args.checkpoint, map_location=device)
    else:
        raise ValueError("Must provide state_dict or checkpoint")
    return model

def run_benchmark():
    parser = ArgumentParser()
    parser.add_argument("--audio", type=str, required=True, help="Path to waveform")
    parser.add_argument("--checkpoint", type=str, help="Path to trained model")
    parser.add_argument("--state_dict", type=str, help="Path to state_dict")
    parser.add_argument("--chunk_time", type=float, default=20, help="Duration of each chunk processed by model")
    parser.add_argument("--step_time", type=float, default=5, help="Increment to process in a step")
    parser.add_argument("--num_runs", type=int, default=5, help="Number of benchmark runs")
    parser.add_argument("--use_cache", action="store_true", help="Use KV-cache for inference")
    parser.add_argument("--plot", action="store_true", help="Create plots for each run")
    parser.add_argument("--output", type=str, default="vap_output.json", help="Path to save output JSON")
    args = parser.parse_args()

    assert Path(args.audio).exists(), f"Audio {args.audio} does not exist"
    assert args.state_dict is not None or args.checkpoint is not None, "Must provide state_dict or checkpoint"

    # Load the Model
    print("Loading Model...")
    model = load_vap_model(args)
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
    avg_duration, std_duration, outputs, durations = benchmark_inference(
        model, waveform, args.chunk_time, args.step_time, args.num_runs, args.use_cache
    )

    print(f"Average inference time: {avg_duration:.4f} seconds (±{std_duration:.4f})")
    print(f"Inference speed: {duration/avg_duration:.2f}x real-time")
    print(f"KV-cache: {'Enabled' if args.use_cache else 'Disabled'}")

    # Save Output
    out = outputs[0]  # Save output from the first run
    out = batch_to_device(out, "cpu")
    data = tensor_dict_to_json(out)
    write_json(data, args.output)
    print(f"Saved output -> {args.output}")

    if args.plot:
        for i, (output, duration) in enumerate(zip(outputs, durations)):
            create_plot(output, duration, i+1, args.use_cache, waveform)

if __name__ == "__main__":
    run_benchmark()