import time
import torch
import numpy as np
from queue import Queue
from threading import Thread
from argparse import ArgumentParser
from vap.modules.VAP import step_extraction
from vap.modules.lightning_module import VAPModule
from vap.utils.audio import load_waveform

class AudioStreamer:
    def __init__(self, model, chunk_time=20, step_time=0.1, sample_rate=16000):
        self.model = model
        self.chunk_time = chunk_time
        self.step_time = step_time
        self.sample_rate = sample_rate
        self.chunk_samples = int(chunk_time * sample_rate)
        self.step_samples = int(step_time * sample_rate)
        self.audio_buffer = torch.zeros((1, 2, self.chunk_samples), dtype=torch.float32)
        self.processing_queue = Queue()
        self.last_processed_time = 0
        self.is_running = False

    def add_audio(self, new_audio):
        # Shift the buffer and add new audio
        self.audio_buffer = torch.cat([self.audio_buffer[:, :, self.step_samples:], new_audio], dim=-1)
        self.processing_queue.put(self.audio_buffer.clone())

    def process_audio(self):
        while self.is_running:
            if not self.processing_queue.empty():
                start_time = time.time()
                audio_chunk = self.processing_queue.get()
                out = step_extraction(audio_chunk, self.model, self.chunk_time, self.step_time, pbar=False, use_cache=True)
                processing_time = time.time() - start_time
                self.last_processed_time = time.time()
                
                # Here you can do something with the output, like sending it to a visualization thread
                print(f"Processed chunk in {processing_time:.4f} seconds")
            else:
                time.sleep(0.001)  # Small sleep to prevent busy waiting

    def start(self):
        self.is_running = True
        self.processing_thread = Thread(target=self.process_audio)
        self.processing_thread.start()

    def stop(self):
        self.is_running = False
        if self.processing_thread.is_alive():
            self.processing_thread.join()

def simulate_audio_stream(streamer, audio_path, duration=60):
    waveform, _ = load_waveform(audio_path, sample_rate=streamer.sample_rate, mono=False)
    waveform = waveform.unsqueeze(0)  # Add batch dimension
    
    start_time = time.time()
    while time.time() - start_time < duration:
        current_time = time.time() - start_time
        sample_index = int(current_time * streamer.sample_rate)
        if sample_index + streamer.step_samples <= waveform.shape[-1]:
            new_audio = waveform[:, :, sample_index:sample_index + streamer.step_samples]
            streamer.add_audio(new_audio)
        time.sleep(streamer.step_time)

def load_vap_model(checkpoint_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VAPModule.load_model(checkpoint_path, map_location=device)
    model = model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    return model

def main():
    parser = ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=False, default="/home/serhan/Desktop/VoiceActivityProjection/serhan-utils/checkpoints/audio_1/epoch=1-step=21761.ckpt", help="Path to trained model checkpoint")
    parser.add_argument("--audio", type=str, required=False, default="/home/erik/projects/data/Fisher/fisher_eng_tr_sp_d1/audio/002/fe_03_00208.wav",  help="Path to audio file for streaming simulation")
    parser.add_argument("--duration", type=int, default=60, help="Duration of streaming simulation in seconds")
    parser.add_argument("--step_time", type=float, default=0.01, help="Time step for audio chunks in seconds")
    args = parser.parse_args()

    model = load_vap_model(args.checkpoint)
    print("Model loaded and moved to", next(model.parameters()).device)

    streamer = AudioStreamer(model, step_time=args.step_time)
    streamer.start()

    # Simulate audio stream
    simulate_audio_stream(streamer, args.audio, duration=args.duration)

    streamer.stop()

if __name__ == "__main__":
    main()