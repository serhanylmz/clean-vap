import torch
import torchaudio

def add_noise_to_stereo_torch(input_file, output_file, noise_level=0.005):
    # Read the audio file
    waveform, sample_rate = torchaudio.load(input_file)
    
    # Ensure the audio is stereo
    if waveform.shape[0] != 2:
        raise ValueError("Input audio must be stereo")
    
    # Generate noise
    noise = torch.randn_like(waveform) * noise_level
    
    # Add noise to the audio data
    noisy_waveform = waveform + noise
    
    # Clip the data to the range [-1, 1]
    noisy_waveform = torch.clamp(noisy_waveform, -1.0, 1.0)
    
    # Write the noisy audio to a new file
    torchaudio.save(output_file, noisy_waveform, sample_rate)

# Usage
input_file = "audio.wav"
output_file = "audio_noisy.wav"
add_noise_to_stereo_torch(input_file, output_file, noise_level=0.005)