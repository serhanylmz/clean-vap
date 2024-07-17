import torch
import torchaudio

def stereo_mixer_with_delay(audio, sample_rate=16000):
    channels, seq_length = audio.shape
    max_delay_samples = int(0.3 * sample_rate)  # 300ms
    
    # Pad the input
    padded_left = torch.nn.functional.pad(audio[0], (max_delay_samples, max_delay_samples))
    padded_right = torch.nn.functional.pad(audio[1], (max_delay_samples, max_delay_samples))
    
    # Generate random scaling factors and delays
    left_to_right_scale = torch.rand(1).item() * 0.5
    right_to_left_scale = torch.rand(1).item() * 0.5
    left_to_right_delay = torch.randint(0, max_delay_samples, (1,)).item()
    right_to_left_delay = torch.randint(0, max_delay_samples, (1,)).item()
    
    # Mix channels
    mixed_left = padded_left[max_delay_samples:-max_delay_samples].clone()
    mixed_right = padded_right[max_delay_samples:-max_delay_samples].clone()
    
    mixed_left += right_to_left_scale * padded_right[max_delay_samples + right_to_left_delay : max_delay_samples + right_to_left_delay + seq_length]
    mixed_right += left_to_right_scale * padded_left[max_delay_samples + left_to_right_delay : max_delay_samples + left_to_right_delay + seq_length]
    
    # Normalize to prevent clipping
    max_val = torch.max(torch.abs(torch.stack([mixed_left, mixed_right])))
    if max_val > 1:
        mixed_left /= max_val
        mixed_right /= max_val
    
    return torch.stack((mixed_left, mixed_right))

def test_stereo_mixer(input_file, output_file):
    # Load the audio file
    waveform, sample_rate = torchaudio.load(input_file)
    
    # Ensure the audio is stereo
    if waveform.shape[0] == 1:
        waveform = waveform.repeat(2, 1)
    
    # Apply the stereo mixing
    mixed_waveform = stereo_mixer_with_delay(waveform, sample_rate)
    
    # Save the mixed audio
    torchaudio.save(output_file, mixed_waveform, sample_rate)
    
    print(f"Mixed audio saved to: {output_file}")

# Example usage
if __name__ == "__main__":
    input_path = "/home/erik/projects/data/Fisher/fisher_eng_tr_sp_d1/audio/002/fe_03_00208.wav"
    output_path = "/home/serhan/Desktop/VoiceActivityProjection/serhan-utils/mixed_audio.wav"
    test_stereo_mixer(input_path, output_path)