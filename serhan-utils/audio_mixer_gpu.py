import torch

def gpu_stereo_mixer_with_delay(audio_batch, sample_rate=44100):
    left, right = audio_batch[:, :, 0], audio_batch[:, :, 1]
    batch_size, seq_length = left.shape
    
    # Calculate max delay in samples
    max_delay_samples = int(0.3 * sample_rate)  # 300ms
    
    # Pad the input
    padded_left = torch.nn.functional.pad(left, (max_delay_samples, max_delay_samples))
    padded_right = torch.nn.functional.pad(right, (max_delay_samples, max_delay_samples))
    
    # Generate uniform random scaling factors between 0 and 0.5
    left_to_right_scale = torch.rand(batch_size, 1, device=audio_batch.device) * 0.5
    right_to_left_scale = torch.rand(batch_size, 1, device=audio_batch.device) * 0.5
    
    # Generate random delays up to 300ms
    left_to_right_delay = torch.randint(0, max_delay_samples, (batch_size,), device=audio_batch.device)
    right_to_left_delay = torch.randint(0, max_delay_samples, (batch_size,), device=audio_batch.device)
    
    # Prepare index tensors for efficient slicing
    batch_indices = torch.arange(batch_size, device=audio_batch.device)[:, None]
    seq_indices = torch.arange(seq_length, device=audio_batch.device)[None, :]
    
    # Mix channels
    mixed_left = padded_left[:, max_delay_samples:-max_delay_samples].clone()
    mixed_right = padded_right[:, max_delay_samples:-max_delay_samples].clone()
    
    # Efficient mixing using advanced indexing
    mixed_left += right_to_left_scale * padded_right[
        batch_indices,
        max_delay_samples + seq_indices - right_to_left_delay[:, None]
    ]
    
    mixed_right += left_to_right_scale * padded_left[
        batch_indices,
        max_delay_samples + seq_indices - left_to_right_delay[:, None]
    ]
    
    # Stack channels
    mixed_audio = torch.stack((mixed_left, mixed_right), dim=2)
    
    # Normalize to prevent clipping
    max_val = torch.max(torch.abs(mixed_audio), dim=2, keepdim=True).values
    max_val = torch.max(max_val, dim=1, keepdim=True).values
    mixed_audio = torch.where(max_val > 1, mixed_audio / max_val, mixed_audio)
    
    return mixed_audio

if __name__ == "__main__":
    import torchaudio

    def test_gpu_stereo_mixer(input_path, output_path):
        # Load the audio file
        waveform, sample_rate = torchaudio.load(input_path)
        
        # Ensure the audio is stereo
        if waveform.shape[0] == 1:
            waveform = waveform.repeat(2, 1)
        
        # Move to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        waveform = waveform.to(device)
        
        # Reshape for batch processing (add batch dimension)
        waveform = waveform.unsqueeze(0).permute(0, 2, 1)
        
        # Apply the gpu_stereo_mixer_with_delay function
        mixed_waveform = gpu_stereo_mixer_with_delay(waveform, sample_rate)
        
        # Reshape back to torchaudio format
        mixed_waveform = mixed_waveform.squeeze(0).permute(1, 0)
        
        # Move back to CPU for saving
        mixed_waveform = mixed_waveform.cpu()
        
        # Save the mixed audio
        torchaudio.save(output_path, mixed_waveform, sample_rate)

    # Usage example:
    input_path = "/home/erik/projects/data/Fisher/fisher_eng_tr_sp_d1/audio/002/fe_03_00208.wav"
    output_path = "/home/serhan/Desktop/VoiceActivityProjection/serhan-utils/mixed_audio.wav"
    test_gpu_stereo_mixer(input_path, output_path)