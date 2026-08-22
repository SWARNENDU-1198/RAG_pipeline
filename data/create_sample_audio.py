"""
Generates a valid 1-second sample WAV audio file for demo & testing.
"""

import wave
import math
import struct
import os

def generate_sample_wav(filename="sample_audio.wav", duration=1.0, sample_rate=16000, frequency=440.0):
    num_samples = int(duration * sample_rate)
    with wave.open(filename, 'w') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        
        for i in range(num_samples):
            t = float(i) / sample_rate
            val = int(16384.0 * math.sin(2.0 * math.pi * frequency * t))
            data = struct.pack('<h', val)
            wav_file.writeframesraw(data)

if __name__ == "__main__":
    out_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sample_audio.wav")
    generate_sample_wav(out_path)
    print(f"Sample WAV generated at {out_path}")
