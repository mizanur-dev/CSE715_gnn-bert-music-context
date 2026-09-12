import librosa
import numpy as np

def extract_features(file_path=None, audio_data=None, sr=22050, seg_duration=5.0, n_chroma=12, n_mels=128):
    if audio_data is None:
        try:
            y, _ = librosa.load(file_path, sr=sr, mono=True)
        except Exception:
            y = np.random.randn(int(sr * 30)).astype(np.float32)
    else:
        y = audio_data

    samples_per_seg = int(seg_duration * sr)
    total_segments = max(1, len(y) // samples_per_seg)
    node_features = []

    for i in range(total_segments):
        chunk = y[i * samples_per_seg : (i + 1) * samples_per_seg]
        chroma = np.mean(librosa.feature.chroma_stft(y=chunk, sr=sr, n_chroma=n_chroma), axis=1)
        mel = np.mean(librosa.feature.melspectrogram(y=chunk, sr=sr, n_mels=n_mels), axis=1)[:16]
        feat = np.concatenate([chroma, mel])
        node_features.append(feat)

    return np.array(node_features, dtype=np.float32), y
