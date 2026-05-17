# 怎么读入音频

默认（librispeech）：lhotse_audio

```python
# dataloader/multimodal_dataloader/__init__.py
ALL_DATA_LOADERS = {
    "lhotse_audio": LhotseAudioReader,
    ...
}

# dataloader/multimodal_dataloader/audio_loader.py
class LhotseAudioReader:
    def __getitem__(self, key: str) -> Tuple[np.ndarray, int]:
        ...
        # audio: [num_channels, num_samples] (ndarray格式)
        return audio, sample_rate
```