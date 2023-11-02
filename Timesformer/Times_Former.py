import av

import numpy as np

from transformers import AutoImageProcessor, TimesformerModel

from huggingface_hub import hf_hub_download

np.random.seed(0)


def read_video_pyav(container, indices):

    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    converted_len = int(clip_len * frame_sample_rate)
    end_idx = np.random.randint(converted_len, seg_len)
    start_idx = end_idx - converted_len
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices


# video clip consists of 300 frames (10 seconds at 30 FPS)

file_path = hf_hub_download(

    repo_id="nielsr/video-demo", filename="eating_spaghetti.mp4", repo_type="dataset"

)

container = av.open(file_path)
# sample 8 frames
indices = sample_frame_indices(clip_len=8, frame_sample_rate=4, seg_len=container.streams.video[0].frames)
video = read_video_pyav(container, indices)
#image_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")
image_processor = AutoImageProcessor.from_pretrained("facebook/timesformer-base-finetuned-k400")
# prepare video for the model
inputs = image_processor(list(video), return_tensors="pt")
# forward pass
model = TimesformerModel.from_pretrained("facebook/timesformer-base-finetuned-k400")
outputs = model(**inputs)
last_hidden_states = outputs.last_hidden_state
list(last_hidden_states.shape)
