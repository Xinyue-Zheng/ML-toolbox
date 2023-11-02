import av

import torch

import numpy as np

from transformers import AutoImageProcessor, TimesformerForVideoClassification

from huggingface_hub import hf_hub_download

np.random.seed(0)


def read_video_pyav(container, indices):

    '''

    Decode the video with PyAV decoder.

    Args:

        container (`av.container.input.InputContainer`): PyAV container.

        indices (`List[int]`): List of frame indices to decode.

    Returns:

        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).

    '''

    frames = []

    container.seek(0)

    start_index = indices[0]

    end_index = indices[-1]

    for i, frame in enumerate(container.decode(video=0)):

        if i > end_index:

            break

        if i >= start_index and i in indices:

            frames.append(frame)

    m=np.stack([x.to_ndarray(format="rgb24") for x in frames])
    return list(m)


def sample_frame_indices(clip_len, frame_sample_rate, seg_len):

    '''

    Sample a given number of frame indices from the video.

    Args:

        clip_len (`int`): Total number of frames to sample.

        frame_sample_rate (`int`): Sample every n-th frame.

        seg_len (`int`): Maximum allowed index of sample's last frame.

    Returns:

        indices (`List[int]`): List of sampled frame indices

    '''

    converted_len = int(clip_len * frame_sample_rate)

    end_idx = np.random.randint(converted_len, seg_len)

    start_idx = end_idx - converted_len

    indices = np.linspace(start_idx, end_idx, num=clip_len)

    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)

    return indices


# video clip consists of 300 frames (10 seconds at 30 FPS)
#
# file_path = hf_hub_download(
#
#     repo_id="nielsr/video-demo", filename="eating_spaghetti.mp4", repo_type="dataset"
#
# )
file_paths=['/home/xinyue/kinetics-dataset/k400/test/crossing river/0xPY4YhnwBk_000002_000012.mp4', '/home/xinyue/kinetics-dataset/k400/test/crossing river/1nG0hiUNCM8_000009_000019.mp4']
containers = [av.open(file_path) for file_path in file_paths]

# sample 8 frames

indices = [sample_frame_indices(clip_len=8, frame_sample_rate=1, seg_len=container.streams.video[0].frames) for container in containers]

video = [read_video_pyav(containers[i], indices[i]) for i in range(len(indices))]

image_processor = AutoImageProcessor.from_pretrained("facebook/timesformer-base-finetuned-k400")

model = TimesformerForVideoClassification.from_pretrained("facebook/timesformer-base-finetuned-k400")

inputs = image_processor(video, return_tensors="pt")

with torch.no_grad():

    outputs = model(**inputs)

    logits = outputs.logits

# model predicts one of the 400 Kinetics-400 classes

predicted_label = logits.argmax(-1).item()

print(model.config.id2label[predicted_label])