import datasets
from datasets import Value, load_dataset
import os
from typing import List, Optional
import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers import AutoImageProcessor, TimesformerForVideoClassification
import numpy as np
import av

path='/home/xinyue/kinetics-dataset/k400/datasets'
cache_dir='/home/xinyue/PycharmProjects/Timesformer/cache'
image_processor = AutoImageProcessor.from_pretrained("facebook/timesformer-base-finetuned-k400")
from transformers import VideoMAEImageProcessor

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

    return list(np.stack([x.to_ndarray(format="rgb24") for x in frames]))


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

def get_label_list(raw_dataset, split="train") -> List[str]:
    """Get the list of labels from a mutli-label dataset"""

    if isinstance(raw_dataset[split]["label"][0], list):
        R=raw_dataset[split]["label"]
        label_list = [label for sample in R for label in sample]
        label_list = list(set(label_list))
    else:
        label_list = raw_dataset[split].unique("label")
    # we will treat the label list as a list of string instead of int, consistent with model.config.label2id
    label_list = [str(label) for label in label_list]
    return label_list

def multi_labels_to_ids(labels: List[str]) -> List[float]:
    ids = [0.0] * len(label_to_id)  # BCELoss requires float as target type
    for label in labels:
        ids[label_to_id[label]] = 1.0
    return ids

def preprocess_function(examples):

    paths=[example for example in examples['path'][:15]]
    containers = [av.open(example) for example in paths]

    # sample 8 frames

    indices =  [sample_frame_indices(clip_len=8, frame_sample_rate=1, seg_len=container.streams.video[0].frames) for container in containers]

    videos = [read_video_pyav(containers[i], indices[i]) for i in range(len(indices))]
    result={}
    result['vedio_feature'] = [image_processor(video, return_tensors="pt") for video in videos]

    result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples["label"][:15]]
    return result

if not os.path.exists(cache_dir):
    os.mkdir(cache_dir)
raw_datasets_o=load_dataset(
            'imdb',
            cache_dir=cache_dir,
            token=None
        )
print(get_label_list(raw_datasets_o,split='train'))
raw_datasets = load_dataset(
            path=path,
            cache_dir=cache_dir,
            token=None
        )
label_list = get_label_list(raw_datasets, split="train")

label_list.sort()
num_labels = len(label_list)
# config = AutoConfig.from_pretrained(
#         model_args.config_name if model_args.config_name else model_args.model_name_or_path,
#         num_labels=num_labels,
#         finetuning_task="text-classification",
#         cache_dir=model_args.cache_dir,
#         revision=model_args.model_revision,
#         token=model_args.token,
#         trust_remote_code=model_args.trust_remote_code,
#     )
#



image_processor = AutoImageProcessor.from_pretrained("facebook/timesformer-base-finetuned-k400")

model = TimesformerForVideoClassification.from_pretrained("facebook/timesformer-base-finetuned-k400")

model.config.problem_type = "multi_label_classification"

label_to_id = {v: i for i, v in enumerate(label_list)}
        # update config with label infos
if model.config.label2id != label_to_id:
    model.config.label2id = label_to_id
    model.config.id2label = {id: label for label, id in model.config.label2id.items()}

raw_datasets = raw_datasets.map(preprocess_function,batched=True,batch_size=15)
splits=['train','test','validation']
for split in splits:
    raw_datasets[split]=raw_datasets[split].remove_columns('path')

print(raw_datasets)