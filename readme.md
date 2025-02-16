# Audio-Segmenter

This repository is a pytorch based implementation of the paper titled [Audio Segmentation for Robust Real Time Recognition Based on Neural Networks](https://aclanthology.org/2016.iwslt-1.4/), it is designed for audio signal segmentation, a task that requires classifying different segments of a given audio file into corresponding categories, depending on the given dataset. It also implements a harmonic layer, proposed in the paper [Harmonic Loss Trains interpretable AI models](https://arxiv.org/abs/2502.01628) as the final classification layer (Do read the short comment on it on modules/net.py file)


## SETUP:
Clone the repository (`git clone <url>`) and `pip install -r requirements.txt`


## DATA FORMAT
The repository supports two dataset formats:
1. MUSAN dataset format 
 
        data/musan--
                    |
                    music
                    |
                    noise
                    |
                    speech

    Here, each class is in its own folder as you can see

2. OpenBMAT format

        data/openbmat--
                    |
                    train
                    |
                    eval
                    |
                    annotations/annotations.json
    
    annotations.json file is formatted as follows:
    ```json
    {
        "annotations": {
            "annotator_a": {
                "file1": {
                    "0": {
                        "start": 0.0,
                        "end": 38.56,
                        "class": "music"
                    },
                    "1": {
                        "start": 38.56,
                        "end": 39.13,
                        "class": "no-music"
                    },
                    "2": {
                        "start": 39.13,
                        "end": 60.0,
                        "class": "music"
                    }
                },
                "file2": {
                    "0": {
                        "start": 0.0,
                        "end": 60.0,
                        "class": "music"
                    }
                }
            }
        }
    }
    ```

    PS: if you happen to have multiple annotators annotating the same dataset, they can be contained in this single `annotations.json` file under different annotators (annotator_a, annotator_b, etc)

    Run the `download_musan.py`

## HOW TO USE
To train on a custom dataset with .wav files, simply run `python train.py --data_dir="data/musan" --ext="wav"`, there are other CLI options you should probably look into with `python train.py --help`

To run inference after training, simply run `python inference.py --file_path="path to audiofile"`


## EXPERIMENTATION BRIEFING (with MUSAN)

The technique in this repository is implemented exactly as described in the original paper, with the exception of the hidden layers sizes, The paper claimed to have used 3 hidden layers with output sizes 30, 20 and 10 respectively, however, I found that this was not the case with my implementation, I had to ramp up the hidden layer sizes to 256, 128 and 64 to get similar levels of accuracy shown in the paper.

A temporal context of $C_f=6$ is used, so the input size is: $(2 \cdot C_f + 1) \cdot t \cdot SR$, where: $t$ is segment duration (10ms), $SR$ is sample rate, which is 16000 for this specific dataset, this makes the input size 2080.
For the MFCC features, 20 mel-frequency bins are used, the window size for the spectrogram is same as the segment duration $t=10\text{ms}$, given that $SR=16000$, this implies a window size of 160. The hop length is also 160 (so overlap between time bins = 0). With all these combined, we see that the MFCC feature is a spectral image of size (20 x 13). The first order stats (mean, std and variance) are collected along the last dimension of the tensor data, reducing the MFCC features from (20 x 13) to (20 x 3). The ZCR (Zero Crossing Rate) is also computed, and its first order stats are collected as well, which is just a 1D tensor of size (3, ). All these are them combined together, to give an input feature size of (20 x 3) + 3 = 63

After training for 100 epochs, with a starting learning rate of $lr=0.01$ and a batch size of 256, I was able to achieve an accuracy, a precision, a recall and an F1 score of up to 90% on the Evaluation split of the MUSAN dataset