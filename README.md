# Akan Cinematic Emotions (AkaCE) Dataset

**The First Multimodal Emotion Dialogue Dataset for an African Language**

> 📝 Citation:  
> David Sasu, Zehui Wu, Ziwei Gong, Run Chen, Pengyuan Shi, Lin Ai, Julia Hirschberg, Natalie Schluter.  
> *Akan Cinematic Emotions: A Multimodal Dataset for Speech Emotion Recognition in Movie Dialogues.* Findings of ACL 2025.

## Overview

The Akan Cinematic Emotions (AkaCE) dataset is the **first multimodal speech emotion recognition (SER) dataset for any African language**, specifically developed for **Akan**, a widely spoken West African language. It includes:
- **385 dialogues** from Akan-language movies
- **6,162 utterances** from **308 speakers** (balanced by gender)
- Rich **multimodal data**: audio, visual, and text
- **Seven emotion labels** (Sadness, Fear, Anger, Surprise, Disgust, Happy, Neutral)
- **Word-level prosodic prominence annotations** — making it the first **prosodically annotated dataset** for any African language

<pre>
### Dataset Structure

```
AkaCE/
├── audio/           # Audio clips segmented by utterance
├── video/           # Video clips or frame sequences per utterance
├── transcripts/     # Manual transcriptions with speaker tags
├── annotations/
│   ├── emotions.csv  # Emotion labels per utterance
│   ├── prosody.csv   # Word-level prosodic prominence annotations
│   └── metadata.csv  # Utterance IDs, timestamps, speaker info
└── README.md        # Dataset documentation
```
</pre>


## Emotion Categories

Each utterance is labeled with one of the following emotions:

- 😐 Neutral
- 😢 Sadness
- 😠 Anger
- 😨 Fear
- 😲 Surprise
- 😖 Disgust
- 😊 Happy

## Modalities

- **Text:** Human-transcribed Akan utterances
- **Audio:** Speech clips with a 16 kHz sampling rate
- **Video:** Video clips from movies used
- **Prosody:** Binary labels indicating prosodic prominence (1 = prominent, 0 = not)

## Annotation Process

- **Transcriptions:** Manual due to limitations in current Akan ASR systems
- **Emotions:** Labeled by trained native speakers using Ekman’s universal emotion framework + Neutral
- **Prosody:** Annotated using a binary scheme following guidelines adapted from prior phonetic work
- **Validation:** Majority voting with expert adjudication for disagreements

## Statistics

| Metric                          | Value       |
|------------------------------- |-------------|
| Total dialogues                 | 385         |
| Total utterances                | 6,162       |
| Unique speakers                 | 308         |
| Gender balance (M/F)           | 155 / 153   |
| Avg. words per utterance       | 19          |
| Avg. utterance length (sec)    | 6.7         |
| Prominent words (binary tags)  | 37,314      |

## Benchmark Results

| Modality          | Weighted F1 | Macro F1 |
|------------------|-------------|----------|
| Text              | 44.58       | 22.29    |
| Audio             | **52.38**   | **29.51**|
| Vision            | 40.57       | 20.02    |
| Text + Audio      | 55.51       | 30.15    |
| Audio + Vision    | 53.84       | 30.42    |
| Text + Audio + Vision (Concat) | 55.81 | 30.97 |
| **Transformer Fusion**         | **56.13** | **31.68** |

## Use Cases

AkaCE supports research in:
- Emotion Recognition in Conversation (ERC)
- Speech Emotion Recognition (SER)
- Low-resource language modeling
- Prosody-aware neural models
- Multimodal representation learning

## Citation

Please cite the following paper when using the dataset:

```bibtex
@inproceedings{sasu2025akace,
  title={Akan Cinematic Emotions: A Multimodal Dataset for Speech Emotion Recognition in Movie Dialogues},
  author={David Sasu and Zehui Wu and Ziwei Gong and Run Chen and Pengyuan Shi and Lin Ai and Julia Hirschberg and Natalie Schluter},
  booktitle={Findings of the Association for Computational Linguistics (ACL)},
  year={2025}
}

License

This dataset is released for non-commercial research purposes only. Please refer to the LICENSE.txt for more details.

