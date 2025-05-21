import torch
from torch import nn
import transformers
import torchaudio
from transformers import AutoTokenizer, WhisperProcessor
from torch.utils.data import DataLoader
import os
import pandas as pd
import numpy as np
import string
import re
import glob
from PIL import Image  # For loading images

TEXT_MAX_LEN = 96
AUDIO_MAX_LEN = 16000 * 30

class Dataset_akan(torch.utils.data.Dataset):
    """
    Multimodal dataset for Akan Speech Emotion Dataset.
    
    Args:
        csv_path: Path to the CSV file.
        modality: A string with modality letters, e.g., 'tav' (text, audio, vision).
        split: One of 'train', 'val', or 'test'.
        text_context_length: Number of previous utterances to include as context.

    """
    def __init__(self, csv_path, modality, split, text_context_length):
        self.modality = modality       
        df = pd.read_csv(csv_path)
        df = df[df['split1'] == split].reset_index(drop=True)
        df = df[df['Emotion'].isin(['Neutral', 'Anger', 'Sadness', 'Joy', 'Surprise', 'Fear', 'Disgust'])].reset_index(drop=True)
        df = df[df['Start Time'] != df['End Time']].reset_index(drop=True)
        df = df[df['Movie Title'] != 'me_sofo_anyim_me_yere'].reset_index(drop=True)
        df = df[df['Movie Title'] == 'king_solomon'].reset_index(drop=True)

        # store labels
        self.labels = df['Emotion']
        
        if 't' in modality:
            # store texts
            self.texts = df['Utterance']
            self.text_context_length = text_context_length
            self.tokenizer = AutoTokenizer.from_pretrained("Ghana-NLP/abena-base-asante-twi-uncased")
        
        if 'a' in modality:
            # store audio
            self.audio_file_paths = []
            for i in range(len(df)):
                movie_title = df['Movie Title'][i]
                start_time = df['Start Time'][i].replace(':', '')
                end_time = df['End Time'][i].replace(':', '')
                audio_file_path = f'../audio_segments/{movie_title}/{movie_title.capitalize()}_{start_time}_{end_time}.wav'
                self.audio_file_paths.append(audio_file_path)
            self.feature_extractor = WhisperProcessor.from_pretrained("openai/whisper-small")

        if 'v' in modality:
            # store video folder paths
            self.video_file_paths = []
            for i in range(len(df)):
                movie_title = df['Movie Title'][i]
                start_time = df['Start Time'][i].replace(':', '')
                end_time = df['End Time'][i].replace(':', '')
                video_folder = f'../extracted_frames/{movie_title}/{movie_title}_{start_time}_{end_time}_frames'
                self.video_file_paths.append(video_folder)
            
            from torchvision import transforms
            # For full frames, resize to 224x224 with ImageNet stats.
            self.vision_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
            ])
    
        self.movie_id, self.sentence_id = df['Movie ID'], df['Sentence No']
        
    def __getitem__(self, index):
        return_dict = {'label': self.labels[index]}

        if 't' in self.modality:
            # tokenize text
            text = str(self.texts[index])         
            tokenized_text = self.tokenizer(
                text,            
                max_length=64,                                
                padding="max_length",     
                truncation=True,          
                add_special_tokens=True,  
                return_attention_mask=True            
            )    

            # load text context
            text_context = ''
            for i in range(1, self.text_context_length + 1):
                if index - i < 0 or self.movie_id[index] != self.movie_id[index - i]:
                    break
                else:
                    context = str(self.texts[index - i])
                    text_context = context + '[SEP]' + text_context

            if text_context.endswith('[SEP]'):
                text_context = text_context[:-5]
            tokenized_context = self.tokenizer(
                text_context,            
                max_length=TEXT_MAX_LEN,                                
                padding="max_length",     
                truncation=True,          
                add_special_tokens=True,  
                return_attention_mask=True            
            )
            return_dict["text_tokens"] = torch.tensor(tokenized_text["input_ids"], dtype=torch.long)
            return_dict["text_masks"] = torch.tensor(tokenized_text["attention_mask"], dtype=torch.long)
            return_dict["text_context_tokens"] = torch.tensor(tokenized_context["input_ids"], dtype=torch.long)
            return_dict["text_context_masks"] = torch.tensor(tokenized_context["attention_mask"], dtype=torch.long)

        if 'a' in self.modality:
            # load audio
            sound, sr = torchaudio.load(self.audio_file_paths[index])
            soundData = torch.mean(sound, dim=0)
            features = self.feature_extractor(
                soundData, 
                sampling_rate=16000, 
                max_length=AUDIO_MAX_LEN, 
                return_tensors="pt", 
                return_attention_mask=True,
                truncation=True, 
                do_normalize=True
            )
            audio_features = torch.tensor(np.array(features['input_features']), dtype=torch.float32).squeeze()    
            audio_masks = torch.tensor(np.array(features['attention_mask']), dtype=torch.long).squeeze()
            return_dict["audio_features"] = audio_features
            return_dict["audio_masks"] = audio_masks
        
        if 'v' in self.modality:
            # Load images from the video folder.
            video_folder = self.video_file_paths[index]
            # List all jpg files in the folder.
            image_paths = sorted(glob.glob(os.path.join(video_folder, "*.jpg")))
            # Filter based on vision_type: if "face", only use files starting with "face", else "frame".

            filtered_paths = [p for p in image_paths if os.path.basename(p).startswith("frame")]
            
            images = []
            for img_path in filtered_paths:
                try:
                    image = Image.open(img_path).convert('RGB')
                    img_tensor = self.vision_transform(image)
                    images.append(img_tensor)
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")
            
            # If no images were found, create a dummy image.
            if len(images) == 0:
                images.append(torch.zeros(3, 224, 224))
            
            return_dict["vision_frames"] = torch.stack(images)
        
        return return_dict    
        
    def __len__(self):
        return len(self.labels)


def custom_collate_fn(batch):
    """
    Custom collate function to handle variable number of vision frames.
    Pads vision_frames (if present) to the maximum number of frames in the batch and
    creates a corresponding vision_frame_mask.
    """
    collated_batch = {}
    
    if 'vision_frames' in batch[0]:
        max_frames = max(sample['vision_frames'].shape[0] for sample in batch)
        max_frames = min(max_frames, 30)  # Limit to 30 frames.
        padded_frames = []
        frame_masks = []
        for sample in batch:
            frames = sample['vision_frames']  # shape: (n_frames, C, H, W)
            n_frames = frames.shape[0]
            
            if n_frames < max_frames:
                pad_tensor = torch.zeros((max_frames - n_frames, frames.shape[1], frames.shape[2], frames.shape[3]), dtype=frames.dtype)
                frames = torch.cat([frames, pad_tensor], dim=0)
                mask = torch.cat([torch.ones(n_frames), torch.zeros(max_frames - n_frames)], dim=0)
            else:
                frames = frames[:max_frames]
                mask = torch.ones(max_frames)
            padded_frames.append(frames)
            frame_masks.append(mask)
        collated_batch["vision_frames"] = torch.stack(padded_frames)
        collated_batch["vision_frame_mask"] = torch.stack(frame_masks)
    
    for key in batch[0].keys():
        if key not in ["vision_frames"]:
            collated_batch[key] = torch.utils.data._utils.collate.default_collate([sample[key] for sample in batch])
    
    return collated_batch


def data_loader(modality, batch_size=8, text_context_length=1):
    csv_path = '../Akan Speech Emotion Dataset cleaned.csv'
    train_data = Dataset_akan(csv_path, modality, 'train', text_context_length)
    val_data = Dataset_akan(csv_path, modality, 'val', text_context_length)
    test_data = Dataset_akan(csv_path, modality, 'test', text_context_length)
    print(f"Train size: {len(train_data)}, Validation size: {len(val_data)}, Test size: {len(test_data)}")
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)
    return train_loader, val_loader, test_loader


def test():
    # Test with vision modality using face images.
    dataloader = data_loader('tav', 8, 1)
    for batch in dataloader[0]:
        print("Batch keys:", list(batch.keys()))
        if "vision_frames" in batch:
            print("Vision frames shape:", batch["vision_frames"].shape)
        break

if __name__ == '__main__':
    test()
