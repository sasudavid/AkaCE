import math
import torch
from torch import nn
from transformers import WhisperModel, WhisperConfig

class AudioEmotionClassifier(nn.Module):
    def __init__(self, num_classes=7, whisper_model_name="openai/whisper-small", freeze_encoder=False):
        """
        Audio emotion classifier using the encoder of a pre-trained Whisper model.

        Args:
            num_classes (int): Number of emotion classes (default: 7).
            whisper_model_name (str): Name or path of the Whisper model to load.
            freeze_encoder (bool): If True, the Whisper encoder parameters will not be updated.
        """
        super(AudioEmotionClassifier, self).__init__()
        # Load the Whisper configuration and model.
        self.config = WhisperConfig.from_pretrained(whisper_model_name)
        # Load the entire model then extract the encoder.
        whisper_model = WhisperModel.from_pretrained(whisper_model_name)
        self.encoder = whisper_model.encoder  # This is an instance of WhisperEncoder.
        
        # Optionally freeze the encoder parameters.
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # The encoder’s output hidden size is given by config.d_model.
        hidden_size = self.config.d_model
        self.classifier = nn.Linear(hidden_size, num_classes)
    
    def forward(self, audio_features, audio_masks=None):
        """
        Args:
            audio_features (torch.Tensor): 
                A tensor containing the preprocessed audio features.
                The Whisper encoder expects shape (batch_size, num_mel_bins, sequence_length).
            audio_masks (torch.Tensor, optional): 
                A binary tensor of shape (batch_size, original_sequence_length) from the processor,
                with 1 for valid frames and 0 for padding. Because the encoder downsamples by a factor 
                (here assumed to be 2 due to the second convolution's stride), this mask will be 
                downsampled before pooling.
        Returns:
            logits (torch.Tensor): A tensor of shape (batch_size, num_classes) containing class logits.
        """
        # Ensure the audio features are in the expected shape:
        # If the last dimension equals num_mel_bins, permute to (batch, num_mel_bins, seq_len).
        if audio_features.dim() == 3:
            if audio_features.size(2) == self.config.num_mel_bins:
                # Permute from (batch, seq_len, num_mel_bins) to (batch, num_mel_bins, seq_len)
                audio_features = audio_features.permute(0, 2, 1)
            elif audio_features.size(1) != self.config.num_mel_bins:
                raise ValueError(
                    f"Expected one dimension to equal {self.config.num_mel_bins} (the number of mel bins), "
                    f"but got audio_features of shape {audio_features.shape}."
                )
        else:
            raise ValueError("audio_features must be a 3D tensor")

        # Forward pass through the encoder.
        # Note: In Whisper's implementation the attention_mask argument is not used,
        # so we pass None. (Any provided audio_masks would need to be downsampled accordingly.)
        encoder_outputs = self.encoder(input_features=audio_features, attention_mask=None)
        # encoder_outputs is an instance of BaseModelOutput containing last_hidden_state.
        hidden_states = encoder_outputs.last_hidden_state  # shape: (batch_size, seq_len_downsampled, hidden_size)

        # If an audio mask is provided, we attempt to downsample it to match the encoder output.
        # The second conv layer has stride=2, so we downsample by taking every 2nd element.
        if audio_masks is not None:
            # Assume audio_masks has shape (batch_size, original_seq_len).
            audio_masks = audio_masks[:, ::2]  # shape: (batch_size, seq_len_downsampled)
            # Expand mask to match hidden_states: (batch_size, seq_len_downsampled, 1)
            mask = audio_masks.unsqueeze(-1).float()
            # Perform masked average pooling over the sequence (time) dimension.
            masked_hidden = hidden_states * mask
            sum_hidden = masked_hidden.sum(dim=1)  # (batch_size, hidden_size)
            valid_counts = mask.sum(dim=1).clamp(min=1)
            pooled_output = sum_hidden / valid_counts
        else:
            # Otherwise, simply average over the time dimension.
            pooled_output = hidden_states.mean(dim=1)
        
        logits = self.classifier(pooled_output)
        return logits, pooled_output, hidden_states, audio_masks
