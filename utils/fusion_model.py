import torch
from torch import nn

from vision_model import VisionEmotionClassifier 
from audio_model import AudioEmotionClassifier 
from text_model import TextEmotionClassifier

from cross_attn_encoder import CMELayer, BertConfig

class FusionEmotionClassifier(nn.Module):
    def __init__(self, num_classes=7, modalities="tav",
                 text_use_context=True,
                 vision_backbone="resnet", vision_resnet_type="resnet18", vision_pretrained=True,
                 audio_whisper_model="openai/whisper-small", audio_freeze_encoder=False,
                 fusion_method="concat"):
        """
        Initializes the FusionEmotionClassifier which fuses features from text, vision, and audio.

        Args:
            num_classes (int): Number of emotion classes.
            modalities (str): A string containing the modalities to fuse.
                              For example, "tav" means fuse text, audio, and vision.
                              "tv" means text and vision, etc.
            text_use_context (bool): Whether to use context in the text model.
            vision_backbone (str): For vision, either "resnet" or "inception".
            vision_resnet_type (str): If using "resnet", the variant (e.g. "resnet18").
            vision_pretrained (bool): Whether to load pretrained vision weights.
            audio_whisper_model (str): Which Whisper model to use for audio.
            audio_freeze_encoder (bool): Whether to freeze the audio encoder.
            fusion_method (str): The fusion method to use. Options: "concat", "cross_attn".
        """
        super(FusionEmotionClassifier, self).__init__()
        self.modalities = modalities.lower()
        self.fusion_method = fusion_method.lower()
        
        # Instantiate text model if requested.
        if "t" in self.modalities:
            self.text_model = TextEmotionClassifier(use_context=text_use_context)
            self.text_pooler_dim = 1536 if text_use_context else 768
        else:
            self.text_model = None
            self.text_pooler_dim = 0
            
        # Instantiate vision model if requested.
        if "v" in self.modalities:
            self.vision_model = VisionEmotionClassifier(num_classes=num_classes,
                                                        backbone=vision_backbone,
                                                        resnet_type=vision_resnet_type,
                                                        pretrained=vision_pretrained)
            self.vision_pooler_dim = self.vision_model.encoder.feature_dim
        else:
            self.vision_model = None
            self.vision_pooler_dim = 0
            
        # Instantiate audio model if requested.
        if "a" in self.modalities:
            self.audio_model = AudioEmotionClassifier(num_classes=num_classes,
                                                      whisper_model_name=audio_whisper_model,
                                                      freeze_encoder=audio_freeze_encoder)
            self.audio_pooler_dim =self.audio_model.config.d_model
        else:
            self.audio_model = None
            self.audio_pooler_dim = 0
        
        if self.fusion_method == "concat" or len(self.modalities) != 3:
            fusion_dim = self.text_pooler_dim + self.vision_pooler_dim + self.audio_pooler_dim
            self.fusion_classifier = nn.Linear(fusion_dim, num_classes)
        elif self.fusion_method == "cross_attn":
            # cls embedding layers
            self.text_cls_emb = nn.Embedding(num_embeddings=1, embedding_dim=768)
            self.audio_cls_emb = nn.Embedding(num_embeddings=1, embedding_dim=768)
            self.vision_cls_emb = nn.Embedding(num_embeddings=1, embedding_dim=768)
            # Use the same hidden size for all modalities.
            Bert_config = BertConfig(num_hidden_layers=5, hidden_size=768, num_attention_heads=12)
            self.AudioText_CME_layers = nn.ModuleList(
                [CMELayer(Bert_config) for _ in range(Bert_config.num_hidden_layers)]
            )
            self.vision_upsample = nn.Linear(self.vision_pooler_dim, 768)
            self.AudioVision_CME_layers = nn.ModuleList(
                [CMELayer(Bert_config) for _ in range(Bert_config.num_hidden_layers)]
            )
            fusion_dim = 768*5
            self.fusion_classifier = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(fusion_dim, 768),
                nn.ReLU(),
                nn.Linear(768, num_classes)
            )

    def prepend_cls(self, inputs, masks, layer_name):
        if layer_name == 'text':
            embedding_layer = self.text_cls_emb
        elif layer_name == 'audio':
            embedding_layer = self.audio_cls_emb
        elif layer_name == 'vision':
            embedding_layer = self.vision_cls_emb
        index = torch.LongTensor([0]).to(device=inputs.device)
        cls_emb = embedding_layer(index)
        cls_emb = cls_emb.expand(inputs.size(0), 1, inputs.size(2))
        outputs = torch.cat((cls_emb, inputs), dim=1)
        
        cls_mask = torch.ones(inputs.size(0), 1).to(device=inputs.device)
        masks = torch.cat((cls_mask, masks), dim=1)
        return outputs, masks
    
    def forward(self, text_inputs=None, vision_inputs=None, audio_inputs=None):
        """
        Forward pass for the fusion model.
        
        Args:
            text_inputs (dict, optional): Dictionary with keys "input_ids", "attention_mask",
                                          and optionally "context_input_ids" and "context_attention_mask".
            vision_inputs (tuple, optional): Tuple (vision_frames, vision_frame_mask).
            audio_inputs (tuple, optional): Tuple (audio_features, audio_masks).
        
        Returns:
            fusion_logits (torch.Tensor): Final emotion logits (batch_size, num_classes).
            fusion_features (torch.Tensor): Concatenated fusion features.
        """
        pooled_features_list = []
        
        if self.text_model is not None:
            text_logits, text_pooler, text_hidden_states, text_masks, context_pooler = self.text_model(
                text_inputs["input_ids"],
                text_inputs["attention_mask"],
                text_inputs.get("context_input_ids", None),
                text_inputs.get("context_attention_mask", None)
            )
            pooled_features_list.append(text_pooler)
        
        if self.vision_model is not None:
            vision_frames, vision_frame_mask = vision_inputs
            vision_logits, vision_pooler, vision_hidden_states, vision_masks = self.vision_model(vision_frames, vision_frame_mask)
            pooled_features_list.append(vision_pooler)
        
        if self.audio_model is not None:
            audio_features, audio_masks = audio_inputs
            audio_logits, audio_pooler, audio_hidden_states, audio_masks = self.audio_model(audio_features, audio_masks)
            pooled_features_list.append(audio_pooler)
        
        if not pooled_features_list:
            raise ValueError("No modalities provided for fusion.")

        if self.fusion_method == "concat" or len(self.modalities) != 3:
            fusion_features = torch.cat(pooled_features_list, dim=1)
            fusion_logits = self.fusion_classifier(fusion_features)
            return fusion_logits, fusion_features
        elif self.fusion_method == "cross_attn":

            vision_hidden_states = self.vision_upsample(vision_hidden_states)


            # cls embedding
            text_hidden_states, text_masks = self.prepend_cls(text_hidden_states, text_masks, 'text')
            audio_hidden_states, audio_masks = self.prepend_cls(audio_hidden_states, audio_masks, 'audio')
            vision_hidden_states, vision_masks = self.prepend_cls(vision_hidden_states, vision_masks, 'vision')

            # cross modality encoder
            for layer_module in self.AudioText_CME_layers:
                audio_hidden_states_t, text_hidden_states  = layer_module(audio_hidden_states, audio_masks, text_hidden_states, text_masks)
            
            for layer_module in self.AudioVision_CME_layers:
                audio_hidden_states_v, vision_hidden_states = layer_module(audio_hidden_states, audio_masks, vision_hidden_states, vision_masks)
            
            # fusion
            # fusion_features = torch.cat([text_hidden_states.mean(dim=1), audio_hidden_states_t.mean(dim=1),
            #                              vision_hidden_states.mean(dim=1), audio_hidden_states_v.mean(dim=1),
            #                              context_pooler], dim=1)
            fusion_features = torch.cat([text_hidden_states[:,0,:], audio_hidden_states_t[:,0,:],
                                         vision_hidden_states[:,0,:], audio_hidden_states_v[:,0,:],
                                         context_pooler], dim=1)
            fusion_logits = self.fusion_classifier(fusion_features)
            return fusion_logits, fusion_features