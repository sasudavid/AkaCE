import torch
from torch import nn
import torchvision.models as models
from facenet_pytorch import InceptionResnetV1
import torch.nn.functional as F

class VisionEncoder(nn.Module):
    def __init__(self, backbone="resnet", resnet_type="resnet18", pretrained=True, pooling="avg"):
        """
        Initializes the VisionEncoder with a selectable backbone and pooling strategy.
        
        Args:
            backbone (str): Either "resnet" or "inception".
                - "resnet" uses a torchvision ResNet (selectable via resnet_type).
                - "inception" uses InceptionResnetV1 pretrained on VGGFace2.
            resnet_type (str): If backbone is "resnet", options include "resnet18", "resnet34",
                               "resnet50", "resnet101", "resnet152".
            pretrained (bool): Whether to load pretrained weights.
            pooling (str): Pooling strategy over frame features; options:
                           "avg" for average pooling (default) or "attn" for attention pooling.
        """
        super(VisionEncoder, self).__init__()
        self.backbone = backbone.lower()
        self.pooling = pooling.lower()
        
        if self.backbone == "resnet":
            resnet_type = resnet_type.lower()
            if resnet_type == "resnet18":
                self.model = models.resnet18(pretrained=pretrained)
                feature_dim = 512
            elif resnet_type == "resnet34":
                self.model = models.resnet34(pretrained=pretrained)
                feature_dim = 512
            elif resnet_type == "resnet50":
                self.model = models.resnet50(pretrained=pretrained)
                feature_dim = 2048
            elif resnet_type == "resnet101":
                self.model = models.resnet101(pretrained=pretrained)
                feature_dim = 2048
            elif resnet_type == "resnet152":
                self.model = models.resnet152(pretrained=pretrained)
                feature_dim = 2048
            else:
                raise ValueError(f"Unsupported resnet type: {resnet_type}")
            self.model.fc = nn.Identity()
        elif self.backbone == "inception":
            self.model = InceptionResnetV1(pretrained='vggface2').eval()
            feature_dim = 512
        else:
            raise ValueError("backbone must be either 'resnet' or 'inception'")
            
        self.feature_dim = feature_dim
        
        if self.pooling == "attn":
            self.attention_fc = nn.Linear(self.feature_dim, 1)
    
    def forward(self, vision_frames, vision_frame_mask):
        """
        Args:
            vision_frames (torch.Tensor): Tensor of shape 
                (batch_size, max_frames, 3, H, W) containing image frames.
            vision_frame_mask (torch.Tensor): Tensor of shape 
                (batch_size, max_frames) with 1 for valid frames and 0 for padded frames.
                
        Returns:
            pooled_features (torch.Tensor): Tensor of shape (batch_size, feature_dim)
                aggregated from valid frames.
            padded_features (torch.Tensor): Tensor of shape (batch_size, max_frames, feature_dim)
                where each sample’s valid features are placed (and padded/truncated to max_frames).
        """
        batch_size, max_frames, C, H, W = vision_frames.shape
        # Create a boolean mask.
        mask_bool = vision_frame_mask.bool()  # (B, max_frames)
        # Gather all valid frames across the batch.
        valid_frames = vision_frames[mask_bool]  # (total_valid, C, H, W)
        
        # Pass valid frames through backbone.
        if self.backbone == "resnet":
            features_valid = self.model(valid_frames)
        elif self.backbone == "inception":
            if H != 160 or W != 160:
                valid_frames = F.interpolate(valid_frames, size=(160, 160), mode="bilinear", align_corners=False)
            features_valid = self.model(valid_frames)
        else:
            raise ValueError("Invalid backbone")
        
        # Get indices of valid frames.
        valid_idx = torch.nonzero(mask_bool, as_tuple=False)  # shape: (total_valid, 2) with [batch_index, frame_index]
        
        pooled_features_list = []
        padded_features_list = []
        for i in range(batch_size):
            # Select features corresponding to sample i.
            sample_mask = valid_idx[:, 0] == i
            if sample_mask.sum() == 0:
                sample_features = torch.zeros(1, self.feature_dim, device=vision_frames.device)
            else:
                sample_features = features_valid[sample_mask]  # shape: (n_valid, feature_dim)
            # Pool valid features.
            if self.pooling == "avg":
                pooled = sample_features.mean(dim=0, keepdim=True)
            elif self.pooling == "attn":
                attn_scores = self.attention_fc(sample_features)
                attn_weights = F.softmax(attn_scores, dim=0)
                pooled = (sample_features * attn_weights).sum(dim=0, keepdim=True)
            else:
                raise ValueError("Unsupported pooling type. Use 'avg' or 'attn'.")
            pooled_features_list.append(pooled)
            
            # Create padded feature tensor: pad (or truncate) to length max_frames.
            n_valid = sample_features.shape[0]
            if n_valid < max_frames:
                pad = torch.zeros(max_frames - n_valid, self.feature_dim, device=vision_frames.device)
                padded = torch.cat([sample_features, pad], dim=0)
            else:
                padded = sample_features[:max_frames]
            padded_features_list.append(padded.unsqueeze(0))
        
        pooled_features = torch.cat(pooled_features_list, dim=0)       # (batch_size, feature_dim)
        padded_features = torch.cat(padded_features_list, dim=0)         # (batch_size, max_frames, feature_dim)
        return pooled_features, padded_features


class VisionEmotionClassifier(nn.Module):
    def __init__(self, num_classes=7, backbone="resnet", resnet_type="resnet18", pretrained=True, pooling="avg"):
        """
        Combines the VisionEncoder with a classification head for emotion detection.
        
        Args:
            num_classes (int): Number of emotion classes.
            backbone (str): "resnet" or "inception".
            resnet_type (str): If backbone is "resnet", the variant to use.
            pretrained (bool): Whether to load pretrained weights.
            pooling (str): Pooling strategy ("avg" or "attn").
        """
        super(VisionEmotionClassifier, self).__init__()
        self.encoder = VisionEncoder(backbone=backbone, resnet_type=resnet_type, pretrained=pretrained, pooling=pooling)
        self.classifier = nn.Linear(self.encoder.feature_dim, num_classes)
        
    def forward(self, vision_frames, vision_frame_mask):
        pooled_features, padded_features = self.encoder(vision_frames, vision_frame_mask)
        logits = self.classifier(pooled_features)
        return logits, pooled_features, padded_features, vision_frame_mask
