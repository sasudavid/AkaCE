import torch
from torch import nn
from torch.optim import AdamW
from tqdm import tqdm
from data_loader import data_loader
import os
from sklearn.metrics import f1_score, classification_report

# ----------------------------------------
# Hyperparameters and Configuration
# ----------------------------------------

EPOCHS = 1
BATCH_SIZE = 16
LEARNING_RATE = 1e-5
TEXT_CONTEXT_LENGTH = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set modality for fusion training using a string
MODALITY = "tav" # "t" for text, "a" for audio, "v" for vision, "ta" for text and audio, "tv" for text and vision, "av" for audio and vision, "tav" for all
FUSION_METHOD = "cross_attn" # "cross_attn" or "concat"
TEXT_USE_CONTEXT = True

VISION_BACKBONE = "resnet"
VISION_RESNET = "resnet18"
POOLING = "avg"

checkpoint_path = f"../checkpoint/best_model_{MODALITY}.pt"

# ----------------------------------------


# ----------------------------------------
# Training and Evaluation Functions
# ----------------------------------------

def train_model(
    model,
    train_loader,
    val_loader,
    device,
    label2id,
    modality="tav",
    epochs=10,
    lr=1e-5,
    patience=3,
    checkpoint_path="best_model.pt"
):
    model.to(device)
    criterion = nn.CrossEntropyLoss()

    # # Create optimizer with different parameter groups for fusion models.
    # if len(modality) > 1:
    #     # Define individual learning rates for each module.
    #     text_lr = 1e-5
    #     vision_lr = 1e-4
    #     audio_lr = 1e-5
    #     fusion_lr = 1e-4
    #     param_groups = []
    #     if "t" in modality:
    #         param_groups.append({"params": model.text_model.parameters(), "lr": text_lr})
    #     if "v" in modality:
    #         param_groups.append({"params": model.vision_model.parameters(), "lr": vision_lr})
    #     if "a" in modality:
    #         param_groups.append({"params": model.audio_model.parameters(), "lr": audio_lr})
    #     # Fusion classifier is always present in the fusion model.
    #     param_groups.append({"params": model.fusion_classifier.parameters(), "lr": fusion_lr})
    #     optimizer = AdamW(param_groups)
    # else:
    optimizer = AdamW(model.parameters(), lr=lr)
    
    best_combined_f1 = -float("inf")
    epochs_without_improve = 0

    model.train()
    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch+1}/{epochs} ---")
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        for batch in tqdm(train_loader, desc="Training", leave=False):
            optimizer.zero_grad()
            
            # Fusion mode: if modality string length > 1, process each modality input separately.
            if len(modality) > 1:
                # Prepare text inputs if "t" is in modality.
                text_inputs = None
                if "t" in modality:
                    text_inputs = {
                        "input_ids": batch["text_tokens"].to(device),
                        "attention_mask": batch["text_masks"].to(device),
                        "context_input_ids": batch["text_context_tokens"].to(device),
                        "context_attention_mask": batch["text_context_masks"].to(device)
                    }
                # Prepare vision inputs if "v" is in modality.
                vision_inputs = None
                if "v" in modality:
                    vision_inputs = (
                        batch["vision_frames"].to(device),
                        batch["vision_frame_mask"].to(device)
                    )
                # Prepare audio inputs if "a" is in modality.
                audio_inputs = None
                if "a" in modality:
                    audio_inputs = (
                        batch["audio_features"].to(device),
                        batch["audio_masks"].to(device)
                    )
                logits,_ = model(text_inputs, vision_inputs, audio_inputs)
            else:
                # Single modality training.
                if 't' in modality:
                    logits,_,_,_,_ = model(
                        input_ids=batch["text_tokens"].to(device),
                        attention_mask=batch["text_masks"].to(device),
                        context_input_ids=batch["text_context_tokens"].to(device),
                        context_attention_mask=batch["text_context_masks"].to(device)
                    )
                elif 'v' in modality:
                    logits,_,_,_ = model(
                        vision_frames=batch["vision_frames"].to(device),
                        vision_frame_mask=batch["vision_frame_mask"].to(device)
                    )
                elif 'a' in modality:
                    logits,_,_,_ = model(
                        audio_features=batch["audio_features"].to(device),
                        audio_masks=batch["audio_masks"].to(device)
                    )
            
            numeric_labels = [label2id[emo] for emo in batch["label"]]
            labels = torch.tensor(numeric_labels, dtype=torch.long).to(device)
            
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            _, preds = torch.max(logits, dim=1)
            total_correct += (preds == labels).sum().item()
        
        avg_train_loss = total_loss / total_samples
        train_acc = total_correct / total_samples
        
        val_loss, val_acc, combined_f1, weighted_f1, macro_f1 = evaluate_model(
            model, val_loader, device, label2id, modality=modality, criterion=criterion
        )
        
        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"Weighted F1: {weighted_f1:.4f}, Macro F1: {macro_f1:.4f}, Combined F1: {combined_f1:.4f}")
        
        if combined_f1 > best_combined_f1:
            best_combined_f1 = combined_f1
            epochs_without_improve = 0
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Combined F1 improved; model saved to {checkpoint_path}")
        else:
            epochs_without_improve += 1
            print(f"No improvement for {epochs_without_improve} epoch(s).")
        
        if epochs_without_improve >= patience:
            print("Early stopping triggered.")
            break

def evaluate_model(model, data_loader, device, label2id, modality="tav", criterion=None):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_preds = []
    all_labels = []
    compute_loss = criterion is not None
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating", leave=False):
            if len(modality) > 1:
                text_inputs = None
                if "t" in modality:
                    text_inputs = {
                        "input_ids": batch["text_tokens"].to(device),
                        "attention_mask": batch["text_masks"].to(device),
                        "context_input_ids": batch["text_context_tokens"].to(device),
                        "context_attention_mask": batch["text_context_masks"].to(device)
                    }
                vision_inputs = None
                if "v" in modality:
                    vision_inputs = (
                        batch["vision_frames"].to(device),
                        batch["vision_frame_mask"].to(device)
                    )
                audio_inputs = None
                if "a" in modality:
                    audio_inputs = (
                        batch["audio_features"].to(device),
                        batch["audio_masks"].to(device)
                    )
                logits, _ = model(text_inputs, vision_inputs, audio_inputs)
            else:
                if "t" in modality:
                    logits,_,_,_,_ = model(
                        input_ids=batch["text_tokens"].to(device),
                        attention_mask=batch["text_masks"].to(device),
                        context_input_ids=batch["text_context_tokens"].to(device),
                        context_attention_mask=batch["text_context_masks"].to(device)
                    )
                elif "v" in modality:
                    logits,_,_,_ = model(
                        vision_frames=batch["vision_frames"].to(device),
                        vision_frame_mask=batch["vision_frame_mask"].to(device)
                    )
                elif "a" in modality:
                    logits,_,_,_ = model(
                        audio_features=batch["audio_features"].to(device),
                        audio_masks=batch["audio_masks"].to(device)
                    )
            
            numeric_labels = [label2id[emo] for emo in batch["label"]]
            labels = torch.tensor(numeric_labels, dtype=torch.long).to(device)
            if compute_loss:
                loss = criterion(logits, labels)
                total_loss += loss.item() * labels.size(0)
            _, preds = torch.max(logits, dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    
    avg_loss = total_loss / total_samples if compute_loss else None
    accuracy = total_correct / total_samples
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted')
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    combined_f1 = weighted_f1 + macro_f1
    report = classification_report(all_labels, all_preds, labels=list(label2id.values()), target_names=[str(k) for k in label2id.values()], zero_division=0)
    print("\nClassification Report:")
    print(report)
    print(f"Weighted F1 Score: {weighted_f1:.4f}, Macro F1 Score: {macro_f1:.4f}")
    
    model.train()
    if compute_loss:
        return avg_loss, accuracy, combined_f1, weighted_f1, macro_f1
    else:
        return accuracy


if __name__ == "__main__":
    if len(MODALITY) > 1:
        from fusion_model import FusionEmotionClassifier
        model = FusionEmotionClassifier(num_classes=7, modalities=MODALITY,
                                        text_use_context=TEXT_USE_CONTEXT,
                                        vision_backbone=VISION_BACKBONE, vision_resnet_type=VISION_RESNET, vision_pretrained=True,
                                        audio_whisper_model="openai/whisper-small", audio_freeze_encoder=False, fusion_method=FUSION_METHOD)
    else:
        if MODALITY == "t":
            from text_model import TextEmotionClassifier
            model = TextEmotionClassifier(use_context=TEXT_USE_CONTEXT).to(DEVICE)
        elif MODALITY == "v":
            from vision_model import VisionEmotionClassifier
            model = VisionEmotionClassifier(num_classes=7, backbone=VISION_BACKBONE, resnet_type=VISION_RESNET, pretrained=True, pooling=POOLING).to(DEVICE)
        elif MODALITY == "a":
            from audio_model import AudioEmotionClassifier
            model = AudioEmotionClassifier(num_classes=7, whisper_model_name="openai/whisper-small", freeze_encoder=False).to(DEVICE)
    
    label2id = {
        "Neutral": 0,
        "Anger": 1,
        "Sadness": 2,
        "Joy": 3,
        "Surprise": 4,
        "Fear": 5,
        "Disgust": 6
    }
    
    train_loader, val_loader, test_loader = data_loader(MODALITY, batch_size=BATCH_SIZE, text_context_length=TEXT_CONTEXT_LENGTH)

    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=DEVICE,
        label2id=label2id,
        modality=MODALITY,
        epochs=EPOCHS,
        lr=LEARNING_RATE,
        patience=3,
        checkpoint_path=checkpoint_path
    )
    
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
        print(f"Loaded best model checkpoint from {checkpoint_path}")
    else:
        print("Checkpoint not found. Testing with current model weights.")
    
    test_loss, test_acc, _, _, _ = evaluate_model(
        model=model,
        data_loader=test_loader,
        device=DEVICE,
        label2id=label2id,
        modality=MODALITY,
        criterion=nn.CrossEntropyLoss()
    )
    print(f"\nTest Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
