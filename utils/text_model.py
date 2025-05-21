import torch
from torch import nn
from transformers import AutoModel

class TextEmotionClassifier(nn.Module):
    def __init__(self, use_context=True):
        """
        Initializes the TextEmotionClassifier.

        Args:
            use_context (bool): If True, use both the main input and context. 
                                If False, only the main input is used.
        """
        super().__init__()
        self.use_context = use_context
        self.roberta_model = AutoModel.from_pretrained("Ghana-NLP/abena-base-asante-twi-uncased")
        # If using context, classifier expects concatenated [pooler_output, context_pooler]
        if self.use_context:
            self.classifier = nn.Linear(768 * 2, 7)
        else:
            self.classifier = nn.Linear(768, 7)

    def forward(self, input_ids, attention_mask, context_input_ids=None, context_attention_mask=None):
        # Process the main input.
        main_output = self.roberta_model(input_ids, attention_mask, return_dict=True)
        # get hidden states
        hidden_states = main_output["last_hidden_state"]
        main_pooler = main_output["pooler_output"]  # [batch_size, 768]

        context_pooler = None
        if self.use_context:
            if context_input_ids is None or context_attention_mask is None:
                raise ValueError("Context inputs are required when use_context is True.")
            context_output = self.roberta_model(context_input_ids, context_attention_mask, return_dict=True)
            context_pooler = context_output["pooler_output"]  # [batch_size, 768]
            # Concatenate the main and context pooler outputs.
            pooled_features = torch.cat((main_pooler, context_pooler), dim=1)  # [batch_size, 1536]
        else:
            pooled_features = main_pooler

        logits = self.classifier(pooled_features)  # [batch_size, 7]
        return logits, pooled_features, hidden_states, attention_mask, context_pooler