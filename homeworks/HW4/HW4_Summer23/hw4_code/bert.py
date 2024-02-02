import torch

torch.manual_seed(10)
import torch.nn as nn
from transformers import BertModel


class BERTClassifier(nn.Module):
    def __init__(self, num_classes):
        """
        Initialize the following modules:
            1. Bert Model using the pretrained 'bert-base-uncased' model (use from_pretrained method), Ref: https://huggingface.co/transformers/v3.0.2/model_doc/bert.html
            2. Linear layer. In dimension should be 768.
            3. Dropout module.

        Args:
            num_classes: Number of classes (labels).

        """
        super(BERTClassifier, self).__init__()

        self.model = BertModel.from_pretrained('bert-base-uncased')

        self.linear = nn.Linear(768, num_classes)
        self.dropout = nn.Dropout(.5)

    def forward(self, inputs, mask):
        """
        Implement the forward function to feed the input through the model and get the output.

        Args:
            inputs: Input data.
            mask: attention_mask

        Returns:
          output: Logits of each label.
        """

        output = self.model(inputs, mask)
        #last_hidden_state = output[0]
        last_hidden_state = output.last_hidden_state
        output = self.dropout(last_hidden_state)
        output = output[:, 0, :]
        output = self.linear(output)

        return output
