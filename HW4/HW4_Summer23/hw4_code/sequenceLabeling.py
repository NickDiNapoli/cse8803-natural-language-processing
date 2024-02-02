import torch

torch.manual_seed(10)
import torch.nn as nn
from transformers import BertModel


class SequenceLabeling(nn.Module):
    def __init__(self, num_classes):
        """
        Initialize the following modules:
            1. Bert Model using the pretrained 'bert-base-uncased' model,
            2. Dropout module.
            3. Linear layer. In dimension should be 768.

        Args:
        num_classes: Number of classes (labels).

        """
        super(SequenceLabeling, self).__init__()

        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.linear = nn.Linear(768, num_classes)
        self.dropout = nn.Dropout(.5)
        self.relu = nn.ReLU()


    def forward(self, inputs, mask, token_type_ids):
        """
        Implement the forward function to feed the input through the bert model with inputs, mask and token type ids.
        The output of bert layer model is then fed to dropout, linear and relu.

        Args:
            inputs: Input data.
            mask: attention_mask
            token_type_ids: token type ids

        Returns:
          output: Logits of each label.
        """
        output = self.model(inputs, mask, token_type_ids)
        # last_hidden_state = output[0]
        last_hidden_state = output.last_hidden_state
        output = self.dropout(last_hidden_state)
        #print(output.shape)
        #output = output[:, 0, :]
        output = self.linear(output)
        output = self.relu(output)

        return output
