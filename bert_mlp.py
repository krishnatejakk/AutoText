from transformers import BertModel
import torch.nn as nn
import torch


class CustomBERTModel(nn.Module):
    def __init__(self, l1, num_classes):
          super(CustomBERTModel, self).__init__()
          self.bert = BertModel.from_pretrained("bert-base-cased")
          ### New layers:
          self.linear1 = nn.Linear(768, l1)
          self.linear2 = nn.Linear(l1, num_classes)

    def forward(self, ids, mask, finetune=False, freeze=False, last=False):
          if freeze:
                with torch.no_grad():
                    sequence_output, pooled_output = self.bert(ids, attention_mask=mask)
                    # sequence_output has the following shape: (batch_size, sequence_length, 768)
                    linear1_output = self.linear1(sequence_output[:,0,:].view(-1,768)) ## extract the 1st token's embeddings
          else:
            if finetune:
                sequence_output, pooled_output = self.bert(ids, attention_mask=mask)
            else:
                with torch.no_grad():
                    sequence_output, pooled_output = self.bert(ids, attention_mask=mask)
            # sequence_output has the following shape: (batch_size, sequence_length, 768)
            linear1_output = self.linear1(sequence_output[:,0,:].view(-1,768)) ## extract the 1st token's embeddings
          linear2_output = self.linear2(linear1_output)
          if last:
              return linear2_output, linear1_output
          else:
              return linear2_output