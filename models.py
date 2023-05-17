from transformers import T5PreTrainedModel, T5EncoderModel
from transformers.modeling_outputs import SequenceClassifierOutput
from torch import nn

class T5Classification(T5PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        if self.num_labels == 1:
            self.problem_type = "regression"
            self.label_transform = True
        else:
            self.problem_type = "classification"
        self.encoder = T5EncoderModel(config) #("nielsr/nt5-small-rc1")
        self.linear = nn.Linear(self.encoder.config.hidden_size, self.num_labels)
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU6()


    def forward(self, input_ids, attention_mask, labels=None):
        encoded_outputs = self.encoder(input_ids)
        last_hidden_state = encoded_outputs.last_hidden_state #(batch size, seq len, 512)
        loss = None
        if labels is not None:
            if self.problem_type == "classification":
                logits = self.linear(last_hidden_state[:,0,:])
                softmax_logits = self.softmax(logits)
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(softmax_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.problem_type == "regression":
                logits = self.relu(last_hidden_state[:,0,:])
                logits = self.linear(logits)
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.squeeze(), labels.type_as(logits).view(-1))
        outputs = SequenceClassifierOutput(loss=loss, 
                                 logits=logits, 
                                 hidden_states=encoded_outputs.hidden_states, 
                                 attentions=encoded_outputs.attentions)
        return outputs
