from torch import nn
from torch.nn.init import normal_, constant_
from utils.basic_ops import ConsensusModule

class TBNClassification(nn.Module):
    def __init__(self, feature_dim, modality, num_class,
                 consensus_type, before_softmax, num_segments):
        super().__init__()
        self.num_class = num_class
        self.modality = modality
        self.reshape = True
        self.consensus = ConsensusModule(consensus_type)
        self.before_softmax = before_softmax
        self.num_segments = num_segments

        if not self.before_softmax:
            self.softmax = nn.Softmax()

        if len(self.modality) > 1:  # Fusion
            self._add_classification_layer(512)
        else:  # Single modality
            self._add_classification_layer(feature_dim)
            
        # Debug: Print classification layer info
        print(f"🔍 TBNClassification Debug:")
        print(f"   Modality count: {len(self.modality)}")
        print(f"   Feature dim: {feature_dim}")
        print(f"   FC input dim: {self.fc_action.in_features if hasattr(self, 'fc_action') else 'N/A'}")
        print(f"   FC output dim: {self.fc_action.out_features if hasattr(self, 'fc_action') else 'N/A'}")

    def _add_classification_layer(self, input_dim):

        std = 0.001
        if isinstance(self.num_class, (list, tuple)):  # Multi-task

            self.fc_verb = nn.Linear(input_dim, self.num_class[0])
            self.fc_noun = nn.Linear(input_dim, self.num_class[1])
            normal_(self.fc_verb.weight, 0, std)
            constant_(self.fc_verb.bias, 0)
            normal_(self.fc_noun.weight, 0, std)
            constant_(self.fc_noun.bias, 0)
        else:
            self.fc_action = nn.Linear(input_dim, self.num_class)
            normal_(self.fc_action.weight, 0, std)
            constant_(self.fc_action.bias, 0)
            self.weight = self.fc_action.weight
            self.bias = self.fc_action.bias

    def forward(self, inputs):

        # Snippet-level predictions and temporal aggregation with consensus
        if isinstance(self.num_class, (list, tuple)):  # Multi-task
            # Verb
            base_out_verb = self.fc_verb(inputs)
            if not self.before_softmax:
                base_out_verb = self.softmax(base_out_verb)
            if self.reshape:
                base_out_verb = base_out_verb.view((-1, self.num_segments) + base_out_verb.size()[1:])
            output_verb = self.consensus(base_out_verb)

            # Noun
            base_out_noun = self.fc_noun(inputs)
            if not self.before_softmax:
                base_out_noun = self.softmax(base_out_noun)
            if self.reshape:
                base_out_noun = base_out_noun.view((-1, self.num_segments) + base_out_noun.size()[1:])
            output_noun = self.consensus(base_out_noun)

            output = (output_verb.squeeze(1), output_noun.squeeze(1))

        else:
            base_out = self.fc_action(inputs)
            if not self.before_softmax:
                base_out = self.softmax(base_out)
            if self.reshape:
                base_out = base_out.view((-1, self.num_segments) + base_out.size()[1:])

            output = self.consensus(base_out)
            output = output.squeeze(1)

        return {'logits': output}
