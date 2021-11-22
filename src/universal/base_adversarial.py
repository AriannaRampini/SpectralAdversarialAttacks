
import torch

class BaseAdversarial():
    def __init__(self,
                 classifier: torch.nn.Module,
                 classifier_args: dict,
                 target = None):
        super().__init__()
        self._classifier = classifier
        self._target = target
        self._classifier_args = classifier_args

    @property
    def classifier(self) -> torch.nn.Module: 
        return self._classifier

    @property
    def perturbed_pos(self) -> torch.Tensor:
        raise NotImplementedError()

    @property
    def logits(self) -> torch.Tensor:
        return self.classifier(self.pos, **self._classifier_args)

    @property
    def perturbed_logits(self) -> torch.Tensor:
        return self.classifier(self.perturbed_pos, **self._classifier_args)

    @property
    def target(self):
        return self._target

    @property
    def is_targeted(self) -> bool: 
        return self._target is not None

    # cached operations
    @property
    def is_successful(self) -> bool:
        prediction = self.logits.argmax().item()
        adversarial_prediction = self.perturbed_logits.argmax().item()

        if self.is_targeted:
          return adversarial_prediction == self.target
        else:
          return  prediction != adversarial_prediction
