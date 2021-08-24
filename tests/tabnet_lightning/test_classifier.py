import pytest
import torch


class TestClassificationHead():
    @pytest.mark.parametrize("batch_size, input_size, num_classes, class_weights",
                             [
                                 (128, 32, [2, 2, 2, 2], None),
                                 # (1, 32, [2, 2, 2, 2], None),
                                 #
                                 # (128, 32, 2, None),
                                 # (1, 32, 2, None),
                                 # (128, 32, 2, [0.4, 0.6]),
                                 # (1, 32, 2, [0.1, 0.9]),
                                 # (128, 32, 4, None),
                                 # (1, 32, 4, None),
                                 # (128, 32, 4, [0.1, 0.3, 0.4, 0.9]),
                                 # (1, 32, 4,[0.1, 0.3, 0.4, 0.9]),
                                 # (128, 32, [2, 2, 2, 2], None),
                                 # (1, 32, [2, 2, 2, 2], None),
                             ])
    def test_forward(self, batch_size, input_size, num_classes, class_weights):
        from tabnet_lightning.classifier import ClassificationHead

        inputs = torch.randn(size=(batch_size, input_size))

        if isinstance(num_classes, int):
            if num_classes == 2:
                labels = torch.randint(0, 2, size=(batch_size, ))
            elif num_classes > 2:
                labels = torch.randint(0, num_classes, size=(batch_size, ))
        elif isinstance(num_classes, list):
            if all(c == 2 for c in num_classes):
                labels = torch.randint(0, 2, size=(batch_size, len(num_classes)))

        classification_head = ClassificationHead(input_size=input_size, num_classes=num_classes, class_weights=class_weights)
        logits, probs, loss = classification_head(inputs, labels)

        assert True


