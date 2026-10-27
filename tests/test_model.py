import unittest

import torch

from vit_scratch.model import TinyViT, ViTConfig, patchify, unpatchify


class ModelTests(unittest.TestCase):
    def test_patch_order_and_reconstruction(self):
        images = torch.arange(2 * 28 * 28).reshape(2, 1, 28, 28).float()
        patches = patchify(images, 4)
        self.assertEqual(patches.shape, (2, 49, 16))
        self.assertTrue(torch.equal(patches[0, 0], images[0, 0, :4, :4].flatten()))
        self.assertTrue(torch.equal(patches[0, 1], images[0, 0, :4, 4:8].flatten()))
        self.assertTrue(torch.equal(unpatchify(patches, 1, 28, 28, 4), images))

    def test_logits_attention_and_gradients(self):
        torch.manual_seed(1)
        model = TinyViT(ViTConfig())
        logits, attention = model(torch.rand(2, 1, 28, 28), return_attention=True)
        self.assertEqual(logits.shape, (2, 10))
        self.assertEqual(attention.shape, (2, 2, 4, 50, 50))
        self.assertTrue(
            torch.allclose(attention.sum(dim=-1), torch.ones(2, 2, 4, 50))
        )
        logits.sum().backward()
        self.assertIsNotNone(model.patch_embedding.projection.weight.grad)
        self.assertIsNotNone(model.class_token.grad)

    def test_one_training_step_changes_weights(self):
        torch.manual_seed(1)
        model = TinyViT()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        before = model.classifier.weight.detach().clone()
        loss = torch.nn.functional.cross_entropy(
            model(torch.rand(4, 1, 28, 28)), torch.tensor([1, 2, 3, 4])
        )
        loss.backward()
        optimizer.step()
        self.assertFalse(torch.equal(before, model.classifier.weight))


if __name__ == "__main__":
    unittest.main()
