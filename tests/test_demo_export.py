import json
import unittest
from pathlib import Path


class DemoExportTests(unittest.TestCase):
    def test_demo_matches_model_shape_and_probability_rules(self):
        demo_path = Path(__file__).resolve().parents[1] / "web/public/data/demo.json"
        demo = json.loads(demo_path.read_text(encoding="utf-8"))
        config = demo["config"]
        token_count = (config["image_size"] // config["patch_size"]) ** 2 + 1
        self.assertEqual(demo["schema_version"], 1)
        self.assertEqual(len(demo["samples"]), config["classes"])
        self.assertEqual(
            [sample["label"] for sample in demo["samples"]],
            list(range(config["classes"])),
        )
        for sample in demo["samples"]:
            self.assertEqual(len(sample["pixels"]), config["image_size"] ** 2)
            self.assertEqual(len(sample["probabilities"]), config["classes"])
            self.assertAlmostEqual(sum(sample["probabilities"]), 1.0, places=4)
            self.assertEqual(sample["prediction"], sample["probabilities"].index(max(sample["probabilities"])))
            self.assertEqual(len(sample["attention"]), config["depth"])
            for layer in sample["attention"]:
                self.assertEqual(len(layer), config["heads"])
                for head in layer:
                    self.assertEqual(len(head), token_count)
                    for row in head:
                        self.assertEqual(len(row), token_count)
                        self.assertAlmostEqual(sum(row), 1.0, places=4)


if __name__ == "__main__":
    unittest.main()
