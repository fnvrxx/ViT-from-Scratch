import argparse
import json
from pathlib import Path

import torch

from .data import CLASS_NAMES, load_datasets
from .model import TinyViT, ViTConfig


def first_example_per_class(test_data) -> list[int]:
    indices = {}
    for index, label in enumerate(test_data.targets.tolist()):
        indices.setdefault(label, index)
        if len(indices) == len(CLASS_NAMES):
            break
    if len(indices) != len(CLASS_NAMES):
        raise ValueError("test set does not contain every class")
    return [indices[label] for label in range(len(CLASS_NAMES))]


def rounded_attention(attention: torch.Tensor) -> list:
    return [
        [
            [[round(float(weight), 6) for weight in row] for row in head]
            for head in layer
        ]
        for layer in attention.squeeze(0)
    ]


def run(args: argparse.Namespace) -> dict:
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint["model"] != "vit":
        raise ValueError("export requires a ViT checkpoint")
    model = TinyViT(ViTConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    _, _, test_data = load_datasets(args.data_dir, checkpoint["seed"])
    samples = []
    with torch.inference_mode():
        for index in first_example_per_class(test_data):
            image, label = test_data[index]
            logits, attention = model(image.unsqueeze(0), return_attention=True)
            probabilities = logits.softmax(dim=-1).squeeze(0)
            samples.append(
                {
                    "test_index": index,
                    "label": label,
                    "prediction": int(probabilities.argmax()),
                    "pixels": image.mul(255).round().to(torch.uint8).flatten().tolist(),
                    "probabilities": [round(float(x), 6) for x in probabilities],
                    "attention": rounded_attention(attention),
                }
            )
    vit_metrics = json.loads(Path(args.vit_metrics).read_text(encoding="utf-8"))
    cnn_metrics = json.loads(Path(args.cnn_metrics).read_text(encoding="utf-8"))
    if vit_metrics["model"] != "vit" or cnn_metrics["model"] != "cnn":
        raise ValueError("metric files must belong to ViT and CNN respectively")
    demo = {
        "schema_version": 1,
        "source": "Fashion-MNIST official test split",
        "selection": "first test example for each class, in class order",
        "classes": list(CLASS_NAMES),
        "config": checkpoint["config"],
        "metrics": {"vit": vit_metrics, "cnn": cnn_metrics},
        "samples": samples,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(demo, separators=(",", ":")) + "\n", encoding="utf-8"
    )
    print(f"Exported {len(samples)} test examples to {output_path}")
    return demo


def main():
    parser = argparse.ArgumentParser(description="Export real ViT examples for the web")
    parser.add_argument("--checkpoint", default="artifacts/vit.pt")
    parser.add_argument("--vit-metrics", default="artifacts/vit_metrics.json")
    parser.add_argument("--cnn-metrics", default="artifacts/cnn_metrics.json")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output", default="web/public/data/demo.json")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
