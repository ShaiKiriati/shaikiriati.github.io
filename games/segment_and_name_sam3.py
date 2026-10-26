#!/usr/bin/env python3
"""
Find and name objects with SAM 3, pick 5-20 at random, and show the image
with the list of names below it.

SAM 3 segments by *concept*: you give it a short noun phrase ("dog", "coffee
mug") and it returns a mask for every matching instance, already named. So
there is no separate naming model; the program just runs SAM 3 over a list of
candidate concepts and keeps whatever it finds.

Setup:
    pip install -U transformers torch pillow numpy matplotlib accelerate
    # facebook/sam3 is a gated model: request access at
    # https://huggingface.co/facebook/sam3 , then log in:
    huggingface-cli login

Run:
    python segment_and_name_sam3.py                    # file dialog
    python segment_and_name_sam3.py photo.jpg
    python segment_and_name_sam3.py photo.jpg --concepts my_concepts.txt --overlay
"""
import argparse
import random
import textwrap
from collections import Counter
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from transformers import Sam3Model, Sam3Processor

SAM3_MODEL = "facebook/sam3"
MAX_SIDE = 1280        # downscale very large images for speed/memory
MIN_AREA = 0.001       # ignore instances smaller than this fraction of the image
DEDUPE_IOU = 0.7       # drop overlapping instances found under different concepts

# Candidate concepts. SAM 3 only looks for what you ask for, so edit freely
# or pass --concepts with one phrase per line. More concepts = slower.
DEFAULT_CONCEPTS = """
person, man, woman, child, dog, cat, horse, cow, sheep, bird, fish, elephant,
giraffe, zebra, bear, car, truck, bus, motorcycle, bicycle, boat, airplane,
train, traffic light, stop sign, bench, chair, sofa, table, desk, bed, pillow,
lamp, bookshelf, cabinet, door, window, curtain, mirror, rug, clock, vase,
potted plant, flower, tree, bush, laptop, monitor, keyboard, computer mouse,
phone, television, remote control, speaker, headphones, camera, cup, mug,
glass, bottle, bowl, plate, fork, knife, spoon, pot, pan, kettle, toaster,
microwave, oven, refrigerator, sink, toilet, towel, book, paper, pen, scissors,
backpack, handbag, suitcase, umbrella, hat, helmet, shirt, jacket, dress, pants,
shoe, glasses, watch, ball, kite, skateboard, surfboard, guitar, teddy bear,
apple, banana, orange, tomato, carrot, broccoli, bread, cake, pizza, sandwich,
fence, building, house, bridge, road, sign, flag, balloon, painting, basket,
box, bucket, candle, mountain, rock, cloud
"""


@dataclass
class Instance:
    name: str
    mask: np.ndarray   # bool, H x W
    score: float


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def choose_file():
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        title="Open an image",
        filetypes=[("Images", "*.png *.jpg *.jpeg"), ("All files", "*.*")],
    )
    root.destroy()
    return path


def load_concepts(path):
    if path:
        with open(path, encoding="utf-8") as f:
            items = [line.strip() for line in f if line.strip()]
    else:
        items = [c.strip() for c in DEFAULT_CONCEPTS.replace("\n", " ").split(",")]
    seen, out = set(), []
    for c in items:
        if c and c not in seen:
            seen.add(c)
            out.append(c)
    return out


@torch.no_grad()
def find_instances(image, concepts, device, threshold, batch_size):
    """Run SAM 3 once per concept (batched) and collect every named instance."""
    print(f"Loading {SAM3_MODEL} on {device}...")
    model = Sam3Model.from_pretrained(SAM3_MODEL).to(device).eval()
    processor = Sam3Processor.from_pretrained(SAM3_MODEL)

    found = []
    for start in range(0, len(concepts), batch_size):
        chunk = concepts[start:start + batch_size]
        # Same image repeated, one text prompt per copy (documented batched usage).
        inputs = processor(images=[image] * len(chunk), text=chunk, return_tensors="pt").to(device)
        outputs = model(**inputs)
        results = processor.post_process_instance_segmentation(
            outputs,
            threshold=threshold,
            mask_threshold=0.5,
            target_sizes=inputs.get("original_sizes").tolist(),
        )
        for name, res in zip(chunk, results):
            for mask, score in zip(res["masks"], res["scores"]):
                m = mask.detach().cpu().numpy().astype(bool)
                if m.mean() >= MIN_AREA:
                    found.append(Instance(name, m, float(score)))
        done = min(start + batch_size, len(concepts))
        print(f"  {done}/{len(concepts)} concepts, {len(found)} instances so far", end="\r")
    print()
    return found


def dedupe(instances):
    """Keep the highest-scoring instance when several concepts hit the same object
    (e.g. 'sofa' and 'couch', or 'person' and 'woman')."""
    kept, small = [], []
    for inst in sorted(instances, key=lambda i: i.score, reverse=True):
        s = inst.mask[::4, ::4]   # downsample for a fast IoU check
        dup = False
        for k_small in small:
            inter = np.logical_and(s, k_small).sum()
            union = np.logical_or(s, k_small).sum()
            if union and inter / union > DEDUPE_IOU:
                dup = True
                break
        if not dup:
            kept.append(inst)
            small.append(s)
    return kept


def show(image, chosen, overlay):
    fig = plt.figure(figsize=(9, 10))
    gs = fig.add_gridspec(2, 1, height_ratios=[4, 1.3])

    ax_img = fig.add_subplot(gs[0])
    shown = np.asarray(image).astype(np.float32)
    if overlay and chosen:
        cmap = plt.get_cmap("tab20")
        for i, inst in enumerate(chosen):
            color = np.array(cmap(i % 20)[:3]) * 255
            m = inst.mask
            if m.shape != shown.shape[:2]:
                continue
            shown[m] = shown[m] * 0.5 + color * 0.5
    ax_img.imshow(shown.astype(np.uint8))
    ax_img.axis("off")

    ax_txt = fig.add_subplot(gs[1])
    ax_txt.axis("off")
    if chosen:
        counts = Counter(i.name for i in chosen)
        labels = [f"{n} ×{c}" if c > 1 else n for n, c in counts.items()]
        body = textwrap.fill(" • ".join(labels), width=80)
        text = f"Objects ({len(chosen)} chosen at random):\n\n{body}"
    else:
        text = "No objects found. Try lowering --threshold or supplying more --concepts."
    ax_txt.text(0.5, 0.95, text, ha="center", va="top", fontsize=12, transform=ax_txt.transAxes)

    fig.tight_layout()
    plt.show()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("image", nargs="?", help="PNG/JPEG path (omit for a file dialog)")
    p.add_argument("--concepts", help="text file with one concept phrase per line")
    p.add_argument("--threshold", type=float, default=0.5, help="SAM 3 detection threshold")
    p.add_argument("--batch", type=int, default=4, help="concepts per forward pass (lower if out of memory)")
    p.add_argument("--overlay", action="store_true", help="tint the chosen objects on the image")
    args = p.parse_args()

    path = args.image or choose_file()
    if not path:
        print("No file selected.")
        return

    image = Image.open(path).convert("RGB")
    image.thumbnail((MAX_SIDE, MAX_SIDE))
    device = pick_device()

    concepts = load_concepts(args.concepts)
    instances = find_instances(image, concepts, device, args.threshold, args.batch)
    instances = dedupe(instances)
    print(f"{len(instances)} distinct objects found")

    n = min(random.randint(5, 20), len(instances))
    chosen = random.sample(instances, n) if n else []
    print("Chosen:", [i.name for i in chosen])
    show(image, chosen, args.overlay)


if __name__ == "__main__":
    main()
