"""Benign primary-color controls for the local attention runtime."""
import contextlib
import json
import time
import torch
from torch.nn.attention import sdpa_kernel, SDPBackend
from PIL import Image
from PIL import ImageDraw, ImageFont
import paper_run as p

torch.set_num_threads(2)
torch.set_num_interop_threads(2)
victim = p.load_victim()
colors = {"red": (255, 0, 0), "green": (0, 255, 0), "blue": (0, 0, 255)}
records = []
for backend in ["auto", "math"]:
    for name, rgb in colors.items():
        ctx = sdpa_kernel(SDPBackend.MATH) if backend == "math" else contextlib.nullcontext()
        t0 = time.time()
        with torch.inference_mode(), ctx:
            answer = victim.inference(["What is the single solid color of this image? Answer with one color word."],
                                      [Image.new("RGB", (448, 448), rgb)], max_new_tokens=16, batch_size=1)[0]
        record = {"backend": backend, "color": name, "response": answer,
                  "correct": name in answer.lower(), "seconds": time.time() - t0}
        records.append(record)
        p.dump(p.RUN / "numerics_probe.json", records)
        print(json.dumps(record), flush=True)

font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 52)
for label in ["ALPHA 4827", "OMEGA 9153"]:
    image = Image.new("RGB", (448, 448), "white")
    draw = ImageDraw.Draw(image)
    draw.text((22, 160), label, font=font, fill="black")
    fixture = p.RUN / ("control_" + label.split()[0].lower() + ".png")
    image.save(fixture)
    for backend in ["auto", "math"]:
        ctx = sdpa_kernel(SDPBackend.MATH) if backend == "math" else contextlib.nullcontext()
        with torch.inference_mode(), ctx:
            answer = victim.inference(["Read the text in this image exactly."], [image], max_new_tokens=32, batch_size=1)[0]
        record = {"backend": backend, "ocr_expected": label, "response": answer,
                  "correct": label in answer}
        records.append(record)
        p.dump(p.RUN / "numerics_probe.json", records)
        print(json.dumps(record), flush=True)

image = Image.new("RGB", (448, 448), "white")
draw = ImageDraw.Draw(image)
draw.ellipse((30, 30, 180, 180), fill="red", outline="black", width=3)
draw.rectangle((240, 30, 400, 180), fill="green", outline="black", width=3)
draw.polygon([(220, 250), (130, 410), (310, 410)], fill="blue", outline="black", width=3)
image.save(p.RUN / "control_shapes.png")
for backend in ["auto", "math"]:
    ctx = sdpa_kernel(SDPBackend.MATH) if backend == "math" else contextlib.nullcontext()
    with torch.inference_mode(), ctx:
        answer = victim.inference(["Name each shape and its color in the image."], [image], max_new_tokens=80, batch_size=1)[0]
    record = {"backend": backend, "control": "three_shapes", "response": answer,
              "correct": all(x in answer.lower() for x in ["red", "green", "blue", "circle", "square", "triangle"])}
    records.append(record)
    p.dump(p.RUN / "numerics_probe.json", records)
    print(json.dumps(record), flush=True)
assert all(x["correct"] for x in records if "ocr_expected" in x or "control" in x)
