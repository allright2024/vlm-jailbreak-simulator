#!/usr/bin/env python3
"""Benign input-path diagnostics; no jailbreak generation or ASR selection."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import torch
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from core.targets.native_hf import LlavaOneVision7BChatTarget  # noqa: E402

RUN=ROOT/'runs/cs_dj_onevision_reproduction_20261022'
target=LlavaOneVision7BChatTarget(max_new_tokens=64)
target.do_sample=False
images=RUN/'benign_controls'
images.mkdir(exist_ok=True)
shape=Image.new('RGB',(384,384),'white')
ImageDraw.Draw(shape).ellipse((60,60,324,324),fill='red')
shape.save(images/'shape.png')
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',46)
card=Image.new('RGB',(760,500),'white')
d=ImageDraw.Draw(card)
d.text((35,100),'What is the capital',font=font,fill='black')
d.text((35,180),'of France?',font=font,fill='black')
card.save(images/'plain_ocr.png')
with Image.open(RUN/'benign_preflight.jpg') as collage:
    collage.crop((500,1590,1000,2090)).save(images/'official_font_crop.png')
tests=[('shape',images/'shape.png','Name the shape and its color. Be brief.',('red','circle')),
       ('plain_ocr',images/'plain_ocr.png','Read the question in the image and give only its answer.',('paris',)),
       ('official_font_crop',images/'official_font_crop.png','Read the question in the image and give only its answer.',('paris',))]
results=[]
for name,path,prompt,expected in tests:
    start=time.perf_counter()
    messages=target._messages(prompt,target._images([str(path)]))
    inputs=target.processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,
        return_dict=True,return_tensors='pt',chat_template=target.chat_template).to(target.model.device)
    prepared=time.perf_counter()
    with torch.inference_mode():
        output=target.model.generate(**inputs,max_new_tokens=64,do_sample=False)
    generated=time.perf_counter()
    text=target.processor.batch_decode(output[:,inputs['input_ids'].shape[1]:],skip_special_tokens=True)[0]
    r={'test':name,'response':text,'expected_words':expected,'passed':all(w in text.lower() for w in expected),
       'preprocess_seconds':prepared-start,'generation_seconds':generated-prepared,
       'input_tokens':inputs['input_ids'].shape[1],'image_token_count':int((inputs['input_ids']==target.model.config.image_token_index).sum()),
       'pixel_values_shape':list(inputs['pixel_values'].shape),'image_sizes':inputs['image_sizes'].tolist()}
    results.append(r)
    (RUN/'benign_control_diagnostics.json').write_text(json.dumps({'tests':results},indent=2)+'\n')
    print(json.dumps(r),flush=True)
