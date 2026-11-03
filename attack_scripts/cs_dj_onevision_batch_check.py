#!/usr/bin/env python3
"""Check greedy-prefix parity and memory for homogeneous CS-DJ batches."""
import json
from pathlib import Path
import sys
import time

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from core.targets.native_hf import LlavaOneVision7BChatTarget  # noqa: E402

RUN=ROOT/'runs/cs_dj_onevision_reproduction_20261022'
data=[json.loads(x) for x in (RUN/'test_cases.jsonl').read_text().splitlines()]
previous={x['custom_id']:x for x in map(json.loads,(RUN/'responses.jsonl').read_text().splitlines()) if not x.get('construction_failed')}
assert len(previous)>=3
chosen=[next(c for c in data if c['custom_id']==cid) for cid in list(previous)[:4]]
target=LlavaOneVision7BChatTarget(max_new_tokens=64)
messages=[target._messages(c['prompt'],target._images([c['image']])) for c in chosen]
start=time.perf_counter()
inputs=target.processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,
    return_dict=True,return_tensors='pt',padding=True,chat_template=target.chat_template).to(target.model.device)
torch.cuda.reset_peak_memory_stats()
with torch.inference_mode():
    output=target.model.generate(**inputs,max_new_tokens=64,do_sample=False)
elapsed=time.perf_counter()-start
checks=[]
for c,out in zip(chosen,output[:,inputs['input_ids'].shape[1]:]):
    reference=previous[c['custom_id']]['generated_token_ids'][:64]
    actual=out[:len(reference)].cpu().tolist()
    checks.append({'custom_id':c['custom_id'],'compared_tokens':len(reference),'greedy_prefix_exact':reference==actual})
value={'batch_size':len(chosen),'seconds':elapsed,'checks':checks,
       'all_checked_prefixes_exact':all(c['greedy_prefix_exact'] for c in checks),
       'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),
       'input_shape':list(inputs['input_ids'].shape),'device_map':target.model.hf_device_map,
       'language_attention_implementation':target.model.config.text_config._attn_implementation}
(RUN/'batching_diagnostic.json').write_text(json.dumps(value,indent=2)+'\n')
print(json.dumps(value,indent=2),flush=True)
