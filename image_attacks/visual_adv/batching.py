"""Batching for the released MiniGPT-4 generation distribution.

Stopping is per response: a row that emits ### is forced to EOS on its next
step, while the other rows continue. Left padding is masked from attention.
Batch size and seeds are recorded, since sampled tokens need not match serial
execution bit for bit. Batch size 1 uses the unmodified released generator.
"""

import torch


def force_finished_rows(ids, scores, done, eos_id):
    done |= ids[:, -1].eq(835)
    if ids.size(1) >= 2:
        done |= ids[:, -2].eq(2277) & ids[:, -1].eq(29937)
    scores[done] = -torch.inf
    scores[done, eos_id] = 0
    return scores


def decode_released(tokenizer, row, eos_id=1):
    # Exclude padding after the first natural/forced EOS, but retain EOS itself
    # as the original serial decoder did. Its input-ID prefix is one BOS token.
    tokens = row.tolist()
    end = next((i + 1 for i in range(1, len(tokens)) if tokens[i] == eos_id), len(tokens))
    tokens = tokens[:end]
    if tokens and tokens[0] == 0:
        tokens = tokens[1:]
    if tokens and tokens[0] == 1:
        tokens = tokens[1:]
    return tokenizer.decode(tokens, add_special_tokens=False).split("###")[0].split("Assistant:")[-1].strip()


def generate_batch(official, image, texts):
    from transformers import LogitsProcessor, LogitsProcessorList

    if len(texts) == 1:
        return [official.generate(image, texts[0])]
    with torch.no_grad():
        wrapper = official.wrapper
        full_texts = [wrapper.minigpt4_chatbot_prompt % ("Please extend the following sentence: " + text)
                      for text in texts]
        prompt = wrapper.Prompt(model=official.model, text_prompts=full_texts,
                                img_prompts=[[image]], device=str(official.device))
        prompt.img_embs *= len(texts)
        prompt.update_context_embs()
        contexts = prompt.context_embs
        length = max(context.size(1) for context in contexts)
        pad_id = official.model.llama_tokenizer.pad_token_id
        pad = official.model.llama_model.model.embed_tokens(
            torch.tensor([pad_id], device=official.device))
        embeddings = pad[None].expand(len(texts), length, -1).clone()
        mask = torch.zeros((len(texts), length), dtype=torch.long, device=official.device)
        for index, context in enumerate(contexts):
            embeddings[index, -context.size(1):] = context[0]
            mask[index, -context.size(1):] = 1
        eos_id = official.model.llama_model.config.eos_token_id
        done = torch.zeros(len(texts), dtype=torch.bool, device=official.device)

        class StopRows(LogitsProcessor):
            def __call__(self, input_ids, scores):
                return force_finished_rows(input_ids, scores, done, eos_id)

        generator = official.generator
        output = official.model.llama_model.generate(
            inputs_embeds=embeddings, attention_mask=mask,
            max_new_tokens=generator.max_new_tokens, num_beams=generator.num_beams,
            do_sample=True, min_length=generator.min_length, top_p=generator.top_p,
            repetition_penalty=generator.repetition_penalty, length_penalty=generator.length_penalty,
            temperature=generator.temperature, pad_token_id=pad_id,
            logits_processor=LogitsProcessorList([StopRows()]),
        )
        return [decode_released(official.model.llama_tokenizer, row, eos_id) for row in output]
