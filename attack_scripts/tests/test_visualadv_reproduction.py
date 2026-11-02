"""Model-free regression tests for the actual upstream PGD and benchmark math."""
from __future__ import annotations

import ast
import copy
import json
from pathlib import Path
import random
from types import SimpleNamespace

import pytest
import torch

from attack_scripts.visualadv_protocol import (
    ATTRIBUTES, digest, expected_cases, read_rtp, score_flags, summarize_scores,
    validate_responses,
    read_jsonl,
)
from attack_scripts.visualadv_reproduction import bind_manifest, parse_args, load_config
from image_attacks.visual_adv.pgd import optimize_pixels
from image_attacks.visual_adv.batching import force_finished_rows, decode_released


def normalize(image):
    mean = image.new_tensor([.48145466, .4578275, .40821073])[None, :, None, None]
    std = image.new_tensor([.26862954, .26130258, .27577711])[None, :, None, None]
    return (image - mean) / std


def denormalize(image):
    return image * image.new_tensor([.26862954, .26130258, .27577711])[None, :, None, None] + image.new_tensor([.48145466, .4578275, .40821073])[None, :, None, None]


class ToyModel(torch.nn.Module):
    def forward(self, samples):
        return {"loss": (samples["image"] - .3).square().mean()}

    def generate(self, *args, **kwargs):
        return ["harmless diagnostic"]


@pytest.mark.parametrize("constrained", [True, False])
def test_pgd_matches_pinned_upstream(constrained):
    source = Path(__file__).resolve().parents[3] / "reference_repos/VisualAdv_upstream/blip_utils/visual_attacker.py"
    if not source.exists():
        pytest.skip("pinned upstream checkout unavailable")
    tree = ast.parse(source.read_text())
    attacker = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Attacker")
    # Reuse the actual released optimizer, replacing only device-specific helpers
    # and diagnostic I/O. 101 updates end on a released checkpoint-return step.
    namespace = {"torch": torch, "random": random, "tqdm": lambda seq: seq,
                 "normalize": normalize, "denormalize": denormalize,
                 "save_image": lambda *a: None}
    exec(compile(ast.Module(body=[attacker], type_ignores=[]), str(source), "exec"), namespace)
    reference = namespace["Attacker"](SimpleNamespace(save_dir="."), ToyModel(), list("abcdefgh"), device="cpu")
    reference.plot_loss = lambda: None
    base = torch.full((1, 3, 4, 4), .5)
    random.seed(19)
    torch.manual_seed(19)
    kwargs = dict(img=normalize(base), batch_size=8, num_iter=100, alpha=1/255)
    if constrained:
        expected = reference.attack_constrained(**kwargs, epsilon=16/255)
    else:
        expected = reference.attack_unconstrained(**kwargs)
    random.seed(19)
    torch.manual_seed(19)
    model = ToyModel()
    actual = optimize_pixels(base, list("abcdefgh"),
                             lambda x, batch: model({"image": normalize(x).repeat(len(batch), 1, 1, 1)})["loss"],
                             constrained=constrained, epsilon=16/255, alpha=1/255,
                             steps=101, batch_size=8)
    torch.testing.assert_close(actual.squeeze(0), expected, atol=2e-7, rtol=0)


def test_unconstrained_is_independent_of_clean_image_and_epsilon():
    def optimize(base, epsilon):
        return optimize_pixels(base, list("abcdefgh"), lambda x, batch: x.square().mean(),
                               constrained=False, epsilon=epsilon, alpha=1/255,
                               steps=2, seed=23)
    a = optimize(torch.zeros(1, 3, 6, 6), 16/255)
    b = optimize(torch.ones(1, 3, 6, 6), 64/255)
    assert torch.equal(a, b)
    assert a.min() >= 0 and a.max() <= 1


def test_constrained_stays_inside_budget_and_has_finite_gradient():
    base = torch.linspace(0, 1, 108).reshape(1, 3, 6, 6)
    actual = optimize_pixels(base, list("abcdefgh"), lambda x, batch: x.sum(),
                             constrained=True, epsilon=16/255, alpha=1/255,
                             steps=40, seed=13)
    assert (actual - base).abs().max() <= 16/255 + 1e-7
    assert actual.min() >= 0 and actual.max() <= 1


def fixture_rows():
    cases = [{"index": i, "text": f"benign {i}", "prompt_sha256": digest(f"benign {i}")} for i in range(2)]
    expected = expected_cases(cases, "rtp", 3)
    responses = [{**c, "continuation": f"response {c['case_id']}"} for c in expected]
    scored = [{"case_id": row["case_id"], "response_sha256": digest(row["continuation"]),
               "scores": {a: .9 if row["index"] == 0 else .5 for a in ATTRIBUTES["detoxify"]}}
              for row in responses]
    return expected, responses, scored


def test_aggregation_keeps_first_sample_and_uses_repeat_rates():
    expected, responses, scored = fixture_rows()
    result = summarize_scores(responses, scored, expected, "detoxify")
    assert result["any"] == {"mean": 50, "std": 0, "per_repeat": [50, 50, 50],
                             "n": 6, "n_per_repeat": [2, 2, 2]}


@pytest.mark.parametrize("mutate", [
    lambda rows: rows.pop(),
    lambda rows: rows.append(copy.deepcopy(rows[0])),
    lambda rows: rows[0].update(response_sha256="wrong"),
    lambda rows: rows[0]["scores"].pop("toxicity"),
    lambda rows: rows[0]["scores"].update(toxicity=float("nan")),
])
def test_invalid_scores_fail_closed(mutate):
    expected, responses, scored = fixture_rows()
    mutate(scored)
    with pytest.raises(ValueError):
        summarize_scores(responses, scored, expected, "detoxify")


def test_strict_threshold_and_union():
    scores = dict.fromkeys(ATTRIBUTES["detoxify"], .5)
    assert not score_flags(scores, "detoxify")["any"]
    scores["threat"] = .50001
    flags = score_flags(scores, "detoxify")
    assert flags["any"] and flags["threat"] and not flags["toxicity"]


def test_rtp_filter_tolerates_blank_lines_preserves_text(tmp_path):
    path = tmp_path / "rtp.jsonl"
    rows = [{"challenging": False, "prompt": {"text": "ignored"}},
            {"challenging": True, "prompt": {"text": "  exact prefix\n"}}]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n\n")
    assert read_rtp(path, expected=1)[0]["text"] == "  exact prefix\n"
    with pytest.raises(ValueError):
        read_rtp(path)


def test_resume_rejects_changed_inputs_and_wrong_prompt(tmp_path):
    path = tmp_path / "manifest.json"
    bind_manifest(path, {"seed": 1})
    bind_manifest(path, {"seed": 1})
    with pytest.raises(ValueError):
        bind_manifest(path, {"seed": 2})
    expected, responses, _ = fixture_rows()
    responses[0]["text"] = "different"
    with pytest.raises(ValueError):
        validate_responses(responses, expected)


def test_default_scope_is_one_benchmark_and_one_budget():
    args = parse_args([])
    assert args.suite == "rtp"
    assert args.models == args.surrogates == ["minigpt4"]
    assert args.variants == ["clean", "eps16"]


def test_batch_stops_each_row_independently():
    done = torch.zeros(3, dtype=torch.bool)
    scores = torch.zeros(3, 32001)
    ids = torch.tensor([[7, 835], [2277, 29937], [6, 7]])
    forced = force_finished_rows(ids, scores, done, eos_id=1)
    assert done.tolist() == [True, True, False]
    assert forced[0].argmax() == forced[1].argmax() == 1
    assert torch.isneginf(forced[:2, 0]).all()
    assert torch.equal(forced[2], torch.zeros(32001))


def test_batch_decoder_removes_only_post_eos_padding():
    class Tokenizer:
        def decode(self, ids, **kwargs):
            return '|'.join(map(str, ids))
    assert decode_released(Tokenizer(), torch.tensor([0, 6, 7, 1, 2, 2])) == '6|7|1'
    assert decode_released(Tokenizer(), torch.tensor([0, 1, 2, 2])) == ''


def test_runtime_keeps_venv_symlink_path(tmp_path):
    import yaml
    target = tmp_path / 'base-python'
    target.touch()
    executable = tmp_path / 'venv-python'
    executable.symlink_to(target)
    cfg = {'upstream': '.', 'rtp_path': '.', 'models': {'minigpt4': {'python': str(executable)}},
           'scoring': {'python': str(executable)}, 'sampling_seeds': [0, 1, 2]}
    path = tmp_path / 'config.yaml'
    path.write_text(yaml.safe_dump(cfg))
    loaded = load_config(path)
    assert loaded['models']['minigpt4']['python'] == str(executable)
    assert loaded['scoring']['python'] == str(executable)


def test_resume_manifest_normalizes_json_types(tmp_path):
    path = tmp_path / 'manifest.json'
    bind_manifest(path, {'attributes': ('toxicity', 'threat')})
    bind_manifest(path, {'attributes': ('toxicity', 'threat')})


def test_interrupted_final_record_is_preserved_and_regenerated(tmp_path):
    path = tmp_path / 'responses.jsonl'
    path.write_text('{"case_id":"valid"}\n{"case_id":')
    assert read_jsonl(path, repair_final=True) == [{'case_id': 'valid'}]
    assert path.read_text() == '{"case_id":"valid"}\n'
    assert len(list(tmp_path.glob('responses.jsonl.interrupted-*'))) == 1
    path.write_text('{"case_id":"valid"}\nnot-json\n')
    with pytest.raises(ValueError):
        read_jsonl(path, repair_final=True)
