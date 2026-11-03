"""CPU regression checks for supported methods and shared Qwen preprocessing."""
import copy
import unittest

import torch

import core.attacks  # register the remaining implementations
from core.composition import load_composition_config, validate_composition_config
from core.registry import get_attack
from defenses import load_defense
from multimodalmodels.qwen2_adapter import Qwen2Adapter


class SupportedMethodsTests(unittest.TestCase):
    def test_remaining_attacks_register_and_removed_names_fail(self):
        for name in ('FigStep', 'CS-DJ', 'MML', 'JOOD', 'JPS', 'VisualAdv', 'HADES', 'JailBound'):
            self.assertEqual(get_attack(name).name, name)
        for name in ('UltraBreak', 'JailbreakInPieces'):
            with self.assertRaises(KeyError):
                get_attack(name)
        with self.assertRaisesRegex(ValueError, 'Unknown defense_name'):
            load_defense('ECSO')

    def test_remaining_recipe_quotas_and_groups_are_consistent(self):
        config = load_composition_config('configs/compositions/jailbreak_dataset_v3.yaml')
        self.assertEqual(config['expected_totals'], {'final': 1450, 'raw': 1740})
        ids = [r['id'] for r in config['recipes']]
        grouped = [r for g in config['execution_groups'] for r in g['recipes']]
        self.assertCountEqual(ids, grouped)
        self.assertEqual(len(ids), 8)

    def test_stale_recipe_cannot_silently_use_a_removed_component(self):
        config = load_composition_config('configs/compositions/jailbreak_dataset_v3.yaml')
        for component, attack in [('ultrabreak_patch', 'UltraBreak'),
                                  ('jailbreak_in_pieces_embedding', 'JailbreakInPieces')]:
            stale = copy.deepcopy(config)
            stale['recipes'][0]['stages'][0].update(component=component, attack=attack)
            with self.assertRaises(ValueError):
                validate_composition_config(stale)

    def test_qwen_patchification_keeps_values_temporal_copies_and_gradients(self):
        adapter = Qwen2Adapter.__new__(Qwen2Adapter)
        pixels = torch.arange(3 * 56 * 28, dtype=torch.float32).reshape(3, 56, 28)
        pixels.requires_grad_()
        patched = adapter.preprocess_patched(pixels, (1, 4, 2))
        self.assertEqual(tuple(patched.shape), (8, 1176))
        self.assertTrue(torch.equal(patched.flatten().sort().values,
                                    pixels.flatten().repeat(2).sort().values))
        expected = pixels[:, :14, :14].unsqueeze(1).expand(-1, 2, -1, -1).reshape(-1)
        self.assertTrue(torch.equal(patched[0], expected))
        patched.square().sum().backward()
        self.assertTrue(torch.equal(pixels.grad, 4 * pixels.detach()))


if __name__ == '__main__':
    unittest.main()
