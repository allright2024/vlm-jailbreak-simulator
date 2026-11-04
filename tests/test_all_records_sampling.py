"""Full benchmark coverage, including advice/opinion and duplicate records."""
from collections import Counter
import copy
import unittest

from core.five_method_dataset import instruction_key, load_config, select_source_parents, plan_jood_variants
from core.schema import Behavior, EvalType


class AllRecordSamplingTests(unittest.TestCase):
    def plan(self):
        config, recipes = load_config('configs/compositions/five_methods_3000.yaml')
        pools = {name: [Behavior(f'{name}_{i}', name, f'{name} question {i}',
                                eval_type=(EvalType.HARMFUL, EvalType.ADVICE, EvalType.OPINION)[i % 3],
                                category=f'category_{i % 7}') for i in range(count)]
                 for name, count in [('SafeBench', 500), ('HADES', 750), ('MM-SafetyBench', 1680)]}
        return config, recipes, pools

    def test_every_original_record_appears_before_seventy_extra_draws(self):
        config, recipes, pools = self.plan()
        parents, plans, meta = select_source_parents(config, recipes, pools, set())
        rows = [b for r in recipes for b in plans[r.id]]
        self.assertEqual(len(parents), 2930)
        self.assertEqual(len(rows), 3000)
        self.assertEqual({b.behavior_id for b in rows[:2930]}, {b.behavior_id for b in parents})
        self.assertEqual(Counter(Counter(b.behavior_id for b in rows).values()), {1: 2860, 2: 70})
        self.assertEqual(Counter(b.dataset for b in rows), {'SafeBench':512, 'HADES':768, 'MM-SafetyBench':1720})
        self.assertEqual(meta['resampled_draws'], 70)
        self.assertEqual(meta['eval_type_policy'], 'all')
        for recipe in recipes:
            self.assertEqual(len(plans[recipe.id]), recipe.count)
        self.assertEqual({b.eval_type for b in parents}, set(EvalType) - {EvalType.COPYRIGHT})

    def test_duplicate_record_ids_are_preserved_with_different_recipes(self):
        config, recipes, pools = self.plan()
        for index in range(20):
            pools['MM-SafetyBench'][index + 100].instruction = pools['SafeBench'][index].instruction
        parents, plans, meta = select_source_parents(config, recipes, pools, set())
        self.assertEqual(len(parents), 2930)
        self.assertEqual(meta['unique_instructions'], 2910)
        pairs = [(r.id, instruction_key(b.instruction)) for r in recipes for b in plans[r.id]]
        self.assertEqual(len(set(pairs)), 3000)

    def test_advice_and_opinion_are_not_filtered(self):
        config, recipes, pools = self.plan()
        parents, _, meta = select_source_parents(config, recipes, pools, set())
        expected = {b.behavior_id for rows in pools.values() for b in rows}
        self.assertEqual({b.behavior_id for b in parents}, expected)
        self.assertTrue(all(p['filtered_records'] == 0 for p in meta['pools'].values()))

    def test_seeded_schedule_is_reproducible(self):
        config, recipes, pools = self.plan()
        first = select_source_parents(config, recipes, pools, set())
        self.assertEqual(first, select_source_parents(config, recipes, pools, set()))
        other = copy.deepcopy(config)
        other['seed'] += 1
        self.assertNotEqual(first[1], select_source_parents(other, recipes, pools, set())[1])

    def test_does_not_silently_drop_training_overlap(self):
        config, recipes, pools = self.plan()
        with self.assertRaisesRegex(ValueError, 'JPS training overlaps'):
            select_source_parents(config, recipes, pools, {instruction_key(pools['SafeBench'][0].instruction)})

    def test_budget_smaller_than_source_pool_is_rejected(self):
        config, recipes, pools = self.plan()
        pools['SafeBench'] += [Behavior(f'extra_{i}', 'SafeBench', f'Extra question {i}') for i in range(71)]
        with self.assertRaisesRegex(ValueError, 'exceed'):
            select_source_parents(config, recipes, pools, set())

    def test_jood_still_covers_all_nineteen_weights_and_five_images(self):
        config, recipes, pools = self.plan()
        _, plans, _ = select_source_parents(config, recipes, pools, set())
        variants = plan_jood_variants(config, recipes, plans, 5)
        self.assertEqual(len(variants), 1840)
        self.assertEqual(set(Counter(a['source_weight'] for a in variants.values()).values()), {96, 97})
        self.assertEqual(set(Counter(a['harmless_index'] for a in variants.values()).values()), {368})
        triples = {(b.dataset, variants[(r.id, b.behavior_id)]['weight_index'],
                    variants[(r.id, b.behavior_id)]['harmless_index'])
                   for r in recipes for b in plans[r.id] if 'JOOD' in r.methods}
        self.assertEqual(len(triples), 285)


if __name__ == '__main__':
    unittest.main()
