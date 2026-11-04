"""Question-by-question sampling across full benchmark pools."""
import copy
import unittest
from collections import Counter

from core.five_method_dataset import load_config, select_source_parents, plan_jood_variants
from core.schema import Behavior, EvalType


class FullPoolSamplingTests(unittest.TestCase):
    def setup_plan(self):
        config, recipes = load_config("configs/compositions/five_methods_3000.yaml")
        config["sources"].update({"balance": "equal", "sampling": "full_pool_cycles"})
        names = config["sources"]["datasets"]
        pools = {
            name: [Behavior(f"{name}_{i}", name, f"{name} unique question {i}",
                            category=f"category_{i % 5}") for i in range(80)]
            for name in names
        }
        return config, recipes, pools

    def test_one_recipe_per_draw_and_full_pool_coverage(self):
        config, recipes, pools = self.setup_plan()
        parents, plans, metadata = select_source_parents(config, recipes, pools, set())
        pairs = [(rid, b.behavior_id) for rid, rows in plans.items() for b in rows]
        self.assertEqual(len(parents), 240)
        self.assertEqual(len(pairs), 3000)
        self.assertEqual(len(set(pairs)), 3000)
        self.assertEqual(metadata["assignment_policy"], "one_recipe_per_sampled_question")
        for name in pools:
            usage = Counter(b.behavior_id for rows in plans.values() for b in rows if b.dataset == name)
            self.assertEqual(len(usage), 80)
            self.assertEqual(sum(usage.values()), 1000)
            self.assertEqual(Counter(usage.values()), {12: 40, 13: 40})
        for recipe in recipes:
            self.assertEqual(len(plans[recipe.id]), recipe.count)
            self.assertEqual(len({b.behavior_id for b in plans[recipe.id]}), recipe.count)

    def test_complete_source_pool_before_reuse(self):
        config, recipes, pools = self.setup_plan()
        _, plans, _ = select_source_parents(config, recipes, pools, set())
        for name in pools:
            sequence = [b.behavior_id for r in recipes for b in plans[r.id] if b.dataset == name]
            for offset in range(0, len(sequence), 80):
                chunk = sequence[offset:offset + 80]
                self.assertEqual(len(chunk), len(set(chunk)))

    def test_deterministic_seeded_selection(self):
        config, recipes, pools = self.setup_plan()
        first = select_source_parents(config, recipes, pools, set())
        second = select_source_parents(config, recipes, pools, set())
        self.assertEqual(first, second)
        changed = copy.deepcopy(config)
        changed["seed"] += 1
        other = select_source_parents(changed, recipes, pools, set())
        self.assertNotEqual([b.behavior_id for b in first[0]], [b.behavior_id for b in other[0]])

    def test_excludes_training_nonharmful_and_cross_source_duplicates(self):
        config, recipes, pools = self.setup_plan()
        names = list(pools)
        pools[names[0]][0].eval_type = EvalType.ADVICE
        training = {pools[names[1]][0].instruction.casefold()}
        pools[names[2]][1].instruction = pools[names[0]][1].instruction
        parents, _, _ = select_source_parents(config, recipes, pools, training)
        keys = [" ".join(b.instruction.casefold().split()) for b in parents]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertTrue(all(b.eval_type == EvalType.HARMFUL for b in parents))
        self.assertTrue(set(keys).isdisjoint(training))
        self.assertEqual(len(parents), 237)

    def test_covers_all_jood_ratios_and_images(self):
        config, recipes, pools = self.setup_plan()
        _, plans, _ = select_source_parents(config, recipes, pools, set())
        assignments = plan_jood_variants(config, recipes, plans, 5)
        self.assertEqual(len(assignments), 1840)
        weights = Counter(row["source_weight"] for row in assignments.values())
        images = Counter(row["harmless_index"] for row in assignments.values())
        self.assertEqual(set(weights), set(config["jood"]["source_weights"]))
        self.assertLessEqual(max(weights.values()) - min(weights.values()), 1)
        self.assertEqual(set(images.values()), {368})

    def test_rejects_pools_too_small_for_a_unique_recipe(self):
        config, recipes, pools = self.setup_plan()
        name = next(iter(pools))
        pools[name] = pools[name][:10]
        with self.assertRaises(ValueError):
            select_source_parents(config, recipes, pools, set())


if __name__ == "__main__":
    unittest.main()

