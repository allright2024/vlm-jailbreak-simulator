"""Attack base class: the single interface all eight attacks implement.

The eight methods we target fall into three families with different control
flow, but they all map one ``Behavior`` to one or more ``TestCase`` objects. The
base class captures that, plus a declaration of what each attack *requires* so
the runner can skip incompatible (attack, dataset) pairs with a clear message
instead of crashing deep inside the attack.

Families (for reference; all use the same ``build`` entry point):
  * Offline generators (FigStep, CS-DJ, DMN, HADES-blackbox): produce test
    cases without touching the target model.
  * Surrogate-optimization (FORCE, HADES-whitebox): optimize an image on a
    white-box surrogate; ``requires_surrogate`` is True.
  * Target-in-the-loop (JOOD, ASO): query the target while attacking.
    These return multiple attempts; the runner scores Pass@k. ``interactive``
    is True and ``build_interactive`` is used instead of ``build``.

An attack that enhances another attack's output (ASO) sets ``is_plugin`` and
receives parent test cases.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import re
from typing import Any, Optional

from .schema import Behavior, TestCase


LINEAGE_SCHEMA_VERSION = 2


def normalized_instruction_hash(text: str) -> str:
    """Stable hash of an instruction, robust to casing/whitespace/trailing punct.

    Used to catch train/eval leakage even when the same request appears under a
    different ``behavior_id`` (e.g. the same prompt in AdvBench and AdvBenchM).
    """
    norm = re.sub(r"\s+", " ", (text or "").strip().lower())
    norm = norm.rstrip(".!?\"' ")
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()[:16]


def file_sha256(path: str) -> Optional[str]:
    """SHA-256 of a file's bytes, or None if it is missing."""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


class Attack:
    """Base class for jailbreak attacks.

    Subclasses set ``name`` (via @register_attack) and implement ``build`` (and,
    for target-in-the-loop attacks, ``build_interactive``).
    """

    name: str = ""

    # Reproducibility metadata.  Subclasses pin the upstream revision they were
    # checked against.  Fidelity is split into two orthogonal axes so the
    # manifest never overclaims:
    #   * algorithm_fidelity  -- does the *generation algorithm* match upstream?
    #       "faithful_port" | "adapted" | "replay"   (a code property, fixed)
    #   * experiment_profile  -- which *settings* this run used, from config
    #       e.g. "quick" (rolled-back, fast) vs "official" (paper hyperparams)
    # substitutes_model marks a model/dataset swap that blocks exact paper
    # reproduction even at the official profile (e.g. a different surrogate).
    official_repo: str = ""
    official_commit: str = ""
    algorithm_fidelity: str = "adapted"
    substitutes_model: bool = False
    # The effective hyperparameters that MUST hold for a run to count as the
    # official profile, keyed by resolved instance-attribute name. Empty means
    # the official profile cannot be verified (so paper_reproduction stays False
    # even if a config merely *labels* itself official).
    official_signature: dict[str, Any] = {}

    # --- capability / requirement declaration (used by the runner) ----------
    requires_images: bool = False      # needs dataset-provided harmful images
    requires_keyword: bool = False     # needs Behavior.keyword
    requires_statement: bool = False   # needs Behavior.statement (declarative rephrase)
    requires_target: bool = False      # needs Behavior.target (desired prefix)
    requires_surrogate: bool = False   # optimizes on a white-box surrogate model
    interactive: bool = False          # queries the target during the attack
    is_plugin: bool = False            # enhances another attack's test cases
    num_attempts: int = 1              # attempts per behavior (Pass@k)

    def __init__(self, config: Optional[dict[str, Any]] = None):
        self.config = config or {}
        self.excluded_behavior_ids: list[str] = []
        # Universal attacks fit an artifact on a train split; by default those
        # train behaviors are removed from same-benchmark evaluation.
        self.exclude_train_from_eval = bool(self.config.get("exclude_train_from_eval", True))
        # Artifact-reuse lineage (set by prepare() for universal attacks).
        self._artifact_reused = False
        self._lineage_verified = False
        self._lineage_record: Optional[dict[str, Any]] = None
        self._lineage_sidecar_sha256: Optional[str] = None

    def provenance(self) -> dict[str, Any]:
        """Machine-readable implementation provenance for the run manifest.

        ``experiment_profile`` comes from config (default "quick"), so rolling a
        config back to fast settings is reflected honestly; ``paper_reproduction``
        is only true for a faithful algorithm run at the official profile with no
        model/dataset substitution.
        """
        profile = str(self.config.get("experiment_profile", "quick"))
        verified = self.profile_verified()
        artifact_reused = bool(self.config.get("adv_image") or self.config.get("delta_path")) \
            or self._artifact_reused
        lineage_ok = (not artifact_reused) or self._lineage_verified
        return {
            "official_repo": self.official_repo,
            "official_commit": self.official_commit,
            "algorithm_fidelity": self.algorithm_fidelity,
            "experiment_profile": profile,
            "effective_config": self.effective_config(),
            "profile_verified": verified,
            "substitutes_model": self.substitutes_model,
            "artifact_reused": artifact_reused,
            "artifact_lineage_verified": self._lineage_verified,
            "artifact_lineage": self._lineage_record,
            "lineage_sidecar_sha256": self._lineage_sidecar_sha256,
            # Config-aware, not label-aware: every result-affecting setting must
            # match the official signature, the algorithm must be a faithful port
            # with no model substitution, and a reused artifact must have verified
            # lineage (built by this exact config + inputs).
            "paper_reproduction": bool(
                self.algorithm_fidelity == "faithful_port"
                and profile == "official"
                and verified
                and not self.substitutes_model
                and lineage_ok
            ),
        }

    def effective_config(self) -> dict[str, Any]:
        """Resolved result-affecting settings, keyed like ``official_signature``.

        Faithful-port attacks override this so ``profile_verified`` compares the
        *complete* effective configuration (not just a couple of fields)."""
        return {}

    def profile_verified(self) -> bool:
        """True iff every official-signature setting matches the effective config.

        An empty signature means "cannot verify" and returns False, so a config
        that only *labels* itself ``official`` never earns paper_reproduction.
        """
        if not self.official_signature:
            return False
        # Exact equality is intentional: adding a new result-affecting setting to
        # ``effective_config`` without also pinning it in the official signature
        # must fail closed rather than silently preserving the official label.
        return self.effective_config() == self.official_signature

    # --- artifact-reuse lineage (universal attacks) -------------------------
    def _lineage_inputs(self) -> dict[str, str]:
        """Map of name -> absolute input file path whose *content* must match for
        a reused artifact to count as built by this config (train CSV, base image,
        target corpus, ...). Overridden by universal attacks."""
        return {}

    def _lineage_auxiliary_files(self, artifact_abs: str) -> dict[str, str]:
        """Additional files that form one logical artifact (for example JPS's
        steer prompt). Their bytes are authenticated alongside the main image."""
        return {}

    def _lineage_code_files(self) -> list[str]:
        """Implementation files whose bytes determine artifact compatibility.

        Subclasses add helpers/adapters that participate in optimization. Keeping
        this fingerprint in the sidecar prevents a cache produced by older attack
        code from being silently reused after an implementation change.
        """
        subclass_file = inspect.getsourcefile(type(self))
        return [p for p in (__file__, subclass_file) if p]

    def _implementation_sha256(self) -> str:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        h = hashlib.sha256()
        for path in sorted(set(os.path.abspath(p) for p in self._lineage_code_files())):
            rel = os.path.relpath(path, repo_root)
            digest = file_sha256(path)
            if digest is None:
                raise FileNotFoundError(f"lineage implementation file missing: {path}")
            h.update(rel.encode("utf-8"))
            h.update(b"\0")
            h.update(digest.encode("ascii"))
            h.update(b"\0")
        return h.hexdigest()

    @staticmethod
    def _hashed_files(paths: dict[str, str], *, strict: bool) -> dict[str, str | None]:
        hashes = {name: file_sha256(path) for name, path in paths.items()}
        missing = [paths[name] for name, digest in hashes.items() if digest is None]
        if strict and missing:
            raise FileNotFoundError("lineage file(s) missing: " + ", ".join(missing))
        return hashes

    def _lineage_payload(self, artifact_abs: str, *, strict: bool = False) -> dict[str, Any]:
        artifact_abs = os.path.abspath(artifact_abs)
        artifact_sha = file_sha256(artifact_abs)
        if strict and artifact_sha is None:
            raise FileNotFoundError(f"lineage artifact missing: {artifact_abs}")
        return {
            "lineage_schema_version": LINEAGE_SCHEMA_VERSION,
            "attack": self.name,
            "official_repo": self.official_repo,
            "official_commit": self.official_commit,
            "algorithm_fidelity": self.algorithm_fidelity,
            "implementation_sha256": self._implementation_sha256(),
            "artifact_sha256": artifact_sha,
            "effective_config": self.effective_config(),
            "inputs": self._hashed_files(self._lineage_inputs(), strict=strict),
            "auxiliary_files": self._hashed_files(
                self._lineage_auxiliary_files(artifact_abs), strict=strict
            ),
            "source_model": getattr(self, "surrogate", getattr(self, "source_model", "")),
            "source_model_revision": getattr(
                self, "surrogate_revision", getattr(self, "source_model_revision", None)
            ),
            "training_behavior_ids": self.training_behavior_ids(),
            "training_instruction_hashes": sorted(self.training_instruction_hashes()),
            "training_behavior_fingerprints": sorted(self.training_behavior_fingerprints()),
        }

    def write_lineage(self, artifact_abs: str) -> None:
        """Record how a freshly-optimized artifact was produced (sidecar json)."""
        payload = self._lineage_payload(artifact_abs, strict=True)
        with open(artifact_abs + ".lineage.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write("\n")
        self._artifact_reused = False
        self._lineage_verified = True
        self._lineage_record = payload
        self._lineage_sidecar_sha256 = file_sha256(artifact_abs + ".lineage.json")

    def verify_lineage(self, artifact_abs: str) -> bool:
        """Verify a reused artifact matches a sidecar built by this exact config."""
        self._artifact_reused = True
        side = artifact_abs + ".lineage.json"
        self._lineage_sidecar_sha256 = file_sha256(side)
        if not os.path.isfile(side):
            self._lineage_verified = False
            self._lineage_record = None
            return False
        try:
            with open(side, encoding="utf-8") as f:
                saved = json.load(f)
            current = self._lineage_payload(artifact_abs, strict=True)
        except (OSError, ValueError):
            self._lineage_verified = False
            self._lineage_record = None
            return False
        self._lineage_record = current
        self._lineage_verified = saved == current
        return self._lineage_verified

    # --- train/eval split (universal attacks) -------------------------------
    def training_behavior_ids(self) -> list[str]:
        """IDs used to fit an attack artifact, if the attack has a train split."""
        return []

    def training_instruction_hashes(self) -> set[str]:
        """Normalized-instruction hashes of the train split, for content-level
        leakage detection across datasets (IDs differ, wording may not)."""
        return set()

    def training_behavior_fingerprints(self) -> set[str]:
        """Hashes covering any additional train-row fields used by optimization.

        Instruction hashes are standardized for leakage detection; attacks whose
        loss also consumes targets, images, or other row metadata override this
        hook so artifact lineage covers those bytes as well.
        """
        return set()

    def evaluation_exclusion_reason(self, behavior: Behavior) -> Optional[str]:
        """Return a reason when ``behavior`` must not enter evaluation.

        Removes behaviors used to fit the adversarial artifact, by ID and by
        normalized-instruction hash (so the same request under a different ID in
        another dataset is also caught).  Kept separate from field requirements
        so train/eval leakage stays visible in logs and manifests.
        """
        if not self.exclude_train_from_eval:
            return None
        if behavior.behavior_id in set(self.training_behavior_ids()):
            return "train_split"
        if normalized_instruction_hash(behavior.instruction) in self.training_instruction_hashes():
            return "train_content_dup"
        return None

    # --- one-time artifact preparation --------------------------------------
    def prepare(self, out_dir: str) -> None:
        """Optimize or load any shared artifact once, before per-behavior build().

        Offline attacks need nothing here.  Universal (surrogate-optimized)
        attacks override this so the artifact is produced independently of which
        eval behaviors survive subsetting/exclusion -- otherwise a fully-excluded
        or ``--limit``ed selection could skip optimization entirely.
        """
        return None

    # --- compatibility check ------------------------------------------------
    def missing_requirements(self, behavior: Behavior) -> list[str]:
        """Return the names of required fields this behavior cannot satisfy.

        The runner calls this before building; a non-empty list means the
        (attack, behavior) pair is skipped with a logged reason rather than
        producing a broken test case.
        """
        missing = []
        if self.requires_images and not behavior.images:
            missing.append("images")
        if self.requires_keyword and not behavior.keyword:
            missing.append("keyword")
        if self.requires_statement and not behavior.statement:
            missing.append("statement")
        if self.requires_target and not behavior.target:
            missing.append("target")
        return missing

    # --- generation entry points -------------------------------------------
    def build(self, behavior: Behavior, out_dir: str) -> list[TestCase]:
        """Produce test cases for one behavior (offline / surrogate attacks).

        ``out_dir`` is where the attack may write generated images (its own
        subdirectory under the run's images folder). Returned ``TestCase.images``
        must be repo-root-relative paths. Implementations should be deterministic
        given a fixed seed so runs are reproducible.
        """
        raise NotImplementedError

    def build_interactive(
        self, behavior: Behavior, out_dir: str, target: "object"
    ) -> list[TestCase]:
        """Produce test cases while querying ``target`` (target-in-the-loop).

        Default: fall back to offline ``build`` and ignore the target. Attacks
        like JOOD/ASO override this to run their adaptive loop and return
        one TestCase per attempt (with increasing ``attempt`` indices).
        """
        return self.build(behavior, out_dir)

    @staticmethod
    def test_case_id(attack: str, behavior_id: str, attempt: int = 0) -> str:
        return f"{attack}:{behavior_id}:{attempt}"
