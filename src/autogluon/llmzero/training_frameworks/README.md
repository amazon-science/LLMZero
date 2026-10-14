# Training Frameworks

This package provides an abstract interface that supplies every piece of
framework-specific knowledge the MCTS engine needs: hyperparameter guidance,
metric descriptions, the stdout grammar, the observer set, checkpoint layout,
the optimization direction, and prompt customization. The agents
(`agents/`), prompts (`prompts/`), execution observers
(`execution/observers/`), and node managers (`managers/`) all talk **only** to
the `BaseTrainingFramework` ABC. Adding a new framework requires **no edits** to
`prompts/`, `execution/observers/`, `constants.py`, or the node managers.

The promise: **one package folder + one registry line + a config value.** Drop a
`training_frameworks/<name>/` package, call `register_framework("<name>", <Class>)`
once in `__init__.py`, and set `training_framework: <name>` in the config YAML.
Almost every override point has a safe default on the base class, so a framework
overrides only what applies and otherwise runs with no framework-specific
guidance. The one required override is `stdout_parser()` (abstract — every
framework owns its stdout grammar).

The registry lives in `__init__.py` and mirrors `compute_infra/`:

| Function | Purpose |
|----------|---------|
| `register_framework(name, cls)` | Add a framework class under a name. |
| `get_framework(name)` | Look up a class by name; raises `KeyError` if unknown. |
| `get_framework_from_config(config)` | Resolve from `config.training_framework`; returns `BaseTrainingFramework` when unset. |
| `validate_framework_config(config)` | Resolve the name (raise on unknown) and run the framework's optional `validate()` hook. Unset is a no-op. |

The built-in framework is `verl_grpo`
(`training_frameworks/verl_grpo/`, the worked example referenced throughout
this document), the verl 0.9.0 GRPO/PPO RL post-training framework.

## The stdout-parsing contract

A framework supplies a **stateless** parser so the generic observers stay
framework-agnostic mechanisms — the observers handle *what to do* with a signal,
the parser handles *how to read it from a line*. Parsers subclass
`BaseStdoutParser` (`stdout_parser.py`). The contract:

- `parse_step_metrics(line) -> (step: int, {metric_name: float}) | None`
  The **only** `@abstractmethod`. Returns the step number and a dict of metric
  values for a metrics line, or `None` for a non-metrics line.

- `parse_checkpoint(line, pending_step=None) -> {'step': int, 'path': str} | {'pending_step': int} | None`
  Returns a completed checkpoint (`{'step', 'path'}`) for a save line. Some
  frameworks print a save **marker** on one line and the checkpoint **path** on a
  later line; for those, the parser returns `{'pending_step': int}` from the
  marker line to *prime* a step, then pairs it with the path on the following
  line. The parser is **stateless** — the **observer** owns the pending-step
  state and passes the primed `pending_step` back in on subsequent calls.

- `parse_progress(line) -> (iteration: int, seconds_per_iter: float) | None`
  Returns the current iteration index and the per-iteration time from a progress
  line, or `None`.

- `select_validation_metric(metrics, pattern, selected_name) -> (score, selected_name)`
  Picks the validation score from a metrics dict. **First-match lock-in:** the
  first metric name matching `pattern` is selected and **returned as
  `selected_name`**; on every later call the caller passes that name back in so
  the **same** metric is reused for the whole run. Returns `(None, selected_name)`
  when a step has no validation metric.

There is **no default parser** — every framework owns its stdout grammar. A
framework subclasses `BaseStdoutParser` and implements at minimum
`parse_step_metrics` (plus `parse_checkpoint` / `parse_progress` if it
checkpoints or reports progress). See `verl_grpo/parser.py` (`VerlStdoutParser`)
for a worked example: a tight step pattern, the generic `key:value` metric-token
grammar, verl's checkpoint marker/path priming with the embedded
`global_step_<N>` step, and tqdm-style progress.

## ABC method reference

Every `BaseTrainingFramework` classmethod, its base default, and the component
that consumes it. All have safe defaults except `stdout_parser()` (abstract —
required) — override only what applies.

| Method | Default | Consumed by |
|--------|---------|-------------|
| `hyperparameter_categories()` | `{}` | Proposer prompt (via `format_hyperparameter_guide`). |
| `hyperparameter_descriptions()` | `{}` | Proposer prompt (via `format_hyperparameter_guide`). |
| `format_hyperparameter_guide()` | renders categories + descriptions, `""` if empty | Proposer prompt. |
| `metric_descriptions()` | `{}` | Proposer + early-stopper prompts (via `format_metric_descriptions`). |
| `format_metric_descriptions(metric_names)` | renders descriptions, `""` if empty | Proposer + early-stopper prompts. |
| `default_text_metrics()` | `[]` | Proposer + early-stopper agents (and the proposer's neutral-template metric fallback). |
| `default_plot_metrics()` | `[]` | Proposer + early-stopper agents (and the proposer's neutral-template metric fallback). |
| `validation_metric_pattern()` | `None` | Best-validation observer (via `build_observers`); overridden by the `validation_metric_pattern` config key. |
| `score_direction()` | `"maximize"` | Node managers (best-tracking, UCT normalization, MCTS selection). |
| `normalize_score(score)` | identity for maximize, negated for minimize | Node managers (UCT normalization + comparisons). |
| `better(candidate, incumbent)` | `normalize_score(candidate) > normalize_score(incumbent)` | Node managers (best/worst tracking, MCTS selection). |
| `checkpoint_resume_info(node, ckpt_config)` | generic resume/from-scratch text | Post-training-coder prompt. |
| `training_config_filename()` | `"training_config.yaml"` | Proposer prompt (locates the previous-config file). |
| `filter_training_config(config)` | identity (returns config unchanged) | Proposer prompt (previous-config display). |
| `stdout_parser()` | **(abstract — required)** | All generic observers (injected via `build_observers`). |
| `build_observers(ctx)` | standard observer set wired with `stdout_parser()` | Executer agent. |
| `validate_checkpoint(checkpoint)` | path is a non-empty directory | Node managers (checkpoint filtering). |
| `prompt_fragments(agent_name)` | `{}` | Proposer + early-stopper prompts (injected into the neutral default templates). |
| `prompt_template(agent_name)` | `None` | Proposer + early-stopper prompts (full template override). |

`agent_name` for the prompt hooks is one of `{"proposer", "early_stopper"}`.

## Score direction (maximize vs minimize)

`score_direction()` returns `"maximize"` (default) or `"minimize"`.
`normalize_score()` and `better()` map every raw validation score into a single
**higher-is-better internal space** (identity for maximize, negated for
minimize), so the node managers run all comparisons and min-max UCT
normalization on the normalized value regardless of direction. The **raw** score
is still what gets stored on the node and reported — normalization is purely an
internal convenience for MCTS.

verl GRPO is a **maximize** framework (higher validation reward is better), so
`verl_grpo` does not override `score_direction()`. A **minimize** framework — for
example an SFT run scored by validation/eval loss — only needs to override
`score_direction()` to return `"minimize"`; `normalize_score`/`better` then do
the right thing automatically, and nothing in the managers changes.

`score_direction()` is threaded one step further upstream too: the default
`build_observers()` passes it to the `ValidationObserver`, which selects the
**best checkpoint within a single run** accordingly — the highest metric value
for maximize, the lowest for minimize. So a minimize framework that overrides
only `score_direction()` is correct end to end (per-run checkpoint selection →
per-node score → MCTS ranking) with no observer or manager edits.

## Validation score mode

`validation_score_mode` (config) chooses how a run's per-step validation metric is
reduced to the single score the node reports to MCTS. Both modes select the metric via
`validation_metric_pattern` (config key or framework default) with first-match lock-in,
and feed the chosen `ValidationObserver` reduction:

- **`best_validation`** (default) — report the direction-aware best across the run
  (highest for maximize, lowest for minimize).
- **`final_validation`** — report the **last** recorded value of the metric.

**Throughput example.** To optimize training throughput, set
`validation_score_mode: final_validation`, point `validation_metric_pattern` at a
running-average seconds-per-step metric your framework logs, and make the framework
`score_direction() == "minimize"`. MCTS then drives the last-reported step time down.

## Two-tier observers

How much a framework customizes execution depends on whether it needs new
*signals* or just a new *grammar* for the existing signals.

**Tier 1 — parser only.** If your framework produces the same kinds of signals
(step metrics, checkpoints, progress, validation score) but logs them with a
different stdout grammar, ship only a `parser.py` (subclass `BaseStdoutParser`)
and return it from `stdout_parser()`. The base `build_observers()` wires that parser into the
standard observer set automatically — no observer code to write. This covers
"same observers, different stdout grammar." `verl_grpo` is a Tier-1 framework: it
provides `VerlStdoutParser` and reuses the default observers.

**Tier 2 — own observers.** If your framework needs *different* observers (new
signals, replaced behavior), ship your own `observers/` package (subclass
`BaseExecutionObserver`) and override `build_observers(ctx)` to add or replace
observers in the returned list. This requires **zero edits** to the core
`execution/observers/` package.

`build_observers(ctx)` receives an `ObserverContext` (`observer_context.py`)
carrying exactly the execution inputs the executer agent knows at submit time:

- `validation_score_mode` (`"best_validation"` or `"final_validation"`) and
  `validation_metric_pattern`
- `checkpoint_optimization_enabled` (whether to attach the checkpoint-path observer)
- `early_stopper_enabled`, `early_stopper_callback`, `best_run_metrics`,
  `best_run_validation_score`, `min_steps_before_early_stop`,
  `checkpoint_interval_seconds`, `is_from_scratch` (the LLM early stopper)

A single framework may use **different parser objects** for different observers
within one `build_observers()` call if it needs to. The base implementation
injects the same framework parser into every observer, including
`CheckpointPathObserver` (which owns the checkpoint grammar). `build_observers`
owns that wiring.

## Prompt customization

Frameworks tailor the proposer and early-stopper prompts in two ways:

- `prompt_fragments(agent_name)` returns a dict of text fragments
  **injected into the neutral default templates**. Recognized keys (all
  optional): `objective`, `role`, `metric_interpretation`. Use this to flavor the
  default prompt without rewriting it.
- `prompt_template(agent_name)` returns a **full template override** string, or
  `None` to keep the neutral default.

`agent_name` is one of `{"proposer", "early_stopper"}`. See
`verl_grpo/prompts/` (`proposer.py`, `early_stopper.py`, dispatched by
`prompts/__init__.py:fragments_for`) for RL-flavored fragments — e.g. an
`objective` of "maximize the validation score" plus metric-interpretation
guidance.

## How to add a framework

Use `verl_grpo/` as the template for the package layout.

### 1. Create the package

```
src/autogluon/llmzero/training_frameworks/<name>/
├── __init__.py          # re-export the framework class
├── framework.py         # subclass BaseTrainingFramework (thin wiring class)
├── parser.py            # optional: BaseStdoutParser subclass
├── hyperparameters.py   # optional: HYPERPARAMETER_CATEGORIES / _DESCRIPTIONS
├── metrics.py           # optional: METRIC_DESCRIPTIONS + default metric lists
├── checkpoints.py       # optional: resume-info text + validate_checkpoint
├── config_filter.py     # optional: filter_training_config
├── observers/           # optional: framework-owned observers (Tier 2 only)
└── prompts/             # optional: prompt fragments / templates
```

Split by concern as `verl_grpo` does; `framework.py` stays a thin class that
delegates to the modules.

### 2. Override only what applies

`stdout_parser` is required (abstract — no default); every other ABC method has a
safe default. A typical RL-style framework also overrides:
`hyperparameter_categories` / `hyperparameter_descriptions`,
`metric_descriptions`, `default_text_metrics` / `default_plot_metrics`,
`validation_metric_pattern`, and `score_direction` (if not maximize). Everything
else (prompt fragments, config filtering, checkpoint validation/resume) is
optional polish.

### 3. Provide a parser (required)

Every framework owns its stdout grammar, so add a `parser.py` and return its
parser from `stdout_parser()` (there is no default). Subclass
`BaseStdoutParser` and implement `parse_step_metrics` (plus
`parse_checkpoint`/`parse_progress` if applicable), as `VerlStdoutParser` does.

### 4. Override `build_observers()` only for framework-owned observers

If you need new or replaced observers, ship an `observers/` package and override
`build_observers(ctx)` (Tier 2). Otherwise leave it — the base wires the standard
set with your parser.

### 5. Register and configure

Add one line to `training_frameworks/__init__.py`:

```python
from .<name> import MyFramework  # noqa: E402

register_framework("<name>", MyFramework)
```

Then select it in the config YAML:

```yaml
training_framework: <name>
```

(There is no dedicated CLI flag for the framework selection; set it in the config
YAML.)

### Minimal skeleton

```python
# training_frameworks/<name>/framework.py
from ..base import BaseTrainingFramework


class MyFramework(BaseTrainingFramework):
    @classmethod
    def hyperparameter_categories(cls) -> dict:
        return {"Optimization": ["learning_rate"]}

    @classmethod
    def hyperparameter_descriptions(cls) -> dict:
        return {"learning_rate": ("Step size for updates.", "Higher = faster but less stable.")}

    @classmethod
    def metric_descriptions(cls) -> dict:
        return {"eval/loss": "Validation loss; lower is better."}

    @classmethod
    def default_text_metrics(cls) -> list:
        return ["eval/loss"]

    @classmethod
    def default_plot_metrics(cls) -> list:
        return ["eval/loss"]

    @classmethod
    def validation_metric_pattern(cls) -> str:
        return r"eval/loss"

    @classmethod
    def score_direction(cls) -> str:
        return "minimize"  # lower eval loss is better

    # REQUIRED: no default parser
    @classmethod
    def stdout_parser(cls):
        from .parser import MyStdoutParser  # subclass of BaseStdoutParser

        return MyStdoutParser()
```

## Onboarding a new model

There is a clean split between what lives in the **task description / config**
and what lives in the **framework plugin**:

- **Task description / config (per run):** the model path, train/val/test data
  files, run-specific output paths, project/experiment names, and node counts.
  These come from the description templates under `constants/`
  (`description_template_lora.txt`, `description_template_fft.txt`,
  `description_template_given_config.txt`); the templates themselves document
  every `{{PLACEHOLDER}}` they expect, and the run-folder layout is covered in
  the example run folders under `examples/`.
- **Framework plugin (per framework):** hyperparameter guidance, metric
  descriptions, checkpoint layout (`validate_checkpoint` /
  `checkpoint_resume_info`), the stdout grammar (`stdout_parser`), the observer
  set (`build_observers`), and the objective direction (`score_direction`).

Because none of the model-specific values live in the framework, **onboarding the
same framework with a new model needs only a new task description** — no
framework code changes. You write framework code only when you adopt a genuinely
new training framework (a different stdout grammar, metric set, checkpoint
layout, or objective direction).
