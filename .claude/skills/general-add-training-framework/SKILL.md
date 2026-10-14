---
name: General - Add Training Framework
description: "[General] Scaffold a new training-framework plugin under training_frameworks/<name>/, deriving stdout-parser regexes from a real completed training log. Engine extension; framework-agnostic."
---

# Add Training Framework

Scaffold a new training-framework plugin under
`src/autogluon/llmzero/training_frameworks/<name>/`, deriving the stdout parser
regexes from a **real completed training log** the user supplies. The framework
contract is defined in `src/autogluon/llmzero/training_frameworks/README.md`;
`training_frameworks/verl_grpo/` is the worked example. Follow them.

## Input

- **Framework name**: registry key + package dir name (e.g. `hf_sft`).
- **Framework dir**: where to scaffold. Default
  `src/autogluon/llmzero/training_frameworks/<name>/`.
- **Completed training log**: path to a finished run's stdout. REQUIRED — the
  parser regexes are derived and verified against this log, not guessed.
- **score_direction**: `maximize` (default) or `minimize`.
- **Tier**: Tier 1 (parser only, reuse the standard observers) or Tier 2 (ship
  framework-owned observers). Default Tier 1.
- **Optional**: hyperparameter categories, metric descriptions.

Ask for any missing input one question at a time. Do not proceed without the log.

## Steps

### 1. Read the contract

Read `src/autogluon/llmzero/training_frameworks/README.md` (the ABC method table,
the stdout-parsing contract, the two-tier observer model) and skim
`training_frameworks/verl_grpo/` for the package layout. There is **no default
parser** — the new framework MUST ship a `BaseStdoutParser` subclass implementing
`parse_step_metrics`.

### 2. Read and mine the training log

Read the supplied log. Identify, with concrete sample lines quoted back to the user:

1. **Step-metrics lines** — recurring per-step lines carrying `key:value` metrics.
   Derive a step anchor regex (something stable that begins these lines) and the
   metric-token regex. Reuse verl's generic token regex unless the log differs:
   `([\w/@-]+):(?:np\.(?:float64|float32|int32|int64)\()?([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\)?`
2. **Checkpoint lines** — a save marker and/or a path line. If the path embeds the
   step (like verl's `global_step_<N>`), capture it; else prime a pending step from
   the marker and pair it with the next path line.
3. **Progress lines** — tqdm/iteration lines with `(iteration, seconds_per_iter)`.
4. **Validation metric** — grep the log for the metric the user wants MCTS to
   optimize; propose a `validation_metric_pattern` regex and show the matched
   metric names. Get the user to confirm.

### 3. Decide the log stream (stdout / stderr / both)

Inspect the sample log to determine which stream the framework writes metrics to.
Many frameworks (anything using loguru's default handler, or Python `logging`'s
default `StreamHandler`) write to **stderr**, not stdout. The executor already
separates the two streams and the base observer routes lines by
`BaseTrainingFramework.log_stream()`:

- Declare `log_stream()` on the framework class: return `"stdout"` (default),
  `"stderr"`, or `"both"`.
- If the run was captured as a single merged file you cannot tell — check the
  framework's logging setup in source (e.g. `logger.add(...)`, `wandb`,
  `print(file=sys.stderr)`), or run a quick capture with stdout/stderr split.
- Getting this wrong is the #1 reason a new framework's metrics are silently
  never parsed (observers see an empty stream).

### 4. Verify regexes against the log BEFORE writing files

For each derived regex, run it against real lines sampled from the log and show
the parsed output:
- `parse_step_metrics`: show `(step, {metric: value})` for 3–5 sampled lines.
- `parse_checkpoint`: show the `{step, path}` (and any pending-step priming).
- `parse_progress`: show `(iteration, seconds_per_iter)`.
- `validation_metric_pattern`: show which metric name it locks onto.

Regex-from-sample is best-effort — the log is ground truth. Only proceed once the
parsed output is correct. Iterate on the regexes with the user if any are wrong.

### 5. Scaffold the package

Create under the framework dir (mirroring `verl_grpo/`):

```
<dir>/
  __init__.py          # re-export the framework class
  framework.py         # subclass BaseTrainingFramework; override applicable methods
  parser.py            # subclass BaseStdoutParser: parse_step_metrics
                       #   (+ parse_checkpoint / parse_progress when present in the log)
  hyperparameters.py   # only if the user provided HP categories/descriptions
  metrics.py           # only if the user provided metric descriptions / default lists
  checkpoints.py       # only if the framework has a checkpoint layout to validate/resume
```

`parser.py` skeleton (fill the regexes derived + verified in steps 2–3):

```python
import re
from typing import Dict, Optional, Tuple

from ..stdout_parser import BaseStdoutParser

_STEP_PATTERN = re.compile(r"<derived step anchor>")
_METRIC_TOKEN = re.compile(
    r"([\w/@-]+):(?:np\.(?:float64|float32|int32|int64)\()?([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\)?"
)


class <Name>StdoutParser(BaseStdoutParser):
    step_pattern = _STEP_PATTERN

    def parse_step_metrics(self, line: str) -> Optional[Tuple[int, Dict[str, float]]]:
        if not self.step_pattern.search(line):
            return None
        step_match = re.search(r"<step number capture>", line)
        if not step_match:
            return None
        current_step = int(step_match.group(1))
        matches = _METRIC_TOKEN.findall(line)
        metrics = {}
        for key, value in matches:
            try:
                metrics[key] = float(value)
            except ValueError:
                continue
        return (current_step, metrics) if metrics else None

    # Add parse_checkpoint / parse_progress only if the log shows those signals.
```

**Stateful parser when metric lines lack a step number.** If a framework's
validation/eval lines do not include the step (only the periodic training-step
lines do), make the parser stateful: store `self._last_step` from each step line
and attach it to the following validation lines — return `(self._last_step, {...})`
for a validation line that carries no step of its own. Without this, those metrics
get a wrong or missing step and MCTS scoring breaks.

`framework.py` skeleton (override only what applies; see README ABC table):

```python
from ..base import BaseTrainingFramework


class <Name>Framework(BaseTrainingFramework):
    @classmethod
    def log_stream(cls) -> str:
        return "<stdout|stderr|both>"  # which stream the framework logs metrics to

    @classmethod
    def score_direction(cls) -> str:
        return "<maximize|minimize>"

    @classmethod
    def validation_metric_pattern(cls) -> str:
        return r"<confirmed validation pattern>"

    @classmethod
    def stdout_parser(cls):
        from .parser import <Name>StdoutParser

        return <Name>StdoutParser()

    # Optional: hyperparameter_categories / hyperparameter_descriptions,
    # metric_descriptions, default_text_metrics / default_plot_metrics,
    # validate_checkpoint / checkpoint_resume_info, filter_training_config.
    # For Tier 2, also override build_observers(ctx) to add your own observers.
```

`__init__.py`:
```python
from .framework import <Name>Framework

__all__ = ["<Name>Framework"]
```

### 6. Ask what checkpoint state to save/load for resume (do NOT guess)

Checkpoint semantics differ per framework and change the resume contract. Before
writing `checkpoints.py` and any resume logic, ASK the user which state a resume
must carry:

- **Weights only** — resume just reloads model params, then re-inits optimizer/
  scheduler fresh (e.g. `verl_grpo`: RL rollouts restart from the policy weights).
- **Weights + optimizer state** (and often scheduler, LR, RNG, global step) — resume
  must restore the full training state to continue an optimization trajectory
  correctly (e.g. `verl_sft`: the Adam optimizer momentum/second-moment + LR schedule
  are part of the checkpoint; losing them corrupts the estimates and spikes the loss).

Pin down concretely:
- **What files/keys constitute a valid checkpoint** — drives `validate_checkpoint`
  (e.g. presence of `optimizer.pt`/`optim_state` vs weights-only shards).
- **What the resume command must pass** — drives `checkpoint_resume_info`. If the
  framework can resume weights-only OR full-state, note the flag that selects it
  (e.g. a `resume_optimizer_only`-style override) so the proposer/executor can
  emit the right resume config.
- **Which state is LOCKED on resume** — e.g. optimizer type/betas, LR schedule shape,
  and model architecture usually must match the checkpoint; feed these into the
  locked-HP block in step 7.

Encode the answer in `checkpoints.py` (`validate_checkpoint`, `checkpoint_resume_info`).
A weights-only framework can skip optimizer-state validation; a full-state framework
must fail a checkpoint that is missing optimizer state rather than silently resuming
weights-only.

### 7. Ask a human for the hyperparameter tuning policy (do NOT guess)

Hyperparameter guidance is domain knowledge. Before writing `hyperparameters.py`
and the proposer fragment, ASK the user:

- **Which HP groups to focus on** (e.g. optimizer/LR, regularization, data mixture).
- **Any constraints** — especially simplex/sum constraints (e.g. a mixture weight
  dict that must sum to 1.0) or coupled bounds (min <= max).
- **Which HPs must stay LOCKED** and why (e.g. model architecture is fixed when
  resuming from a checkpoint; a metric-defining or budget HP must not move).

Encode locked HPs and hard constraints explicitly in the proposer prompt fragment
(e.g. verl's `entropy_coeff=0` rule, or a dedicated locked-HP block listing the
architecture/precision/budget HPs that must not move). List only the tunable HPs
in `HYPERPARAMETER_CATEGORIES`.

### 8. Register and configure

Append to `src/autogluon/llmzero/training_frameworks/__init__.py`:
```python
from .<name> import <Name>Framework  # noqa: E402

register_framework("<name>", <Name>Framework)
```

Tell the user to set `training_framework: <name>` in their config YAML. Do NOT
silently change the package default in `configs/default.yaml`.

### 9. Verify the registration imports

Run (with the project test interpreter):
```bash
python -c \
  "from autogluon.llmzero.training_frameworks import get_framework; \
   p = get_framework('<name>').stdout_parser(); print(type(p).__name__)"
```
Expect it to print `<Name>StdoutParser` with no import error.

### 10. Remind the user to add a test

Point at `tests/test_training_framework.py` for the pattern (parse a real log line
through the parser; assert `build_observers` returns the standard set). Suggest:
```bash
python -m pytest tests/test_training_framework.py -q
```

### 11. Report

Tell the user:
- The package path created and which optional modules were generated.
- The derived parser regexes and the validation pattern (with the verified parses).
- The `register_framework` line added.
- The config key to set (`training_framework: <name>`).
- That they should add a test and run pytest.
