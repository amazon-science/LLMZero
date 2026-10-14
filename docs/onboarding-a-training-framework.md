# Onboarding a New Training Framework into LLMZero

This document has been reorganized into a folder with detailed references.

See: **[docs/onboarding-verl-sft/](onboarding-verl-sft/)** — a non-RFT / SFT (supervised fine-tuning) worked example (verl LoRA SFT with optimizer-state continuous training), with a three-file structure:
- `README.md` — Overview, gotchas, generic checklist
- `description.txt` — Reference `llmzero_input/description.txt` for verl_sft runs
- `cli-commands.md` — CLI invocations (standalone run, fresh MCTS run, continuous training with checkpoint chaining, monitoring)

The `/general-add-training-framework` skill scaffolds the framework plugin package;
the docs folder above is the narrative and operational guide.
