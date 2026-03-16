# AGENTS.md

## Scope
This folder contains dataloader, augmentation, preprocessing, and patch-sampling code for the 3D medical imaging pipeline.

## Goal
Refactor internals safely without breaking the training pipeline.

## Rules
- Preserve current external behavior.
- Do not change public APIs, function signatures, argument names, return types, or config field names unless explicitly requested.
- Do not break compatibility with the current dataloader, sampler, TorchIO queue, dataset creation flow, or training loop.
- Do not change patch shapes, batch structure, subject keys, or CT/mask alignment semantics unless explicitly requested.
- Do not introduce unnecessary dependencies.

## Allowed
- Clean up internal logic
- Add/remove private helpers
- Improve typing, comments, logging, and validation
- Reduce duplication
- Improve readability and maintainability
- Make safe internal optimizations

## Priorities
1. Correctness
2. Backward compatibility
3. Clarity
4. Robustness
5. Efficiency

## When editing
Prefer conservative changes. If a change might alter outputs or integration behavior, preserve the current behavior.