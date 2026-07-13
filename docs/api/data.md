# Data API

Loading CSVs into contrastive groups and wrapping text in prompt templates.

## Datasets

`LabeledDataset` splits one CSV by a binary label; `TwoGroupDataset` loads two CSVs as the
two groups. Both expose `as_groups()` returning `(group_a, group_b)`.

::: cap.data.loaders.LabeledDataset

::: cap.data.loaders.TwoGroupDataset

## Prompt templates

A `PromptTemplate` wraps a row's text into the question the model is actually asked. The
built-in templates are collected in `PROMPT_TYPES` (the keys are the `--type` values).

::: cap.data.prompt_templates.PromptTemplate

::: cap.data.prompt_templates.PROMPT_TYPES
