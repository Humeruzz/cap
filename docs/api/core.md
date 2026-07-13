# Core API

The low-level building blocks: activation capture, statistics, neuron patching, HDF5
storage, and the evaluator interface. Most of these are re-exported from the top-level
`cap` package.

## Activation capture

::: cap.core.capture.ActivationCapture

## Statistics

::: cap.core.stats.ActivationStats

## Activation patching

::: cap.core.patch.ActivationPatcher

## HDF5 storage

::: cap.utils.h5_utils.H5Store

## Evaluators

Any object implementing `EvaluatorProtocol` can be passed to `cap patch --evaluator`.
`CSVEvaluator` is the batteries-included implementation for a labelled CSV.

::: cap.core.evaluation.EvaluatorProtocol

::: cap.core.evaluation.CSVEvaluator
