from pathlib import Path

from datasets import load_dataset

from ..utils.answer_utils import clean_number


class DatasetLoader:
    @staticmethod
    def load_gsm8k(*, n_samples, split="train", seed=42, n_shots=0):
        dataset = load_dataset("gsm8k", "main", split=split)

        few_shot_examples = []
        if n_shots > 0:
            few_shot_data = dataset.shuffle(seed=seed + 1000).select(range(n_shots))
            few_shot_examples = [
                {"question": ex["question"], "answer": ex["answer"]} for ex in few_shot_data
            ]

        dataset = dataset.shuffle(seed=seed).select(range(min(n_samples, len(dataset))))

        prompts = []
        answers = []
        for ex in dataset:
            parts = [
                f"Question: {fs['question']}\nAnswer: {fs['answer']}\n" for fs in few_shot_examples
            ]
            parts.append(f"Question: {ex['question']}\nLet's think step by step.\nAnswer:")
            prompts.append("\n".join(parts))
            answers.append(clean_number(ex["answer"].split("####")[-1].strip()))

        return prompts, answers

    @staticmethod
    def load_math(*, n_samples, split="test", seed=42, n_shots=0):
        dataset = load_dataset("hendrycks/competition_math", split=split)

        few_shot_examples = []
        if n_shots > 0:
            few_shot_data = dataset.shuffle(seed=seed + 1000).select(range(n_shots))
            few_shot_examples = [
                {"question": ex["problem"], "answer": ex["solution"]} for ex in few_shot_data
            ]

        dataset = dataset.shuffle(seed=seed).select(range(min(n_samples, len(dataset))))

        prompts = []
        answers = []
        for ex in dataset:
            parts = [
                f"Question: {fs['question']}\nAnswer: {fs['answer']}\n" for fs in few_shot_examples
            ]
            parts.append(f"Question: {ex['problem']}\nLet's think step by step.\nAnswer:")
            prompts.append("\n".join(parts))
            answers.append(ex["solution"])

        return prompts, answers

    @staticmethod
    def load_mmlu(*, n_samples, split="test", seed=42, n_shots=5, subjects=None):
        dataset = load_dataset("cais/mmlu", "all", split=split)

        if subjects:
            dataset = dataset.filter(lambda x: x["subject"] in subjects)

        few_shot_examples = []
        if n_shots > 0:
            few_shot_data = dataset.shuffle(seed=seed + 1000).select(range(n_shots))
            few_shot_examples = [
                {
                    "question": ex["question"],
                    "choices": ex["choices"],
                    "answer": chr(65 + ex["answer"]),
                }
                for ex in few_shot_data
            ]

        dataset = dataset.shuffle(seed=seed).select(range(min(n_samples, len(dataset))))

        def format_mcq(question, choices, answer=None):
            choice_text = "\n".join(f"{chr(65 + i)}. {c}" for i, c in enumerate(choices))
            line = f"{question}\n{choice_text}\nAnswer:"
            if answer:
                line += f" {answer}\n"
            return line

        prompts = []
        answers = []
        for ex in dataset:
            parts = [
                format_mcq(fs["question"], fs["choices"], fs["answer"]) for fs in few_shot_examples
            ]
            parts.append(format_mcq(ex["question"], ex["choices"]))
            prompts.append("\n".join(parts))
            answers.append(chr(65 + ex["answer"]))

        return prompts, answers

    @staticmethod
    def load_gpqa(*, n_samples, split="train", seed=42, n_shots=5):
        dataset = load_dataset("Idavidrein/gpqa", "gpqa_main", split=split)

        few_shot_examples = []
        if n_shots > 0:
            few_shot_data = dataset.shuffle(seed=seed + 1000).select(range(n_shots))
            few_shot_examples = [
                {
                    "question": ex["Question"],
                    "choices": [
                        ex["Correct Answer"],
                        ex["Incorrect Answer 1"],
                        ex["Incorrect Answer 2"],
                        ex["Incorrect Answer 3"],
                    ],
                    "answer": "A",
                }
                for ex in few_shot_data
            ]

        dataset = dataset.shuffle(seed=seed).select(range(min(n_samples, len(dataset))))

        def format_mcq(question, choices, answer=None):
            choice_text = "\n".join(f"{chr(65 + i)}. {c}" for i, c in enumerate(choices))
            line = f"{question}\n{choice_text}\nAnswer:"
            if answer:
                line += f" {answer}\n"
            return line

        prompts = []
        answers = []
        for ex in dataset:
            choices = [
                ex["Correct Answer"],
                ex["Incorrect Answer 1"],
                ex["Incorrect Answer 2"],
                ex["Incorrect Answer 3"],
            ]
            parts = [
                format_mcq(fs["question"], fs["choices"], fs["answer"]) for fs in few_shot_examples
            ]
            parts.append(format_mcq(ex["Question"], choices))
            prompts.append("\n".join(parts))
            answers.append("A")

        return prompts, answers

    @staticmethod
    def load_hellaswag(*, n_samples, split="validation", seed=42):
        # Returns (prompts, answers) like all other loaders — not a raw HF Dataset
        dataset = load_dataset("hellaswag", split=split)
        dataset = dataset.shuffle(seed=seed).select(range(min(n_samples, len(dataset))))

        prompts = []
        answers = []
        for ex in dataset:
            prompts.append(ex["ctx"])
            answers.append(int(ex["label"]))

        return prompts, answers

    @staticmethod
    def build_dataset_path(dataset_name):
        return Path(__file__).resolve().parents[1] / "data" / f"{dataset_name}.csv"

    @staticmethod
    def load_fact_check_csv(csv_path, n_samples=0, seed=42):
        """Load a fact-check CSV. Required columns: 'text', 'label' (0 = no error, 1 = factual problem)."""
        dataset = load_dataset("csv", data_files=csv_path, split="train")
        if n_samples > 0:
            dataset = dataset.shuffle(seed=seed).select(range(min(n_samples, len(dataset))))
        else:
            dataset = dataset.shuffle(seed=seed)

        prompts = []
        answers = []
        for row in dataset:
            prompts.append(
                f"Check next sentence on factuality: {row['text']}. "
                f'Answer only 0 or 1, where 0 means "no error" and 1 "There is factual problem". '
                f"Any other answer will be interpreted as wrong. Answer: "
            )
            answers.append(int(row["label"]))
        return prompts, answers

    @staticmethod
    def load_faithfulness_csv(csv_path, n_samples=0, seed=42):
        """Load a faithfulness CSV. Required columns: 'source', 'summary', 'label' (1 = faithful, 0 = not)."""
        dataset = load_dataset("csv", data_files=csv_path, split="train")
        if n_samples > 0:
            dataset = dataset.shuffle(seed=seed).select(range(min(n_samples, len(dataset))))
        else:
            dataset = dataset.shuffle(seed=seed)

        prompts = []
        answers = []
        for row in dataset:
            prompts.append(
                f"Your task is to check whether the summary is fully entailed by and grounded in the provided source texts. "
                f"Ignore any external knowledge.\n"
                f"Source texts: {row['source']}\n"
                f"Summary: {row['summary']}\n"
                f"Answer 1 if every statement in the summary can be directly verified from the source texts, "
                f"or 0 if any statement is unsupported or contradicted. Answer only 0 or 1. Answer: "
            )
            answers.append(int(row["label"]))
        return prompts, answers
