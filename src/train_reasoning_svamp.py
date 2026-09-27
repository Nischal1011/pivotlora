#!/usr/bin/env python3
"""
run_svamp.py

Controlled additional-dataset replication:

    Vanilla LoRA
        versus
    Response-position PivotLoRA

This script preserves the frozen GSM8K protocol for:

    - Joint prompt/completion tokenization
    - Character-offset completion-only masking
    - Response-predictor-position activation pooling
    - Randomized rank-8 PCA with no random rotation
    - Custom LoRA parameterization
    - FP64 output-preserving QR pivot after epoch 3
    - AdamW-state reset for both transformed factors
    - Optimizer, scheduler, and training hyperparameters
    - Greedy generation and numeric answer extraction

Only dataset-specific components are changed:

    - Dataset: ChilleD/SVAMP
    - Official 700-example train split
    - Official 300-example test split
    - 128 sampled training examples per seed
    - Prompt constructed from Body and Question
    - Completion constructed from Equation and Answer
    - Evaluation against the numeric Answer field
    - Per-example JSONL prediction files

Full study:
    python -u run_svamp.py

Only first paired seed:
    python -u run_svamp.py --seeds 1

Only PivotLoRA:
    python -u run_svamp.py --methods pivotlora --seeds 3

Smoke test:
    python -u run_svamp.py --smoke
"""

# =============================================================================
# Environment: set before importing torch
# =============================================================================

import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "true"


# =============================================================================
# Imports
# =============================================================================

import argparse
import csv
import gc
import json
import math
import re
import time
import traceback
import warnings
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

warnings.filterwarnings("ignore")


# =============================================================================
# Fixed protocol
# =============================================================================

MODEL_NAME = "meta-llama/Llama-3.1-8B"
DATASET_NAME = "ChilleD/SVAMP"

RANK = 8
ALPHA = 16

N_TRAIN = 128
EXPECTED_TRAIN_EXAMPLES = 700
EXPECTED_TEST_EXAMPLES = 300

EPOCHS = 15
PIVOT_EPOCH = 3

BATCH_SIZE = 4
GRAD_ACCUM = 2
EFFECTIVE_BATCH_SIZE = BATCH_SIZE * GRAD_ACCUM

LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.01

MAX_SEQ_LEN = 512
MAX_NEW_TOKENS = 512

SEED_LIST = [42, 43, 44]
METHODS = ["lora", "pivotlora"]

TARGET_MODULES = [
    "q_proj",
    "v_proj",
]

RESULTS_CSV = "results_svamp_response_pivot_matched.csv"
SMOKE_RESULTS_CSV = "results_svamp_response_pivot_matched_smoke.csv"

DIAGNOSTICS_DIR = "pivot_diagnostics_svamp_response_matched"
SMOKE_DIAGNOSTICS_DIR = "pivot_diagnostics_svamp_response_matched_smoke"

PREDICTIONS_DIR = "predictions_svamp_response_matched"
SMOKE_PREDICTIONS_DIR = "predictions_svamp_response_matched_smoke"

PROTOCOL_ID = "svamp_response_pca_qr_matched_v1"

CSV_FIELDS = [
    "protocol",
    "dataset",
    "model",
    "method",
    "seed",
    "n_train",
    "eval_examples",
    "correct",
    "svamp_numeric_accuracy",
    "parse_rate",
    "train_wall_time_seconds",
    "predictions_file",
    "diagnostics_file",
]

PROMPT_TEMPLATE = (
    "Question: {problem}\n"
    "Answer: "
)

COMPLETION_TEMPLATE = (
    "Equation: {equation}\n"
    "#### {answer}"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Pivot validation tolerance for the actual FP32-stored factors.
PRESERVATION_TOLERANCE = 1e-5

# Preserve the PCA initialization scale from the frozen GSM8K protocol.
PCA_INITIALIZATION_SCALE_MULTIPLIER = math.sqrt(5.0)

# Smoke mode must reach the epoch-3 pivot.
SMOKE_EPOCHS = 4
SMOKE_EVAL_EXAMPLES = 50


# =============================================================================
# Reproducibility and hardware reporting
# =============================================================================

def configure_reproducibility(seed):
    set_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    try:
        torch.use_deterministic_algorithms(
            True,
            warn_only=True,
        )
    except TypeError:
        pass


def report_hardware():
    print(f"Running on: {DEVICE}")
    print(f"PyTorch version: {torch.__version__}")

    if not torch.cuda.is_available():
        print("CUDA is not available.")
        return

    print(
        f"GPUs available: "
        f"{torch.cuda.device_count()}"
    )

    for gpu_index in range(
        torch.cuda.device_count()
    ):
        properties = (
            torch.cuda.get_device_properties(
                gpu_index
            )
        )

        print(
            f"  GPU {gpu_index}: "
            f"{torch.cuda.get_device_name(gpu_index)} "
            f"({properties.total_memory / 1024**3:.1f} GB)"
        )

    print(
        "CUDA capability:",
        torch.cuda.get_device_capability(0),
    )

    print(
        "PyTorch CUDA runtime:",
        torch.version.cuda,
    )

    print(
        "Compiled CUDA architectures:",
        torch.cuda.get_arch_list(),
    )


# =============================================================================
# SVAMP formatting
# =============================================================================

def clean_text(value):
    return " ".join(
        str(value).strip().split()
    )


def build_problem(example):
    body = clean_text(example["Body"])
    question = clean_text(example["Question"])

    if body and question:
        return f"{body} {question}"

    return body or question


def build_prompt(example):
    return PROMPT_TEMPLATE.format(
        problem=build_problem(example)
    )


def build_completion(example):
    return COMPLETION_TEMPLATE.format(
        equation=clean_text(
            example["Equation"]
        ),
        answer=clean_text(
            example["Answer"]
        ),
    )


# =============================================================================
# Answer extraction and normalization
# =============================================================================

# This is intentionally the same basic numeric extraction policy as the
# frozen GSM8K implementation.
NUMBER_PATTERN = re.compile(
    r"-?\d+(?:,\d+)*(?:\.\d+)?"
)


def extract_answer(text):
    """
    Extract the numeric answer after the final #### marker.

    If the marker is absent, use the final generated number. This mirrors
    the frozen GSM8K evaluation behavior.
    """

    if "####" in text:
        answer_portion = (
            text.split("####")[-1].strip()
        )

        numbers = NUMBER_PATTERN.findall(
            answer_portion
        )

        if numbers:
            return numbers[0].replace(",", "")

    numbers = NUMBER_PATTERN.findall(text)

    if numbers:
        return numbers[-1].replace(",", "")

    return ""


def normalize_numeric_answer(answer):
    answer = str(answer).strip().replace(
        ",",
        "",
    )

    if not answer:
        return ""

    try:
        numeric_value = float(answer)

        if not math.isfinite(numeric_value):
            return answer

        if numeric_value.is_integer():
            return str(int(numeric_value))

        return format(
            numeric_value,
            ".15g",
        )

    except ValueError:
        return answer


# =============================================================================
# Completion-only tokenization
# =============================================================================

def format_train(example, tokenizer):
    """
    Build a completion-only causal-LM example using character offsets.

    Prompt and completion are tokenized together. This avoids assuming that
    independently tokenizing the prompt and prompt-plus-completion produces
    an identical token prefix.

    Labels:
        prompt and special-prefix tokens: -100
        completion tokens: token ID
        appended EOS: token ID
        padding: -100

    A token is supervised when it contains at least one completion character.
    Therefore, a tokenizer token crossing the prompt/completion character
    boundary is correctly treated as supervised.
    """

    prompt = build_prompt(example)
    completion = build_completion(example)

    text_without_eos = prompt + completion
    response_start_character = len(prompt)

    encoded = tokenizer(
        text_without_eos,
        add_special_tokens=True,
        truncation=True,
        max_length=MAX_SEQ_LEN - 1,
        return_attention_mask=True,
        return_offsets_mapping=True,
    )

    input_ids = list(
        encoded["input_ids"]
    )
    attention_mask = list(
        encoded["attention_mask"]
    )
    offsets = list(
        encoded["offset_mapping"]
    )

    labels = []
    supervised_response_tokens = 0
    first_supervised_index = None

    for token_index, (
        token_id,
        offset,
    ) in enumerate(
        zip(
            input_ids,
            offsets,
        )
    ):
        start_character = int(offset[0])
        end_character = int(offset[1])

        # BOS and other special tokens generally have offset (0, 0).
        is_special_offset = (
            start_character == 0
            and end_character == 0
        )

        # Preserve the frozen GSM8K masking policy: supervise every token
        # containing at least one response character.
        is_response_token = (
            not is_special_offset
            and end_character
            > response_start_character
        )

        if is_response_token:
            labels.append(token_id)
            supervised_response_tokens += 1

            if first_supervised_index is None:
                first_supervised_index = (
                    token_index
                )
        else:
            labels.append(-100)

    if supervised_response_tokens == 0:
        raise RuntimeError(
            "No completion tokens survived tokenization. "
            "The prompt may have consumed MAX_SEQ_LEN."
        )

    if first_supervised_index is None:
        raise RuntimeError(
            "Completion-only boundary verification failed."
        )

    first_offset = offsets[
        first_supervised_index
    ]

    if int(first_offset[1]) <= response_start_character:
        raise RuntimeError(
            "The first supervised token does not contain "
            "a completion character."
        )

    # Every preceding non-special token must end at or before the response
    # boundary under the frozen character-offset masking policy.
    for index in range(first_supervised_index):
        start_character = int(
            offsets[index][0]
        )
        end_character = int(
            offsets[index][1]
        )

        is_special_offset = (
            start_character == 0
            and end_character == 0
        )

        if (
            not is_special_offset
            and end_character
            > response_start_character
        ):
            raise RuntimeError(
                "A response token was masked before the "
                "recorded completion boundary."
            )

        if labels[index] != -100:
            raise RuntimeError(
                "A prompt token was unexpectedly supervised."
            )

    # Explicitly append and supervise EOS.
    input_ids.append(
        tokenizer.eos_token_id
    )
    attention_mask.append(1)
    labels.append(
        tokenizer.eos_token_id
    )

    if len(input_ids) > MAX_SEQ_LEN:
        raise RuntimeError(
            "Internal tokenization error: sequence "
            "exceeded MAX_SEQ_LEN after EOS append."
        )

    padding_length = (
        MAX_SEQ_LEN - len(input_ids)
    )

    input_ids.extend(
        [tokenizer.pad_token_id]
        * padding_length
    )

    attention_mask.extend(
        [0] * padding_length
    )

    labels.extend(
        [-100] * padding_length
    )

    if not (
        len(input_ids)
        == len(attention_mask)
        == len(labels)
        == MAX_SEQ_LEN
    ):
        raise RuntimeError(
            "Tokenized example has inconsistent lengths."
        )

    return {
        "input_ids": torch.tensor(
            input_ids,
            dtype=torch.long,
        ),
        "attention_mask": torch.tensor(
            attention_mask,
            dtype=torch.long,
        ),
        "labels": torch.tensor(
            labels,
            dtype=torch.long,
        ),
    }


class SVAMPDataset(Dataset):
    def __init__(
        self,
        examples,
        tokenizer,
    ):
        self.data = [
            format_train(
                example=example,
                tokenizer=tokenizer,
            )
            for example in examples
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


# =============================================================================
# Data validation, loading, and paired sampling
# =============================================================================

def validate_svamp_split(
    examples,
    split_name,
):
    required_fields = {
        "ID",
        "Body",
        "Question",
        "Equation",
        "Answer",
    }

    seen_ids = set()

    for index, example in enumerate(
        examples
    ):
        missing_fields = (
            required_fields
            - set(example.keys())
        )

        if missing_fields:
            raise RuntimeError(
                f"SVAMP {split_name} row {index} "
                f"is missing fields: "
                f"{sorted(missing_fields)}"
            )

        example_id = str(
            example["ID"]
        )

        if example_id in seen_ids:
            raise RuntimeError(
                f"Duplicate SVAMP ID {example_id!r} "
                f"in {split_name}."
            )

        seen_ids.add(example_id)

        if not clean_text(example["Question"]):
            raise RuntimeError(
                f"SVAMP {split_name} row {index} "
                "has an empty Question."
            )

        if not clean_text(example["Equation"]):
            raise RuntimeError(
                f"SVAMP {split_name} row {index} "
                "has an empty Equation."
            )

        normalized_answer = (
            normalize_numeric_answer(
                example["Answer"]
            )
        )

        if not normalized_answer:
            raise RuntimeError(
                f"SVAMP {split_name} row {index} "
                "has an unparseable Answer."
            )

    return seen_ids


def sample_training_examples(
    training_examples,
    seed,
):
    """
    Preserve the frozen GSM8K sampling implementation.
    """

    rng = np.random.default_rng(seed)

    selected_indices = rng.choice(
        len(training_examples),
        size=N_TRAIN,
        replace=False,
    )

    return [
        training_examples[int(index)]
        for index in selected_indices
    ]


def load_svamp_splits():
    dataset = load_dataset(
        DATASET_NAME
    )

    if (
        "train" not in dataset
        or "test" not in dataset
    ):
        raise RuntimeError(
            f"{DATASET_NAME} must expose official "
            "train and test splits."
        )

    training_examples = list(
        dataset["train"]
    )

    test_examples = list(
        dataset["test"]
    )

    print(
        f"[DATA] train_pool="
        f"{len(training_examples)} "
        f"| test={len(test_examples)}"
    )

    if (
        len(training_examples)
        != EXPECTED_TRAIN_EXAMPLES
    ):
        raise RuntimeError(
            "Unexpected official SVAMP training size: "
            f"expected {EXPECTED_TRAIN_EXAMPLES}, "
            f"received {len(training_examples)}."
        )

    if (
        len(test_examples)
        != EXPECTED_TEST_EXAMPLES
    ):
        raise RuntimeError(
            "Unexpected official SVAMP test size: "
            f"expected {EXPECTED_TEST_EXAMPLES}, "
            f"received {len(test_examples)}."
        )

    train_ids = validate_svamp_split(
        examples=training_examples,
        split_name="train",
    )

    test_ids = validate_svamp_split(
        examples=test_examples,
        split_name="test",
    )

    overlap = train_ids & test_ids

    if overlap:
        raise RuntimeError(
            "Official SVAMP train and test splits "
            f"overlap on {len(overlap)} IDs."
        )

    test_answers = [
        normalize_numeric_answer(
            example["Answer"]
        )
        for example in test_examples
    ]

    return (
        training_examples,
        test_examples,
        test_answers,
    )


# =============================================================================
# LoRA layer
# =============================================================================

class LoRALayer(nn.Module):
    """
    Stored-coordinate convention:

        A: [d_in, rank]
        B: [rank, d_out]

        adapter(x) = (alpha / rank) * x @ A @ B

    This is the transpose of the common paper convention:

        A_paper: [rank, d_in]
        B_paper: [d_out, rank]
    """

    def __init__(
        self,
        original_layer,
        rank,
        alpha,
        init_A=None,
    ):
        super().__init__()

        self.original_layer = original_layer
        self.rank = rank
        self.scaling = alpha / rank

        input_dimension = (
            original_layer.in_features
        )

        output_dimension = (
            original_layer.out_features
        )

        device = (
            original_layer.weight.device
        )
        adapter_dtype = torch.float32

        if init_A is None:
            # Preserve the standard frozen-protocol LoRA initialization.
            standard_A = torch.empty(
                rank,
                input_dimension,
                device=device,
                dtype=adapter_dtype,
            )

            nn.init.kaiming_uniform_(
                standard_A,
                a=math.sqrt(5.0),
            )

            initialized_A = (
                standard_A
                .transpose(0, 1)
                .contiguous()
            )

        else:
            if tuple(init_A.shape) != (
                input_dimension,
                rank,
            ):
                raise ValueError(
                    "PCA initializer shape mismatch: "
                    f"expected {(input_dimension, rank)}, "
                    f"received {tuple(init_A.shape)}"
                )

            scale = (
                PCA_INITIALIZATION_SCALE_MULTIPLIER
                / math.sqrt(input_dimension)
            )

            initialized_A = (
                init_A.to(
                    device=device,
                    dtype=adapter_dtype,
                )
                * scale
            ).contiguous()

        self.lora_A = nn.Parameter(
            initialized_A
        )

        # Zero-B initialization preserves the pretrained output.
        self.lora_B = nn.Parameter(
            torch.zeros(
                rank,
                output_dimension,
                device=device,
                dtype=adapter_dtype,
            )
        )

        original_layer.weight.requires_grad = (
            False
        )

        if original_layer.bias is not None:
            original_layer.bias.requires_grad = (
                False
            )

    def forward(self, x):
        base_output = self.original_layer(x)

        adapter_input = x.to(
            dtype=self.lora_A.dtype
        )

        adapter_output = (
            adapter_input
            @ self.lora_A
            @ self.lora_B
        )

        adapter_output = adapter_output.to(
            dtype=base_output.dtype
        )

        return (
            base_output
            + self.scaling * adapter_output
        )


# =============================================================================
# Model utilities
# =============================================================================

def get_llama_layers(model):
    return model.model.layers


def get_model_input_device(model):
    return next(
        get_llama_layers(model)[0]
        .parameters()
    ).device


def load_base_model():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "A CUDA GPU is required for "
            "meta-llama/Llama-3.1-8B."
        )

    model = (
        AutoModelForCausalLM
        .from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
    )

    model = model.to(DEVICE)
    model.config.use_cache = False

    return model


# =============================================================================
# Response-predictor-position PCA
# =============================================================================

def compute_pca_basis(
    activations,
    rank,
    random_state,
):
    """
    activations: [number_of_examples, hidden_dimension]

    PCA mean-centers over examples internally. Only the ordered PCA basis
    is returned. No random rotation is applied.
    """

    number_of_samples, dimension = (
        activations.shape
    )

    number_of_components = min(
        rank,
        number_of_samples - 1,
        dimension,
    )

    if number_of_components < rank:
        raise RuntimeError(
            f"Need at least rank+1 pooled examples "
            f"for rank-{rank} PCA; received "
            f"{number_of_samples}."
        )

    pca = PCA(
        n_components=rank,
        svd_solver="randomized",
        random_state=random_state,
        n_oversamples=max(
            10,
            2 * rank,
        ),
        iterated_power=4,
    )

    pca.fit(activations)

    # sklearn components: [rank, dimension]
    # stored LoRA A:      [dimension, rank]
    basis = pca.components_.T.copy()

    # Preserve column order while removing numerical non-orthogonality.
    basis, _ = np.linalg.qr(
        basis,
        mode="reduced",
    )

    basis = basis[:, :rank]

    return torch.tensor(
        basis,
        dtype=torch.float32,
    )


def get_response_position_subspaces(
    model,
    train_dataset,
    rank,
    seed,
):
    """
    Collect one pooled response-predictor activation per example and layer.

    For causal-LM training, hidden state t predicts token t+1. Therefore,
    hidden state position t is selected when label t+1 is supervised.
    """

    print(
        "[SUBSPACE] Computing "
        "response-position PCA..."
    )

    model.eval()
    model.config.use_cache = False

    layers = get_llama_layers(model)
    pooled_by_layer = defaultdict(list)

    active_predictor_mask = {
        "value": None
    }

    def make_pre_hook(layer_index):
        def pre_hook(module, inputs):
            del module

            hidden_states = inputs[0]
            predictor_mask = (
                active_predictor_mask["value"]
            )

            if predictor_mask is None:
                raise RuntimeError(
                    "Predictor mask was not set before "
                    "the model forward pass."
                )

            sequence_length = (
                hidden_states.shape[1]
            )

            predictor_mask_local = (
                predictor_mask[
                    :,
                    :sequence_length,
                ]
                .to(
                    device=hidden_states.device,
                    dtype=torch.bool,
                )
            )

            valid_examples = (
                predictor_mask_local.any(
                    dim=1
                )
            )

            if not valid_examples.any():
                return

            selected_states = (
                hidden_states[
                    valid_examples
                ]
            )

            selected_mask = (
                predictor_mask_local[
                    valid_examples
                ]
            )

            weights = (
                selected_mask
                .unsqueeze(-1)
                .to(
                    dtype=selected_states.dtype
                )
            )

            denominator = (
                weights.sum(dim=1)
                .clamp_min(1.0)
            )

            pooled = (
                (
                    selected_states.detach()
                    * weights
                ).sum(dim=1)
                / denominator
            )

            pooled_by_layer[
                layer_index
            ].append(
                pooled.float().cpu()
            )

        return pre_hook

    handles = []

    # q_proj and v_proj receive the same adapter input.
    for layer_index, layer in enumerate(
        layers
    ):
        handle = (
            layer.self_attn.q_proj
            .register_forward_pre_hook(
                make_pre_hook(
                    layer_index
                )
            )
        )

        handles.append(handle)

    device = get_model_input_device(
        model
    )

    loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    try:
        with torch.no_grad():
            for batch in loader:
                labels = batch["labels"]

                attention_mask = (
                    batch["attention_mask"]
                    .bool()
                )

                predictor_mask = (
                    torch.zeros_like(
                        labels,
                        dtype=torch.bool,
                    )
                )

                # Hidden state t predicts supervised label t+1.
                predictor_mask[:, :-1] = (
                    labels[:, 1:] != -100
                )

                # Exclude padded predictor positions.
                predictor_mask &= (
                    attention_mask
                )

                if not predictor_mask.any():
                    raise RuntimeError(
                        "A PCA extraction batch contained "
                        "no supervised predictor positions."
                    )

                active_predictor_mask[
                    "value"
                ] = predictor_mask.to(
                    device
                )

                model(
                    input_ids=(
                        batch["input_ids"]
                        .to(device)
                    ),
                    attention_mask=(
                        batch["attention_mask"]
                        .to(device)
                    ),
                    use_cache=False,
                )

    finally:
        active_predictor_mask["value"] = (
            None
        )

        for handle in handles:
            handle.remove()

    subspaces = defaultdict(dict)

    if len(pooled_by_layer) != len(
        layers
    ):
        raise RuntimeError(
            "Activation extraction did not "
            "cover every layer: got "
            f"{len(pooled_by_layer)} of "
            f"{len(layers)}."
        )

    for layer_index in range(
        len(layers)
    ):
        pooled_batches = (
            pooled_by_layer[layer_index]
        )

        if not pooled_batches:
            raise RuntimeError(
                f"No pooled activations for "
                f"layer {layer_index}."
            )

        activation_matrix = (
            torch.cat(
                pooled_batches,
                dim=0,
            )
            .numpy()
        )

        if (
            activation_matrix.shape[0]
            != len(train_dataset)
        ):
            raise RuntimeError(
                f"Layer {layer_index} produced "
                f"{activation_matrix.shape[0]} "
                f"pooled examples; expected "
                f"{len(train_dataset)}."
            )

        layer_basis = compute_pca_basis(
            activations=activation_matrix,
            rank=rank,
            random_state=(
                seed * 1000
                + layer_index
            ),
        )

        subspaces[layer_index][
            "q_proj"
        ] = layer_basis

        subspaces[layer_index][
            "v_proj"
        ] = layer_basis.clone()

        if layer_index % 8 == 0:
            print(
                f"  [SUBSPACE] "
                f"layer={layer_index:>2} "
                f"| examples="
                f"{activation_matrix.shape[0]} "
                f"| d="
                f"{activation_matrix.shape[1]}"
            )

    del pooled_by_layer
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("[SUBSPACE] Done.")

    return dict(subspaces)


# =============================================================================
# LoRA injection
# =============================================================================

def inject_lora(
    model,
    rank,
    subspaces=None,
):
    adapter_count = 0

    for layer_index, layer in enumerate(
        get_llama_layers(model)
    ):
        for target_name in TARGET_MODULES:
            original_layer = getattr(
                layer.self_attn,
                target_name,
            )

            initialization = None

            if subspaces is not None:
                initialization = subspaces[
                    layer_index
                ][target_name]

            adapter = LoRALayer(
                original_layer=original_layer,
                rank=rank,
                alpha=ALPHA,
                init_A=initialization,
            )

            setattr(
                layer.self_attn,
                target_name,
                adapter,
            )

            adapter_count += 1

    # Freeze everything, then unfreeze only adapter factors.
    for parameter in model.parameters():
        parameter.requires_grad = False

    for module in model.modules():
        if isinstance(module, LoRALayer):
            module.lora_A.requires_grad = True
            module.lora_B.requires_grad = True

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    expected_adapters = (
        len(get_llama_layers(model))
        * len(TARGET_MODULES)
    )

    if adapter_count != expected_adapters:
        raise RuntimeError(
            f"Injected {adapter_count} adapters; "
            f"expected {expected_adapters}."
        )

    if adapter_count != 64:
        raise RuntimeError(
            f"Expected 64 q_proj/v_proj adapters "
            f"for Llama-3.1-8B; received "
            f"{adapter_count}."
        )

    print(
        f"[INJECT] adapters={adapter_count} "
        f"| trainable={trainable_parameters:,}"
    )

    return model


# =============================================================================
# Matched optimizer
# =============================================================================

def build_matched_optimizer(model):
    """
    Both methods use this exact optimizer.

    There are no method-specific parameter groups, eigenvalue rates,
    differential rates, or post-pivot learning-rate overrides.
    """

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    if not trainable_parameters:
        raise RuntimeError(
            "No trainable parameters found."
        )

    return torch.optim.AdamW(
        trainable_parameters,
        lr=LEARNING_RATE,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=WEIGHT_DECAY,
    )


# =============================================================================
# Output-preserving QR pivot
# =============================================================================

class PivotCallback(TrainerCallback):
    """
    Stored coordinates:

        A_old: [d, r]
        B_old: [r, m]

    QR:

        A_old = Q R

    Pivot:

        A_new = Q
        B_new = R B_old

    Preservation:

        A_new B_new
        = Q R B_old
        = A_old B_old

    The transformation is computed in FP64 and stored in FP32.
    AdamW state is reset for both transformed factors. The ordinary
    Trainer linear learning-rate scheduler continues unchanged.
    """

    def __init__(
        self,
        pivot_epoch,
        diagnostics_path,
    ):
        self.pivot_epoch = pivot_epoch
        self.diagnostics_path = (
            diagnostics_path
        )
        self.pivot_done = False
        self.pivot_global_step = None
        self.summary = None

    @staticmethod
    def _frobenius_product_norm_squared(
        A,
        B,
    ):
        """
        Compute ||AB||_F^2 using rank-by-rank Gram matrices.

        A: [d, r]
        B: [r, m]
        """

        gram_A = (
            A.transpose(0, 1) @ A
        )
        gram_B = (
            B @ B.transpose(0, 1)
        )

        value = (
            gram_A
            * gram_B.transpose(0, 1)
        ).sum()

        return value.clamp_min(0.0)

    @staticmethod
    def _frobenius_product_inner(
        A_left,
        B_left,
        A_right,
        B_right,
    ):
        """
        Compute <A_left B_left, A_right B_right>_F.
        """

        cross_A = (
            A_left.transpose(0, 1)
            @ A_right
        )

        cross_B = (
            B_right
            @ B_left.transpose(0, 1)
        )

        return (
            cross_A
            * cross_B.transpose(0, 1)
        ).sum()

    @classmethod
    def _relative_preservation_error(
        cls,
        A_old,
        B_old,
        A_new,
        B_new,
    ):
        old_squared = (
            cls._frobenius_product_norm_squared(
                A_old,
                B_old,
            )
        )

        new_squared = (
            cls._frobenius_product_norm_squared(
                A_new,
                B_new,
            )
        )

        cross = (
            cls._frobenius_product_inner(
                A_old,
                B_old,
                A_new,
                B_new,
            )
        )

        difference_squared = (
            old_squared
            + new_squared
            - 2.0 * cross
        ).clamp_min(0.0)

        denominator = (
            old_squared.clamp_min(
                1e-30
            )
        )

        return torch.sqrt(
            difference_squared
            / denominator
        ).item()

    @staticmethod
    def _orthogonality_error(A):
        identity = torch.eye(
            A.shape[1],
            device=A.device,
            dtype=A.dtype,
        )

        return (
            A.transpose(0, 1)
            @ A
            - identity
        ).norm().item()

    @staticmethod
    def _reset_optimizer_state(
        optimizer,
        parameter,
    ):
        if optimizer is None:
            raise RuntimeError(
                "Pivot callback did not "
                "receive the optimizer."
            )

        optimizer.state.pop(
            parameter,
            None,
        )

    def on_epoch_end(
        self,
        args,
        state,
        control,
        model=None,
        optimizer=None,
        **kwargs,
    ):
        del args
        del kwargs

        if self.pivot_done:
            return control

        if state.epoch is None:
            return control

        if (
            state.epoch + 1e-8
            < self.pivot_epoch
        ):
            return control

        if model is None:
            raise RuntimeError(
                "Pivot callback did not "
                "receive model."
            )

        if optimizer is None:
            raise RuntimeError(
                "Pivot callback did not "
                "receive optimizer."
            )

        print(
            f"[PivotLoRA] Executing "
            f"output-preserving QR pivot "
            f"at epoch={state.epoch:.6f}, "
            f"global_step={state.global_step}"
        )

        diagnostics = []
        pivoted_modules = 0

        for module_name, module in (
            model.named_modules()
        ):
            if not isinstance(
                module,
                LoRALayer,
            ):
                continue

            with torch.no_grad():
                A_old_64 = (
                    module.lora_A
                    .detach()
                    .to(dtype=torch.float64)
                    .clone()
                )

                B_old_64 = (
                    module.lora_B
                    .detach()
                    .to(dtype=torch.float64)
                    .clone()
                )

                orthogonality_before = (
                    self._orthogonality_error(
                        A_old_64
                    )
                )

                # A_old = Q R.
                Q_64, R_64 = (
                    torch.linalg.qr(
                        A_old_64,
                        mode="reduced",
                    )
                )

                A_new_64 = Q_64
                B_new_64 = (
                    R_64 @ B_old_64
                )

                A_new_stored = (
                    A_new_64.to(
                        device=(
                            module.lora_A.device
                        ),
                        dtype=(
                            module.lora_A.dtype
                        ),
                    )
                )

                B_new_stored = (
                    B_new_64.to(
                        device=(
                            module.lora_B.device
                        ),
                        dtype=(
                            module.lora_B.dtype
                        ),
                    )
                )

                # Validate actual stored factors.
                A_check_64 = (
                    A_new_stored.to(
                        dtype=torch.float64
                    )
                )

                B_check_64 = (
                    B_new_stored.to(
                        dtype=torch.float64
                    )
                )

                preservation_error = (
                    self._relative_preservation_error(
                        A_old=A_old_64,
                        B_old=B_old_64,
                        A_new=A_check_64,
                        B_new=B_check_64,
                    )
                )

                orthogonality_after = (
                    self._orthogonality_error(
                        A_check_64
                    )
                )

                if not math.isfinite(
                    preservation_error
                ):
                    raise RuntimeError(
                        "Non-finite pivot "
                        "preservation error in "
                        f"{module_name}."
                    )

                if not math.isfinite(
                    orthogonality_after
                ):
                    raise RuntimeError(
                        "Non-finite post-pivot "
                        "orthogonality error in "
                        f"{module_name}."
                    )

                if (
                    preservation_error
                    > PRESERVATION_TOLERANCE
                ):
                    raise RuntimeError(
                        "Pivot preservation error "
                        "exceeded tolerance in "
                        f"{module_name}: "
                        f"{preservation_error:.3e} > "
                        f"{PRESERVATION_TOLERANCE:.3e}"
                    )

                module.lora_A.copy_(
                    A_new_stored
                )

                module.lora_B.copy_(
                    B_new_stored
                )

                # Preserve the frozen GSM8K optimizer-state policy.
                self._reset_optimizer_state(
                    optimizer,
                    module.lora_A,
                )

                self._reset_optimizer_state(
                    optimizer,
                    module.lora_B,
                )

                diagnostics.append(
                    {
                        "module": module_name,
                        "preservation_error": (
                            preservation_error
                        ),
                        "orthogonality_before": (
                            orthogonality_before
                        ),
                        "orthogonality_after": (
                            orthogonality_after
                        ),
                        "A_old_norm": (
                            A_old_64.norm().item()
                        ),
                        "A_new_norm": (
                            A_check_64.norm().item()
                        ),
                        "B_old_norm": (
                            B_old_64.norm().item()
                        ),
                        "B_new_norm": (
                            B_check_64.norm().item()
                        ),
                    }
                )

                pivoted_modules += 1

        if not diagnostics:
            raise RuntimeError(
                "Pivot callback found no "
                "LoRA modules."
            )

        if pivoted_modules != 64:
            raise RuntimeError(
                f"Expected 64 pivoted adapters, "
                f"received {pivoted_modules}."
            )

        maximum_preservation_error = max(
            item["preservation_error"]
            for item in diagnostics
        )

        maximum_orthogonality_after = max(
            item["orthogonality_after"]
            for item in diagnostics
        )

        minimum_orthogonality_before = min(
            item["orthogonality_before"]
            for item in diagnostics
        )

        diagnostics_summary = {
            "protocol": PROTOCOL_ID,
            "dataset": DATASET_NAME,
            "pivot_epoch_requested": (
                self.pivot_epoch
            ),
            "epoch_executed": float(
                state.epoch
            ),
            "global_step": int(
                state.global_step
            ),
            "pivoted_modules": (
                pivoted_modules
            ),
            "optimizer_state_reset": [
                "lora_A",
                "lora_B",
            ],
            "learning_rate_override": False,
            "preservation_tolerance": (
                PRESERVATION_TOLERANCE
            ),
            "maximum_preservation_error": (
                maximum_preservation_error
            ),
            "minimum_orthogonality_before": (
                minimum_orthogonality_before
            ),
            "maximum_orthogonality_after": (
                maximum_orthogonality_after
            ),
            "all_updates_preserved": (
                maximum_preservation_error
                <= PRESERVATION_TOLERANCE
            ),
            "all_bases_orthonormal": (
                maximum_orthogonality_after
                <= PRESERVATION_TOLERANCE
            ),
            "modules": diagnostics,
        }

        os.makedirs(
            os.path.dirname(
                self.diagnostics_path
            )
            or ".",
            exist_ok=True,
        )

        with open(
            self.diagnostics_path,
            "w",
            encoding="utf-8",
        ) as diagnostics_file:
            json.dump(
                diagnostics_summary,
                diagnostics_file,
                indent=2,
            )

        self.pivot_done = True
        self.pivot_global_step = int(
            state.global_step
        )
        self.summary = diagnostics_summary

        print(
            f"[PivotLoRA] Pivoted "
            f"{pivoted_modules} adapters"
        )

        print(
            "[PivotLoRA] Maximum "
            "preservation error="
            f"{maximum_preservation_error:.3e}"
        )

        print(
            "[PivotLoRA] Maximum post-pivot "
            "orthogonality error="
            f"{maximum_orthogonality_after:.3e}"
        )

        print(
            "[PivotLoRA] Optimizer state reset "
            "for A and B; Trainer LR schedule "
            "unchanged"
        )

        return control


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_svamp(
    model,
    tokenizer,
    test_examples,
    test_answers,
    predictions_path,
    max_eval=None,
):
    model.eval()

    try:
        model.gradient_checkpointing_disable()
    except Exception:
        pass

    model.config.use_cache = True

    device = get_model_input_device(
        model
    )

    if max_eval is not None:
        test_examples = (
            test_examples[:max_eval]
        )
        test_answers = (
            test_answers[:max_eval]
        )

    evaluation_count = len(
        test_examples
    )

    if evaluation_count == 0:
        raise RuntimeError(
            "Evaluation set is empty."
        )

    if (
        len(test_answers)
        != evaluation_count
    ):
        raise RuntimeError(
            "Test examples and answer counts "
            "do not match."
        )

    number_correct = 0
    valid_predictions = 0
    prediction_records = []

    for example_index, (
        example,
        ground_truth,
    ) in enumerate(
        zip(
            test_examples,
            test_answers,
        )
    ):
        prompt = build_prompt(
            example
        )

        encoded_prompt = tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=True,
        )

        input_ids = (
            encoded_prompt["input_ids"]
            .to(device)
        )

        attention_mask = (
            encoded_prompt[
                "attention_mask"
            ]
            .to(device)
        )

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=(
                    MAX_NEW_TOKENS
                ),
                do_sample=False,
                pad_token_id=(
                    tokenizer.eos_token_id
                ),
                eos_token_id=(
                    tokenizer.eos_token_id
                ),
                use_cache=True,
            )

        generated_ids = output_ids[
            0,
            input_ids.shape[1]:,
        ]

        generated_text = (
            tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
            )
        )

        raw_prediction = extract_answer(
            generated_text
        )

        predicted_answer = (
            normalize_numeric_answer(
                raw_prediction
            )
        )

        parsed = bool(
            predicted_answer
        )

        if parsed:
            valid_predictions += 1

        correct = (
            predicted_answer
            == ground_truth
        )

        if correct:
            number_correct += 1

        prediction_records.append(
            {
                "index": example_index,
                "id": str(
                    example["ID"]
                ),
                "body": clean_text(
                    example["Body"]
                ),
                "question": clean_text(
                    example["Question"]
                ),
                "equation": clean_text(
                    example["Equation"]
                ),
                "reference_answer_raw": (
                    clean_text(
                        example["Answer"]
                    )
                ),
                "reference_answer_normalized": (
                    ground_truth
                ),
                "prompt": prompt,
                "generated_completion": (
                    generated_text
                ),
                "extracted_prediction": (
                    raw_prediction
                ),
                "normalized_prediction": (
                    predicted_answer
                ),
                "parsed": parsed,
                "correct": correct,
            }
        )

        processed = example_index + 1

        if processed % 100 == 0:
            print(
                f"  [EVAL] {processed}/"
                f"{evaluation_count} "
                f"running_acc="
                f"{number_correct / processed:.4f}"
            )

    accuracy = (
        number_correct
        / evaluation_count
    )

    parse_rate = (
        valid_predictions
        / evaluation_count
    )

    os.makedirs(
        os.path.dirname(
            predictions_path
        )
        or ".",
        exist_ok=True,
    )

    with open(
        predictions_path,
        "w",
        encoding="utf-8",
    ) as predictions_file:
        for record in prediction_records:
            predictions_file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(
        f"  [EVAL] SVAMP numeric accuracy: "
        f"{accuracy:.6f} "
        f"({number_correct}/"
        f"{evaluation_count})"
    )

    print(
        f"  [EVAL] Parsed predictions: "
        f"{valid_predictions}/"
        f"{evaluation_count} "
        f"({parse_rate:.6f})"
    )

    print(
        f"  [EVAL] Predictions: "
        f"{predictions_path}"
    )

    return (
        accuracy,
        number_correct,
        evaluation_count,
        parse_rate,
    )


# =============================================================================
# Single experiment
# =============================================================================

def run_experiment(
    seed,
    method,
    training_pool,
    test_examples,
    test_answers,
    tokenizer,
    smoke=False,
):
    print("\n" + "=" * 80)

    print(
        f"SVAMP | protocol={PROTOCOL_ID} "
        f"| method={method} | seed={seed}"
    )

    print(
        f"N={N_TRAIN} | d/N="
        f"{4096 / N_TRAIN:.1f} "
        f"| rank={RANK} | alpha={ALPHA}"
    )

    print("=" * 80)

    configure_reproducibility(
        seed
    )

    sampled_training_examples = (
        sample_training_examples(
            training_examples=(
                training_pool
            ),
            seed=seed,
        )
    )

    selected_ids = sorted(
        str(example["ID"])
        for example in (
            sampled_training_examples
        )
    )

    print(
        f"[DATA] seed={seed} "
        f"selected_train={len(selected_ids)}"
    )

    print(
        "[DATA] selected_train_IDs="
        + "|".join(selected_ids)
    )

    train_dataset = SVAMPDataset(
        examples=(
            sampled_training_examples
        ),
        tokenizer=tokenizer,
    )

    model = load_base_model()

    model.config.pad_token_id = (
        tokenizer.pad_token_id
    )

    callbacks = []
    diagnostics_path = ""

    diagnostics_directory = (
        SMOKE_DIAGNOSTICS_DIR
        if smoke
        else DIAGNOSTICS_DIR
    )

    predictions_directory = (
        SMOKE_PREDICTIONS_DIR
        if smoke
        else PREDICTIONS_DIR
    )

    predictions_path = os.path.join(
        predictions_directory,
        (
            f"{PROTOCOL_ID}_"
            f"{method}_seed_{seed}"
            f"{'_smoke' if smoke else ''}"
            ".jsonl"
        ),
    )

    if method == "lora":
        inject_lora(
            model=model,
            rank=RANK,
            subspaces=None,
        )

    elif method == "pivotlora":
        subspaces = (
            get_response_position_subspaces(
                model=model,
                train_dataset=train_dataset,
                rank=RANK,
                seed=seed,
            )
        )

        inject_lora(
            model=model,
            rank=RANK,
            subspaces=subspaces,
        )

        del subspaces
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        os.makedirs(
            diagnostics_directory,
            exist_ok=True,
        )

        diagnostics_path = os.path.join(
            diagnostics_directory,
            (
                f"{PROTOCOL_ID}_"
                f"seed_{seed}"
                f"{'_smoke' if smoke else ''}"
                ".json"
            ),
        )

        callbacks.append(
            PivotCallback(
                pivot_epoch=PIVOT_EPOCH,
                diagnostics_path=(
                    diagnostics_path
                ),
            )
        )

    else:
        raise ValueError(
            f"Unknown method: {method}"
        )

    # Enable checkpointing only after adapter injection and freezing.
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    optimizer = build_matched_optimizer(
        model
    )

    epochs = (
        SMOKE_EPOCHS
        if smoke
        else EPOCHS
    )

    output_directory = os.path.join(
        "tmp",
        (
            f"{PROTOCOL_ID}_"
            f"{method}_{seed}"
            f"{'_smoke' if smoke else ''}"
        ),
    )

    training_argument_values = {
        "output_dir": output_directory,
        "learning_rate": LEARNING_RATE,
        "lr_scheduler_type": "linear",
        "warmup_steps": 0,
        "num_train_epochs": epochs,
        "per_device_train_batch_size": (
            BATCH_SIZE
        ),
        "gradient_accumulation_steps": (
            GRAD_ACCUM
        ),
        "save_strategy": "no",
        "logging_steps": 10,
        "report_to": "none",
        "seed": seed,
        "data_seed": seed,
        "dataloader_num_workers": 0,
        "fp16": False,
        "bf16": True,
        "tf32": True,
        "disable_tqdm": False,
        "remove_unused_columns": False,
        "gradient_checkpointing": True,
    }

    try:
        training_arguments = (
            TrainingArguments(
                eval_strategy="no",
                **training_argument_values,
            )
        )

    except TypeError:
        training_arguments = (
            TrainingArguments(
                evaluation_strategy="no",
                **training_argument_values,
            )
        )

    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=train_dataset,
        callbacks=callbacks,
        optimizers=(
            optimizer,
            None,
        ),
    )

    train_start_time = (
        time.perf_counter()
    )

    training_result = trainer.train()

    train_wall_time = (
        time.perf_counter()
        - train_start_time
    )

    print(
        f"  [TRAIN] Wall time: "
        f"{train_wall_time:.2f}s"
    )

    print(
        f"  [TRAIN] Final metrics: "
        f"{training_result.metrics}"
    )

    if method == "pivotlora":
        pivot_callback = callbacks[0]

        if not pivot_callback.pivot_done:
            raise RuntimeError(
                "Training finished without "
                "executing the PivotLoRA "
                "QR pivot."
            )

        if (
            pivot_callback.pivot_global_step
            is None
        ):
            raise RuntimeError(
                "Pivot callback did not "
                "record its global step."
            )

        if pivot_callback.summary is None:
            raise RuntimeError(
                "Pivot callback did not "
                "create diagnostics."
            )

        if (
            pivot_callback.summary[
                "pivoted_modules"
            ]
            != 64
        ):
            raise RuntimeError(
                "Pivot callback did not "
                "process 64 adapters."
            )

        if not pivot_callback.summary[
            "all_updates_preserved"
        ]:
            raise RuntimeError(
                "At least one adapter update "
                "was not preserved."
            )

        if not pivot_callback.summary[
            "all_bases_orthonormal"
        ]:
            raise RuntimeError(
                "At least one post-pivot "
                "basis was not orthonormal."
            )

    maximum_evaluation_examples = (
        SMOKE_EVAL_EXAMPLES
        if smoke
        else None
    )

    (
        accuracy,
        number_correct,
        evaluation_count,
        parse_rate,
    ) = evaluate_svamp(
        model=model,
        tokenizer=tokenizer,
        test_examples=test_examples,
        test_answers=test_answers,
        predictions_path=(
            predictions_path
        ),
        max_eval=(
            maximum_evaluation_examples
        ),
    )

    del trainer
    del optimizer
    del model
    del train_dataset

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "accuracy": accuracy,
        "correct": number_correct,
        "eval_examples": (
            evaluation_count
        ),
        "parse_rate": parse_rate,
        "train_wall_time": (
            train_wall_time
        ),
        "predictions_file": (
            predictions_path
        ),
        "diagnostics_file": (
            diagnostics_path
        ),
    }


# =============================================================================
# Results persistence
# =============================================================================

def load_completed_results(
    results_csv,
):
    completed = {}

    if not os.path.exists(
        results_csv
    ):
        return completed

    with open(
        results_csv,
        newline="",
        encoding="utf-8",
    ) as results_file:
        reader = csv.DictReader(
            results_file
        )

        for row in reader:
            try:
                if (
                    row["protocol"]
                    != PROTOCOL_ID
                ):
                    continue

                if (
                    int(row["n_train"])
                    != N_TRAIN
                ):
                    continue

                key = (
                    row["method"],
                    int(row["seed"]),
                )

                completed[key] = {
                    "accuracy": float(
                        row[
                            "svamp_numeric_accuracy"
                        ]
                    ),
                    "correct": int(
                        row["correct"]
                    ),
                    "eval_examples": int(
                        row["eval_examples"]
                    ),
                    "parse_rate": float(
                        row["parse_rate"]
                    ),
                    "train_wall_time": float(
                        row[
                            "train_wall_time_seconds"
                        ]
                    ),
                    "predictions_file": (
                        row.get(
                            "predictions_file",
                            "",
                        )
                    ),
                    "diagnostics_file": (
                        row.get(
                            "diagnostics_file",
                            "",
                        )
                    ),
                }

            except (
                KeyError,
                TypeError,
                ValueError,
            ):
                continue

    return completed


def write_protocol_manifest(
    smoke=False,
):
    manifest_path = (
        "svamp_response_pivot_matched_"
        f"{'smoke_' if smoke else ''}"
        "manifest.json"
    )

    manifest = {
        "protocol": PROTOCOL_ID,
        "source_protocol": (
            "gsm8k_response_pca_qr_matched_v1"
        ),
        "model": MODEL_NAME,
        "dataset": DATASET_NAME,
        "task": (
            "SVAMP completion-only SFT"
        ),
        "official_train_examples": (
            EXPECTED_TRAIN_EXAMPLES
        ),
        "official_test_examples": (
            EXPECTED_TEST_EXAMPLES
        ),
        "n_train": N_TRAIN,
        "full_eval_examples": (
            EXPECTED_TEST_EXAMPLES
        ),
        "rank": RANK,
        "alpha": ALPHA,
        "targets": TARGET_MODULES,
        "epochs": (
            SMOKE_EPOCHS
            if smoke
            else EPOCHS
        ),
        "pivot_epoch": PIVOT_EPOCH,
        "batch_size": BATCH_SIZE,
        "gradient_accumulation": (
            GRAD_ACCUM
        ),
        "effective_batch_size": (
            EFFECTIVE_BATCH_SIZE
        ),
        "base_learning_rate": (
            LEARNING_RATE
        ),
        "weight_decay": WEIGHT_DECAY,
        "adam_betas": [
            0.9,
            0.999,
        ],
        "adam_epsilon": 1e-8,
        "scheduler": "linear",
        "warmup_steps": 0,
        "training_format": (
            "Question: {Body} {Question}\\n"
            "Answer: Equation: {Equation}\\n"
            "#### {Answer}"
        ),
        "completion_masking": (
            "joint tokenization with "
            "character offsets"
        ),
        "pca_source": (
            "mean-pooled hidden states at "
            "positions predicting supervised "
            "response tokens"
        ),
        "pca_solver": "randomized",
        "pca_random_rotation": False,
        "pca_initialization_scale_multiplier": (
            PCA_INITIALIZATION_SCALE_MULTIPLIER
        ),
        "eigenvalue_learning_rates": False,
        "differential_B_learning_rate": False,
        "pivot": (
            "FP64 output-preserving reduced QR"
        ),
        "pivot_optimizer_reset": [
            "A",
            "B",
        ],
        "post_pivot_lr_override": False,
        "seeds": SEED_LIST,
        "max_sequence_length": (
            MAX_SEQ_LEN
        ),
        "max_new_tokens": (
            MAX_NEW_TOKENS
        ),
        "evaluation": (
            "greedy generation; numeric Answer "
            "field accuracy"
        ),
        "per_example_predictions": True,
    }

    with open(
        manifest_path,
        "w",
        encoding="utf-8",
    ) as manifest_file:
        json.dump(
            manifest,
            manifest_file,
            indent=2,
        )

    print(
        f"[PROTOCOL] Manifest: "
        f"{manifest_path}"
    )


# =============================================================================
# Study runner
# =============================================================================

def run_study(
    methods=None,
    seeds=None,
    smoke=False,
):
    methods = methods or METHODS
    seeds = seeds or SEED_LIST

    results_csv = (
        SMOKE_RESULTS_CSV
        if smoke
        else RESULTS_CSV
    )

    write_protocol_manifest(
        smoke=smoke
    )

    print("Loading tokenizer...")

    tokenizer = (
        AutoTokenizer.from_pretrained(
            MODEL_NAME,
            use_fast=True,
        )
    )

    if not tokenizer.is_fast:
        raise RuntimeError(
            "A fast tokenizer is required "
            "for offset-based response masking."
        )

    if tokenizer.eos_token_id is None:
        raise RuntimeError(
            "Tokenizer does not define EOS."
        )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = (
            tokenizer.eos_token
        )

    tokenizer.padding_side = "right"

    (
        training_pool,
        test_examples,
        test_answers,
    ) = load_svamp_splits()

    print(
        f"[RESULTS] File: "
        f"{results_csv}"
    )

    completed = (
        load_completed_results(
            results_csv
        )
    )

    if completed:
        print(
            f"[RESUME] Found "
            f"{len(completed)} valid "
            "completed rows"
        )

    needs_header = (
        not os.path.exists(results_csv)
        or os.path.getsize(
            results_csv
        ) == 0
    )

    results_handle = open(
        results_csv,
        "a",
        newline="",
        encoding="utf-8",
    )

    writer = csv.DictWriter(
        results_handle,
        fieldnames=CSV_FIELDS,
    )

    if needs_header:
        writer.writeheader()
        results_handle.flush()

    all_scores = defaultdict(dict)

    try:
        # Preserve the frozen GSM8K study-loop ordering.
        for method in methods:
            for seed in seeds:
                key = (
                    method,
                    seed,
                )

                if key in completed:
                    existing = completed[key]

                    all_scores[method][seed] = (
                        existing["accuracy"]
                    )

                    print(
                        f"[SKIP] {method}-s{seed}: "
                        f"{existing['accuracy']:.6f} "
                        f"({existing['correct']}/"
                        f"{existing['eval_examples']})"
                    )

                    continue

                try:
                    result = run_experiment(
                        seed=seed,
                        method=method,
                        training_pool=(
                            training_pool
                        ),
                        test_examples=(
                            test_examples
                        ),
                        test_answers=(
                            test_answers
                        ),
                        tokenizer=tokenizer,
                        smoke=smoke,
                    )

                    all_scores[method][seed] = (
                        result["accuracy"]
                    )

                    writer.writerow(
                        {
                            "protocol": (
                                PROTOCOL_ID
                            ),
                            "dataset": (
                                DATASET_NAME
                            ),
                            "model": (
                                MODEL_NAME
                            ),
                            "method": method,
                            "seed": seed,
                            "n_train": N_TRAIN,
                            "eval_examples": (
                                result[
                                    "eval_examples"
                                ]
                            ),
                            "correct": (
                                result[
                                    "correct"
                                ]
                            ),
                            "svamp_numeric_accuracy": (
                                f"{result['accuracy']:.9f}"
                            ),
                            "parse_rate": (
                                f"{result['parse_rate']:.9f}"
                            ),
                            "train_wall_time_seconds": (
                                f"{result['train_wall_time']:.2f}"
                            ),
                            "predictions_file": (
                                result[
                                    "predictions_file"
                                ]
                            ),
                            "diagnostics_file": (
                                result[
                                    "diagnostics_file"
                                ]
                            ),
                        }
                    )

                    results_handle.flush()

                except Exception as exception:
                    print(
                        f"[CRASH] "
                        f"{method}-s{seed}: "
                        f"{exception}"
                    )

                    traceback.print_exc()

                    gc.collect()

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

    finally:
        results_handle.close()

    print("\n" + "=" * 80)

    print(
        "SVAMP ADDITIONAL-DATASET RESULT: "
        "MATCHED-OPTIMIZER RESPONSE-POSITION "
        "PIVOTLORA"
    )

    print(
        f"Protocol: {PROTOCOL_ID}"
    )

    print(
        f"Model: {MODEL_NAME}"
    )

    print(
        f"Dataset: {DATASET_NAME}"
    )

    print(
        f"N={N_TRAIN} | d/N="
        f"{4096 / N_TRAIN:.1f}"
    )

    print("=" * 80)

    print(
        f"{'Method':<14}"
        f"{'Seeds':>7}"
        f"{'Mean':>12}"
        f"{'Std':>12}"
    )

    print("-" * 80)

    for method in METHODS:
        method_results = (
            all_scores.get(
                method,
                {},
            )
        )

        if not method_results:
            continue

        ordered_seeds = sorted(
            method_results
        )

        scores = np.asarray(
            [
                method_results[seed]
                for seed in ordered_seeds
            ],
            dtype=np.float64,
        )

        mean_score = float(
            scores.mean()
        )

        standard_deviation = (
            float(
                scores.std(ddof=1)
            )
            if len(scores) > 1
            else 0.0
        )

        per_seed = " ".join(
            f"s{seed}="
            f"{method_results[seed]:.4f}"
            for seed in ordered_seeds
        )

        print(
            f"{method:<14}"
            f"{len(scores):>7}"
            f"{mean_score:>12.6f}"
            f"{standard_deviation:>12.6f} "
            f"{per_seed}"
        )

    lora_results = all_scores.get(
        "lora",
        {},
    )

    pivot_results = all_scores.get(
        "pivotlora",
        {},
    )

    common_seeds = sorted(
        set(lora_results)
        & set(pivot_results)
    )

    if common_seeds:
        paired_differences = (
            np.asarray(
                [
                    pivot_results[seed]
                    - lora_results[seed]
                    for seed in common_seeds
                ],
                dtype=np.float64,
            )
        )

        print(
            "\nPaired gains "
            "(PivotLoRA - LoRA):"
        )

        for seed, difference in zip(
            common_seeds,
            paired_differences,
        ):
            print(
                f"  seed {seed}: "
                f"{difference:+.6f} "
                f"({100 * difference:+.3f} pp)"
            )

        print(
            f"Mean paired gain: "
            f"{paired_differences.mean():+.6f} "
            f"({100 * paired_differences.mean():+.3f} pp)"
        )

        if len(paired_differences) > 1:
            print(
                f"Paired-gain std: "
                f"{paired_differences.std(ddof=1):.6f}"
            )

    print(
        "\nProtocol controls:"
    )

    print(
        "  - implementation copied from the "
        "frozen GSM8K protocol"
    )

    print(
        "  - identical AdamW optimizer for "
        "both methods"
    )

    print(
        "  - identical base LR and linear "
        "scheduler"
    )

    print(
        "  - no eigenvalue or differential "
        "B learning rates"
    )

    print(
        "  - no random PCA-basis rotation"
    )

    print(
        "  - joint prompt/completion "
        "tokenization with offset masking"
    )

    print(
        "  - completion-only "
        "response-position PCA"
    )

    print(
        f"  - output-preserving QR after "
        f"epoch {PIVOT_EPOCH}"
    )

    print(
        "  - Adam state reset for both "
        "transformed factors"
    )

    print(
        "  - official 700-example train and "
        "300-example test splits"
    )

    print("=" * 80)


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    report_hardware()

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Run both methods for four epochs "
            "on seed 42 and evaluate 50 examples. "
            "This smoke test reaches the "
            "epoch-3 pivot."
        ),
    )

    parser.add_argument(
        "--methods",
        nargs="+",
        choices=METHODS,
        default=None,
    )

    parser.add_argument(
        "--seeds",
        type=int,
        default=None,
        help=(
            "Use the first K seeds from "
            "[42, 43, 44]. For example, "
            "--seeds 1 runs seed 42."
        ),
    )

    arguments = parser.parse_args()

    if (
        arguments.seeds is not None
        and not 1
        <= arguments.seeds
        <= len(SEED_LIST)
    ):
        parser.error(
            f"--seeds must be between 1 "
            f"and {len(SEED_LIST)}"
        )

    selected_seeds = (
        SEED_LIST[:arguments.seeds]
        if arguments.seeds is not None
        else SEED_LIST
    )

    if arguments.smoke:
        run_study(
            methods=METHODS,
            seeds=[42],
            smoke=True,
        )
    else:
        run_study(
            methods=arguments.methods,
            seeds=selected_seeds,
            smoke=False,
        )
