"""
run_gsm8k.py

Controlled reasoning-transfer experiment:

    Vanilla LoRA
        versus
    Response-position PivotLoRA

The two methods use identical optimization. They differ only in:

1. initialization of the LoRA input factor A:
       LoRA:       Kaiming-uniform
       PivotLoRA: response-predictor-position PCA

2. PivotLoRA applies one output-preserving QR reparameterization
   after epoch 3.

Protocol v2:

    The prompt ends at "Answer:" and the completion is " " + answer field.
    v1 ended the prompt with "Answer: ", so at inference the model was
    conditioned on a lone-space token that never preceded a completion in
    training. The joined training text is identical in both versions;
    format_train now asserts that the inference prompt is an exact token
    prefix of each training sequence. Each run saves per-example
    predictions and records the fraction of generations ending in
    "#### <number>", and a frozen base-model reference ("base") is
    evaluated once.

Primary full study (base reference, then LoRA and PivotLoRA over three seeds):
    python -u run_gsm8k.py

Only first paired seed:
    python -u run_gsm8k.py --seeds 1

Only PivotLoRA:
    python -u run_gsm8k.py --methods pivotlora --seeds 3

Pipeline/pivot smoke test:
    python -u run_gsm8k.py --smoke
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

RANK = 8
ALPHA = 16

N_TRAIN = 128
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

# "base" evaluates the frozen pretrained model without adapters or training.
# It is run once, on the first requested seed, as a reference point.
METHODS = ["base", "lora", "pivotlora"]

TARGET_MODULES = [
    "q_proj",
    "v_proj",
]

# New filename: deliberately does not reuse v1 results.
RESULTS_CSV = "results_gsm8k_response_pivot_matched_v2.csv"
SMOKE_RESULTS_CSV = "results_gsm8k_response_pivot_matched_v2_smoke.csv"

DIAGNOSTICS_DIR = "pivot_diagnostics_response_matched_v2"

PREDICTIONS_DIR = "predictions_gsm8k_response_matched_v2"
SMOKE_PREDICTIONS_DIR = "predictions_gsm8k_response_matched_v2_smoke"

# v2: the prompt ends at "Answer:" and the completion begins with a space,
# so the inference prompt is an exact token prefix of every training
# sequence. The joined training text is unchanged from v1.
PROTOCOL_ID = "gsm8k_response_pca_qr_matched_v2"

CSV_FIELDS = [
    "protocol",
    "method",
    "seed",
    "n_train",
    "eval_examples",
    "correct",
    "gsm8k_exact_match",
    "parse_rate",
    "format_compliance",
    "train_wall_time_seconds",
    "predictions_file",
]

PROMPT_TEMPLATE = (
    "Question: {question}\n"
    "Answer:"
)

# Prepended to the dataset answer field to form the completion.
COMPLETION_PREFIX = " "

# A generation complies with the trained format when it ends with a
# numeric #### answer, as every GSM8K reference solution does.
FORMAT_PATTERN = re.compile(
    r"[\s\S]*\n####\s*-?\d[\d,]*(?:\.\d+)?\s*"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Pivot validation tolerance for the actual FP32-stored factors.
PRESERVATION_TOLERANCE = 1e-5

# PCA initialization scale used by the prior implementation.
# Q has unit-norm columns, so this gives each PCA direction the
# submitted initialization scale before training.
PCA_INITIALIZATION_SCALE_MULTIPLIER = math.sqrt(5.0)

# Smoke mode must reach the pivot.
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
# Answer extraction and normalization
# =============================================================================


NUMBER_PATTERN = re.compile(
    r"-?\d+(?:,\d+)*(?:\.\d+)?"
)


def extract_answer(text):
    """
    GSM8K reference answers normally contain '#### number'.

    For generated text, use the final numeric expression if the marker
    is absent.
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
    answer = answer.strip().replace(",", "")

    if not answer:
        return ""

    try:
        numeric_value = float(answer)

        if not math.isfinite(numeric_value):
            return answer

        if numeric_value.is_integer():
            return str(int(numeric_value))

        return format(numeric_value, ".15g")

    except ValueError:
        return answer


# =============================================================================
# Completion-only tokenization
# =============================================================================


def format_train(example, tokenizer):
    """
    Build a completion-only causal-LM example using character offsets.

    This avoids assuming that independently tokenizing the prompt and
    the prompt-plus-answer produces an identical token prefix.

    Labels:
        prompt and special-prefix tokens: -100
        answer tokens: token ID
        appended EOS: token ID
        padding: -100
    """

    question = example["question"]
    answer = example["answer"]

    prompt = PROMPT_TEMPLATE.format(
        question=question
    )

    text_without_eos = (
        prompt
        + COMPLETION_PREFIX
        + answer
    )
    response_start_character = len(prompt)

    encoded = tokenizer(
        text_without_eos,
        add_special_tokens=True,
        truncation=True,
        max_length=MAX_SEQ_LEN - 1,
        return_attention_mask=True,
        return_offsets_mapping=True,
    )

    input_ids = list(encoded["input_ids"])
    attention_mask = list(
        encoded["attention_mask"]
    )
    offsets = list(encoded["offset_mapping"])

    labels = []
    supervised_answer_tokens = 0

    for token_id, offset in zip(
        input_ids,
        offsets,
    ):
        start_character = int(offset[0])
        end_character = int(offset[1])

        # BOS and other special tokens generally have offset (0, 0).
        is_special_offset = (
            start_character == 0
            and end_character == 0
        )

        # Supervise every token containing at least one answer
        # character. This also correctly handles a token that crosses
        # the prompt/answer character boundary.
        is_response_token = (
            not is_special_offset
            and end_character
            > response_start_character
        )

        if is_response_token:
            labels.append(token_id)
            supervised_answer_tokens += 1
        else:
            labels.append(-100)

    if supervised_answer_tokens == 0:
        raise RuntimeError(
            "No answer tokens survived tokenization. "
            "The prompt may have consumed MAX_SEQ_LEN."
        )

    # The inference prompt must tokenize to exactly the unsupervised prefix
    # of the training sequence. Otherwise generation starts from a token
    # boundary never seen in training.
    prompt_ids = tokenizer(
        prompt,
        add_special_tokens=True,
    )["input_ids"]

    first_supervised_index = next(
        index
        for index, label in enumerate(labels)
        if label != -100
    )

    if (
        input_ids[:len(prompt_ids)] != prompt_ids
        or first_supervised_index != len(prompt_ids)
    ):
        raise RuntimeError(
            "The inference prompt is not a token prefix of the "
            "training sequence."
        )

    # Explicitly append and supervise EOS.
    input_ids.append(tokenizer.eos_token_id)
    attention_mask.append(1)
    labels.append(tokenizer.eos_token_id)

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


class GSM8KDataset(Dataset):
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
# Data loading and paired sampling
# =============================================================================


def sample_training_examples(
    training_examples,
    seed,
):
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


def load_gsm8k_splits():
    dataset = load_dataset(
        "openai/gsm8k",
        "main",
    )

    training_examples = list(
        dataset["train"]
    )

    test_examples = list(
        dataset["test"]
    )

    test_answers = [
        normalize_numeric_answer(
            extract_answer(
                example["answer"]
            )
        )
        for example in test_examples
    ]

    print(
        f"[DATA] train_pool="
        f"{len(training_examples)} "
        f"| test={len(test_examples)}"
    )

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

    This is the transpose of the paper convention:

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

        device = original_layer.weight.device
        adapter_dtype = torch.float32

        if init_A is None:
            # Standard LoRA initialization:
            # create [rank, d_in], initialize with Kaiming-uniform,
            # then transpose into stored [d_in, rank] coordinates.
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
                standard_A.transpose(0, 1)
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

        # Standard zero-B initialization preserves the pretrained
        # model output exactly at initialization.
        self.lora_B = nn.Parameter(
            torch.zeros(
                rank,
                output_dimension,
                device=device,
                dtype=adapter_dtype,
            )
        )

        original_layer.weight.requires_grad = False

        if original_layer.bias is not None:
            original_layer.bias.requires_grad = False

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
        get_llama_layers(model)[0].parameters()
    ).device


def load_base_model():
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

    PCA mean-centers over examples internally. Only the ordered PCA
    basis is returned. No random rotation is applied.
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
            f"Need at least rank+1 pooled examples for "
            f"rank-{rank} PCA; received "
            f"{number_of_samples}."
        )

    pca = PCA(
        n_components=rank,
        svd_solver="randomized",
        random_state=random_state,
        n_oversamples=max(10, 2 * rank),
        iterated_power=4,
    )

    pca.fit(activations)

    # sklearn components: [rank, dimension]
    # stored LoRA A:      [dimension, rank]
    basis = pca.components_.T.copy()

    # Re-orthogonalize for numerical cleanliness while retaining
    # column order. QR may flip signs, which does not change the span.
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
    Collect one pooled response-predictor activation per example and
    layer.

    For causal LM training, hidden state t predicts token t+1.
    Therefore, hidden state position t is selected when label t+1 is
    supervised.

    Activations are immediately mean-pooled within each example.
    Full token-level activation collections are never retained.
    """

    print(
        "[SUBSPACE] Computing response-position PCA..."
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

            sequence_length = hidden_states.shape[1]

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
                predictor_mask_local.any(dim=1)
            )

            if not valid_examples.any():
                return

            selected_states = hidden_states[
                valid_examples
            ]

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

            # Only one vector per valid example is stored.
            pooled_by_layer[
                layer_index
            ].append(
                pooled.float().cpu()
            )

        return pre_hook

    handles = []

    # q_proj and v_proj receive the same adapter input, so one hook
    # per transformer layer is sufficient.
    for layer_index, layer in enumerate(
        layers
    ):
        handle = (
            layer.self_attn.q_proj
            .register_forward_pre_hook(
                make_pre_hook(layer_index)
            )
        )

        handles.append(handle)

    device = get_model_input_device(model)

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

                predictor_mask = torch.zeros_like(
                    labels,
                    dtype=torch.bool,
                )

                # Correct causal response-position shift:
                # hidden state t predicts supervised label t+1.
                predictor_mask[:, :-1] = (
                    labels[:, 1:] != -100
                )

                # Exclude padded predictor positions.
                predictor_mask &= attention_mask

                if not predictor_mask.any():
                    raise RuntimeError(
                        "A PCA extraction batch contained no "
                        "supervised predictor positions."
                    )

                active_predictor_mask["value"] = (
                    predictor_mask.to(device)
                )

                model(
                    input_ids=(
                        batch["input_ids"].to(device)
                    ),
                    attention_mask=(
                        batch["attention_mask"].to(device)
                    ),
                    use_cache=False,
                )

    finally:
        active_predictor_mask["value"] = None

        for handle in handles:
            handle.remove()

    subspaces = defaultdict(dict)

    if len(pooled_by_layer) != len(layers):
        raise RuntimeError(
            "Activation extraction did not cover every "
            f"layer: got {len(pooled_by_layer)} of "
            f"{len(layers)}."
        )

    for layer_index in range(len(layers)):
        pooled_batches = pooled_by_layer[
            layer_index
        ]

        if not pooled_batches:
            raise RuntimeError(
                f"No pooled activations for layer "
                f"{layer_index}."
            )

        activation_matrix = torch.cat(
            pooled_batches,
            dim=0,
        ).numpy()

        if activation_matrix.shape[0] != len(
            train_dataset
        ):
            raise RuntimeError(
                f"Layer {layer_index} produced "
                f"{activation_matrix.shape[0]} pooled "
                f"examples; expected {len(train_dataset)}."
            )

        layer_basis = compute_pca_basis(
            activations=activation_matrix,
            rank=rank,
            random_state=(
                seed * 1000 + layer_index
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
                f"  [SUBSPACE] layer={layer_index:>2} "
                f"| examples="
                f"{activation_matrix.shape[0]} "
                f"| d={activation_matrix.shape[1]}"
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
    differential rates, or post-pivot LR overrides.
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

    The transformation is computed in FP64 and then stored in FP32.
    AdamW state is reset for both transformed factors. The ordinary
    Trainer linear LR scheduler continues unchanged.
    """

    def __init__(
        self,
        pivot_epoch,
        diagnostics_path,
    ):
        self.pivot_epoch = pivot_epoch
        self.diagnostics_path = diagnostics_path
        self.pivot_done = False
        self.pivot_global_step = None

    @staticmethod
    def _frobenius_product_norm_squared(
        A,
        B,
    ):
        """
        Compute ||A B||_F^2 using only rank-by-rank Gram matrices.

        A: [d, r]
        B: [r, m]
        """

        gram_A = A.transpose(0, 1) @ A
        gram_B = B @ B.transpose(0, 1)

        value = (
            gram_A * gram_B.transpose(0, 1)
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

        denominator = old_squared.clamp_min(
            1e-30
        )

        return torch.sqrt(
            difference_squared / denominator
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
                "Pivot callback did not receive "
                "the optimizer."
            )

        # Removing the complete state entry makes AdamW initialize
        # fresh moments and a fresh step counter on the next update.
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

        if state.epoch + 1e-8 < self.pivot_epoch:
            return control

        if model is None:
            raise RuntimeError(
                "Pivot callback did not receive model."
            )

        if optimizer is None:
            raise RuntimeError(
                "Pivot callback did not receive optimizer."
            )

        print(
            f"[PivotLoRA] Executing output-preserving "
            f"QR pivot at epoch={state.epoch:.6f}, "
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
                    module.lora_A.detach()
                    .to(dtype=torch.float64)
                    .clone()
                )

                B_old_64 = (
                    module.lora_B.detach()
                    .to(dtype=torch.float64)
                    .clone()
                )

                orthogonality_before = (
                    self._orthogonality_error(
                        A_old_64
                    )
                )

                # A_old = Q R.
                Q_64, R_64 = torch.linalg.qr(
                    A_old_64,
                    mode="reduced",
                )

                A_new_64 = Q_64
                B_new_64 = R_64 @ B_old_64

                # Store in the actual FP32 adapter parameter dtype.
                A_new_stored = A_new_64.to(
                    device=module.lora_A.device,
                    dtype=module.lora_A.dtype,
                )

                B_new_stored = B_new_64.to(
                    device=module.lora_B.device,
                    dtype=module.lora_B.dtype,
                )

                # Validate the actual stored factors after conversion.
                A_check_64 = A_new_stored.to(
                    dtype=torch.float64
                )

                B_check_64 = B_new_stored.to(
                    dtype=torch.float64
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
                        "Non-finite pivot preservation "
                        f"error in {module_name}."
                    )

                if not math.isfinite(
                    orthogonality_after
                ):
                    raise RuntimeError(
                        "Non-finite post-pivot "
                        f"orthogonality error in "
                        f"{module_name}."
                    )

                if (
                    preservation_error
                    > PRESERVATION_TOLERANCE
                ):
                    raise RuntimeError(
                        "Pivot preservation error exceeded "
                        f"tolerance in {module_name}: "
                        f"{preservation_error:.3e} > "
                        f"{PRESERVATION_TOLERANCE:.3e}"
                    )

                module.lora_A.copy_(
                    A_new_stored
                )

                module.lora_B.copy_(
                    B_new_stored
                )

                # Both parameter tensors changed coordinates.
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
                "Pivot callback found no LoRA modules."
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
            "maximum_preservation_error": (
                maximum_preservation_error
            ),
            "minimum_orthogonality_before": (
                minimum_orthogonality_before
            ),
            "maximum_orthogonality_after": (
                maximum_orthogonality_after
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

        print(
            f"[PivotLoRA] Pivoted "
            f"{pivoted_modules} adapters"
        )

        print(
            "[PivotLoRA] Maximum preservation "
            f"error={maximum_preservation_error:.3e}"
        )

        print(
            "[PivotLoRA] Maximum post-pivot "
            f"orthogonality error="
            f"{maximum_orthogonality_after:.3e}"
        )

        print(
            "[PivotLoRA] Optimizer state reset for "
            "A and B; Trainer LR schedule unchanged"
        )

        return control


# =============================================================================
# Evaluation
# =============================================================================


def evaluate_gsm8k(
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

    device = get_model_input_device(model)

    if max_eval is not None:
        test_examples = test_examples[:max_eval]
        test_answers = test_answers[:max_eval]

    evaluation_count = len(test_examples)

    if evaluation_count == 0:
        raise RuntimeError(
            "Evaluation set is empty."
        )

    number_correct = 0
    invalid_predictions = 0
    compliant_predictions = 0
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
        prompt = PROMPT_TEMPLATE.format(
            question=example["question"]
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
            encoded_prompt["attention_mask"]
            .to(device)
        )

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=MAX_NEW_TOKENS,
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

        generated_text = tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        )

        raw_prediction = extract_answer(
            generated_text
        )

        predicted_answer = (
            normalize_numeric_answer(
                raw_prediction
            )
        )

        if not predicted_answer:
            invalid_predictions += 1

        format_compliant = bool(
            FORMAT_PATTERN.fullmatch(
                generated_text
            )
        )

        if format_compliant:
            compliant_predictions += 1

        correct = (
            predicted_answer
            == ground_truth
        )

        if correct:
            number_correct += 1

        prediction_records.append(
            {
                "index": example_index,
                "question": example["question"],
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
                "parsed": bool(
                    predicted_answer
                ),
                "format_compliant": (
                    format_compliant
                ),
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
        number_correct / evaluation_count
    )

    parse_rate = (
        (evaluation_count - invalid_predictions)
        / evaluation_count
    )

    format_compliance = (
        compliant_predictions
        / evaluation_count
    )

    os.makedirs(
        os.path.dirname(predictions_path)
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
        f"  [EVAL] GSM8K exact match: "
        f"{accuracy:.6f} "
        f"({number_correct}/"
        f"{evaluation_count})"
    )

    print(
        f"  [EVAL] Invalid predictions: "
        f"{invalid_predictions}"
    )

    print(
        f"  [EVAL] Format-compliant predictions: "
        f"{compliant_predictions}/"
        f"{evaluation_count} "
        f"({format_compliance:.6f})"
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
        format_compliance,
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
        f"GSM8K | protocol={PROTOCOL_ID} "
        f"| method={method} | seed={seed}"
    )

    print(
        f"N={N_TRAIN} | d/N="
        f"{4096 / N_TRAIN:.1f} "
        f"| rank={RANK} | alpha={ALPHA}"
    )

    print("=" * 80)

    configure_reproducibility(seed)

    sampled_training_examples = (
        sample_training_examples(
            training_examples=training_pool,
            seed=seed,
        )
    )

    train_dataset = GSM8KDataset(
        examples=sampled_training_examples,
        tokenizer=tokenizer,
    )

    model = load_base_model()

    model.config.pad_token_id = (
        tokenizer.pad_token_id
    )

    callbacks = []

    predictions_path = os.path.join(
        (
            SMOKE_PREDICTIONS_DIR
            if smoke
            else PREDICTIONS_DIR
        ),
        (
            f"{PROTOCOL_ID}_"
            f"{method}_seed_{seed}"
            f"{'_smoke' if smoke else ''}"
            ".jsonl"
        ),
    )

    maximum_evaluation_examples = (
        SMOKE_EVAL_EXAMPLES
        if smoke
        else None
    )

    if method == "base":
        (
            accuracy,
            number_correct,
            evaluation_count,
            parse_rate,
            format_compliance,
        ) = evaluate_gsm8k(
            model=model,
            tokenizer=tokenizer,
            test_examples=test_examples,
            test_answers=test_answers,
            predictions_path=predictions_path,
            max_eval=maximum_evaluation_examples,
        )

        del model
        del train_dataset

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {
            "accuracy": accuracy,
            "correct": number_correct,
            "eval_examples": evaluation_count,
            "parse_rate": parse_rate,
            "format_compliance": format_compliance,
            "train_wall_time": 0.0,
            "predictions_file": predictions_path,
        }

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
            DIAGNOSTICS_DIR,
            exist_ok=True,
        )

        diagnostics_path = os.path.join(
            DIAGNOSTICS_DIR,
            (
                f"{PROTOCOL_ID}_"
                f"seed_{seed}.json"
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
        training_arguments = TrainingArguments(
            eval_strategy="no",
            **training_argument_values,
        )

    except TypeError:
        training_arguments = TrainingArguments(
            evaluation_strategy="no",
            **training_argument_values,
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

    train_start_time = time.perf_counter()

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
                "Training finished without executing "
                "the PivotLoRA QR pivot."
            )

        if (
            pivot_callback.pivot_global_step
            is None
        ):
            raise RuntimeError(
                "Pivot callback did not record its "
                "global step."
            )

    # Release the optimizer state and trainer before generation so the
    # evaluation KV cache does not compete with them for GPU memory.
    del trainer
    del optimizer

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    (
        accuracy,
        number_correct,
        evaluation_count,
        parse_rate,
        format_compliance,
    ) = evaluate_gsm8k(
        model=model,
        tokenizer=tokenizer,
        test_examples=test_examples,
        test_answers=test_answers,
        predictions_path=predictions_path,
        max_eval=maximum_evaluation_examples,
    )

    del model
    del train_dataset

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "accuracy": accuracy,
        "correct": number_correct,
        "eval_examples": evaluation_count,
        "parse_rate": parse_rate,
        "format_compliance": format_compliance,
        "train_wall_time": train_wall_time,
        "predictions_file": predictions_path,
    }


# =============================================================================
# Results persistence
# =============================================================================


def load_completed_results(results_csv):
    completed = {}

    if not os.path.exists(results_csv):
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

                if int(row["n_train"]) != N_TRAIN:
                    continue

                key = (
                    row["method"],
                    int(row["seed"]),
                )

                completed[key] = {
                    "accuracy": float(
                        row[
                            "gsm8k_exact_match"
                        ]
                    ),
                    "correct": int(
                        row["correct"]
                    ),
                    "eval_examples": int(
                        row["eval_examples"]
                    ),
                    "train_wall_time": float(
                        row[
                            "train_wall_time_seconds"
                        ]
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
        "gsm8k_response_pivot_matched_v2_"
        f"{'smoke_' if smoke else ''}"
        "manifest.json"
    )

    manifest = {
        "protocol": PROTOCOL_ID,
        "model": MODEL_NAME,
        "task": "GSM8K completion-only SFT",
        "n_train": N_TRAIN,
        "full_eval_examples": 1319,
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
        "gradient_accumulation": GRAD_ACCUM,
        "effective_batch_size": (
            EFFECTIVE_BATCH_SIZE
        ),
        "base_learning_rate": (
            LEARNING_RATE
        ),
        "weight_decay": WEIGHT_DECAY,
        "scheduler": "linear",
        "warmup_steps": 0,
        "pca_source": (
            "mean-pooled hidden states at "
            "positions predicting supervised "
            "response tokens"
        ),
        "pca_random_rotation": False,
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
        "max_sequence_length": MAX_SEQ_LEN,
        "max_new_tokens": MAX_NEW_TOKENS,
        "prompt_template": PROMPT_TEMPLATE,
        "completion": (
            f"{COMPLETION_PREFIX!r} + answer field"
        ),
        "prompt_is_token_prefix": True,
        "format_compliance": (
            "fraction of generations ending in "
            "'\\n#### <number>'"
        ),
        "base_reference": (
            "frozen pretrained model, no adapter "
            "or training, evaluated once"
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

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=True,
    )

    if not tokenizer.is_fast:
        raise RuntimeError(
            "A fast tokenizer is required for "
            "offset-based response masking."
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
    ) = load_gsm8k_splits()

    print(
        f"[RESULTS] File: {results_csv}"
    )

    completed = load_completed_results(
        results_csv
    )

    if completed:
        print(
            f"[RESUME] Found "
            f"{len(completed)} valid completed rows"
        )

    needs_header = (
        not os.path.exists(results_csv)
        or os.path.getsize(results_csv) == 0
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
        for method in methods:
            # Greedy evaluation of the frozen model does not depend on
            # the training sample, so the base reference runs once.
            method_seeds = (
                seeds[:1]
                if method == "base"
                else seeds
            )

            for seed in method_seeds:
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
                        training_pool=training_pool,
                        test_examples=test_examples,
                        test_answers=test_answers,
                        tokenizer=tokenizer,
                        smoke=smoke,
                    )

                    all_scores[method][seed] = (
                        result["accuracy"]
                    )

                    writer.writerow(
                        {
                            "protocol": PROTOCOL_ID,
                            "method": method,
                            "seed": seed,
                            "n_train": N_TRAIN,
                            "eval_examples": (
                                result[
                                    "eval_examples"
                                ]
                            ),
                            "correct": (
                                result["correct"]
                            ),
                            "gsm8k_exact_match": (
                                f"{result['accuracy']:.9f}"
                            ),
                            "parse_rate": (
                                f"{result['parse_rate']:.9f}"
                            ),
                            "format_compliance": (
                                f"{result['format_compliance']:.9f}"
                            ),
                            "train_wall_time_seconds": (
                                f"{result['train_wall_time']:.2f}"
                            ),
                            "predictions_file": (
                                result[
                                    "predictions_file"
                                ]
                            ),
                        }
                    )

                    results_handle.flush()

                except Exception as exception:
                    print(
                        f"[CRASH] {method}-s{seed}: "
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
        "GSM8K RESULT: MATCHED-OPTIMIZER "
        "RESPONSE-POSITION PIVOTLORA"
    )

    print(
        f"Protocol: {PROTOCOL_ID}"
    )

    print(
        f"Model: {MODEL_NAME}"
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
        method_results = all_scores.get(
            method,
            {},
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
            float(scores.std(ddof=1))
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
        paired_differences = np.asarray(
            [
                pivot_results[seed]
                - lora_results[seed]
                for seed in common_seeds
            ],
            dtype=np.float64,
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
        "  - identical AdamW optimizer for both methods"
    )

    print(
        "  - identical base LR and linear scheduler"
    )

    print(
        "  - no eigenvalue or differential B learning rates"
    )

    print(
        "  - no random PCA-basis rotation"
    )

    print(
        "  - completion-only response-position PCA"
    )

    print(
        f"  - output-preserving QR after epoch "
        f"{PIVOT_EPOCH}"
    )

    print(
        "  - Adam state reset for both transformed factors"
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
            "Run both methods for four epochs on "
            "seed 42 and evaluate 50 examples. "
            "This smoke test reaches the epoch-3 pivot."
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
            "[42, 43, 44]."
        ),
    )

    arguments = parser.parse_args()

    if (
        arguments.seeds is not None
        and not 1 <= arguments.seeds <= len(
            SEED_LIST
        )
    ):
        parser.error(
            f"--seeds must be between 1 and "
            f"{len(SEED_LIST)}"
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
