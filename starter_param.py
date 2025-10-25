# In this assignment, you will implement the core components of the
# Group-Reward Policy Optimization (GRPO) algorithm. You'll be working with a
# math-solving task called "Countdown," where the model has to generate an
# equation to reach a target number using a given set of numbers.
#
# You will need to implement six key functions:
# 1. `_extract_answer`: To parse the model's response.
# 2. `_validate_numbers`: To ensure the generated equation uses the correct numbers.
# 3. `_evaluate_equation`: To safely calculate the result of the generated equation.
# 4. `reward_fn`: To score the model's generated equations using the helpers.
# 5. `compute_group_normalized_advantages`: To calculate advantages.
# 6. `compute_loss`: To compute per-token loss.
# 7. `masked_mean`: To compute the mean of masked tensor, used in GRPO
# 8. `masked_mean_drgrpo`: To compute the mean of masked tensor, used in DR-GRPO
# ==============================================================================

import os
import datetime
import random
from typing import Callable, Dict, List, Tuple, Any
import logging
import warnings

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from vllm import LLM, SamplingParams
from vllm.model_executor import set_random_seed as vllm_set_random_seed
from unittest.mock import patch
from datasets import load_dataset
from torch.utils.tensorboard import SummaryWriter
import re
from dotenv import load_dotenv

load_dotenv()

logging.getLogger("vllm.engine.scheduler").setLevel(logging.ERROR)
os.environ["VLLM_USE_V1"] = "0"

from torch.optim.lr_scheduler import LambdaLR


def get_constant_schedule_with_warmup(optimizer: torch.optim.Optimizer, num_warmup_steps: int, last_epoch: int = -1):
    def lr_lambda(current_step: int):
        return min(1.0, float(current_step) / float(max(1, num_warmup_steps)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


# -------------------------
# Prompting helpers (Countdown-style)
# -------------------------
TEMPLATE = """Answer the given mathematical reasoning question {question}.
Show your reasoning in <think> </think> tags. And return the final equation in <answer> </answer> tags. Keep your reasoning under {max_tokens} tokens.
For example, question = Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?, the answer is <answer>72</answer>."""


# vLLM utilities
def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85) -> LLM:
    vllm_set_random_seed(seed)
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None,
    )
    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=2048
        )


def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM) -> None:
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())


def init_sampling_params(temperature: float, min_tokens: int, max_tokens: int) -> SamplingParams:
    sp = SamplingParams(
        temperature=temperature,
        top_p=1.0,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
        logprobs=0,
    )
    sp.stop = ["</answer>"]
    sp.include_stop_str_in_output = True
    return sp


# Tokenization utilities
def tokenize_prompt_and_output(prompt_strs: List[str], output_strs: List[str], tokenizer: AutoTokenizer) -> Dict[
    str, torch.Tensor]:
    batch_data = []
    max_len = 0
    for prompt, output in zip(prompt_strs, output_strs):
        prompt_tokens = tokenizer(prompt)["input_ids"]
        output_tokens = tokenizer(output)["input_ids"]
        combined_tokens = prompt_tokens + output_tokens
        max_len = max(max_len, len(combined_tokens))
        batch_data.append({
            "tokens": combined_tokens, "prompt_len": len(prompt_tokens), "total_len": len(combined_tokens)
        })
    batch_size = len(batch_data)
    input_ids = torch.full((batch_size, max_len - 1), tokenizer.eos_token_id, dtype=torch.long)
    labels = torch.full((batch_size, max_len - 1), tokenizer.eos_token_id, dtype=torch.long)
    response_mask = torch.zeros((batch_size, max_len - 1), dtype=torch.bool)
    for i, data in enumerate(batch_data):
        tokens, seq_len = torch.tensor(data["tokens"]), len(data["tokens"])
        input_ids[i, :seq_len - 1], labels[i, :seq_len - 1] = tokens[:-1], tokens[1:]
        response_start, response_end = data["prompt_len"] - 1, seq_len - 1
        if response_end > response_start:
            response_mask[i, response_start:response_end] = True
    return {"input_ids": input_ids, "labels": labels, "response_mask": response_mask}


# Reward function and evaluation (Countdown solution)

# ==============================================================================
# TASK 1: Implement Reward Helper Functions
# Before you can write the reward function, you need to create three helper
# functions to parse and evaluate the model's generated text.
# ==============================================================================

def _extract_answer(solution_str: str) -> str | None:
    """
    Extract the content from the last <answer>...</answer> tag in the given string.

    Hint: Use the `re` module. `re.finditer` can find all occurrences of a pattern.
    Args:
        solution_str: The full text generated by the model.

    Returns:
        The stripped string content of the last answer tag, or None if no tag is found.
    """
    if not isinstance(solution_str, str) or not solution_str:
        return None

    # pattern - <\s*answer\s*>(.*?)</\s*answer\s*>
    matches = list(re.finditer(r"<\s*answer\s*>(.*?)</\s*answer\s*>", solution_str,
                               flags=re.IGNORECASE | re.DOTALL))

    if not matches:
        return None

    content = matches[-1].group(1).strip()
    return content if content else None


def _validate_numbers(equation_str: str, available_numbers: List[int]) -> bool:
    """
    Check if the numbers used in the equation are exactly the ones available.

    Hint:
    1. Use `re.findall(r"\d+", equation_str)` to get all number strings from the equation.
    2. Remember to handle potential errors and return False.

    Args:
        equation_str: The equation string to validate.
        available_numbers: A list of integers that are allowed to be used.

    Returns:
        True if the equation uses the correct numbers, False otherwise.
    """
    try:
        if not isinstance(equation_str, str) or not equation_str.strip():
            return False

        if "=" in equation_str:
            equation_str = equation_str.split("=", 1)[0]

        # removing decimal number answers
        if "." in equation_str:
            return False

        number_tokens = re.findall(r"\d+", equation_str)
        used = [int(x) for x in number_tokens]

        from collections import Counter
        return Counter(used) == Counter(int(x) for x in available_numbers)
    except Exception:
        return False


def _evaluate_equation(equation_str: str) -> float | None:
    """
    Safely evaluate a mathematical equation string.
    Args:
        equation_str: The equation string to evaluate.
    Returns:
        The result of the equation as a float, or None if it's invalid or unsafe.
    """
    if not isinstance(equation_str, str) or not equation_str.strip():
        return None

    if not re.fullmatch(r"[0-9\+\-\*/\(\)\s]+", equation_str):
        return None

    import ast, math
    ALLOWED_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div)
    ALLOWED_UNARY = (ast.UAdd, ast.USub)

    try:
        node = ast.parse(equation_str, mode='eval')
    except Exception:
        return None

    def _eval(n) -> float:
        if isinstance(n, ast.Expression):
            return _eval(n.body)
        elif isinstance(n, ast.Num):
            return float(n.n)
        elif isinstance(n, ast.Constant):
            if isinstance(n.value, int):
                return float(n.value)

            return None
        elif isinstance(n, ast.UnaryOp) and isinstance(n.op, ALLOWED_UNARY):
            val = _eval(n.operand)
            if val is None:
                return None

            return +val if isinstance(n.op, ast.UAdd) else -val
        elif isinstance(n, ast.BinOp) and isinstance(n.op, ALLOWED_BINOPS):
            left = _eval(n.left)
            right = _eval(n.right)

            if left is None or right is None:
                return None

            if isinstance(n.op, ast.Div):
                if abs(right) < 1e-12:
                    return None
                return left / right
            elif isinstance(n.op, ast.Add):
                return left + right
            elif isinstance(n.op, ast.Sub):
                return left - right
            elif isinstance(n.op, ast.Mult):
                return left * right
            else:
                return None

        # Disallow everything else: names, calls, floor-div, pow, etc.
        return None

    try:
        result = _eval(node)
        if result is None or not math.isfinite(result):
            return None

        return float(result)
    except Exception:
        return None


# ==============================================================================
# TASK 2: Implement the Reward Function
# ==============================================================================
def reward_fn(generated_text: str, ground_truth: int) -> float:
    """
    Reward function for countdown problems.
    - 1.0 (Perfect): The equation is valid, uses the correct numbers, and evaluates to the target.
    - 0.1 (Partial): The text contains an <answer> tag, but the equation is incorrect for any reason.
    - 0.0 (Failed): The `generated_text` does not contain an <answer> tag.

    Args:
        generated_text: The full text output from the language model.
        ground_truth: A dictionary containing `target` and `numbers`.

    Returns:
        A float value representing the reward, such as 1.0, 0.1, or 0.0
    """
    math_ans = _extract_answer(generated_text)
    if math_ans is None:
        return 0.0

    #result = int(math_ans)
    target = ground_truth

    # malformed ground truth
    if target is None:
        return 0.1

    if not _validate_numbers(math_ans, [target]):
        return 0.1

    result = _evaluate_equation(math_ans)
    if result is None:
        return 0.1

    return 1.0 if abs(result - float(target)) < 1e-6 else 0.1


def evaluate_model(llm: LLM, sampling_params: SamplingParams, eval_prompts: List[str], eval_answers: List[int]) -> \
Dict[str, Any]:
    rollouts = llm.generate(eval_prompts, sampling_params)
    examples, rewards, output_token_lengths = [], [], []
    for rollout, gt in zip(rollouts, eval_answers):
        response_text = rollout.outputs[0].text
        reward_value = reward_fn(response_text, gt)
        equation = _extract_answer(response_text)

        result = _evaluate_equation(equation) if equation is not None else None
        output_tokens = len(llm.llm_engine.tokenizer.encode(response_text))
        output_token_lengths.append(output_tokens)

        examples.append({
            "prompt": rollout.prompt, "response": response_text, "answer": gt, "equation": equation,
            "result": result, "reward": reward_value, "output_tokens": output_tokens,
        })
        rewards.append(reward_value)
    rewards_tensor = torch.tensor(rewards) if rewards else torch.tensor([0.0])
    tol = 1e-8
    count_correct = sum(1 for r in rewards if abs(r - 1.0) < tol)
    count_partial = sum(1 for r in rewards if abs(r - 0.1) < tol)
    count_failed = sum(1 for r in rewards if abs(r - 0.0) < tol)
    accuracy = (count_correct / len(rewards)) * 100 if rewards else 0.0
    avg_output_tokens = sum(output_token_lengths) / len(output_token_lengths) if output_token_lengths else 0.0
    return {
        "mean_reward": float(rewards_tensor.mean().item()),
        "std_reward": float(rewards_tensor.std().item()) if rewards_tensor.numel() > 1 else 0.0,
        "num_examples": len(rewards), "examples": examples, "count_correct": count_correct,
        "count_partial": count_partial, "count_failed": count_failed, "accuracy": accuracy,
        "avg_output_tokens": avg_output_tokens,
    }


def _format_eval_example(example: Dict[str, Any]) -> str:
    target = example["answer"]["target"] if isinstance(example.get("answer"), dict) and "target" in example[
        "answer"] else "?"
    numbers = example["answer"].get("numbers") if isinstance(example.get("answer"), dict) else None
    return (
        f"Prompt: {example.get('prompt', '')}\n"
        f"Response: {example.get('response', '')}\n"
        f"Equation: {example.get('equation', None)} | Result: {example.get('result', None)} | Target: {target} | Numbers: {numbers}\n"
        f"Reward: {example.get('reward', 0.0):.3f}\n"
    )


def log_train(rollout_batch_loss: float, grad_norm: float, reward_metadata: Dict[str, Any], avg_output_tokens: float,
              writer: SummaryWriter | None, step: int) -> None:
    writer.add_scalar("train/loss", float(rollout_batch_loss), global_step=step)
    writer.add_scalar("train/grad_norm", float(grad_norm), global_step=step)
    writer.add_scalar("train/reward_mean", float(reward_metadata["mean"]), global_step=step)
    writer.add_scalar("train/reward_std", float(reward_metadata["std"]), global_step=step)
    writer.add_scalar("train/avg_output_tokens", float(avg_output_tokens), global_step=step)
    print(
        f"Step {step} | Loss: {rollout_batch_loss:.4f} | Grad norm: {grad_norm:.4f} | Reward mean: {float(reward_metadata['mean']):.4f} | Reward std: {float(reward_metadata['std']):.4f} | Avg output tokens: {avg_output_tokens:.1f}")


def log_eval(metrics: Dict[str, Any], writer: SummaryWriter | None, step: int) -> None:
    examples = metrics.get("examples", []) or []
    if not examples: return
    tol = 1e-8
    correct_examples = [ex for ex in examples if abs(float(ex.get("reward", 0.0)) - 1.0) < tol][:10]
    partial_examples = [ex for ex in examples if abs(float(ex.get("reward", 0.0)) - 0.1) < tol][:10]
    failed_examples = [ex for ex in examples if abs(float(ex.get("reward", 0.0)) - 0.0) < tol][:10]
    if correct_examples:
        print(f"\n=== Eval examples (CORRECT, reward=1.0) @ step {step} ===")
        for idx, ex in enumerate(correct_examples[:2], 1): print(f"[CORRECT #{idx}]\n" + _format_eval_example(ex))
    if partial_examples:
        print(f"\n=== Eval examples (PARTIAL, reward=0.1) @ step {step} ===")
        for idx, ex in enumerate(partial_examples[:2], 1): print(f"[PARTIAL #{idx}]\n" + _format_eval_example(ex))
    if failed_examples:
        print(f"\n=== Eval examples (FAILED, reward=0.0) @ step {step} ===")
        for idx, ex in enumerate(failed_examples[:2], 1): print(f"[FAILED #{idx}]\n" + _format_eval_example(ex))
    if writer:
        correct_text = "\n\n".join([_format_eval_example(ex) for ex in correct_examples]) or ""
        partial_text = "\n\n".join([_format_eval_example(ex) for ex in partial_examples]) or ""
        failed_text = "\n\n".join([_format_eval_example(ex) for ex in failed_examples]) or ""
        if correct_text: writer.add_text("eval/examples_correct", correct_text, global_step=step)
        if partial_text: writer.add_text("eval/examples_partial", partial_text, global_step=step)
        if failed_text: writer.add_text("eval/examples_failed", failed_text, global_step=step)
    print(f"Eval @ step {step}: accuracy={metrics['accuracy']:.1f}% mean_reward={metrics['mean_reward']:.4f} "
          f"avg_tokens={metrics['avg_output_tokens']:.1f} | correct:{metrics['count_correct']} "
          f"partial:{metrics['count_partial']} failed:{metrics['count_failed']}")


# -------------------------
# Advantages and GRPO-clip loss
# -------------------------
# ==============================================================================
# TASK 3: Implement Group Normalized Advantages
# ==============================================================================
def compute_group_normalized_advantages(
        rollout_responses: List[str],
        repeated_ground_truths: List[Dict],
        reward_fn: Callable[[str, int], float],
        group_size: int,
        advantage_eps: float,
        normalize_by_std: bool,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """ Computes advantages by normalizing rewards within groups.

    Args:
        rollout_responses: List of generated responses (length = batch_size * group_size)
        repeated_ground_truths: Ground truth repeated for each response
        reward_fn: Function to compute rewards
        group_size: Number of responses per question (G)
        advantage_eps: Small constant for numerical stability (epsilon)
        normalize_by_std: If True, normalize by std (GRPO); if False, don't (DR-GRPO)

    Returns:
        advantages: Flattened tensor of advantages (shape: [batch_size * group_size])
        raw_rewards: Original rewards before normalization
        metadata: Dictionary with 'mean', 'std', 'max', 'min' of raw rewards
            (e.g. metadata = {
                "mean": torch.mean(raw_rewards),
                "std": torch.std(raw_rewards),
                "max": torch.max(raw_rewards),
                "min": torch.min(raw_rewards),
            })
    """

    # Steps:
    # 1. Calculate the raw reward for each response using the provided `reward_fn`.
    # 2. Reshape the 1D tensor of raw rewards into a 2D tensor of shape (-1, `group_size`).
    # 3. Calculate the mean reward for each group.
    # 4. Compute the advantage by subtracting the group's mean reward.
    # 5. If `normalize_by_std` is True, normalize advantages by `(group_std + advantage_eps)`.
    # 6. Flatten the advantages tensor back into a 1D tensor.
    # 7. Create a `metadata` dictionary with overall statistics of the raw rewards.
    advantages, raw_rewards, metadata = None, None, {}

    raw_rewards_list = [float(reward_fn(r, gt)) for r, gt in zip(rollout_responses, repeated_ground_truths)]
    raw_rewards = torch.tensor(raw_rewards_list, dtype=torch.float32)

    if raw_rewards.numel() == 0:
        advantages = torch.tensor([], dtype=torch.float32)
        metadata = {'mean': 0.0, 'std': 0.0, 'max': 0.0, 'min': 0.0}

        return advantages, raw_rewards, metadata

    if raw_rewards.numel() % group_size != 0:
        usable = (raw_rewards.numel() // group_size) * group_size
        raw_rewards = raw_rewards[:usable]

    group_rewards = raw_rewards.view(-1, group_size)

    group_mean = group_rewards.mean(dim=1, keepdim=True)

    adv = group_rewards - group_mean

    if normalize_by_std:
        group_std = group_rewards.std(dim=1, keepdim=True, unbiased=False)
        adv = adv / (group_std + advantage_eps)

    advantages = adv.reshape(-1)

    metadata = {
        "mean": float(raw_rewards.mean().item()),
        "std": float(raw_rewards.std(unbiased=False).item()),
        "max": float(raw_rewards.max().item()),
        "min": float(raw_rewards.min().item()),
    }
    return advantages, raw_rewards, metadata


# ==============================================================================
# TASK 4: Implement per-token Loss
# ==============================================================================
def compute_loss(
        advantages: torch.Tensor,
        policy_log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        clip_range: float,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Computes the per-token PPO clipped surrogate loss.

    Why omit the KL divergence term?
    For simplicity and following trends in recent RLVR (e.g Dr.GRPO),
    we omit the KL penalty term often found in PPO. This simplifies the implementation
    and has been shown to work well in practice.

    Steps:
    1. Calculate the probability ratio `pi_ratio = exp(policy_log_probs - old_log_probs)`.
    2. Calculate the unclipped term: `advantages * pi_ratio`.
    3. Calculate the clipped term by clipping `pi_ratio` to `[1-clip_range, 1+clip_range]`
       and then multiplying by `advantages`.
    4. The final loss is `-torch.minimum(unclipped_term, clipped_term)`.
    """

    pi_ratio = torch.exp(policy_log_probs - old_log_probs)
    unclipped = advantages * pi_ratio
    clipped_ratio = torch.clamp(pi_ratio, 1.0 - clip_range, 1.0 + clip_range)
    clipped = advantages * clipped_ratio

    loss_per_token = -torch.minimum(unclipped, clipped)

    with torch.no_grad():
        clipped_mask = (pi_ratio < (1.0 - clip_range)) | (pi_ratio > (1.0 + clip_range))
        metadata = {
            "pi_ratio_mean": pi_ratio.mean().detach(),
            "pi_ratio_std": pi_ratio.std(unbiased=False).detach(),
            "clip_fraction": clipped_mask.float().mean().detach(),
        }
    return loss_per_token, metadata


def masked_mean(tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
        Compute the mean of tensor values where mask=True for each row, then average across the batch.
    """
    mask_f = mask.to(dtype=tensor.dtype)
    per_row_sum = (tensor * mask_f).sum(dim=-1)
    per_row_count = mask_f.sum(dim=-1).clamp_min(1.0)
    per_row_mean = per_row_sum / per_row_count

    return per_row_mean.mean()


def masked_mean_drgrpo(tensor: torch.Tensor, mask: torch.Tensor, num_tokens: int) -> torch.Tensor:
    """
        Compute the sum of tensor values where mask=True, divided by num_tokens, then average across the batch.
        This is used for the DR-GRPO loss
    """
    if num_tokens <= 0:
        num_tokens = 1

    mask_f = mask.to(dtype=tensor.dtype)
    per_row_sum = (tensor * mask_f).sum(dim=-1)
    per_row_scaled = per_row_sum / num_tokens

    return per_row_scaled.mean()


def get_response_log_probs(model: PreTrainedModel, input_ids: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    logits = model(input_ids).logits
    log_probs = F.log_softmax(logits, dim=-1)
    log_probs = torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    return log_probs


# Training helpers and loop
def duplicate_data(arr: List, group_size: int) -> List:
    return [x for x in arr for _ in range(group_size)]


def rollout_with_vllm(policy: PreTrainedModel, llm: LLM, sampling_params: SamplingParams, prompts_batch: List[str],
                      group_size: int) -> Tuple[List[str], List[str], List[int]]:
    load_policy_into_vllm_instance(policy, llm)
    prompts_dup = duplicate_data(prompts_batch, group_size)
    vllm_rollouts = llm.generate(prompts_dup, sampling_params, use_tqdm=False)
    rollout_input_text, rollout_response_text, rollout_output_tokens = [], [], []
    for rollout in vllm_rollouts:
        for r in rollout.outputs:
            rollout_input_text.append(rollout.prompt)
            rollout_response_text.append(r.text)
            rollout_output_tokens.append(len(llm.llm_engine.tokenizer.encode(r.text)))
    return rollout_input_text, rollout_response_text, rollout_output_tokens


def tokenize_rollouts(rollout_input_text: List[str], rollout_response_text: List[str], tokenizer: AutoTokenizer) -> \
Dict[str, torch.Tensor]:
    return tokenize_prompt_and_output(rollout_input_text, rollout_response_text, tokenizer)


def grpo_microbatch_step(
        policy: PreTrainedModel, input_ids: torch.Tensor, labels: torch.Tensor, response_mask: torch.Tensor,
        advantages_per_seq: torch.Tensor, gradient_accumulation_steps: int, clip_range: float,
        loss_type: str = "grpo", max_completion_length: int = 256,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    policy_log_probs = get_response_log_probs(policy, input_ids, labels)
    old_log_probs = policy_log_probs.detach()
    advantages = advantages_per_seq.unsqueeze(-1)
    loss_per_token, metadata = compute_loss(advantages, policy_log_probs, old_log_probs, clip_range)
    if loss_type == "grpo":
        loss = masked_mean(loss_per_token, response_mask)
    elif loss_type == "dr_grpo":
        loss = masked_mean_drgrpo(loss_per_token, response_mask, max_completion_length)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
    loss = loss / gradient_accumulation_steps
    loss.backward()
    return loss.detach(), metadata


def train(
        policy: PreTrainedModel, tokenizer: AutoTokenizer, llm: LLM, sampling_params: SamplingParams, *,
        train_prompts: List[str], train_answers: List[Dict], eval_prompts: List[str], eval_answers: List[int],
        optimizer: torch.optim.Optimizer, scheduler, n_grpo_steps: int, rollout_batch_size: int,
        group_size: int, gradient_accumulation_steps: int, clip_range: float, use_std_normalization: bool,
        advantage_eps: float, device: str, eval_every: int = 5, writer: SummaryWriter = None, seed: int,
        loss_type: str = "grpo", max_completion_length: int = 256,
) -> None:
    n_prompts_per_rollout_batch = rollout_batch_size // group_size
    micro_train_batch_size = rollout_batch_size // gradient_accumulation_steps
    random.seed(seed)
    train_step = 0

    metrics = evaluate_model(llm, sampling_params, eval_prompts, eval_answers)
    if writer:
        for k in ["accuracy", "mean_reward", "std_reward", "avg_output_tokens", "count_correct", "count_partial",
                  "count_failed"]:
            writer.add_scalar(f"eval/{k}", metrics[k], global_step=train_step)
        log_eval(metrics, writer, train_step)

    for _ in range(n_grpo_steps):
        sampled = random.sample(list(zip(train_prompts, train_answers)), n_prompts_per_rollout_batch)
        prompts_batch, answers_batch = [p for p, _ in sampled], [a for _, a in sampled]
        rollout_input, rollout_response, rollout_tokens = rollout_with_vllm(policy, llm, sampling_params, prompts_batch,
                                                                            group_size)
        answers_dup = duplicate_data(answers_batch, group_size)
        avg_output_tokens = sum(rollout_tokens) / len(rollout_tokens) if rollout_tokens else 0.0
        advantages, _, reward_meta = compute_group_normalized_advantages(
            rollout_response, answers_dup, reward_fn, group_size, advantage_eps, use_std_normalization
        )
        tokenized = tokenize_rollouts(rollout_input, rollout_response, tokenizer)
        optimizer.zero_grad()
        rollout_loss = 0.0
        for micro_idx in range(0, rollout_batch_size, micro_train_batch_size):
            s = slice(micro_idx, micro_idx + micro_train_batch_size)
            loss, _ = grpo_microbatch_step(
                policy, tokenized["input_ids"][s].to(device), tokenized["labels"][s].to(device),
                tokenized["response_mask"][s].to(device), advantages[s].to(device),
                gradient_accumulation_steps, clip_range, loss_type=loss_type,
                max_completion_length=max_completion_length
            )
            rollout_loss += float(loss.item())
        grad_norm = torch.nn.utils.clip_grad_norm_([p for p in policy.parameters() if p.grad is not None], 1.0)
        optimizer.step()
        scheduler.step()
        rollout_loss /= (rollout_batch_size / micro_train_batch_size)
        train_step += 1
        print(f"Step {train_step} | Loss: {rollout_loss:.4f} | Grad: {grad_norm:.4f} | "
              f"Reward mean: {reward_meta['mean']:.4f} | Reward std: {reward_meta['std']:.4f}")
        log_train(rollout_loss, grad_norm, reward_meta, avg_output_tokens, writer, train_step)
        if train_step % eval_every == 0:
            metrics = evaluate_model(llm, sampling_params, eval_prompts, eval_answers)
            log_eval(metrics, writer, train_step)


def init_policy(model_id: str, device: str) -> Tuple[PreTrainedModel, AutoTokenizer]:
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", use_cache=False
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model.to(device).train()
    return model, tokenizer


def main() -> None:
    # Hyperparameters
    model_id = "Qwen/Qwen3-1.7B"
    device = "cuda"
    seed, gpu_mem_util = 42, 0.4
    n_grpo_steps, rollout_batch_size, group_size, grad_acc_steps = 80, 128, 8, 32
    lr, clip_range, adv_eps = 7e-6, 0.2, 1e-6
    temperature, min_tokens = 1, 4
    eval_every = 10

    # CHANGING HYPERPARAMETERS for main assignment
    loss_type = "grpo"  # "grpo" # or "dr_grpo"
    max_tokens = 512  # or 512, 1024

    # Initialization
    use_std_norm = loss_type == "grpo"
    policy, tokenizer = init_policy(model_id=model_id, device=device)
    llm = init_vllm(model_id=model_id, device=device, seed=seed, gpu_memory_utilization=gpu_mem_util)
    sampling_params = init_sampling_params(temperature=temperature, min_tokens=min_tokens, max_tokens=max_tokens)

    # Dataset
    def build_dataset(split):
        data = []
        for ex in split:
            answer_value = ex['answer'].split('#### ')[-1].strip().replace(',', '')
            prompt = TEMPLATE.format(question=ex["question"], max_tokens=max_tokens)
            prompt = tokenizer.apply_chat_template(
                [dict(role="system", content="You are a smart helpful mathematical assistant."),
                 dict(role="user", content=prompt)],
                add_generation_prompt=True, tokenize=False)
            data.append({"prompt": prompt, "answer": answer_value, })
        return data

    # Load properly split dataset
    train_data = load_dataset("openai/gsm8k", "main")['train']
    eval_data = load_dataset("openai/gsm8k", "main")['test']

    train_examples = build_dataset(train_data)
    eval_examples = build_dataset(eval_data)

    # Optimizer and Scheduler
    # weight decay - 1e-2
    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=1e-2, betas=(0.9, 0.95))
    scheduler = get_constant_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=0)

    # Logging
    timestamp = int(datetime.datetime.now(datetime.timezone.utc).timestamp())
    log_dir = os.path.join("./output", "tb", f"hw_a2_{loss_type}", str(timestamp))
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    # Training
    train(
        policy=policy, tokenizer=tokenizer, llm=llm, sampling_params=sampling_params,
        train_prompts=[ex["prompt"] for ex in train_examples], train_answers=[ex["answer"] for ex in train_examples],
        eval_prompts=[ex["prompt"] for ex in eval_examples], eval_answers=[ex["answer"] for ex in eval_examples],
        optimizer=optimizer, scheduler=scheduler, n_grpo_steps=n_grpo_steps,
        rollout_batch_size=rollout_batch_size, group_size=group_size,
        gradient_accumulation_steps=grad_acc_steps, clip_range=clip_range,
        use_std_normalization=use_std_norm, advantage_eps=adv_eps, device=device,
        eval_every=eval_every, writer=writer, seed=seed, loss_type=loss_type,
        max_completion_length=max_tokens
    )

    # Save model
    out_dir = os.path.join("./output", f"hw_a2_solution_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)
    policy.save_pretrained(out_dir)
    try:
        policy.push_to_hub(f'qwen_model_{loss_type}_gsm8k_{max_tokens}_change_param')
                           #token=<hf-writing-token>)
    except Exception as exc:
        print(exc)

    try:
        tokenizer.push_to_hub(f'qwen_model_{loss_type}_gsm8k_{max_tokens}_change_param')
                              #token=<hf-writing-token>)
    except Exception as exc:
        print(exc)

    tokenizer.save_pretrained(out_dir)
    print(f"Saved model and tokenizer to {out_dir}")
    writer.close()


if __name__ == "__main__":
    main()