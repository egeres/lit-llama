"""
Instruction-tuning with LoRA on the Alpaca dataset.

Note: If you run into a CUDA error "Expected is_sm80 to be true, but got false", uncomment the line
`torch.backends.cuda.enable_flash_sdp(False)` in the script below (see https://github.com/Lightning-AI/lit-llama/issues/101).
"""

# pip install -U deepspeed

import json
import os
import pathlib
import re
import sys
import time
from pathlib import Path

import lightning as L
import matplotlib
import numpy as np
import torch
from matplotlib import pyplot as plt
from rich import print

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from generate import generate
from lit_llama.lora import lora, lora_state_dict, mark_only_lora_as_trainable
from lit_llama.model import LLaMA, LLaMAConfig
from lit_llama.tokenizer import Tokenizer
from scripts.prepare_alpaca import generate_prompt

print("[#4287f5]Imported stuff...", flush=True)

instruction_validation_ = "This is the default text to test this stuff"
instruction_tuning = False
eval_interval = 20
save_interval = 20
eval_iters = 100
log_interval = 20

# Hyperparameters
learning_rate = 3e-4
batch_size = 128
micro_batch_size = 2
gradient_accumulation_iters = batch_size // micro_batch_size
assert gradient_accumulation_iters > 0
# max_iters = 50000 * 3 // micro_batch_size
# max_iters = 1000 * 3 // micro_batch_size
max_iters = 5000 * 3 // micro_batch_size
weight_decay = 0.0
max_seq_length = 512  # see scripts/prepare_alpaca.py
lora_r = 64
lora_alpha = 16
lora_dropout = 0.05
warmup_iters = 100


def get_max_epoch(path_dir_root: str | pathlib.Path) -> int | None:
    assert os.path.exists(path_dir_root)

    numbers: list[int] = []
    for i in os.listdir(path_dir_root):
        iter_match = re.search(r"iter-(\d+)-ckpt.pth", i)
        starting_iter = int(iter_match.group(1)) if iter_match else 0
        numbers.append(starting_iter)
    if len(numbers) == 0:
        return None
    return max(numbers)


def main(
    data_dir: str = "data/alpaca",
    pretrained_path: str = "checkpoints/lit-llama/7B/lit-llama.pth",
    tokenizer_path: str = "checkpoints/lit-llama/tokenizer.model",
    out_dir: str = "out/lora/alpaca",
    previous_checkpoint: str | None = None,
    devices: int = 2,
    instruction_validation: str | None = None,
    max_iters_multiplier: int | None = None,
):
    # Ugly code because I'm in a rush
    if instruction_validation is not None:
        global instruction_validation_
        instruction_validation_ = instruction_validation
    if max_iters_multiplier is not None:
        global max_iters
        max_iters = max_iters_multiplier * 3 // micro_batch_size
    print(f"[#4287f5]Current max iters ...{max_iters}", flush=True)

    print(f"[#4287f5]Starting Fabric... with {devices} devices", flush=True)
    t0 = time.time()
    fabric = L.Fabric(
        accelerator="cuda",
        devices=devices,
        precision="bf16-true",
        # strategy="ddp",
        # strategy="deepspeed",
    )
    fabric.launch()
    fabric.seed_everything(1337 + fabric.global_rank)
    print(f"[#4287f5]Loaded fabric in {time.time() - t0:.2f}s", flush=True)

    if fabric.global_rank == 0:
        os.makedirs(out_dir, exist_ok=True)

    print("[#4287f5]Loading datasets...", flush=True)
    t0 = time.time()
    train_data, val_data = load_datasets(data_dir=data_dir)
    print(f"[#4287f5]Loaded datasets in {time.time() - t0:.2f}s", flush=True)

    config = LLaMAConfig.from_name("7B")
    config.block_size = max_seq_length

    print("[#4287f5]Max sequence length is " + str(max_seq_length), flush=True)
    print("[#4287f5]Loading model...")
    t0 = time.time()
    checkpoint = torch.load(pretrained_path)
    print(f"[#4287f5]Loaded pretrained weights in {time.time() - t0:.2f}s")

    with fabric.init_module(), lora(
        r=lora_r, alpha=lora_alpha, dropout=lora_dropout, enabled=True
    ):
        model = LLaMA(config)
        # model = nn.DataParallel(LLaMA(config))
        # model = nn.parallel.DistributedDataParallel(model)

        # strict=False because missing keys due to LoRA weights not contained in checkpoint state
        # model.load_state_dict(checkpoint, strict=False)

        state_dict = checkpoint
        # Load the fine-tuned adapter weights, if provided
        if previous_checkpoint is not None and os.path.isfile(previous_checkpoint):
            lora_checkpoint = torch.load(previous_checkpoint)
            state_dict.update(lora_checkpoint)
        # strict=False because missing keys due to adapter weights not containted in state dict
        model.load_state_dict(state_dict, strict=False)

    mark_only_lora_as_trainable(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    model, optimizer = fabric.setup(model, optimizer)

    print("[#4287f5]Starting training...")
    t0 = time.time()
    train(
        fabric,
        model,
        optimizer,
        train_data,
        val_data,
        tokenizer_path,
        out_dir,
        previous_checkpoint,
    )
    print(f"[#4287f5]Training took {time.time() - t0:.2f}s")

    # Save the final LoRA checkpoint at the end of training
    checkpoint = lora_state_dict(model)
    fabric.save(os.path.join(out_dir, "lit-llama-lora-finetuned.pth"), checkpoint)


def train(
    fabric: L.Fabric,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_data: np.ndarray,
    val_data: np.ndarray,
    tokenizer_path: str,
    out_dir: str,
    previous_checkpoint: str | None = None,
) -> None:
    """The training loop.

    Loosely based on the nanoGPT implementation: https://github.com/karpathy/nanoGPT.
    """
    step_count = 0
    losses_training_sub = []
    losses_training_so_far = []
    losses_validation_so_far = []

    if os.path.exists(os.path.join(out_dir, "training.json")):
        with open(os.path.join(out_dir, "training.json"), "r") as f:
            losses_training_so_far = json.load(f)
    if os.path.exists(os.path.join(out_dir, "validation.json")):
        with open(os.path.join(out_dir, "validation.json"), "r") as f:
            losses_validation_so_far = json.load(f)

    for iter_num in range(max_iters):
        if step_count <= warmup_iters:
            # linear warmup
            lr = learning_rate * step_count / warmup_iters
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

        t0 = time.time()

        input_ids, targets = get_batch(fabric, train_data)
        with fabric.no_backward_sync(
            model, enabled=((iter_num + 1) % gradient_accumulation_iters != 0)
        ):
            logits = model(input_ids)
            loss = loss_fn(logits, targets)
            losses_training_sub.append(loss.item())
            fabric.backward(loss / gradient_accumulation_iters)

        if (iter_num + 1) % gradient_accumulation_iters == 0:
            optimizer.step()
            optimizer.zero_grad()
            step_count += 1

            if step_count % eval_interval == 0:
                val_loss = validate(fabric, model, val_data, tokenizer_path)

                # Plot eval
                losses_validation_so_far.append(val_loss)
                plt.clf()
                plt.plot(losses_validation_so_far, label="validation")
                plt.savefig(os.path.join(out_dir, "validation.png"))

                # Plot train
                losses_training_so_far.append(np.mean(losses_training_sub))
                losses_training_sub = []
                plt.clf()
                plt.plot(losses_training_so_far, label="training")
                plt.savefig(os.path.join(out_dir, "training.png"))

                with open(os.path.join(out_dir, "training.json"), "w") as f:
                    json.dump(losses_training_so_far, f)
                with open(os.path.join(out_dir, "validation.json"), "w") as f:
                    json.dump(losses_validation_so_far, f)

                fabric.print(f"step {iter_num}: val loss {val_loss:.4f}")
                fabric.barrier()

            if step_count % save_interval == 0:
                print(f"Saving LoRA weights to {out_dir}")
                # We are only saving the LoRA weights
                # TODO: Provide a function/script to merge the LoRA weights with pretrained weights
                checkpoint = lora_state_dict(model)
                fabric.save(
                    os.path.join(out_dir, f"iter-{iter_num:06d}-ckpt.pth"), checkpoint
                )

        dt = time.time() - t0
        loss_item = loss.item()

        if iter_num % log_interval == 0:
            fabric.print(
                f"iter {iter_num}: loss {loss_item:.4f}, time: {dt*1000:.2f}ms"
            )


def generate_response(model, instruction, tokenizer_path):
    tokenizer = Tokenizer(tokenizer_path)
    sample = {"instruction": instruction, "input": ""}
    prompt = instruction
    if instruction_tuning:
        prompt = generate_prompt(sample)
    encoded = tokenizer.encode(prompt, bos=True, eos=False, device=model.device)

    output = generate(
        model,
        idx=encoded,
        max_seq_length=max_seq_length,
        max_new_tokens=100,
    )
    output = tokenizer.decode(output)
    return output  # output.split("### Response:")[1].strip()


@torch.no_grad()
def validate(
    fabric: L.Fabric, model: torch.nn.Module, val_data: np.ndarray, tokenizer_path: str
) -> torch.Tensor:
    global instruction_validation_
    fabric.print("Validating ...")
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        input_ids, targets = get_batch(fabric, val_data)
        logits = model(input_ids)
        loss = loss_fn(logits, targets)
        losses[k] = loss.item()
    out = losses.mean()

    # produce an example:
    # instruction = (
    #     "Recommend a movie for me to watch during the weekend and explain the reason."
    # )
    instruction = instruction_validation_

    output = generate_response(model, instruction, tokenizer_path)
    fabric.print(instruction)
    fabric.print(output)

    model.train()
    return out.item()


def loss_fn(logits, targets):
    # shift the targets such that output n predicts token n+1
    logits = logits[..., :-1, :].contiguous()
    targets = targets[..., 1:].contiguous()
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
    )
    return loss


def get_batch(fabric: L.Fabric, data: list):
    ix = torch.randint(len(data), (micro_batch_size,))

    input_ids = [data[i]["input_ids"].type(torch.int64) for i in ix]
    labels = [data[i]["labels"].type(torch.int64) for i in ix]

    max_len = max(len(s) for s in input_ids)

    def pad_right(x, pad_id):
        # pad right based on the longest sequence
        n = max_len - len(x)
        return torch.cat((x, torch.full((n,), pad_id, dtype=x.dtype)))

    x = torch.stack([pad_right(x, pad_id=0) for x in input_ids])
    y = torch.stack([pad_right(x, pad_id=-1) for x in labels])
    x, y = fabric.to_device((x.pin_memory(), y.pin_memory()))
    return x, y


def load_datasets(data_dir):
    train_data = torch.load(os.path.join(data_dir, "train.pt"))

    size_in_mb = (
        len(train_data)
        * (
            train_data[0]["input_ids"].element_size()
            * train_data[0]["input_ids"].numel()
            + train_data[0]["labels"].element_size() * train_data[0]["labels"].numel()
        )
        / 1024
        / 1024
    )
    print(f"[#4287f5]Loaded training data ({size_in_mb:.2f} MB)", flush=True)

    val_data = torch.load(os.path.join(data_dir, "test.pt"))
    return train_data, val_data


if __name__ == "__main__":
    # Uncomment this line if you see an error: "Expected is_sm80 to be true, but got false"
    # torch.backends.cuda.enable_flash_sdp(False)
    torch.set_float32_matmul_precision("high")

    from jsonargparse.cli import CLI

    print("[#4287f5]Running main...", flush=True)

    CLI(main)
