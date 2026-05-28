#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _import_or_die():
    global torch, transformers, peft, trl, datasets
    import torch
    import transformers
    import peft
    import trl
    import datasets


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct",
                   help="HuggingFace model id (Qwen2.5-VL, InternVL3, LLaVA-OneVision, Molmo)")
    p.add_argument("--train", default="data/dpo/train.jsonl")
    p.add_argument("--val", default="data/dpo/val.jsonl")
    p.add_argument("--output-dir", default="checkpoints/wmw_dpo")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--lora-targets", default="auto",
                   help="Comma-list of target modules, or 'auto' for sensible defaults")
    p.add_argument("--beta", type=float, default=0.1, help="DPO beta")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--per-device-bs", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--max-prompt-length", type=int, default=2048)
    p.add_argument("--max-length", type=int, default=4096)
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--gradient-checkpointing", action="store_true", default=True)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--max-train-samples", type=int, default=None)
    p.add_argument("--text-only", action="store_true",
                   help="Skip image inputs (text-only DPO over trace JSON)")
    args = p.parse_args()

    _import_or_die()
    from transformers import AutoTokenizer, AutoProcessor, AutoModelForCausalLM
    from peft import LoraConfig
    from trl import DPOConfig, DPOTrainer
    from datasets import load_dataset

    print(f"\n══ DPO training: {args.model} ══")
    print(f"  Train: {args.train}")
    print(f"  Val:   {args.val}")
    print(f"  Output: {args.output_dir}")


    ds = load_dataset(
        "json",
        data_files={"train": args.train, "validation": args.val},
    )
    if args.max_train_samples:
        ds["train"] = ds["train"].select(range(min(args.max_train_samples, len(ds["train"]))))
    print(f"  Train samples: {len(ds['train'])}  Val samples: {len(ds['validation'])}")


    is_vlm = not args.text_only
    if is_vlm:
        try:
            processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
            tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
        except Exception as e:
            print(f"  WARN: could not load processor ({e}); falling back to text-only")
            is_vlm = False
            processor = None
            tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    else:
        processor = None
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


    if args.lora_targets == "auto":

        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"]
    else:
        target_modules = [t.strip() for t in args.lora_targets.split(",")]
    lora_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout, bias="none",
        target_modules=target_modules,
        task_type="CAUSAL_LM",
    )
    print(f"  LoRA: r={args.lora_r} alpha={args.lora_alpha} dropout={args.lora_dropout}")
    print(f"  Targets: {target_modules}")


    model_kwargs = dict(
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        trust_remote_code=True,
    )


    try:
        from transformers import AutoModelForVision2Seq
        model = AutoModelForVision2Seq.from_pretrained(args.model, **model_kwargs)
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False


    cfg = DPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_bs,
        per_device_eval_batch_size=args.per_device_bs,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        beta=args.beta,
        max_prompt_length=args.max_prompt_length,
        max_length=args.max_length,
        logging_steps=10,
        save_steps=200,
        eval_strategy="steps",
        eval_steps=100,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        seed=args.seed,
        remove_unused_columns=False,
        save_total_limit=2,
        report_to=[],
    )

    trainer = DPOTrainer(
        model=model,
        args=cfg,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        peft_config=lora_cfg,
        tokenizer=tokenizer,
        processor=processor if is_vlm else None,
    )

    print(f"\n══ Beginning training ({args.epochs} epoch(s)) ══")
    trainer.train()
    trainer.save_model(args.output_dir)
    if processor:
        processor.save_pretrained(args.output_dir)
    else:
        tokenizer.save_pretrained(args.output_dir)


    with open(Path(args.output_dir) / "wmw_run_config.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"\n══ Done. Adapter saved to {args.output_dir} ══")


if __name__ == "__main__":
    main()
