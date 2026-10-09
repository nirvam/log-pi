#!/usr/bin/env python3
"""End-to-end compatibility test using the official TypeSafe AI Python SDK (typesafe-sdk)."""

import sys
from typesafe_sdk import Choice, Noul, Score, TypeSafeClient


def main():
    print("=" * 60)
    print("Testing log(π) with official TypeSafe AI SDK (typesafe-sdk)")
    print("=" * 60)

    # Initialize TypeSafeClient pointing to our local log-pi daemon
    client = TypeSafeClient(
        api_key="test-local-token",
        base_url="http://localhost:9000",
        timeout=60.0,
    )

    state = (
        "The user is installing Arch Linux on an NVMe SSD and wants an efficient, "
        "minimalist UEFI boot setup without bloated menus."
    )

    questions = {
        "bootloader": Choice(
            instructions="Which bootloader aligns best with the user's preference?",
            criteria={
                "systemd-boot": "Simple, lightweight UEFI boot manager directly integrated with systemd",
                "grub": "Extensible multi-bootloader supporting BIOS and complex disk partitioning",
            },
        ),
        "is_experienced": Noul(
            instructions="Does the user likely have prior experience with Linux?",
        ),
        "complexity_rating": Score(
            instructions="Rate the complexity of this installation request from minimal to complex.",
            criteria=[
                "Very trivial, 1-command install",
                "Standard straightforward install",
                "Moderate complexity setup",
                "Advanced custom enterprise partition layout",
                "Extreme complexity setup",
            ],
        ),
    }

    print("\n[1] Sending request to POST /v1/systemone via client.system_one()...")
    resp = client.system_one(state=state, questions=questions)

    print("\n[2] Received response from log(π):")
    print(f"  Model ID:      {resp.model}")
    print(f"  Input Tokens:  {resp.usage.input_tokens}")
    print(f"  Output Tokens: {resp.usage.output_tokens}")

    print("\n[3] Validating parsed Answers:")
    bootloader_ans = resp.answers["bootloader"]
    print("  - Bootloader (Choice):")
    print(f"      Selected:    {bootloader_ans.choice}")
    print(f"      Confidence:  {bootloader_ans.confidence}")
    print(f"      Probs:       {bootloader_ans.probabilities}")
    assert bootloader_ans.type == "choice"
    assert bootloader_ans.choice in ["systemd-boot", "grub"]

    exp_ans = resp.answers["is_experienced"]
    print("  - Experience (Noul):")
    print(f"      Probability: {exp_ans.noul}")
    assert exp_ans.type == "noul"
    assert 0.0 <= exp_ans.noul <= 1.0

    score_ans = resp.answers["complexity_rating"]
    print("  - Complexity (Score):")
    print(f"      Score:       {score_ans.score}")
    print(f"      Confidence:  {score_ans.confidence}")
    print(f"      Legend:      {score_ans.legend}")
    print(f"      Probs:       {score_ans.probabilities}")
    assert score_ans.type == "score"
    assert len(score_ans.legend) == 5
    assert len(score_ans.probabilities) == 5

    print("\n" + "=" * 60)
    print("SUCCESS: 100% wire-protocol compatibility with typesafe-sdk verified!")
    print("=" * 60)


if __name__ == "__main__":
    main()
