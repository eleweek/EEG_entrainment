#!/usr/bin/env python3

import argparse
import hashlib
import random
import sys
from typing import List, Set


# A small built-in dictionary to compose memorable keys.
DICTIONARY_WORDS: List[str] = [
    "apple", "river", "forest", "stone", "cloud", "paper", "silver", "gold",
    "orange", "violet", "indigo", "scarlet", "crimson", "marble", "granite", "sand",
    "ocean", "island", "harbor", "valley", "meadow", "prairie", "canyon", "desert",
    "ember", "willow", "maple", "cedar", "spruce", "pine", "bamboo", "lotus",
    "falcon", "otter", "badger", "fox", "lynx", "panther", "tiger", "leopard",
    "orchid", "rose", "tulip", "daisy", "iris", "lilac", "peony", "violet",
    "aurora", "comet", "meteor", "nebula", "galaxy", "quasar", "cosmos", "orbit",
    "cobalt", "amber", "sable", "ivory", "onyx", "topaz", "jasper", "agate",
    "pepper", "cinnamon", "ginger", "saffron", "clove", "nutmeg", "basil", "thyme",
    "harvest", "solstice", "equinox", "midnight", "sunrise", "sunset", "twilight", "dawn",
    "harpoon", "anchor", "keel", "rudder", "sail", "beacon", "lighthouse", "harvest",
    "mariner", "voyage", "beach", "dune", "coral", "kelp", "reef", "lagoon",
]


def _resolve_blind_cond(key: str, session_num: int) -> str:
    """
    Determine blinded condition from a secret key and session number.

    This mirrors the logic used in scripts/run_trials.py to ensure keys
    map to identical conditions here and during data collection.
    """
    # TODO: deduplicate this function
    h = hashlib.sha256(key.encode("utf-8")).hexdigest()
    last_bit = int(h[-1], 16) & 1
    base = "T" if last_bit == 1 else "P"
    if session_num == 2:
        base = ("P" if base == "T" else "T")
    return base


def generate_blinding_keys(
    n: int = 10,
    session: int = 1,
    seed: int | None = None,
    prefix_number: bool = False,
) -> List[str]:
    """
    Generate n unique keys composed of a dictionary word and a digit 1..9,
    such that the resulting conditions split 1:1 between P and T for the
    specified session.
    """
    if n % 2 != 0:
        raise ValueError("n must be even to split 1:1 between P and T")
    if session not in (1, 2):
        raise ValueError("session must be 1 or 2")

    rng = random.Random(seed)

    target_per_cond = n // 2
    pool_words = DICTIONARY_WORDS[:]
    rng.shuffle(pool_words)

    digits = [str(d) for d in range(1, 10)]

    keys_p: List[str] = []
    keys_t: List[str] = []
    used: Set[str] = set()

    # Keep sampling word+digit combinations until we reach the target split
    attempts = 0
    max_attempts = 10000
    while (len(keys_p) < target_per_cond or len(keys_t) < target_per_cond) and attempts < max_attempts:
        attempts += 1
        word = rng.choice(pool_words)
        digit = rng.choice(digits)
        candidate = f"{digit}{word}" if prefix_number else f"{word}{digit}"
        if candidate in used:
            continue
        cond = _resolve_blind_cond(candidate, session)
        if cond == "P" and len(keys_p) < target_per_cond:
            keys_p.append(candidate)
            used.add(candidate)
        elif cond == "T" and len(keys_t) < target_per_cond:
            keys_t.append(candidate)
            used.add(candidate)

    if len(keys_p) != target_per_cond or len(keys_t) != target_per_cond:
        raise RuntimeError(
            f"Failed to generate the requested split after {attempts} attempts; try a different --seed."
        )

    # Combine and shuffle for a globally mixed list
    keys = keys_p + keys_t
    rng.shuffle(keys)
    return keys


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Generate blinded keys (word+digit) using the same logic as scripts/run_trials.py. "
            "By default, prints keys only without revealing their conditions."
        )
    )
    ap.add_argument("--n", type=int, default=10, help="Number of keys to generate (must be even)")
    ap.add_argument("--session", type=int, choices=[1, 2], default=1, help="Session number affecting blinding")
    ap.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    ap.add_argument(
        "--prefix-number",
        action="store_true",
        help="Prefix the number (e.g., 3river) instead of suffix (e.g., river3)",
    )
    ap.add_argument(
        "--show-mapping",
        action="store_true",
        help="Also print the mapped condition for each key (for internal use only)",
    )
    args = ap.parse_args()

    try:
        keys = generate_blinding_keys(
            n=args.n, session=args.session, seed=args.seed, prefix_number=args.prefix_number
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(2)

    if args.show_mapping:
        for k in keys:
            cond = _resolve_blind_cond(k, args.session)
            print(f"{k}\t{cond}")
    else:
        for k in keys:
            print(k)


if __name__ == "__main__":
    main()


