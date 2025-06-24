#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 13 13:34:00 2025

@author: danielroche
"""

#!/usr/bin/env python3
"""
Gustify – GC3 optimization + DRACH-scrubbing tool
-------------------------------------------------
"""

#from __future__ import annotations
import re, random, sys
from typing import List, Tuple, Dict, Set

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from Bio.Seq import Seq
from Bio.Data import CodonTable

# ==============================================================
#  General helpers
# ==============================================================

DNA_RE = re.compile(r"[^GATC]", re.I)


def clean_sequence(nuc: str) -> str:
    """Keep only A/T/G/C (upper-case)."""
    return DNA_RE.sub("", nuc.upper())


def translate_sequence(
    nuc: str,
    *,
    table: int | str = 1,
    trim_incomplete: bool = True,
    stop_at_first: bool = False,
) -> str:
    """Robust wrapper around BioPython translate()."""
    nuc = clean_sequence(nuc)
    if trim_incomplete and len(nuc) % 3:
        nuc = nuc[: -(len(nuc) % 3)]
        if not nuc:
            return ""
    try:
        aa = Seq(nuc).translate(table=table, to_stop=stop_at_first)
    except CodonTable.TranslationError as err:
        raise ValueError(f"Translation failed: {err}") from err
    return str(aa)


def translation_ok(seq: str, reference: str, label: str) -> bool:
    """Assert silent mutation; emit warning if not."""
    ok = translate_sequence(seq) == translate_sequence(reference)
    print(f"Protein translation preserved after {label}: {ok}")
    if not ok:
        print("⚠️  WARNING: amino-acid sequence changed!")
    return ok


def save_fasta(seq: str, fname: str = "Gustified_Sequence.fasta") -> None:
    with open(fname, "w") as fh:
        fh.write(">GustifiedSequence\n")
        fh.write(seq)
    print(f"✅  Sequence written to '{fname}'")

# ── Simple site / repeat checks ─────────────────────────────────────
def bspqi_check(seq: str) -> bool:      # BspQI  = GCTCTTC
    return "GCTCTTC" not in seq

def dam_check(seq: str) -> bool:        # Dam    = GATC
    return "GATC" not in seq

def dcm_check(seq: str) -> bool:        # Dcm    = CCWGG
    return all(x not in seq for x in ("CCTGG", "CCAGG"))

def repeat_check(seq: str, n: int = 10) -> bool:
    """Fail if any mono-nucleotide run ≥ n bp exists."""
    return not any(re.search(f"{b}{{{n},}}", seq) for b in "ATGC")


# ==============================================================
#  GC / GC3 utilities
# ==============================================================


def gc_percent(seq: str) -> float:
    g = seq.count("G") + seq.count("C")
    return 100 * g / len(seq) if seq else 0.0


def gc3_percent(seq: str) -> float:
    thirds = seq[2::3]
    return 100 * (thirds.count("G") + thirds.count("C")) / len(thirds) if thirds else 0.0


def sliding_gc(seq: str, win: int) -> Tuple[np.ndarray, np.ndarray]:
    vals = np.convolve(
        [1 if b in "GC" else 0 for b in seq], np.ones(win, int), "valid"
    )
    return np.arange(len(vals)), 100 * vals / win


# ==============================================================
#  DRACH finder / editor
# ==============================================================


def find_drach(seq: str) -> List[Tuple[int, str]]:
    """Return (start, 5-mer) for every DRACH motif."""
    seq = seq.upper()
    hits = []
    for i in range(len(seq) - 4):
        w = seq[i : i + 5]
        if w[0] in "AGT" and w[1] in "AG" and w[2] == "A" and w[3] == "C" and w[4] in "ACT":
            hits.append((i, w))
    return hits


def phase_drach(cds_start: int, hits) -> List[Tuple[int, str, int]]:
    out = []
    for start, motif in hits:
        phase = (start + 2 - cds_start) % 3
        out.append((start, motif, phase))
    return out


def remove_drach_motifs_if_possible(
    seq: str, drach_details, cds_start: int = 0
) -> Tuple[str, List[Tuple[int, str, int, str]]]:
    """Synonymously mutate codons to break DRACH motifs."""
    phase_rules: Dict[int, Dict[str, str]] = {
        0: {"AAA": "AAG", "AGA": "AGG", "GAA": "GAG", "GGA": "GGG", "TAA": "TAG", "TGA": "TAA"},
        1: {"AAC": "AAT", "GAC": "GAT"},
        2: {"ACA": "ACG", "ACC": "ACG", "ACT": "ACG", "AAA": "AAG", "GAA": "GAG", "TAA": "TAG"},
    }

    seq_l = list(seq)
    removed = []

    for start, motif, phase in drach_details:
        central = start + 2
        codon_start = central - phase
        if codon_start < 0 or codon_start + 3 > len(seq_l):
            continue

        old_codon = "".join(seq_l[codon_start : codon_start + 3])
        new_codon = phase_rules.get(phase, {}).get(old_codon)
        if not new_codon:
            continue

        alt_seq_l = seq_l.copy()
        alt_seq_l[codon_start : codon_start + 3] = list(new_codon)
        if translate_sequence(seq) == translate_sequence("".join(alt_seq_l)):
            seq_l = alt_seq_l
            removed.append((start, motif, phase, new_codon))

    return "".join(seq_l), removed


# ==============================================================
#  GC3 optimization
# ==============================================================


def optimize_gc3(
    seq: str, target_gc3: float, max_attempts: int = 100
) -> Tuple[
    str,
    float,
    float,
    float,
    float,
    Set[int],
    Set[int],
    int,
    int,
]:
    """
    Synonymously tweak 3rd bases toward a GC3 target.
    Returns: (new_seq, orig_gc, new_gc, orig_gc3, new_gc3,
              at→gc_indices, gc→at_indices, at→gc_count, gc→at_count)
    """
    std_table = CodonTable.unambiguous_dna_by_id[1].forward_table
    aa_to_codons: Dict[str, List[str]] = {}
    for codon, aa in std_table.items():
        aa_to_codons.setdefault(aa, []).append(codon)

    codons = [seq[i : i + 3] for i in range(0, len(seq) - len(seq) % 3, 3)]
    new_codons = codons[:]

    orig_gc = gc_percent(seq)
    orig_gc3 = gc3_percent(seq)

    at_to_gc_idx: Set[int] = set()
    gc_to_at_idx: Set[int] = set()
    at_to_gc = gc_to_at = 0

    attempts = 0
    while attempts < max_attempts:
        current_gc3 = gc3_percent("".join(new_codons))
        if abs(current_gc3 - target_gc3) <= 1:
            break

        idx = random.randrange(len(new_codons))
        codon = new_codons[idx]
        aa = std_table.get(codon, Seq(codon).translate())

        syn = [c for c in aa_to_codons.get(aa, []) if c[:2] == codon[:2]]
        if not syn:
            attempts += 1
            continue

        gc_pref = "GC" if target_gc3 > orig_gc3 else "AT"
        choices = [c for c in syn if c[2] in gc_pref]
        if not choices:
            attempts += 1
            continue

        alt = random.choice(choices)
        if alt == codon:
            attempts += 1
            continue

        tmp = new_codons[:]
        tmp[idx] = alt
        if translate_sequence(seq) != translate_sequence("".join(tmp)):
            attempts += 1
            continue

        # Accept change
        new_codons = tmp
        if codon[2] in "AT":
            at_to_gc += 1
            at_to_gc_idx.add(idx * 3 + 2)
        else:
            gc_to_at += 1
            gc_to_at_idx.add(idx * 3 + 2)

        attempts += 1

    new_seq = "".join(new_codons)
    return (
        new_seq,
        orig_gc,
        gc_percent(new_seq),
        orig_gc3,
        gc3_percent(new_seq),
        at_to_gc_idx,
        gc_to_at_idx,
        at_to_gc,
        gc_to_at,
    )


# ==============================================================
#  Plots
# ==============================================================


def plot_gc(seq: str, window: int = 20, chunk_size=1000):
    pos, vals = sliding_gc(seq, window)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    ax1.plot(pos, vals)
    ax1.set(title="Sliding-window GC%", xlabel="Position", ylabel="GC%")
    ax1.set_ylim(0, 100)
    ax1.legend()
    ax1.grid()

    ax2.hist(vals, bins=30)
    ax2.set(title="GC% distribution", xlabel="GC%", ylabel="Frequency")
    plt.tight_layout()
    plt.show()


def plot_nucleotide_changes(
    new_seq: str,
    old_seq: str,
    at_to_gc_idx: Set[int],
    gc_to_at_idx: Set[int],
    title: str = "Nucleotide changes"):
    """
    Draw per-base change map with the same visual grammar
    as plot_drach_motifs (colour bar + markers).
    """
    fig, ax = plt.subplots(figsize=(15, 3))

    # Base-colour bar (same palette as plot_drach_motifs)
    cmap = {"A": "#4e79a7", "T": "#f28e2c", "C": "#e15759", "G": "#59a14f"}
    for i, b in enumerate(new_seq):
        ax.add_patch(Rectangle((i, 0.7), 1, 0.2, color=cmap.get(b, "black"), lw=0))

    # Markers for every position
    for i, (new_b, old_b) in enumerate(zip(new_seq, old_seq)):
        if new_b == old_b:
            ax.plot(i, 1.4, "o", color="blue", ms=2)              # unchanged
        else:
            col = "green" if i in at_to_gc_idx else "red"
            ax.plot(i, 1.6, "o", color=col, ms=6)
            ax.text(i, 1.8, f"{new_b}", ha="center", va="bottom",
                    fontsize=6, color=col)

    ax.set_xlim(0, len(new_seq))
    ax.set_ylim(0.6, 2.1)
    ax.set_yticks([])
    ax.set_xlabel("Nucleotide position")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


def plot_drach_motifs(
    seq: str,
    drach_info,
    title: str = "DRACH motifs",
    removed=None,
):
    fig, ax = plt.subplots(figsize=(15, 3))
    phase_y = {0: 1.6, 1: 1.8, 2: 2.0}
    phase_col = {0: "blue", 1: "green", 2: "red"}

    for pos, motif, phase in drach_info:
        y, c = phase_y[phase], phase_col[phase]
        ax.plot(pos, y, "v", color=c, ms=9)
        ax.text(pos, y - 0.15, f"{pos}", ha='center', fontsize=6, color=c)
        ax.text(pos, y + 0.12, motif, ha="center", fontsize=6, rotation=45, color=c)

    if removed:
        for pos, old5, phase, _ in removed:
            ax.text(pos, 1 + phase * 0.4, "X", ha="center", color="black", fontweight="bold")
            ax.text(pos, y - 0.15, f"{pos}", ha='center', fontsize=6, color=c)


    # sequence colour bar
    cmap = {"A": "#4e79a7", "T": "#f28e2c", "C": "#e15759", "G": "#59a14f"}
    for i, b in enumerate(seq):
        ax.add_patch(Rectangle((i, 0.7), 1, 0.2, color=cmap.get(b, "black"), lw=0))

    ax.set_xlim(0, len(seq))
    ax.set_ylim(0.6, 2.3)
    ax.set_yticks([])
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


# ==============================================================
#  Main workflow
# ==============================================================


def gustify() -> Tuple[str, List[Tuple[int, str, int]]]:
    """
    Interactive workflow.
    Returns
    -------
    final_sequence : str
    removed_first_pass : list[(start, motif, phase)]
        DRACH motifs successfully removed in the first pass,
        *before* any GC3 optimization.
    """
    original_seq = input("Enter your nucleotide sequence: ").strip().upper()

    # 0) cleaning
    cleaned = clean_sequence(original_seq)
    print(f"Initial GC3%: {gc3_percent(cleaned):.2f}%")
    translation_ok(cleaned, original_seq, "cleaning")

    # 1) DRACH detection ────────────────────────────────────────────
    drach_list = phase_drach(0, find_drach(cleaned))
    if drach_list:
        print("Detected DRACH motifs (pos, motif, phase):")
        for pos, mot, ph in drach_list:
            print(f"  {pos:<5} {mot}  phase={ph}")
    else:
        print("No DRACH motifs found.")
    plot_drach_motifs(cleaned, drach_list, "Original DRACH motifs")

    # 1a) optional DRACH removal (first pass) ───────────────────────
    subset = choose_drach_subset(
        input(
            "\nRemove DRACH motifs?\n"
            "  • 'max' / 'y' – remove all\n"
            "  • 'no' / 'n'   – skip\n"
            "  • list of positions (e.g. 2,18,123)\n> "
        ),
        drach_list,
    )

    seq_after_drach1, removed1 = (
        remove_drach_motifs_if_possible(cleaned, subset) if subset else (cleaned, [])
    )

    if removed1:
        print(f"\n{len(removed1)} DRACH motif(s) removed:")
        for pos, old5, ph, new_codon in removed1:
            print(f"  pos {pos:>4} : {old5} → {new_codon}  (phase {ph})")
    else:
            print("No motifs removed in first pass.")
    

    plot_drach_motifs(
        seq_after_drach1, drach_list, "After DRACH removal (pass 1)", removed1
    )
    translation_ok(seq_after_drach1, original_seq, "first DRACH removal")

    # 2) GC3 optimization (optional) ────────────────────────────────
    current_gc3 = gc3_percent(seq_after_drach1)
    wants_gc3 = input(
        f"\nCurrent GC3% is {current_gc3:.2f}. "
        "Target a specific GC3%? (y/n): "
    ).lower()

    if wants_gc3 == "y":
        try:
            tgt = float(input("Enter target GC3%: "))
        except ValueError:
            print("❌  Not a number – skipping optimization.")
            seq_after_gc3, idx_a2g, idx_g2a = seq_after_drach1, set(), set()
        else:
            opt = optimize_gc3(seq_after_drach1, tgt)
            if isinstance(opt, str):
                print(f"❌  GC3 optimization failed: {opt}")
                seq_after_gc3, idx_a2g, idx_g2a = seq_after_drach1, set(), set()
            else:
                (
                    seq_after_gc3,
                    orig_gc,
                    new_gc,
                    orig_gc3,
                    new_gc3,
                    idx_a2g,
                    idx_g2a,
                    n_a2g,
                    n_g2a,
                ) = opt
                print(
                    f"GC3 {orig_gc3:.2f}% → {new_gc3:.2f}%   "
                    f"(GC {orig_gc:.2f}% → {new_gc:.2f}%)"
                )
                plot_gc(seq_after_gc3)
                plot_nucleotide_changes(seq_after_gc3, seq_after_drach1, idx_a2g, idx_g2a)
    else:
        seq_after_gc3 = seq_after_drach1
        print("GC3 optimization skipped.")

    translation_ok(seq_after_gc3, original_seq, "GC3 optimization")

    # 3) optional second DRACH round (only if GC3 step was run) ──────
    if wants_gc3 == "y" and input("Re-check DRACH motifs? (y/n): ").lower() == "y":
        drach2 = phase_drach(0, find_drach(seq_after_gc3))
        
        print("Detected DRACH motifs (pos, motif, phase):")
        for pos, mot, ph in drach2:
            print(f"  {pos:<5} {mot}  phase={ph}")
        
        subset2 = choose_drach_subset(
            input(
                "\nRemove DRACH motifs (2nd pass)?\n"
                "  • 'max' / 'y' – remove all\n"
                "  • 'no' / 'n'   – skip\n"
                "  • list of positions\n> "
            ),
            drach2,
        )
        seq_final, removed2 = (
            remove_drach_motifs_if_possible(seq_after_gc3, subset2)
            if subset2
            else (seq_after_gc3, [])
        )
        
        if removed1:
            print(f"\n{len(removed1)} DRACH motif(s) removed:")
            for pos, old5, ph, new_codon in removed1:
                print(f"  pos {pos:>4} : {old5} → {new_codon}  (phase {ph})")
        else:
                print("No motifs removed in first pass.")
        
        
        plot_drach_motifs(seq_final, drach2, "After DRACH removal (pass 2)", removed2)
    else:
        seq_final = seq_after_gc3

    translation_ok(seq_final, original_seq, "second DRACH removal")
    save_fasta(seq_final)

    # 4) simple site / repeat checks + summary ───────────────────────
    start_drach = len(drach_list)          # ← already computed up-front
    final_drach = len(find_drach(seq_final))
    
    print("\nDone ✅ – key stats:")
    print(f"Length : {len(seq_final)} bp")
    print(f"GC     : {gc_percent(seq_final):.2f}%")
    print(f"GC3    : {gc3_percent(seq_final):.2f}%")
    print(f"DRACHs : {len(find_drach(seq_final))}")
    print(f"DRACHs : {start_drach}  →  {final_drach}")   # ← NEW
    print("Restriction / repeat checks:")
    print(f"  • BspQI  (GCTCTTC)          : {'Pass' if bspqi_check(seq_final) else 'Fail'}")
    print(f"  • Dam    (GATC)             : {'Pass' if dam_check(seq_final)   else 'Fail'}")
    print(f"  • Dcm    (CCTGG/CCAGG)      : {'Pass' if dcm_check(seq_final)   else 'Fail'}")
    print(f"  • Long mono-nt repeats ≥10  : {'Pass' if repeat_check(seq_final) else 'Fail'}")
    print(seq_final)

    return seq_final, [(pos, mot, ph) for pos, mot, ph, _ in removed1]


# ==============================================================
#  DRACH prompt helper
# ==============================================================


def choose_drach_subset(user_input: str, phased):
    text = user_input.strip().lower()
    if text in {"y", "yes", "max"}:
        return phased
    if text in {"n", "no"}:
        return []
    try:
        wanted = {int(x) for x in re.split(r"\s*,\s*", text) if x}
    except ValueError:
        print("⚠️  Could not parse list – skipping.")
        return []
    subset = [t for t in phased if t[0] in wanted]
    missing = wanted - {t[0] for t in subset}
    if missing:
        print("⚠️  No DRACH motif at positions:", sorted(missing))
    return subset


# ==============================================================
#  Entry-point
# ==============================================================

if __name__ == "__main__":
    try:
        gustify()
    except KeyboardInterrupt:
        sys.exit("\nInterrupted by user.")
