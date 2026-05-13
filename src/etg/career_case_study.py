# src/etg/career_case_study.py
"""MISQ Design Science Case Study: DGT semantic shift tracking and forecasting.

Produces two outputs:
  1. A curated shift-trajectory table for 6 recognisable cybersecurity terms
     across all 12 spells, with annotated real-world event context.
  2. A watchlist of top-predicted shifting terms for the next spell,
     filtered to substantive security vocabulary.

Public API
----------
run_case_study(emb_dict, predictions_df, output_dir, figure=True)
    -> dict with keys 'shift_trajectories', 'watchlist', figure path.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ── Spell metadata ──────────────────────────────────────────────────────────

SPELL_META: dict[int, dict] = {
    1:  {"period": "Jan 2016 – Nov 2016", "label": "S01"},
    2:  {"period": "Nov 2016 – Sep 2017", "label": "S02"},
    3:  {"period": "Sep 2017 – Jul 2018", "label": "S03"},
    4:  {"period": "Jul 2018 – Jun 2019", "label": "S04"},
    5:  {"period": "Jun 2019 – Apr 2020", "label": "S05"},
    6:  {"period": "Apr 2020 – Feb 2021", "label": "S06"},
    7:  {"period": "Feb 2021 – Jan 2022", "label": "S07"},
    8:  {"period": "Jan 2022 – Nov 2022", "label": "S08"},
    9:  {"period": "Nov 2022 – Sep 2023", "label": "S09"},
    10: {"period": "Sep 2023 – Aug 2024", "label": "S10"},
    11: {"period": "Aug 2024 – Jun 2025", "label": "S11"},
    12: {"period": "Jun 2025 – Apr 2026", "label": "S12"},
}

# ── Curated case-study terms ────────────────────────────────────────────────

CURATED_TERMS: list[dict] = [
    {
        "term": "iot",
        "display": "iot",
        "narrative": (
            "IoT security enters hacker discourse. The Mirai botnet (Oct 2016) "
            "weaponised hundreds of thousands of unsecured IoT devices for a "
            "record-breaking DDoS attack, driving an abrupt vocabulary shift as "
            "the community re-contextualised IoT devices from consumer products "
            "to attack infrastructure."
        ),
        "peak_spell": 2,
        "peak_event": "Mirai botnet DDoS (Oct 2016)",
    },
    {
        "term": "rce",
        "display": "rce",
        "narrative": (
            "Remote-code-execution discourse surges. EternalBlue and WannaCry "
            "(May 2017) placed RCE at the centre of community discussion, "
            "shifting 'rce' from a technical abbreviation used among researchers "
            "to a mainstream shorthand for the dominant exploit class of the era."
        ),
        "peak_spell": 3,
        "peak_event": "WannaCry / EternalBlue (May 2017)",
    },
    {
        "term": "ransomware",
        "display": "ransomware",
        "narrative": (
            "The Ransomware-as-a-Service model transforms community understanding. "
            "GandCrab (2018) and REvil (2019) pioneered the affiliate RaaS model, "
            "fundamentally changing how ransomware was discussed — from malware "
            "campaigns to a criminal business ecosystem with operators, affiliates, "
            "and negotiation protocols."
        ),
        "peak_spell": 5,
        "peak_event": "GandCrab / REvil RaaS emergence (2018–2019)",
    },
    {
        "term": "stealer",
        "display": "stealer",
        "narrative": (
            "Info-stealer malware rises as a distinct market category. "
            "RedLine Stealer (2020) and Raccoon Stealer catalysed an "
            "underground market for stolen credentials, cookies, and "
            "cryptocurrency wallets. 'Stealer' shifted from a generic term "
            "to a recognised product category with versioned releases and "
            "subscription pricing on dark-web forums."
        ),
        "peak_spell": 7,
        "peak_event": "RedLine / Raccoon Stealer market surge (2020–2021)",
    },
    {
        "term": "ivanti",
        "display": "ivanti",
        "narrative": (
            "Ivanti becomes synonymous with critical zero-day exploitation. "
            "CVE-2023-46805 and CVE-2024-21887 (Ivanti Connect Secure) were "
            "exploited by nation-state actors before patches were available. "
            "The term 'ivanti' shifted rapidly from a vendor name to a shared "
            "shorthand for enterprise VPN zero-days, appearing in threat-actor "
            "posts alongside exploitation proof-of-concept code."
        ),
        "peak_spell": 10,
        "peak_event": "Ivanti Connect Secure zero-days (2023–2024)",
    },
    {
        "term": "jailbreak",
        "display": "jailbreak",
        "narrative": (
            "Semantic drift from mobile to AI: the clearest example of "
            "vocabulary repurposing in the corpus. Through spells 1–9, "
            "'jailbreak' referred primarily to iOS and Android firmware "
            "bypass. From spell 10 onward, the community's usage shifted "
            "to LLM safety-filter circumvention — the same cognitive frame "
            "(bypassing a vendor's intended restrictions) applied to a new "
            "class of system. DGT detected the drift in real time as "
            "ChatGPT and GPT-4 entered the threat-actor toolkit."
        ),
        "peak_spell": 11,
        "peak_event": "LLM jailbreaking surge post-ChatGPT (2023–2024)",
    },
]

# ── Prediction watchlist filter ──────────────────────────────────────────────

# Terms to skip: noise, Turkish fragments, usernames, domain-like strings
_NOISE_PATTERNS = [
    r"^\d+",          # starts with digit
    r"\.",            # contains dot (domain-like)
    r"^[a-f0-9]{6,}$",  # hex hash
    r"(dosya|veya|ekli|yorum|olan|ndan|inde|ederim|sunar|rkiye|yapma|yapmak|"
    r"sorun|destek|fatih|yabgu|zdanlar|sati|eline|konuyu|kurulum|ileti|telif|"
    r"anlamda|genel|zamanda|arkaday|rkda)",  # Turkish fragments
]

# Curated security interpretations for top watchlist terms
_WATCHLIST_ANNOTATIONS: dict[str, str] = {
    # AI / LLM security
    "jailbreak":         "LLM safety-bypass techniques — semantic drift iOS→AI continues",
    "prompt":            "Prompt injection and AI input manipulation",
    "generat":           "AI-generated malware and social-engineering content",
    # Access control / auth
    "authorization":     "Broken access-control vulnerabilities (OWASP A01)",
    "low-privileg":      "Local privilege-escalation exploit chains",
    "authenticator":     "MFA bypass and authenticator-app phishing",
    "contributor":       "Role-confusion supply-chain attacks (e.g. package maintainer hijack)",
    "contributor-level": "Role-confusion supply-chain attacks (e.g. package maintainer hijack)",
    # Memory safety
    "specially-craft":   "Memory corruption via attacker-controlled input",
    "refcount":          "Reference-count bugs leading to use-after-free / UAF",
    "over-read":         "Out-of-bounds read (information disclosure, ASLR bypass)",
    "taint":             "Taint-tracking / data-flow exploit primitives",
    # Credential / identity crime
    "darkweb":           "Dark-web marketplace activity and data brokering",
    "stealer":           "Info-stealer credential and session-cookie harvesting",
    "bruteforce":        "Credential-stuffing and password-spray campaigns",
    "cashout":           "Financial fraud cashout and money-mule infrastructure",
    "onlyfake":          "AI-generated identity fraud and synthetic document forgery",
    "passport":          "Identity-document fraud and KYC bypass",
    # Network / infra
    "botnet":            "Botnet C2 infrastructure and recruitment",
    "downloader":        "Dropper / loader malware (initial-access broker stage)",
    "miner":             "Cryptomining malware and container escape for mining",
    "subdomain":         "Subdomain takeover for phishing and C2 hosting",
    "outbound":          "Outbound-traffic exfiltration and C2 beacon detection evasion",
    # Vendor / product
    "ivanti":            "Continued Ivanti / enterprise VPN zero-day exploitation",
    "pan-o":             "Palo Alto Networks PAN-OS critical vulnerability discourse",
    "chromium":          "Browser-engine security: V8 / Chromium CVEs",
    "woocommerce":       "WordPress / WooCommerce plugin supply-chain vulnerabilities",
    "elementor":         "WordPress Elementor plugin RCE and XSS vulnerability pattern",
    "magento":           "E-commerce platform skimming and Magento injection",
    "mcafee":            "Endpoint-security driver exploitation (kernel privilege)",
    "irfanview":         "Image-parser CVEs (memory-corruption via crafted files)",
    "juniper":           "Juniper Networks OS vulnerability discourse",
    "aruba":             "Aruba Networks / HP networking device exploitation",
    "xwiki":             "XWiki/wiki-platform SSTI and RCE vulnerability pattern",
    "sequoia":           "macOS Sequoia / Apple platform security discourse",
    # Web / app
    "dom-bas":           "DOM-based XSS and client-side injection",
    "shortcode":         "WordPress shortcode injection — server-side code execution",
    "nonce":             "WordPress nonce bypass enabling CSRF and state changes",
    "affiliate":         "RaaS affiliate programme recruitment and operations",
    "penetration":       "Red-team / penetration-testing tool discourse surge",
    # Red-team tooling
    "specially-craft":   "Memory corruption via attacker-controlled input (CVE pattern)",
}


# ── helpers ─────────────────────────────────────────────────────────────────

def _cosine_shifts(emb_dict: dict, term: str, n_spells: int = 12) -> list[float]:
    """Return per-transition cosine distances for *term* across all spell pairs."""
    word_list = emb_dict["words"].tolist()
    if term not in word_list:
        return [float("nan")] * (n_spells - 1)
    idx = word_list.index(term)
    shifts = []
    for t in range(1, n_spells):
        prev = emb_dict[f"G_{t:02d}"][idx].astype(np.float64)
        curr = emb_dict[f"G_{t+1:02d}"][idx].astype(np.float64)
        pn = np.linalg.norm(prev)
        cn = np.linalg.norm(curr)
        if pn < 1e-8 or cn < 1e-8:
            shifts.append(float("nan"))
        else:
            shifts.append(float(np.clip(1.0 - np.dot(prev / pn, curr / cn), 0.0, 2.0)))
    return shifts


def _is_noise(term) -> bool:
    if not isinstance(term, str):
        return True
    for pat in _NOISE_PATTERNS:
        if re.search(pat, term, re.I):
            return True
    return len(term) < 4


def _spark(values: list[float], chars: str = "▁▂▃▄▅▆▇█") -> str:
    """Return a sparkline string for a list of floats, ignoring NaN."""
    valid = [v for v in values if not (v != v)]
    if not valid:
        return "?" * len(values)
    hi = max(valid) or 1.0
    out = []
    for v in values:
        if v != v:  # nan
            out.append("·")
        else:
            out.append(chars[min(int(v / hi * (len(chars) - 1)), len(chars) - 1)])
    return "".join(out)


# ── main public function ─────────────────────────────────────────────────────

def run_case_study(
    emb_dict: dict[str, np.ndarray],
    predictions_df: pd.DataFrame,
    output_dir: Path,
    n_spells: int = 12,
    watchlist_top_n: int = 15,
    annotated_only: bool = True,
    figure: bool = True,
) -> dict:
    """Generate the MISQ case-study outputs.

    Parameters
    ----------
    emb_dict        : loaded from dgt_embeddings.npz — keys 'words', 'G_01'..'G_12'.
    predictions_df  : loaded from dgt_shift_predictions.csv — columns
                      'word', 'predicted_next_shift', 'heldout_shift', 'abs_error'.
    output_dir      : directory to write JSON, CSV, and optional PNG outputs.
    n_spells        : number of DGT spells (default 12).
    watchlist_top_n : number of terms to include in the watchlist.
    annotated_only  : if True (default), only include watchlist terms that have a
                      curated security interpretation in _WATCHLIST_ANNOTATIONS,
                      suppressing generic forecast noise (e.g. plugin fragments,
                      Turkish tokens that slipped past _is_noise).
    figure          : whether to produce a matplotlib PNG.

    Returns
    -------
    dict with keys:
      'shift_trajectories': list of per-term dicts,
      'watchlist': list of top watchlist term dicts,
      'figure_path': path to PNG or None.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Shift trajectories for curated terms ───────────────────────────
    trajectories = []
    for term_meta in CURATED_TERMS:
        term = term_meta["term"]
        shifts = _cosine_shifts(emb_dict, term, n_spells)
        peak_idx = int(np.nanargmax(shifts))
        peak_shift = shifts[peak_idx]
        trajectories.append({
            "term": term,
            "display": term_meta["display"],
            "narrative": term_meta["narrative"],
            "peak_event": term_meta["peak_event"],
            "peak_spell_transition": f"S{peak_idx+1:02d}→S{peak_idx+2:02d}",
            "peak_cosine_shift": round(peak_shift, 4),
            "per_spell_shifts": {
                f"S{i+1:02d}→S{i+2:02d}": round(s, 4)
                for i, s in enumerate(shifts)
            },
            "sparkline": _spark(shifts),
        })
        log.info(
            "Case study term '%s': peak %.3f at S%02d→S%02d  spark=%s",
            term, peak_shift, peak_idx + 1, peak_idx + 2, _spark(shifts),
        )

    # ── 2. Prediction watchlist ───────────────────────────────────────────
    preds = predictions_df.copy()
    preds = preds[~preds["word"].apply(_is_noise)]
    preds = preds.sort_values("predicted_next_shift", ascending=False)

    watchlist_rows = []
    for _, row in preds.iterrows():
        word = row["word"]
        if annotated_only and word not in _WATCHLIST_ANNOTATIONS:
            continue
        annotation = _watchlist_annotations_lookup(word)
        watchlist_rows.append({
            "term": word,
            "predicted_shift": round(float(row["predicted_next_shift"]), 4),
            "recent_shift": round(float(row["heldout_shift"]), 4),
            "interpretation": annotation,
        })
        if len(watchlist_rows) >= watchlist_top_n:
            break

    log.info("Watchlist: %d terms (top predicted next-spell shift)", len(watchlist_rows))

    # ── 3. Persist JSON ───────────────────────────────────────────────────
    result = {
        "shift_trajectories": trajectories,
        "watchlist": watchlist_rows,
    }
    (output_dir / "case_study_shift_trajectories.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )

    # ── 4. Persist CSV summary tables ─────────────────────────────────────
    traj_rows = []
    for t in trajectories:
        traj_rows.append({
            "term": t["term"],
            "peak_transition": t["peak_spell_transition"],
            "peak_shift": t["peak_cosine_shift"],
            "peak_event": t["peak_event"],
            "sparkline": t["sparkline"],
        })
    pd.DataFrame(traj_rows).to_csv(
        output_dir / "case_study_trajectories.csv", index=False
    )
    pd.DataFrame(watchlist_rows).to_csv(
        output_dir / "case_study_watchlist.csv", index=False
    )

    # ── 5. Optional figure ────────────────────────────────────────────────
    figure_path = None
    if figure:
        figure_path = _make_figure(trajectories, output_dir, n_spells)
        result["figure_path"] = str(figure_path)

    return result


def _watchlist_annotations_lookup(word: str) -> str:
    """Return a security interpretation for *word*, or a generic label."""
    if word in _WATCHLIST_ANNOTATIONS:
        return _WATCHLIST_ANNOTATIONS[word]
    # Generic fallback
    return "Emerging exploit or threat discourse — monitor for context"


def _make_figure(
    trajectories: list[dict],
    output_dir: Path,
    n_spells: int,
) -> Path:
    """Produce a multi-panel line chart of shift trajectories."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
    except ImportError:
        log.warning("matplotlib not available — skipping figure")
        return None

    n = len(trajectories)
    fig, axes = plt.subplots(n, 1, figsize=(10, 2.2 * n), sharex=True)
    if n == 1:
        axes = [axes]

    spell_labels = [f"S{i+1:02d}→S{i+2:02d}" for i in range(n_spells - 1)]
    x = list(range(len(spell_labels)))

    COLORS = ["#4C78A8", "#F58518", "#E45756", "#72B7B2", "#54A24B", "#B279A2"]

    for ax, traj, color in zip(axes, trajectories, COLORS):
        shifts = list(traj["per_spell_shifts"].values())
        valid_x = [xi for xi, s in zip(x, shifts) if s == s]
        valid_y = [s for s in shifts if s == s]

        ax.fill_between(valid_x, valid_y, alpha=0.15, color=color)
        ax.plot(valid_x, valid_y, color=color, linewidth=2, marker="o",
                markersize=4, label=traj["display"])

        # Mark peak
        peak_xi = int(np.nanargmax(shifts))
        ax.axvline(x=peak_xi, color=color, linewidth=1, linestyle="--", alpha=0.6)
        ax.annotate(
            traj["peak_event"],
            xy=(peak_xi, shifts[peak_xi]),
            xytext=(min(peak_xi + 0.3, len(x) - 3), shifts[peak_xi] * 0.7 + 0.1),
            fontsize=7.5,
            color=color,
            arrowprops=dict(arrowstyle="-", color=color, lw=0.8),
        )

        ax.set_ylabel(f"'{traj['display']}'\ncosine shift", fontsize=8)
        ax.set_ylim(bottom=0)
        ax.yaxis.set_major_locator(ticker.MaxNLocator(3))
        ax.tick_params(axis="y", labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(spell_labels, rotation=35, ha="right", fontsize=7.5)
    axes[-1].set_xlabel("Spell transition", fontsize=9)

    fig.suptitle(
        "DGT Semantic Shift Trajectories — HackerSignal Corpus (2016–2026)",
        fontsize=11, y=1.01,
    )
    fig.tight_layout()

    path = output_dir / "case_study_shift_trajectories.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("Figure saved to %s", path)
    return path
