# LLM-Assisted Audit Reliability Metrics

**Important:** These are agreement statistics for two automated LLM judge prompts, not human interrater reliability.

## task1_exploit_relevance

| Metric | Estimate | 95% CI |
|---|---:|---:|
| Percent agreement | 0.9625 | [0.9303, 0.9801] |
| Cohen kappa | 0.9188 | [0.8622, 0.9648] |
| Krippendorff alpha nominal | 0.9188 | [0.8631, 0.9648] |
| Gwet AC1 | 0.9304 | [0.8821, 0.9706] |
| PABAK | 0.9250 | [0.8750, 0.9667] |
| Positive specific agreement | 0.9707 | [0.9498, 0.9875] |
| Negative specific agreement | 0.9480 | [0.9114, 0.9783] |
| Consensus vs benchmark agreement | 0.7532 | [0.6938, 0.8044] |

Confusion matrix, CTI triage rows by vulnerability-research columns:

{
  "0": {
    "0": 82,
    "1": 0
  },
  "1": {
    "0": 9,
    "1": 149
  }
}

## task2_cve_linkage

| Metric | Estimate | 95% CI |
|---|---:|---:|
| Percent agreement | 0.9708 | [0.9410, 0.9858] |
| Cohen kappa | 0.9417 | [0.8990, 0.9831] |
| Krippendorff alpha nominal | 0.9418 | [0.8919, 0.9832] |
| Gwet AC1 | 0.9417 | [0.9000, 0.9833] |
| PABAK | 0.9417 | [0.8917, 0.9833] |
| Positive specific agreement | 0.9705 | [0.9464, 0.9910] |
| Negative specific agreement | 0.9712 | [0.9478, 0.9913] |
| Consensus precision estimate | 0.4936 | [0.4300, 0.5573] |

Confusion matrix, CTI triage rows by vulnerability-research columns:

{
  "False": {
    "False": 118,
    "True": 4
  },
  "True": {
    "False": 3,
    "True": 115
  }
}
