import sys
p = sys.argv[1]; s = open(p).read()
GH = r"\href{https://github.com/BenAmpel/hackersignal}{\texttt{github.com/BenAmpel/hackersignal}}"
E = []
def rep(a, b, n=1):
    c = s.count(a)
    assert c == n, (c, a[:80])
    E.append((a, b))
# 1. links
rep(r"\href{https://huggingface.co/datasets/DatasetSubmission/HackerSignal}{\texttt{hf.co/datasets/DatasetSubmission/HackerSignal}}",
    r"\href{https://huggingface.co/datasets/BenAmpel/HackerSignal}{\texttt{hf.co/datasets/BenAmpel/HackerSignal}}", n=2)
rep(r"\href{https://anonymous.4open.science/r/hackersignal-580F/}{\texttt{anonymous.4open.science/r/hackersignal-580F}}", GH, n=s.count(r"\href{https://anonymous.4open.science/r/hackersignal-580F/}{\texttt{anonymous.4open.science/r/hackersignal-580F}}"))
# 2. ladder descriptions
rep(r"recurrent neural networks (RNN, GRU, LSTM, BiLSTM), a domain-specific transformer (SecBERT), and hybrid sparse+dense retrieval",
    r"recurrent neural networks (RNN, GRU, LSTM, BiLSTM), three cybersecurity-domain transformers (SecBERT, SecureBERT, CySecBERT), and hybrid sparse+dense retrieval")
rep(r"recurrent neural networks (RNN, GRU, LSTM, BiLSTM), and a domain-specific transformer (SecBERT~\citep{secbert2022}). We report macro-F1 and per-class F1 for the three hardest classes in Table~\ref{tab:task2-results}.",
    r"recurrent neural networks (RNN, GRU, LSTM, BiLSTM), and three cybersecurity-domain transformers (SecBERT~\citep{secbert2022}, SecureBERT~\citep{aghaei2023securebert}, and CySecBERT~\citep{bayer2024cysecbert}). We report macro-F1 (mean $\pm$ SD over three seeds) and per-class F1 for the three hardest classes in Table~\ref{tab:task2-results}.")
rep(r"and eight classification approaches (Task~2)", r"and ten classification approaches (Task~2)")
rep(r"recurrent neural networks (RNN, GRU, LSTM, BiLSTM), and a domain-specific transformer (SecBERT).",
    r"recurrent neural networks (RNN, GRU, LSTM, BiLSTM), and three domain-specific transformers (SecBERT, SecureBERT, CySecBERT).")
# 3. Table 5
old_t = s[s.index(r"\caption{Task 2 (ETC) test results on 1,735"):s.index(r"\end{tabular}", s.index(r"\caption{Task 2 (ETC) test results on 1,735"))]
new_t = r"""\caption{Task 2 (ETC) test results on 1,735 temporally held-out samples. Macro-F1 is the mean $\pm$ SD over three seeds; per-class F1 for the three hardest classes is the seed mean. Bootstrap 95\% intervals on macro-F1 (2,000 resamples) have a half-width of about 0.02; the top two models do not differ significantly. BoW = bag-of-words features.}
\label{tab:task2-results}
\begin{tabular}{llcccc}
\toprule
& \textbf{Model} & \textbf{Macro-F1} & \textbf{info\_disc.} & \textbf{mem\_corr.} & \textbf{auth.} \\
\midrule
\multirow{3}{*}{\rotatebox[origin=c]{90}{\footnotesize BoW}}
& Decision Tree     & .702 $\pm$ .007 & .45 & .64 & .60 \\
& TF-IDF + LR       & .772 $\pm$ .000 & .66 & .69 & .79 \\
& SVM (Linear)      & .799 $\pm$ .000 & .70 & .68 & .82 \\
\midrule
\multirow{4}{*}{\rotatebox[origin=c]{90}{\footnotesize Recurrent}}
& RNN               & .436 $\pm$ .031 & .30 & .42 & .44 \\
& GRU               & .858 $\pm$ .001 & .70 & .84 & .81 \\
& LSTM              & .824 $\pm$ .013 & .63 & .77 & .78 \\
& BiLSTM            & .856 $\pm$ .003 & .69 & .84 & .81 \\
\midrule
\multirow{3}{*}{\rotatebox[origin=c]{90}{\footnotesize Transf.}}
& SecBERT           & .852 $\pm$ .007 & .67 & .82 & .83 \\
& SecureBERT        & .892 $\pm$ .008 & \textbf{.80} & .90 & .85 \\
& CySecBERT         & \textbf{.898 $\pm$ .003} & .79 & \textbf{.90} & \textbf{.86} \\
\bottomrule
"""
E.append((old_t, new_t))
# 4. Task 2 paragraph
old_p = s[s.index("The best bag-of-words baseline (SVM) attains macro-F1 of 0.799"):s.index("preventing models from memorizing era-specific features.") + len("preventing models from memorizing era-specific features.")]
new_p = (r"The best bag-of-words baseline (SVM) attains macro-F1 of 0.799. Gated recurrent models improve on it: GRU (0.858) and BiLSTM (0.856) are level, and both exceed LSTM (0.824; GRU$-$LSTM paired $\Delta$ 0.029, 95\% CI [0.016, 0.043]). "
         r"The vanilla RNN is far weaker (0.436): 32\% of test exploits exceed the 300-token window, and without gating the signal from early tokens fades over long sequences. "
         r"Domain pre-training helps most. CySecBERT (0.898) and SecureBERT (0.892) are the strongest models, and their difference is not significant (paired $\Delta$ 0.006, 95\% CI [$-$0.004, 0.016]), while SecBERT (0.852) is level with the best recurrent models. "
         r"The hard classes (information disclosure, memory corruption, authentication/access) have substantial overlap. For example, disclosure exploits often involve injection (SSRF, XXE), while privilege escalation and authentication share patterns with RCE. Decision Tree drops to macro-F1=0.702, with disclosure at 0.45. "
         r"Lexical classes such as XSS and injection are easier (mean F1 0.94 and 0.86 across models), indicating that exploit signals are partly lexical. Temporal splits make it harder as exploit names and vulnerability patterns evolve, preventing models from memorizing era-specific features.")
E.append((old_p, new_p))
# 5. Task 3 delta
rep(r"(paired $\Delta$MRR 0.005, 95\% CI [$-$0.011, 0.020])", r"(paired $\Delta$MRR 0.004, 95\% CI [$-$0.012, 0.020])")
# 6. conclusion
rep(r"BiLSTM at macro-F1~0.87 for 8-class exploit type classification", r"domain-pretrained transformers at macro-F1~0.90 for 8-class exploit type classification")
# 7. Table 3 caption (D2)
rep(r"Tasks~1 and~3 use temporal splits (train $<$2022, test 2024+) over the full 340K NVD corpus;",
    r"Task~3 splits by CVE publication year (train $<$2022, val 2022--2023, test 2024+) and Task~1 uses the same test queries by source date (92 of its 1,990 test queries link to CVEs published before 2024), both over the full 340K NVD corpus;")
rep(r"$^\dagger$Same corpus; zero CVE overlap between train/val/test gold sets.", r"$^\dagger$Same corpus; zero CVE overlap between Task~3 train/val/test gold sets.")
# 8. App C (D1)
rep(r"Finally, we verify that no CVE identifier appears in both the train and test for any task. For Task~3 (temporal generalization), CVE-disjointness is enforced by construction: $|C_{\text{train}} \cap C_{\text{test}}| = 0$, supported programmatically at build time.",
    r"Finally, we check gold-CVE overlap between splits. For Task~3 (temporal generalization), CVE-disjointness is enforced by construction: $|C_{\text{train}} \cap C_{\text{test}}| = 0$, supported programmatically at build time. Task~1 does not impose disjointness; 9 of its 1,876 test CVEs (9 of 1,990 test queries) also occur in training.")
# 9. App D dense, BoW, neural, compute
rep(r"Batch size 128, max sequence length 512 tokens, $\ell_2$-normalized embeddings, cosine similarity scoring. Corpus encoded in 5K-document chunks on CPU with multi-process pooling.",
    r"Inputs are the first 384 characters of each query and document, max sequence length 512 tokens, batch size 128, $\ell_2$-normalized fp32 embeddings, cosine similarity scoring; BGE is used without its query instruction. MRR is computed over the top 10 (MRR@10), with ties broken by corpus order. Two settings matter: encoding full texts instead of the first 384 characters raises Task~1 MRR for the base-size models by about 0.03 (e.g., BGE 0.744, E5 0.741) and lowers MiniLM by 0.025, and adding BGE's query instruction raises its MRR to 0.725 (Task~1) and 0.737 (Task~3). E5's prefixes change MRR by less than 0.001.")
rep(r"as the feature extractor. \texttt{DecisionTreeClassifier}", r"as the feature extractor, applied to the first 2,000 characters of each text. \texttt{DecisionTreeClassifier}")
rep(r"RNN, GRU, LSTM, and BiLSTM use a word-level vocabulary (min frequency $\geq$3, yielding ${\sim}$19K tokens) with 128-dim embeddings, 128-dim hidden states, max sequence length 300, batch size 128, and Adam optimizer (lr=$10^{-3}$). Class-weighted cross-entropy loss with gradient clipping at 1.0. Trained for 8 epochs with best checkpoint selected by validation macro-F1. SecBERT~\citep{secbert2022} uses \texttt{jackaduma/SecBERT} with max sequence length 256, batch size 16, AdamW (lr=$2{\times}10^{-5}$, weight decay 0.01), linear warmup over 10\% of steps, trained for 3 epochs on a 15K subsample. All neural models use MPS (Apple Silicon) acceleration.",
    r"RNN, GRU, LSTM, and BiLSTM use a word-level vocabulary (min frequency $\geq$3, yielding 175,062 tokens) with 128-dim embeddings, 128-dim hidden states, max sequence length 300 (right-padded), batch size 128, and Adam optimizer (lr=$10^{-3}$). The classifier reads the hidden state at the last non-padding token (both directions for BiLSTM, via packed sequences). Class-weighted cross-entropy loss with gradient clipping at 1.0. Trained for 8 epochs with best checkpoint selected by validation macro-F1. "
    r"SecBERT~\citep{secbert2022} (\texttt{jackaduma/SecBERT}), SecureBERT~\citep{aghaei2023securebert} (\texttt{ehsanaghaei/SecureBERT}), and CySecBERT~\citep{bayer2024cysecbert} (\texttt{markusbayer/CySecBERT}) are fine-tuned on the first 1,000 characters of each text with max sequence length 256, batch size 16, AdamW (lr=$2{\times}10^{-5}$, weight decay 0.01), linear warmup over 10\% of steps, and unweighted cross-entropy for 3 epochs on a 15K training subsample; the epoch is selected by validation macro-F1. "
    r"All Task~2 models are run with seeds 42, 43, and 44 (the 15K subsample is drawn per seed); we report the mean and SD. Fine-tuning on all 64,413 training rows is reported in the released results.")
old_c = s[s.index(r"All baselines ran on a single macOS arm64 workstation (Apple M-series, 16\, GB RAM)"):s.index(r"All baseline code is available at", s.index(r"All baselines ran on a single macOS arm64 workstation"))]
new_c = (r"All baselines ran on one Apple M5 Pro workstation (48\,GB unified memory, macOS 27) with Python 3.12.15, PyTorch 2.14.1 (MPS backend), sentence-transformers 6.2.0, transformers 5.21.0, and scikit-learn 1.9.1; no CUDA GPUs were used. "
         r"BM25 uses an exact sparse reimplementation of \texttt{rank\_bm25} (maximum score difference $3{\times}10^{-12}$ on a 50-query check): 5--8\,s to index the 340K corpus and 15--25\,s to score each test set. "
         r"Dense encoding of the 340K corpus took ${\sim}$4\,min (MiniLM) to ${\sim}$30\,min (base-sized models). Task~2 training per seed: BoW 7--40\,s; RNN/GRU/LSTM 5--36\,min; BiLSTM ${\sim}$60\,min; SecBERT ${\sim}$17\,min and SecureBERT/CySecBERT ${\sim}$40--45\,min on the 15K subsample. Per-run timings, seeds, model revisions, and data hashes are in \texttt{results/camera\_ready/provenance.json}. ")
E.append((old_c, new_c))
# 10. checklist
rep(r"\item[] Answer: \answerNo{}" + "\n" + r"    \item[] Justification: No error bars or significance tests are reported.",
    r"\item[] Answer: \answerYes{}" + "\n" + r"    \item[] Justification: Tables~\ref{tab:task1-results}--\ref{tab:task3-results} report test-set bootstrap 95\% intervals (10,000 resamples for MRR, 2,000 for macro-F1) and paired bootstrap tests for the top two models; Task~2 reports the mean $\pm$ SD over three seeds.")
old_j = s[s.index(r"Retrieval baselines are fully deterministic") if r"Retrieval baselines are fully deterministic" in s else s.index(r"BM25 and dense-retrieval baselines are fully deterministic"):]
old_j = old_j[:old_j.index("\n")]
E.append((old_j, r"BM25 and dense-retrieval baselines are deterministic given a fixed corpus and queries (ties are broken by corpus order). MPS training is not bit-deterministic, which the seed SD captures."))
rep(r"Baselines were run on a single macOS arm64 workstation with Python 3.12 and PyTorch 2.x. Appendix~\ref{app:hyperparams} reports wall-clock times per model and task.",
    r"Baselines were run on one Apple M5 Pro workstation (48\,GB) with PyTorch 2.14.1 on the MPS backend. Appendix~\ref{app:hyperparams} reports wall-clock times per model and task.")
# 11. bibitems
rep(r"\bibitem[Jackaduma(2022)]{secbert2022}", r"""\bibitem[Aghaei et~al.(2023)]{aghaei2023securebert}
Aghaei, E., Niu, X., Shadid, W., \& Al-Shaer, E. (2023). SecureBERT: A Domain-Specific Language Model for Cybersecurity. In \textit{Security and Privacy in Communication Networks (SecureComm 2022)}, pp. 39--56. Springer. \url{https://doi.org/10.1007/978-3-031-25538-0_3}.

\bibitem[Bayer et~al.(2024)]{bayer2024cysecbert}
Bayer, M., Kuehn, P., Shanehsaz, R., \& Reuter, C. (2024). CySecBERT: A Domain-Adapted Language Model for the Cybersecurity Domain. \textit{ACM Transactions on Privacy and Security}, 27(2), 1--20. \url{https://doi.org/10.1145/3652594}.

\bibitem[Jackaduma(2022)]{secbert2022}""")
for a, b in E:
    assert s.count(a) >= 1, a[:80]
    s = s.replace(a, b)
open(p, "w").write(s)
print(f"{len(E)} edits applied to proposed copy")

# ---- second pass: Tables 4/6 and dependent text from results.csv (audit values) ----
s = open(p).read()
T = [
 ("BM25 & 0.594 & 0.690 & 0.716 & 0.636 \\\\", "BM25 & 0.595 & 0.688 & 0.715 & 0.636 \\\\"),
 ("MiniLM-L6-v2 & 0.656 & 0.776 & 0.809 & 0.709 \\\\", "MiniLM-L6-v2 & 0.655 & 0.776 & 0.809 & 0.709 \\\\"),
 ("mpnet-base-v2 & 0.626 & 0.756 & 0.792 & 0.683 \\\\", "mpnet-base-v2 & 0.626 & 0.756 & 0.792 & 0.682 \\\\"),
 ("BGE-base-v1.5 & 0.654 & \\textbf{0.789} & 0.822 & \\textbf{0.714} \\\\", "BGE-base-v1.5 & 0.653 & \\textbf{0.789} & 0.823 & \\textbf{0.714} \\\\"),
 ("E5-base-v2 & 0.649 & \\textbf{0.789} & \\textbf{0.825} & 0.710 \\\\", "E5-base-v2 & 0.649 & \\textbf{0.789} & \\textbf{0.824} & 0.710 \\\\"),
 ("Hybrid BM25+mpnet & \\textbf{0.678} & 0.755 & 0.770 & 0.712 \\\\", "Hybrid BM25+mpnet & \\textbf{0.679} & 0.753 & 0.768 & 0.711 \\\\"),
 ("BM25 & 0.619 & 0.714 & 0.741 & 0.660 \\\\", "BM25 & 0.620 & 0.713 & 0.740 & 0.660 \\\\"),
 ("MiniLM-L6-v2 & 0.671 & 0.788 & 0.820 & 0.723 \\\\", "MiniLM-L6-v2 & 0.670 & 0.788 & 0.820 & 0.722 \\\\"),
 ("mpnet-base-v2 & 0.647 & 0.777 & 0.810 & 0.704 \\\\", "mpnet-base-v2 & 0.647 & 0.777 & 0.809 & 0.703 \\\\"),
 ("BGE-base-v1.5 & 0.668 & \\textbf{0.799} & 0.828 & 0.726 \\\\", "BGE-base-v1.5 & 0.667 & 0.798 & 0.828 & 0.726 \\\\"),
 ("E5-base-v2 & 0.663 & 0.798 & \\textbf{0.831} & 0.721 \\\\", "E5-base-v2 & 0.663 & \\textbf{0.799} & \\textbf{0.830} & 0.721 \\\\"),
 ("Hybrid BM25+mpnet & \\textbf{0.697} & 0.774 & 0.788 & \\textbf{0.730} \\\\", "Hybrid BM25+mpnet & \\textbf{0.698} & 0.772 & 0.786 & \\textbf{0.730} \\\\"),
 ("cluster tightly (MRR 0.683--0.714)", "cluster tightly (MRR 0.682--0.714)"),
 ("but its difference from the hybrid (0.712) is not significant", "but its difference from the hybrid (0.711) is not significant"),
 ("the hybrid leads at R@1 (0.678)", "the hybrid leads at R@1 (0.679)"),
 ("Bootstrap 95\\% intervals on MRR (10,000 resamples) have a half-width of about 0.02;", "Bootstrap 95\\% intervals on MRR@10 (10,000 resamples) have a half-width of about 0.02;"),
]
for a, b in T:
    assert s.count(a) == 1, (s.count(a), a)
    s = s.replace(a, b)
open(p, "w").write(s)
print(f"{len(T)} table/text edits applied")
