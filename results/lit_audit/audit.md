# Literature audit: is the per-call generation cap reported?

Method. PDFs downloaded from arXiv / ACL Anthology, text extracted with `pdftotext -layout`
(files in `pdf/`). Two regex passes over the full text including appendices: phrases for
generation limits (`hits_*.txt`) and number-plus-token patterns (`num_*.txt`); then every hit
and every passage stating decoding settings (temperature, vLLM, greedy) was read by hand.
Line numbers below refer to `pdf/<name>.txt`.

Scope. A convenience sample: the papers cited in our related work that evaluate agent or
multi-agent systems under a token or cost criterion. Not a systematic review. SRR (2609.07786)
is excluded: it scores multiple-choice answers from next-token logits and generates nothing.

Q1 = the paper states the per-call generation cap used **at evaluation**.
Q2 = the paper states that compared systems share that cap.

| Paper | Multi-agent? | Q1 | Q2 | Evidence |
|---|---|---|---|---|
| AgentPrune (2410.02506) | yes | **no** | no | §Implementation details gives model, temperature 1, rounds K; no generation limit (586) |
| AgentDropout (ACL 2025) | yes | **no** | no | temperature 1, sample count, dropout rates; no generation limit (448) |
| Optima (2410.08115) | yes | **no** | no | no generation limit anywhere in text |
| Agent-GSPO (2510.22477) | yes | **no** | no | no generation limit anywhere in text |
| Gated-Memory Routing (2609.00237) | yes | **no** | no | only a 128k context window (1128); "small budgets truncate useful reasoning" (583) |
| Total Cost of Agency (2609.23790) | yes | **no** | no | decomposes billed output tokens but states no cap |
| CoRL (2511.02755) | yes | training only | no | "maximum response length of 500" in RL training setup (805); evaluation cap not stated |
| BAMAS (2511.21572) | yes | **yes, own system only** | **no** | per-agent caps 384 / 1,024 by dataset (Table 7, 673-693); baselines' caps not stated |
| Rethinking MAS Efficiency (2609.05933) | yes | yes | **yes** | "Every inference call uses max_tokens = 2048 as the per-call generation budget" (1362), made part of a shared protocol |
| ACON (2510.00615) | single agent | **no** | no | temperature 0, seed 42 (1142); no generation limit |
| MEM1 (2506.15841) | single agent | **no** | no | temperature 0.01 at inference, vLLM (898-900); no generation limit |
| Agent-Omit (2602.04284) | single agent | training only | no | "maximum response length is set to 32k tokens" during RL (428) |
| AdaTurn (2607.14547) | single agent | **no** | no | budgets are in turns; discusses turn-level "catastrophic truncation" but states no token cap |
| Inference-Time Budget Control (2605.05701) | single agent | **no** | no | total output-token budget B_tok (163); no per-call cap |
| TALE (ACL Findings 2025) | single call | **no** | no | budget is a prompt instruction; no generation limit stated |
| BATS (2511.17006) | single agent | yes | yes (global setting) | "maximum number of new tokens ... 65,536 for Gemini ... 64,000" (1052) |
| Context-Folding (2510.11967) | single agent | yes | yes | response length 65,536 (283); baseline with "same 327K max length" (342) |
| BAGEN (2606.00198) | single agent | yes | yes | Max Tokens 800 per model (Table 5, 1639-1654); 512-token cap at RL evaluation (1450) |
| Plan-and-Budget (2505.16122) | single call | yes | yes | "A hard cutoff of 8192 tokens is applied" to all models and baselines (505) |
| BudgetThinker (2508.17196) | single call | yes | yes | the cap is the independent variable (303-304) |
| Markovian Thinker (2510.06557) | single chain | yes | yes | chunk / total thinking budgets equal across compared runs (668) |

## Counts (21 papers)

| | all | multi-agent (9) |
|---|---|---|
| state the evaluation cap (Q1 yes) | 8 | 2 |
| of which only for their own system | 1 | 1 (BAMAS) |
| training cap only | 2 | 1 |
| no cap stated | 11 | 6 |
| state that compared systems share it (Q2) | 7 | 1 |

**Multi-agent papers: 7 of 9 do not state an evaluation-time per-call cap for the systems they
compare; only one (Rethinking MAS Efficiency) fixes it as a shared protocol.** The one other
that reports caps (BAMAS) sets them per agent role as part of its own configuration, including
a 1,024-token executor cap on MATH, and does not state the baselines'.

Papers that study reasoning length directly (BudgetThinker, Markovian Thinker, Plan-and-Budget)
all report and equalise the cap, which is consistent with our reading: the communities that
treat generation length as the object of study control it; those that treat the agent
architecture as the object of study mostly do not.

## Caveats to state in the paper
- Convenience sample of 21, drawn from our own related work.
- "Not stated" means not stated in the paper text. Released code may fix a cap; absent from the
  paper, a reader cannot tell whether compared systems shared it.
- Extraction is regex plus manual reading; a cap written in an image-only table would be missed.
