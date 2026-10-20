# ARR Responsible NLP Checklist — draft answers (fill in the OpenReview form)

Section numbers refer to the current main.pdf. Items marked TODO need a decision or a fact check.

## A. For every submission
- **A1 Limitations?** Yes — section "Limitations" (after §6).
- **A2 Potential risks?** No — the method only decides when to call a more expensive pipeline of the same
  model; its guarantee bounds accuracy loss relative to that pipeline under an i.i.d. assumption, and the
  Limitations discuss when that assumption fails. (TODO: or add one sentence on misuse if you prefer "Yes".)

## B. Did you use or create scientific artifacts?  Yes
- **B1 Cite creators?** Yes — §5.1 (gpt-oss, Gemma 4, MATH-500, FRAMES, MuSiQue, BBEH), Appendix B.
- **B2 License / terms of use?** TODO — the paper does not state licenses. Either answer "No" with a
  justification, or add one line to Table 5 / Appendix B. Licenses to verify before writing them:
  MATH (MIT), FRAMES (Apache-2.0), MuSiQue (CC BY 4.0), BBEH (Apache-2.0), gpt-oss (Apache-2.0),
  Gemma 4 (check the model card's license).
- **B3 Use consistent with intended use?** Yes — all are research benchmarks / openly released models used
  for evaluation only. (Same place as B2 if a sentence is added.)
- **B4 Personal / offensive content?** No — public benchmarks; FRAMES uses Wikipedia articles.
- **B5 Documentation of artifacts?** Yes — Appendix B (splits, documents, Table 5).
- **B6 Statistics (splits, sizes)?** Yes — §5.1 and Table 5 (400 / 660 / 1,934 / 874 tasks, 20/80 dev/test).

## C. Did you run computational experiments?  Yes
- **C1 Parameters and compute budget?** Partly — token costs are reported throughout (§5, Appendix F
  calibration cost: 1.5M–79M tokens). TODO: GPU type/hours are no longer in the paper (serving details
  were removed); answer "Partly" with that justification, or add one sentence back.
- **C2 Experimental setup / hyperparameters?** Yes — §5.1, Appendix B (sampling, caps), Appendix C
  (threshold grid, δ, splits).
- **C3 Descriptive statistics?** Yes — means over 200 random splits, violation rates, i.i.d. check with
  upper 95% limit (§5.2, Appendix D).
- **C4 Existing packages?** Yes — math-verify (Appendix B), official BBEH evaluation. (vLLM is no longer
  named in the paper.)

## D. Human annotators / participants?  No
- Answer label validation was done by the authors (Appendix B "Scoring", Appendix F.7), not by recruited
  annotators, so D1–D5 are N/A.

## E. AI assistants in research or writing?
- TODO — answer truthfully according to ARR policy (e.g. coding and writing assistance); this is the
  authors' declaration.
