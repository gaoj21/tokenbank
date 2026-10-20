"""Single resolver for (family, arm, benchmark tag, budget k) -> trace file. Mirrors the
explicit map in _gen_materials.py; every analysis script should go through this rather than
format a pattern, because the Qwen governed 24k cells live under a different name."""
import os
GOV = {'Qwen':'V5.1-Ab7-NoVerbal','Gemma':'V5.1-Ab7-NoVerbal',
       'GLM':'V5.2-ReasonLadder','DeepSeek':'V5.2-ReasonLadder'}
def path(fam, arm, t, k):
    k = str(k)
    if fam == 'Qwen':
        if arm == 'V5-Accum-Only': return f'traces/tasks_sweep_Qwen3.8-27B_acc_{t}{k}k.jsonl'
        if arm in ('V5-Unconstrained','V5-Unc-Fmt'): return f'traces/tasks_sweep_Qwen3.8-27B_q38_{t}{k}.jsonl'
        for p in (f'traces/tasks_sweep_Qwen3.8-27B_q38ab7_{t}{k}.jsonl',
                  f'traces/tasks_sweep_Qwen3.8-27B_q38_ab7_{t}.jsonl' if k=='24' else ''):
            if p and os.path.exists(p): return p
        return None
    if fam == 'Gemma':
        if arm == 'V5-Accum-Only': return f'traces/tasks_sweep_gemma-4-31B-bf16_gcmp_{t}{k}.jsonl'
        return f'traces/tasks_sweep_gemma-4-31B-bf16_gor_{t}{k}.jsonl'
    if fam == 'GLM': return f'traces/tasks_sweep_GLM-5.3-Flash_glm4_{t}{k}.jsonl'
    if fam == 'DeepSeek': return f'traces/tasks_sweep_DeepSeek-V4.1-Flash_ds_{t}{k}.jsonl'
