"""semantic_entropy.py - does a local assistant know when it is guessing?

After the assistant drafts an answer, ask the same question a few more times and check whether the
answers MEAN the same thing. When they scatter, the model is likely confabulating, however fluent
each answer sounds. Follows the idea in Farquhar et al., "Detecting hallucinations in large language
models using semantic entropy" (Nature, 2024), with one judge call in place of pairwise entailment.

  check(prompt, system, answer, n=3) -> {"verdict": consistent|mixed|uncertain, "entropy": bits,
                                          "clusters": k, "mean_cos": x, "samples": [...]}
  - n extra samples from the local model at temperature 0.8;
  - the same model groups the answers by whether they make the same factual claims;
  - entropy over the group sizes (answer + samples); uncertain when entropy >= UNCERTAIN_BITS;
  - embedding-similarity grouping is only the fallback when the judge call fails.
Fail-soft: any error returns {"verdict": "unavailable"} and the caller keeps its answer unchanged.
Off switch: SE_ENABLED=0. Runs entirely on a local Ollama server; no network calls leave the machine.

`ollama_client` is the caller's own thin wrapper around a local Ollama server (generate + embed).
Pass your own `gen`, `embed` and `gen_judge` callables to use any other backend.
Run `python3 semantic_entropy.py` for the self-test (no model needed).
"""
from __future__ import annotations
import math, os, re

CLUSTER_COS = float(os.environ.get("SE_COS", "0.86"))
UNCERTAIN_BITS = float(os.environ.get("SE_BITS", "0.9"))
N = int(os.environ.get("SE_N", "3"))
SMALLTALK = re.compile(r"^\s*(hi|hey|hello|thanks|thank you|ok|okay|good (morning|night|evening)|yo|go|stop|undo)\b[\s!.?]*$", re.I)


def _cos(a, b):
    n = sum(x * y for x, y in zip(a, b)); da = math.sqrt(sum(x * x for x in a)); db = math.sqrt(sum(y * y for y in b))
    return n / (da * db) if da and db else 0.0


def _norm(t):
    t = re.sub(r"\((sure|fairly sure|not sure[^)]*)\)\s*$", "", str(t or ""), flags=re.I)
    return re.sub(r"\s+", " ", t).strip()


def clusters(vecs, thr=CLUSTER_COS):
    cl = []
    for i, v in enumerate(vecs):
        for c in cl:
            if _cos(v, vecs[c[0]]) >= thr:
                c.append(i); break
        else:
            cl.append([i])
    return cl


JUDGE = ("You check whether several answers to the same question make the SAME factual claims. Ignore wording, tone, "
         "confidence words like (sure), and extra suggestions about where to look. Two answers are the same if they assert the "
         "same facts (same numbers, names, yes/no, dates) or both say the information is not available. Return ONLY JSON: "
         "{\"groups\": [[answer numbers that agree], ...]} using 1-based answer numbers, every answer in exactly one group.")


def judge_groups(question, texts, gen_judge):
    """Meaning-equivalence grouping by the local model (the paper's bidirectional-entailment step, as one call)."""
    import json as _j
    listing = "\n".join("%d. %s" % (i + 1, t[:300]) for i, t in enumerate(texts))
    out = gen_judge("QUESTION: %s\n\nANSWERS:\n%s\n\nJSON:" % (question, listing))
    m = re.search(r"\{.*\}", out or "", re.S)
    if not m:
        return None
    g = _j.loads(m.group(0)).get("groups")
    flat = sorted(i for grp in g for i in grp)
    if flat != list(range(1, len(texts) + 1)):
        return None
    return [[i - 1 for i in grp] for grp in g]


def entropy(cl, total):
    return -sum((len(c) / total) * math.log2(len(c) / total) for c in cl if c)


def applies(question):
    return bool(question) and len(question.strip()) > 10 and not SMALLTALK.match(question)


def check(prompt, system, answer, n=None, gen=None, embed=None, question="", gen_judge=None):
    try:
        if os.environ.get("SE_ENABLED", "1") == "0":
            return {"verdict": "off"}
        n = n or N
        if gen is None or embed is None:
            import ollama_client as oc
            model = os.environ.get("OLLAMA_REASON_MODEL", "qwen3.6:35b-a3b")
            gen = gen or (lambda: oc.generate(prompt, system=system, model=model, timeout=60,
                                              options={"num_predict": 160, "temperature": 0.8}, think=False))
            embed = embed or (lambda texts: oc.embed(texts, model="nomic-embed-text", timeout=60))
            gen_judge = gen_judge or (lambda p: oc.generate(p, system=JUDGE, model=model, timeout=60,
                                                             options={"num_predict": 120, "temperature": 0.0}, think=False))
        samples = [s for s in (gen() for _ in range(n)) if s]
        texts = [_norm(answer)] + [_norm(s) for s in samples]
        if len(texts) < 3:
            return {"verdict": "unavailable", "reason": "too few samples"}
        cl, method = None, "judge"
        if gen_judge:
            try:
                cl = judge_groups(question or prompt[-300:], texts, gen_judge)
            except Exception:
                cl = None
        vecs = embed(texts)
        if not vecs:
            return {"verdict": "unavailable", "reason": "embed failed"}
        if cl is None:
            cl, method = clusters(vecs), "embedding"
        h = entropy(cl, len(texts))
        pairs = [(i, j) for i in range(len(vecs)) for j in range(i + 1, len(vecs))]
        mc = sum(_cos(vecs[i], vecs[j]) for i, j in pairs) / len(pairs)
        verdict = "consistent" if len(cl) == 1 else ("uncertain" if h >= UNCERTAIN_BITS else "mixed")
        return {"verdict": verdict, "entropy": round(h, 3), "clusters": len(cl), "mean_cos": round(mc, 3), "method": method,
                "samples": [s[:160] for s in samples]}
    except Exception as e:
        return {"verdict": "unavailable", "reason": type(e).__name__}


def relabel(answer, se):
    """If the samples disagreed, the reply says so — replacing a 'sure'/'fairly sure' tail or appending one."""
    if (se or {}).get("verdict") != "uncertain":
        return answer
    base = re.sub(r"\s*\((sure|fairly sure)\)\s*$", "", answer.strip(), flags=re.I)
    if re.search(r"\(not sure", base, re.I):
        return base
    return base + " (not sure — I gave different answers when I re-checked; verify before acting)"


def selftest():
    res = []
    same = lambda texts: [[1.0, 0.0, 0.0]] * len(texts)
    scatter = lambda texts: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.7, 0.7, 0.0]][:len(texts)]
    g = lambda: "x"
    a = check("p", "s", "The meeting is at 6 (sure)", n=3, gen=g, embed=same)
    res.append(("agreeing samples -> consistent, entropy 0", a["verdict"] == "consistent" and a["entropy"] == 0))
    b = check("p", "s", "The meeting is at 6 (sure)", n=3, gen=g, embed=scatter)
    res.append(("scattered samples -> uncertain", b["verdict"] == "uncertain" and b["entropy"] >= UNCERTAIN_BITS))
    res.append(("uncertain relabels a 'sure' answer", relabel("Game is at 6. (sure)", b).endswith("verify before acting)")))
    res.append(("consistent leaves the answer alone", relabel("Game is at 6. (sure)", a) == "Game is at 6. (sure)"))
    res.append(("small talk is exempt", not applies("Good morning!") and applies("What time is the meeting on Friday?")))
    res.append(("failure is fail-soft", check("p", "s", "a", n=3, gen=lambda: None, embed=same)["verdict"] == "unavailable"))
    for n_, ok in res:
        print(("PASS " if ok else "FAIL ") + n_)
    print("SELFTEST %s" % ("PASS" if all(ok for _, ok in res) else "FAIL"))
    return 0 if all(ok for _, ok in res) else 1


if __name__ == "__main__":
    import sys
    sys.exit(selftest())
