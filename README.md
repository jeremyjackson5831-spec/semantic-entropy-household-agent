# Does my assistant know when it's guessing?

**Semantic entropy on a household AI agent.** A field note by Jeremy Jackson, November 2026.

I designed and operate this system. The code was built with Claude under my direction and review.

## Summary

I run a personal AI agent for my household on a Mac mini with a local 35B model. I added a semantic-entropy check so the agent can tell me when it is guessing, following Farquhar et al. (Nature, 2024). On 15 real questions it never invented an answer to something it could not know, and it caught two confident-sounding guesses that my existing confidence signal missed.

It also failed in two ways worth writing down. It passed one answer that was consistent and wrong, because every sample read the same misleading context. And in 20 days of live use it flagged 46% of answers as uncertain, which is too many to be useful.

## The problem

My agent already ended each answer with a confidence word: sure, fairly sure, or not sure. That word came from how fresh its sources were. If the calendar was read an hour ago, the answer said "sure".

Freshness says nothing about whether the model used the source correctly. A model can read current data and still produce a fluent answer the data does not support. I wanted a second signal that looked at the model's own behavior, cost nothing per call, and ran on the same machine.

## Method

The check runs after the agent drafts an answer and before it is sent.

1. Ask the same question three more times at temperature 0.8, using the same local model (Qwen 3.6, 35B) and the same context.
2. Have the model group the four answers by whether they make the same factual claims: same numbers, names, dates and yes/no. Wording and tone are ignored. Two answers that both say the information is unavailable count as the same.
3. Compute the entropy of the group sizes. One group is 0 bits. A 3-1 split is 0.81 bits. A 2-2 split is 1.0 bit.
4. At 0.9 bits or more, relabel the answer: "not sure, I gave different answers when I re-checked; verify before acting".

The threshold tolerates one odd sample and flags an even split or worse. The check adds 4 to 7 seconds, runs entirely on the local machine, and fails soft: any error leaves the original answer untouched. Small talk is exempt.

My first version grouped answers by embedding similarity (nomic-embed-text, cosine 0.85 or higher). It flagged nearly everything. Short answers that mean the same thing, such as "29." and "There are 29 jobs", embed far apart. Judging claims directly, as the paper does with entailment, fixed that. Embedding grouping remains only as the fallback when the judge call fails.

## Calibration on 15 real questions

| Kind of question | Asked | Consistent | Flagged |
| --- | --- | --- | --- |
| Grounded: the answer is in the agent's state | 6 | 5 | 1 |
| Unanswerable: the answer is in no state | 6 | 6 | 0 |
| Speculative: no fact settles it | 3 | 1 | 2 |

**Unanswerable questions were the result I cared about most.** All six times, all four samples said the information was not available and pointed to where it would be. Nothing was invented.

**The check caught what freshness could not.** One speculative answer, a prediction about finishing applications by a holiday, carried the label "sure" from the freshness signal. The samples split and the check relabeled it.

**The one grounded flag was useful.** The first answer was correct, but the re-samples disagreed about a decision I had made. That told me recall of that decision was unstable.

**The miss.** A question about a family member's college choice came back consistent and wrong. All four samples said the search was on hold. It was active. Two unrelated items sat side by side in one stored decision row, and my recall step cut each row at 220 characters, so every sample read the same misleading excerpt. I fixed the source row and added the case to my regression set.

## Twenty days in production

The check flagged 52 of 114 live answers (46%) between October 22 and November 10, 2026. That is far above the rate where a warning label still means something. I had planned to review the flag rate after one week and did not; this count is that overdue review.

| Where the question came from | Checks | Consistent | Mixed (3-1) | Flagged uncertain |
| --- | --- | --- | --- | --- |
| Typed or texted by me | 63 | 17 | 17 | 29 (46%) |
| Auto-generated status card ("Tell me about...") | 27 | 14 | 5 | 8 (30%) |
| Auto-generated advice card ("What should I do on...") | 24 | 3 | 6 | 15 (62%) |

The judge produced the grouping in 102 checks. The embedding fallback, the method I had already found to over-flag, produced it in the other 12.

The pattern suggests a cause, which I have not yet tested. Advice questions have no single correct answer, so honest samples legitimately differ, and the check reads that as guessing. Status questions, which do have one answer, were flagged half as often. The check measures disagreement between samples. Disagreement means confabulation only when the question has a fact behind it.

## Limits and next tests

**Self-consistency catches guessing. It cannot catch a shared mistake.** When the context is wrong, every sample is wrong the same way and the entropy is zero. The defense is clean source data and a regression set, not more sampling.

**This is a small study.** It covers 15 calibration questions and 114 production checks, with one model, one user and one judge. The judge is the same model that wrote the answers, so its errors may correlate with theirs. I have no labeled ground truth for the 114 production answers, so I can report the flag rate but not the precision.

What I would test next, in order:

1. Label a sample of the 52 flagged answers as right or wrong, to measure how many flags were real.
2. Classify each question as factual or open-ended before checking, and skip or re-threshold the open-ended ones.
3. Compare N = 3 against N = 5 and N = 10 samples. With four answers, entropy can only take a few values.
4. Use a different model as the judge and measure how often the two judges agree.
5. Remove the embedding fallback, or log its flags separately, since it accounts for 12 checks of known lower quality.

## Code

`semantic_entropy.py` is the check as it runs, with household names and file paths removed. `python3 semantic_entropy.py` runs its self-test without a model.

## Source

Farquhar, Kossen, Kuhn and Gal, "Detecting hallucinations in large language models using semantic entropy", Nature, 2024.
