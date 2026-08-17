# Label-Efficient Prompt-Injection Detection

*Study design, written 2026-10-08 before any experiment was run. The hypotheses below are fixed here so that the results can prove them wrong.*

## 1. The problem

A detector that flags prompt-injection text is a classifier, and a classifier needs labeled examples. Labels are the expensive part: each one is a person reading a text and deciding whether it tries to take over an AI system. Two ideas can cut that cost, and this study measures both.

- **Active learning.** Instead of labeling texts at random, the current model picks the texts it would learn the most from, and only those get labeled. The question is how many labels this saves for injection detection specifically.
- **A cheaper labeler.** A local language model can label texts for free, but it makes mistakes. Active learning is known to be sensitive to label noise, because it deliberately asks about the hardest, most ambiguous items, which are exactly the ones a noisy labeler gets wrong. The question is whether the savings survive, and whether routing only the model's low-confidence items to a human recovers them.

## 2. Research questions and pre-registered hypotheses

| # | Question | Hypothesis |
|---|---|---|
| RQ1 | How many labels does active learning save over random labeling? | **H1.** Uncertainty sampling reaches 95% of the full-data F1 with fewer than half the labels random sampling needs. |
| RQ2 | How accurate is a local 8B language model as a labeler? | **H2.** Agreement with the ground-truth labels is between 80% and 90%, and most errors are benign texts that talk about instructions or rules. |
| RQ3 | Does labeler noise erase the active-learning advantage? | **H3.** With LLM labels, uncertainty sampling loses more F1 relative to perfect labels than random sampling does. |
| RQ4 | How many human labels does a hybrid labeler save? | **H4.** Sending only the LLM's low-confidence items to a human reaches the F1 of perfect labels while using fewer than 30% human labels. |
| RQ5 | Does a detector trained on a public dataset transfer to a separate attack set? | **H5.** Recall on the out-of-distribution attacks is well below in-distribution recall, and lowest on payloads written as document text. |

## 3. Data

- **Main dataset:** `xTRam1/safe-guard-prompt-injection` on Hugging Face, 8,236 training and 2,060 test texts, each labeled injection or benign. The training split is the labeling pool; the test split is never used for training or for choosing what to label. Its license is not stated, so the scripts download it and this repository never redistributes it.
- **Out-of-distribution test sets:** the 45 attacks and 7 benign questions from a companion project (the Prompt Injection Lab), and the test split of `deepset/prompt-injections` (Apache-2.0).

## 4. Method

- **Features.** Each text is turned into a 768-number vector by a local embedding model (`nomic-embed-text`) once, and the vectors are cached.
- **Classifier.** Logistic regression on the vectors: fast, deterministic, and cheap enough to retrain hundreds of times.
- **Query strategies.** Random (the baseline); uncertainty (label what the model is least sure about); diversity (label what best covers the pool); and a hybrid of the two.
- **Protocol.** Start from 20 randomly chosen labels, add 20 per round, stop at 600. Every configuration is repeated with 10 random seeds, and results are reported as the mean with the spread across seeds.
- **Labelers.** Perfect (the dataset's labels); LLM only (a local 8B instruction model); and hybrid (the LLM labels each queried item, and items below a confidence threshold go to the human, here simulated by the dataset label, with every human label counted).
- **Metrics.** F1 for the injection class and AUROC on the test split; learning curves of F1 against labels used; the number of labels needed to reach 95% of the full-data F1; the number of human labels in the hybrid setting; recall on the out-of-distribution sets.

## 5. What a result looks like

A set of learning curves, one per strategy and labeler, with bands showing the spread across seeds, and a table of labels needed to reach the target. The headline will take the form: *"Active learning reached X% of full-data F1 with N labels instead of M; with an LLM labeler the advantage shrank to …; a hybrid labeler recovered it with H human labels."* The numbers are unknown until the runs.

## 6. Limits known in advance

One dataset, one embedding model, one classifier family, and one labeling model. The "human" in the hybrid labeler is simulated by the dataset's labels, which may themselves contain errors. Results describe this setup, not active learning in general.
