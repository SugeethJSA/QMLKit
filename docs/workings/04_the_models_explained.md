# The Models, Explained

All models answer the same question: *given this row of dog-reaction numbers,
is the sample more likely cancer-associated or not?* They differ in how they
think.

## The classical team (fast, proven)

| Model | Kitchen-table intuition |
|---|---|
| **Logistic Regression** | Draws one straight dividing line; each feature gets a simple +/− vote weight |
| **SVM (linear / RBF)** | Finds the widest gap between the two groups. RBF version can bend the gap into curves |
| **Random Forest** | Asks hundreds of yes/no questions ("is pressure-front > 300?") and takes a majority vote |
| **XGBoost** | Like Random Forest but each new question focuses on the mistakes of the previous ones — usually our strongest single model |
| **MLP** | A small neural network: layers of weighted votes stacked so it can learn combos |

## The quantum team (experimental)

These run on a **quantum computer simulator** — software that mimics a quantum
computer's mathematics. No quantum advantage is claimed; the point is to test
whether the representation helps.

| Model | Intuition |
|---|---|
| **QSVM** | An SVM whose notion of "similarity" comes from a tiny quantum circuit instead of a classical formula. Two dogs get similar scores if their encoded quantum states overlap |
| **VQC** | A quantum circuit with dials (trainable angles). Training turns the dials so that cancer samples tip an output needle toward +1 and healthy toward −1 |
| **QCNN** | A quantum mini-convolutional-net: neighbouring qubits "chat" in layers, then weaker ones are progressively ignored — a pattern-hierarchy machine |
| **Quantum reservoir** | A random fixed quantum system is poked with data; we read out how it sloshes. Only the readout is trained |

## The BioZZ trick (our paper's contribution) → next doc

The quantum models need to encode 6–8 numbers into qubits. Our **BioZZ /
CW-ZZ** map lets *relationships in the training data* decide how strongly
qubit-pairs interact. See [05_why_biozz_is_special.md](05_why_biozz_is_special.md).

## The hybrid recipes (the actual winners)

- **Quantum-Augmented-XGB:** train a tiny VQC on reduced features; take its
  cancer-probability output as ONE extra column; glue it onto the original
  features; hand everything to XGBoost. The quantum model becomes a specialist
  witness; XGBoost remains the judge.
- **Soft voting:** four models vote; we average their probability outputs.
- **Stacking:** collect each base model's opinion *on data it wasn't trained on*
  (to keep everyone honest), then train a small Logistic Regression that learns
  whom to trust and by how much.

In the real VOC benchmark these hybrids scored **0.97–0.99 ROC-AUC**, ahead of
every pure-classical and pure-quantum configuration. Details:
[`docs/reports/hybrid_search_report.md`](../reports/hybrid_search_report.md).
