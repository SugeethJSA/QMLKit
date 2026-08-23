# How QMLKit Works — Start Here

This series explains the whole system in plain language. No physics degree or
veterinary background needed.

| Doc | What it covers |
|---|---|
| [01_the_big_picture.md](01_the_big_picture.md) | The idea in one page: a dog, a kennel full of sensors, and two kinds of machine learning |
| [02_the_kennel_hardware.md](02_the_kennel_hardware.md) | What each sensor measures and how the ESP32 turns wiggles into numbers |
| [03_from_sensors_to_features.md](03_from_sensors_to_features.md) | How raw telemetry becomes a tidy row of features a model can read |
| [04_the_models_explained.md](04_the_models_explained.md) | Classical vs quantum models — SVM, Random Forest, XGBoost, QSVM, VQC, QCNN, reservoirs |
| [05_why_biozz_is_special.md](05_why_biozz_is_special.md) | The "correlation-aware" trick from our paper, explained with an analogy |
| [06_training_testing_fairly.md](06_training_testing_fairly.md) | Why we split data the way we do, cross-validation, ablations, robustness |
| [07_the_hybrid_lab.md](07_the_hybrid_lab.md) | How the Training Lab mixes-and-matches models to find the best recipe |
| [08_datasets.md](08_datasets.md) | The three datasets: real VOC chemistry, dog ECG notes, simulated kennel trials |
| [09_running_everything.md](09_running_everything.md) | Copy-paste commands for every task |

## The one-paragraph version

A trained dog sniffs a breath sample in a smart kennel. Sensors record how the
dog *reacts* — weight shifts on floor pads, head position, distance to the
sample, heart rate, tiny tremors from a collar sensor. An ESP32 chip streams
those readings 100 times per second to our software, which condenses each
viewing session into one row of numbers. Machine-learning models then learn
which reaction patterns tend to accompany cancer-positive samples versus
healthy ones. We compare **classical** models (fast, proven) with **quantum**
models (experimental, from a quantum computer simulator), and — the interesting
part — **hybrids** that feed quantum-computed patterns into classical learners.
Everything is packaged as a desktop app with a live dashboard so a handler can
watch sessions in real time.

> Important honesty note throughout the docs: this is a **research screening
> aid**, not a medical device. "Cancer-associated" predictions are leads for
> clinicians, never diagnoses.
