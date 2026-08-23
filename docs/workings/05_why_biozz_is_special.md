# Why BioZZ Is Special

## The problem it solves

A quantum feature map must translate ordinary numbers into quantum states. The
standard recipe (a "ZZ feature map") treats **every pair of numbers equally**:
heart-rate and head-distance get the same interaction strength as two
essentially-random sensor readings.

But our measurements aren't all equally related. Heart rate rises *together*
with movement when a dog gets excited; two unrelated readings vary
independently. Throwing that knowledge away wastes information.

## The idea

While training (and only while training — never on test data), we compute the
**correlation matrix**: for every pair of features, how much do they move
together, from −1 (opposites) through 0 (unrelated) to +1 (twins)?

Then we build the quantum encoding so those correlations literally scale the
interactions:

> interaction strength(i, j) = correlation(i, j) × (standard ZZ phase)

Pairs with strong relationships produce big quantum effects; unrelated pairs
barely interact. Positively- and negatively-correlated pairs rotate in opposite
directions.

An everyday analogy: a good investigator weighs testimony between witnesses who
corroborate each other more heavily than testimony from strangers. BioZZ wires
that instinct into the quantum circuit.

## Guardrails that keep it honest

1. **Train-only estimation.** Correlations come from the training split only,
   and are frozen before test data is touched. Each cross-validation fold
   re-derives its own matrix from its own training portion.
2. **The permuted-control experiment.** We also run BioZZ with the correlation
   values *shuffled* (right magnitudes, wrong pairings). If BioZZ's edge came
   merely from "having extra knobs" rather than real relationships, shuffled-BioZZ
   would score just as well. This is the control that makes any claim about
   correlation-awareness scientifically meaningful.
3. Same qubit count, same repetitions, same data splits as the plain-ZZ baseline
   it is compared against.

In our first VOC ablation: Angle 0.557 ≈ plain ZZ 0.557 < CW-ZZ 0.658, with the
permuted control at 0.689 — i.e., correlation-aware encoding beats conventional
maps, but at this sample size we can't yet separate "real relationship
knowledge" from "extra expressivity." Larger-N runs are queued.
