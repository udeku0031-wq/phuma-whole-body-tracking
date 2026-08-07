# formal_v1 Learning Gap Diagnosis

## Frozen Scope

formal_v1 results are frozen and must not be modified or overwritten.

Frozen methods:

- M0
- M4
- M5
- M6
- M7
- M7-Raw diagnostic ablation

Do not run or overwrite for now:

- M1/M2/M3 formal training
- M0 seed43/44
- M7 seed43/44
- 3000/12000 data-size experiments
- Test

## Git Anchors

- V1 frozen tag: experiments-v1
- M7-Raw diagnostic branch: ablation/m7-raw-v1

## Key Diagnosis Chain

- M4 > M0
- M5 < M4
- M6 > M5
- M6 < M4
- M7 > M6
- M7 > M4
- M7-Raw > M7

## Selected Validation Results

M7 selected checkpoint:

- checkpoint: model_33500.pt
- Micro Success: 0.8740
- Macro Success: 0.8863
- Completion: 0.9290
- Body Error: 0.05189
- Joint L2: 0.82921
- Failures: 962

M7-Raw selected checkpoint:

- checkpoint: model_33999.pt
- Micro Success: 0.8849
- Macro Success: 0.8930
- Completion: 0.9341
- Body Error: 0.04992
- Joint L2: 0.80858
- Failures: 879

## Conclusion

Under the same Quality Gate and Cluster Diversity constraints, Raw Error outperforms the current Learning Gap v1 design.

Therefore, formal_v1 is frozen as diagnostic evidence, and the project should enter Learning Gap v2 development.

The new target is not to beat old M7, but to beat the stronger baseline:

M7-Raw = Raw Error + Quality Gate + Cluster Diversity.
