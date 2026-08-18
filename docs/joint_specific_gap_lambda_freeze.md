# Joint-Specific Gap Lambda Freeze

Selected lambda_joint: 0.025

Candidate set: 0.000, 0.025, 0.050, 0.100.

Frozen-policy checkpoints:
- model_10000.pt: 9208411ee441c427f96d91ac8ffbea695ee129feebd402d42d6ab3a19ba4e45a
- model_20000.pt: f3358d168eade5c701ed8797ed9fb31c2a531121741796f6c066de2d85c20c7b
- model_30000.pt: ebc3e5280172d8897d9070a631f1917309f5b886c38e2d55c7e6c4d14ce72240
- model_33999.pt: b7d7b685bd4ee9baa3aeb08a426b3452717b45c7167c7fa53a5746dafe430c02

Snapshot coverage:
- model_10000.pt: eligible=21485 observed=21460 cold=25
- model_20000.pt: eligible=21485 observed=21485 cold=0
- model_30000.pt: eligible=21485 observed=21485 cold=0
- model_33999.pt: eligible=21485 observed=21485 cold=0

Selection uses only training-distribution sampling perturbation statistics from frozen-policy proxy snapshots.
No Validation, Test, reward, success, completion, or Joint L2 metrics are used.

Proxy limitation: snapshots are deterministic frozen-policy diagnostic rollouts on the training motion library, not reconstructed M7-Raw historical per-joint EMA state.

Safety rule: cluster and motion probabilities must be exact, entropy drop <= 1%, Top1/Top5 mass increase <= 5%, median per-motion TV <= 0.01, p95 per-motion TV <= 0.03, Spearman >= 0.98, no new cap pathology, and p90 active boost >= 1%.

Selection reason:
lambda_joint=0.025 is the smallest nonzero candidate that is safe and measurable on every selected snapshot. Larger candidates were not chosen because STEP 7 freezes the minimum effective intervention, not the largest safe perturbation.
