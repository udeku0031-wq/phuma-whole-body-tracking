"""English manuscript material; empirical values are checked by the asset builder."""

TITLE = (
    "Quality-Gated Diversity-Constrained Hierarchical Error Sampling for "
    "Humanoid Whole-Body Motion Tracking"
)

ABSTRACT = (
    "Training a shared humanoid motion-tracking policy requires allocating experience across "
    "heterogeneous trajectories with unequal reliability and learning progress. We present "
    "quality-gated diversity-constrained hierarchical error sampling (QD-HES), which combines "
    "reference-quality knowledge, unsupervised motion structure, and online tracking statistics. "
    "A segment-level quality gate restricts episode-start eligibility, fixed motion clusters "
    "allocate diversity budgets, and tracking errors prioritize motions and segments within "
    "each cluster. The method changes the training distribution while retaining the underlying "
    "PPO controller, observation space, and reward design. Experiments use a local conversion "
    "of PHUMA with 6,000 training, 7,636 validation, and 7,592 test motions. At 50k iterations, "
    "QD-HES exceeds its quality-disabled counterpart by 1.784 percentage points in validation "
    "macro success. Validation screening selects a 59k QD-HES checkpoint. On the final test "
    "split, it achieves 92.31% micro success, 90.79% macro success, and 95.54% completion. "
    "Compared with the validation-selected 54k diversity-only checkpoint, it reduces failures "
    "from 731 to 584 and joint-position L2 error by 7.50%. Source-group-aware paired bootstrap "
    "intervals support a positive test-set difference. These final checkpoints have unequal "
    "training and selection budgets, and all reported training runs use one seed. The results "
    "provide evidence for structured sampling in this setting without establishing a universal "
    "training limit or superiority across random seeds."
)

HIGHLIGHTS = [
    "Quality, diversity, and tracking error guide hierarchical motion sampling.",
    "Quality gating changes episode starts without deleting entire trajectories.",
    "At 50k iterations, quality gating improves validation macro success by 1.78 pp.",
    "The selected model achieves 92.31% micro success on 7,592 test motions.",
    "Paired analysis preserves dependence among chunks of the same source motion.",
]

SECTIONS = [
    ("1. Introduction and contribution boundary", [
        "Large motion libraries contain both repeated movement patterns and infrequent behaviors. "
        "A sampler that responds only to tracking error can devote substantial experience to a narrow "
        "part of the library. Conversely, uniformly allocating starts does not distinguish unreliable "
        "reference segments from segments that are physically plausible but currently difficult for the policy. "
        "This motivates separating reference eligibility, coverage across motion families, and adaptive "
        "allocation within those families.",
        "The proposed contribution is a training-data allocation mechanism, not a new robot, motion "
        "dataset, or PPO architecture. The project builds on the BeyondMimic motion-tracking code [1] "
        "and PHUMA reference motions [2]. It adds a shared multi-motion library, segment metadata, "
        "quality gating, online learning statistics, and diversity-constrained hierarchical sampling. "
        "The local experiments concern simulated Unitree G1 tracking; real-robot transfer and diffusion "
        "control results from upstream work must not be attributed to this study.",
        "Three contributions are supported by the available artifacts: (i) an explicit factorization "
        "of cluster, motion, and segment sampling with independent quality and coverage constraints; "
        "(ii) a modular implementation supporting controlled sampler variants without changing the "
        "controller; and (iii) fixed-budget validation comparisons, checkpoint diagnostics, and paired "
        "analysis of validation-selected policies on a separate test split. The evidence is strongest "
        "for the observed single-seed setting. QD-HES is a proposed paper name for the repository's "
        "M7-Raw variant, not the difficulty-calibrated variant named M7.",
    ]),
    ("2. Related-work positioning", [
        "BeyondMimic develops a compact tracking formulation and a downstream guided-diffusion "
        "controller [1]. This project reuses its tracking foundation and studies how training references "
        "are allocated. PHUMA addresses physically reliable humanoid reference data [2]; the present "
        "quality audit concerns residual artifacts and reset eligibility in the local converted corpus, "
        "rather than replacing PHUMA's dataset construction procedure.",
        "GMT studies general whole-body tracking with adaptive motion sampling and a mixture-of-experts "
        "policy [3]. Thus, adaptive sampling alone should not be presented as unprecedented. The "
        "distinguishing design here is the separation of a reference-quality start mask, an error-independent "
        "cluster budget, and error-dependent conditional distributions. No reproduced GMT benchmark "
        "is available in this package, so no numerical superiority over GMT is claimed. PPO [4] is "
        "the optimization backbone and is held fixed across the implemented variants.",
        "For KBS, the methodological emphasis is explicit knowledge-guided experience allocation: "
        "reference-quality metadata defines eligibility, unsupervised motion structure defines coverage, "
        "and online tracking statistics estimate where additional experience is needed. This is not "
        "a claim of symbolic reasoning or formal knowledge-base inference. The four verified core "
        "references supplied here are a starting bibliography, not an exhaustive literature review.",
    ]),
    ("3. Problem formulation and system overview", [
        "A single policy pi_theta maps tracking observations to 29 joint-position actions. At each "
        "episode assignment, the sampler chooses a motion m, a segment s within that motion, and a "
        "legal reference frame t0 within the segment. Subsequent simulation follows the reference "
        "trajectory until natural completion, physical early termination, or administrative truncation. "
        "The sampling distribution affects future on-policy experience; it is not a replay-buffer "
        "priority distribution and does not add an importance-weighted off-policy loss.",
        "Figure 1 separates offline metadata construction, online PPO training, and evaluation. "
        "Only training references enter metadata fitting. Validation screens and selects checkpoints; "
        "the selected checkpoints are evaluated on Test without an arrow back into the sampler. "
        "Figure 2 expands the sampling mechanism. Module design, interfaces, and implementation "
        "locations are catalogued in Tables 1 and 2 and in the implementation evidence index.",
    ]),
    ("3.1. Data conversion and multi-motion reference library", [
        "The converter reads PHUMA G1 trajectories and exports the WBT NPZ fields required by the "
        "tracker: joint positions, joint velocities, world-frame body positions and quaternions, "
        "and body linear and angular velocities. The local converted corpus contains 76,086 motions. "
        "This corpus size is not the number used to train a policy: training uses the fixed "
        "6,000-motion subset. The NPZ schema contains 29 joints and 30 body entries; the saved G1 "
        "tracking configuration selects 14 bodies for tracking metrics. These counts serve different "
        "purposes and must not be conflated.",
        "The loader accepts a single file, a directory, or an ordered manifest. Per-motion frame "
        "counts, frame rates, offsets, and body indexing map heterogeneous references into shared "
        "tensors. Each parallel environment retains its own motion identifier and reference-frame "
        "index. Manifest identity and metadata compatibility checks prevent accidental reuse of "
        "metadata built for a different training pool.",
    ]),
    ("3.2. Stage 0: segment indexing and state management", [
        "Each motion is partitioned into nominal one-second segments. With frame rate f_m and "
        "length T_m, the segment width is L_m = max(1, round(f_m)) and the number of segments is "
        "ceil(T_m / L_m). The training pool yields 21,575 segments. A stable global index maps a "
        "motion/local-segment pair to metadata, eligibility, sample counts, and online statistics.",
        "The infrastructure separates reference observations from episode outcomes, supports "
        "probability validation, and persists adaptive state during checkpointing. With research "
        "modules disabled, the implementation preserves the legacy uniform sampling path. This "
        "is an engineering control for ablation validity, not a new sampling objective.",
    ]),
    ("3.3. Module 1: reference-quality gate", [
        "The offline audit examines finite values, quaternion validity, URDF joint limits, velocity "
        "consistency, isolated acceleration and jerk spikes, temporal discontinuities, ground "
        "penetration, and foot sliding. These are reference-data tests, not policy success labels. "
        "Weighted violation severities produce a bounded quality score. The profile combines "
        "score thresholds (pass at 0.90; reject below 0.55) with hard-rejection and metric-specific "
        "rules; the scalar score alone is therefore not a complete label definition.",
        "The audit labels 15,155 segments pass, 6,330 borderline, and 90 reject. Borderline starts "
        "remain eligible. Eligibility intersects the Stage-0 legal-start mask with the quality "
        "mask; 99.5768% of legal candidate start frames remain, and 5,998 motions retain at least "
        "one eligible start. The 0.4171% rejected-segment fraction differs from the rejected-frame "
        "fraction because segment lengths and legal start ranges differ.",
        "The gate applies at assignment_start. It does not remove trajectories from the corpus, "
        "rewrite references, or guarantee that a rollout never traverses a rejected segment. "
        "It is a heuristic reference-quality mechanism rather than a proof of physical feasibility. "
        "All active thresholds and weights are exported in Table S4 and the configuration snapshots.",
    ]),
    ("3.4. Module 2: intrinsic difficulty for ablation variants", [
        "Intrinsic difficulty uses 28 policy-independent kinematic and contact descriptors. "
        "Derivatives are computed before segmentation to avoid artificial boundary spikes. "
        "Training-only robust normalization uses a median and a 1.4826-MAD scale with documented "
        "fallbacks for near-constant features. A weighted score is mapped through an empirical "
        "CDF into ten approximately balanced difficulty bins.",
        "Difficulty provides a reference for comparing online errors among similarly difficult "
        "segments. It neither rejects segments nor directly determines the final M7-Raw sampling "
        "probabilities. M7-Raw and D-only disable runtime difficulty calibration. The diversity "
        "builder reuses raw kinematic/contact feature metadata produced by the offline feature "
        "pipeline; it excludes the difficulty score and bin from clustering. Thus, a shared "
        "feature-extraction dependency is not an active learning-gap mechanism.",
    ]),
    ("3.5. Module 3: online error estimation", [
        "At every control step, body-position, joint-position, and orientation errors are attributed "
        "to the segment actually visited. Episode outcomes add physical termination, completion, "
        "and natural-success observations. Administrative time limits censor outcome components "
        "rather than automatically labeling them physical failures. Statistics accumulate during "
        "warm-up; initialized exponential moving averages are committed per iteration with decay "
        "rho = 0.95. The first available observation initializes an EMA rather than mixing with zero.",
        "Tracking components are normalized by 0.30 m (body), 0.50 rad (joint L2), and 0.40 rad "
        "(orientation geodesic angle), then clipped to [0,5]. Segment error is an active-component "
        "weighted mean of these three quantities, termination, one minus completion, and one minus "
        "success, with weights [1,1,1,1,0.5,0.5]. Missing or censored components do not contribute "
        "to the denominator. A reliable segment requires at least 32 reference observations and "
        "initialized tracking components.",
        "Motion error aggregates the observation-weighted segment mean, the unweighted segment "
        "P90, and motion-level termination, incomplete execution, and failure, with weights "
        "[1,0.25,0.5,0.5,0.5]. Motion reliability additionally requires at least eight episode "
        "outcomes. Cold items retain a neutral treatment and exploration support; initial zero "
        "statistics are not interpreted as demonstrated competence.",
        "For learning-gap ablations, a segment's error is centered and scaled relative to reliable "
        "training segments in its frozen difficulty bin, with a standard-deviation floor of 0.10 "
        "and clipping to [-5,5]. Sparse bins fall back to global reliable-segment statistics. "
        "Motion scores aggregate positive excess, whereas conditional segment scores subtract "
        "the motion's median gap. Joint-specific gap variants add a joint-level correction and "
        "remain diagnostic variants, not components of QD-HES.",
    ]),
    ("3.6. Module 4: diversity-constrained hierarchical sampling", [
        "Seventeen selected segment descriptors are aggregated into 30 motion-level features. "
        "The train-only robustly normalized features are clustered with K-means++ initialization "
        "and K = 8. The stored cluster sizes are [1902,456,503,948,603,557,728,303]; the reported "
        "silhouette score is 0.2779 on a diagnostic subsample. These clusters need not coincide "
        "with semantic categories. Source-category labels, policy errors, difficulty scores, "
        "and quality scores do not enter the clustering distance.",
        "For nonempty eligible clusters, let n_c be the number of eligible motions and C their "
        "count. The allocation is P(c) = f/C + (1-f)n_c^alpha / sum_j n_j^alpha, with f = 0.5 "
        "and alpha = 0.5. Empty clusters receive zero probability. The quality gate may change "
        "eligible counts, so raw cluster counts are not identical to gated runtime counts. "
        "The budget has no online error feedback and no count-aware deficit correction in this version.",
        "Within each selected cluster, motion and segment distributions use raw motion and segment "
        "error, respectively. For eligible item i, the logit is (clip(E_i,-10,10) + "
        "0.25/sqrt(N_i+1))/tau with tau = 1. Softmax probabilities are mixed with 0.15 uniform "
        "probability and projected to a capped simplex by proportional water filling. The "
        "configured motion cap is 0.02; within a small cluster it is relaxed to at least "
        "1/n_c to retain feasibility. The conditional segment cap is 1.0. Ineligible items "
        "have exactly zero probability, and invalid score sets use an eligible-uniform fallback.",
        "The resulting distribution is P(c,m,s) = P(c) P(m|c) P(s|m). A legal start frame is "
        "sampled uniformly inside the chosen segment. During the first 1,000 iterations, diversity "
        "budgets remain active while the conditional motion and segment distributions are uniform. "
        "After warm-up, score and probability refreshes occur every 50 completed iterations; "
        "cached distributions keep the reset path inexpensive. This policy does not imply that "
        "the entire hierarchical distribution is uniform during warm-up.",
    ]),
    ("3.7. Controller, reproducibility, and method matrix", [
        "The PPO actor and critic use hidden widths [512,256,128] with ELU activations. Actor and "
        "critic observation dimensions are 160 and 286; the action dimension is 29. Training uses "
        "3,072 environments and 24 control steps per iteration, corresponding to 73,728 "
        "transitions per iteration. Simulation dt is 0.005 s with decimation four, giving a "
        "50-Hz controller. Training episodes have a 10-s administrative limit; the evaluator "
        "uses a 60-s horizon. Saved run parameters, rather than generic defaults, determine "
        "the settings tabulated in Table 5.",
        "Reward weights and scales are listed in Table 6. Physical terminations include anchor "
        "height error above 0.25 m, end-body vertical position error above 0.25 m, and an "
        "absolute difference above 0.8 between the reference and robot body-frame gravity-vector "
        "z components. The latter is a dimensionless projected-gravity test, not a 0.8-rad "
        "orientation-angle threshold. Online orientation error uses a different, angular metric.",
        "Table 4 distinguishes M0-M7, the flat GlobalRaw baselines, D-only, and M7-Raw. "
        "M2 and M3 have pilot evidence but no available formal full-validation result. They "
        "must not be silently filled with zeros or omitted from a claim of a complete M0-M7 "
        "factorial study. D-only disables quality relative to M7-Raw at the sampler level, "
        "but comparing their final selected Test checkpoints does not isolate that switch "
        "because checkpoint and search budgets also differ.",
    ]),
    ("4. Experimental design", [
        "The local grouped-random split contains 60,858 training-pool motions, 7,636 validation "
        "motions, and 7,592 test motions. Training draws a fixed 6,000-motion subset from the "
        "training pool. The 500-motion probe is a subset of full Validation. Source groups "
        "are derived from normalized source sequence identities with explicit chunk suffixes "
        "removed, and groups are disjoint across splits. This prevents identified sequence "
        "chunks from crossing splits; it does not establish subject-disjoint or independently "
        "deduplicated semantic splits. Table 3 and the evidence manifest record identities.",
        "Every evaluation motion starts at frame zero and is evaluated once with deterministic "
        "actions, seed 42, randomization disabled, and 3,072 environments. Training-time adaptive "
        "start sampling does not choose evaluation motions. Results are aligned by unique motion "
        "path, not CSV row order. Complete final Test coverage and matching manifest hashes "
        "were checked against each model's evaluation configuration.",
        "Micro success is the fraction of motions reaching the final reference frame before "
        "early termination. Macro success is the unweighted mean of 17 source-category success "
        "rates. Completion is the mean completed-frame fraction. Body error averages the "
        "Euclidean error over the 14 configured tracked bodies after the implementation's "
        "root/yaw alignment, then over observed steps and motions. Joint L2 averages the norm "
        "of the 29-joint error vector; joint RMS equals joint L2 divided by sqrt(29). Error "
        "means cover observed trajectories only, so premature failure can make an error "
        "look small. Success and completion must therefore accompany error metrics.",
        "The recorded checkpoint rule uses validation macro first, with an epsilon of 0.002; "
        "near ties compare micro success, completion, body error, and finally earlier iteration. "
        "The epsilon is a selection tolerance, not a significance threshold. The probe screens "
        "candidates; full Validation adjudicates available candidates. Joint L2 and Test "
        "performance are not checkpoint-selection criteria.",
        "The evidence has three distinct comparison scopes: a shared 34k training budget, "
        "the 50k endpoint comparison between D-only and M7-Raw, and a final Test comparison "
        "of separately validation-selected checkpoints. For the last comparison, GlobalRaw "
        "was trained to about 34k and selected at 33.5k, D-only was extended to 59k and "
        "selected at 54k, and M7-Raw was explored to 70k and selected at 59k. Reporting "
        "only the selected checkpoint iteration would hide the unequal search expenditure.",
        "All available principal runs use training seed 42. Paired 95% percentile intervals "
        "use 10,000 bootstrap resamples of 3,610 source groups; macro differences resample "
        "groups within each category before averaging categories. These intervals describe "
        "evaluation-corpus variation conditional on the fitted policies. They neither estimate "
        "training-seed variance nor correct for the preceding checkpoint search. Because an "
        "earlier Test phase exists in the project history, this split is not described as "
        "never accessed throughout development. The v2 choices followed the recorded validation "
        "sequence; the directory name alone is not a cryptographic pre-Test preregistration.",
    ]),
    ("5. Results", [
        "The following results distinguish matched-budget validation, checkpoint-search diagnostics, "
        "and the final selected-policy Test comparison. Tables 8-14 provide complete principal "
        "metrics, while supplementary tables and the archival evaluation inventory retain pilot "
        "and diagnostic evidence separately.",
    ]),
    ("5.1. Fixed-budget validation results", [
        "At the common 34k budget, D-only achieves the best available macro success, 89.9319%, "
        "compared with 88.1403% for GlobalRaw and 89.3026% for M7-Raw (Table 8; Fig. 4). "
        "M7-Raw has marginally higher micro success than D-only, 88.4887% versus 88.4494%, "
        "but this does not reverse the macro-first choice. The full-validation outcomes "
        "do not establish a uniform benefit of every added module. In particular, the "
        "learning-gap variants do not outperform the strongest raw-error configuration.",
        "At the 50k endpoints, M7-Raw exceeds D-only in micro success by 1.7810 percentage "
        "points, macro success by 1.7836 points, and completion by 1.1997 points. Failures "
        "decrease from 861 to 725; joint L2 decreases from 0.861048 to 0.784055 rad. This "
        "same-endpoint comparison supports a quality-gate benefit in the observed longer-budget "
        "run, while the 34k result shows that the benefit is not budget-independent. Multiple "
        "independent training seeds are needed before attributing the effect robustly to the module.",
    ]),
    ("5.2. Budget scaling and checkpoint diagnostics", [
        "M7-Raw's available full-validation macro scores rise from 89.3026% at 34k to "
        "90.5345% at 50k and 91.3028% at 59k (Table 9; Fig. 5). The selected 59k checkpoint "
        "has 91.6841% micro success, 95.2066% completion, and 635 failures. At 60k, micro "
        "success remains close (91.6056%) but macro drops to 89.9619%; the 70k endpoint "
        "has 89.8613% macro. Therefore, longer training is not monotonically better in "
        "these measurements. The 59k checkpoint is best among evaluated candidates under "
        "the adopted rule, not a demonstrated global optimum or a model-capacity limit.",
        "The dense probe (Fig. S1) records a near-zero-success region. In particular, two "
        "65000 evaluations share the checkpoint hash, manifest hash, recorded evaluator "
        "commit and settings, and both report zero success with approximately 3.09% "
        "completion. This corroborates poor observed behavior at that checkpoint; it "
        "does not by itself isolate optimization, sampler-state, or simulator mechanisms. "
        "Reported body/joint errors during such short failures are not evidence of good "
        "tracking. No new simulation was launched to diagnose the mechanism in this package.",
    ]),
    ("5.3. Final Test performance", [
        "Table 10 and Fig. 6 report the three validation-selected policies on 7,592 Test "
        "motions. M7-Raw completes 7,008 motions successfully, compared with 6,861 for "
        "D-only and 6,595 for GlobalRaw. Its micro success, macro success, and completion "
        "are 92.3077%, 90.7906%, and 95.5444%, respectively. Mean body error is 0.047820 m "
        "and mean joint L2 is 0.761526 rad. These are the best values among the three "
        "tested policies, not a claim against all external tracking methods.",
        "Relative to D-only 54k, M7-Raw 59k improves micro by 1.9362 percentage points, "
        "macro by 2.6305 points, and completion by 1.0491 points. Failures decrease by "
        "147 (20.11%); body and joint L2 errors decrease by 5.27% and 7.50%. Relative "
        "to GlobalRaw 33.5k, micro improves by 5.4399 points and failures decrease by "
        "413. These are selected-policy comparisons with the unequal budgets documented "
        "in Table 10, rather than controlled estimates of the quality gate alone.",
        "Paired outcomes sharpen this interpretation: 297 motions succeed only for M7-Raw, "
        "150 only for D-only, 6,711 for both, and 434 for neither. Thus the net gain is "
        "147, not 297. The group-bootstrap interval for the micro difference is "
        "[1.283,2.583] percentage points and for macro [1.003,4.511] points (Table 11; "
        "Fig. 7). Both intervals are positive under the stated resampling procedure, "
        "but they do not demonstrate repeatability across training seeds.",
    ]),
    ("5.4. Category-level and failure analysis", [
        "M7-Raw improves Test success in 10 categories, ties in four, and decreases in "
        "three compared with D-only (Table 13; Fig. 8). Fitness contributes 119 fewer "
        "failures out of the overall net reduction of 147. This concentration explains "
        "much of the micro gain, while the macro result gives equal weight to each "
        "category. Large proportional changes in dance (9 motions), animation (10), "
        "or custom (12) should not be interpreted as equally precise evidence.",
        "The three regressions are EgoBody, humanml, and perform. Humanml incurs five "
        "additional failures, EgoBody two, and perform one. GRAB remains tied at "
        "73.3668% success with 53 failures for every tested policy, indicating a "
        "persistent limitation in these data. End-body position termination remains "
        "the most frequent failure label (Table 14). A recorded label is a terminal "
        "condition, not a causal diagnosis of why a behavior failed.",
        "Table 12 contrasts Validation and Test metrics for the same selected checkpoint. "
        "The two splits have different category and motion compositions; a numerically "
        "higher Test micro score is not evidence that the policy learned from Test. "
        "The provenance and selection procedure, not the direction of a score difference, "
        "determine whether evaluation leakage occurred.",
    ]),
    ("6. Discussion and limitations", [
        "The observations favor separating reliable episode starts, coverage constraints, "
        "and adaptive error prioritization. However, the dataset-level gate removes only "
        "a small share of potential starts, so its measured benefit should not be "
        "explained as simply discarding a large low-quality subset. A causal account "
        "would require targeted start-distribution, failure-case, and learning-dynamics "
        "analyses beyond the aggregate outcomes available here.",
        "The principal limitations are one training seed, one robot and simulator setup, "
        "missing formal M2/M3 evaluations, unequal search budgets for final Test policies, "
        "and prior Test exposure during the wider project. The quality and online-learning "
        "profiles also retain a provisional metadata flag. Their exact values and hashes "
        "are preserved, but this is not evidence of an independently audited hyperparameter "
        "freeze. No real-robot result, matched external-method benchmark, runtime-overhead "
        "measurement, or publication-ready policy rollout sequence is invented.",
        "The supplementary tables retain negative diagnostic results, including difficulty-gap "
        "and joint-gap variants, separately from the final Test comparison. This avoids "
        "combining pilot probes with full Validation or presenting optional modules as "
        "if they contributed to the final model. The present evidence supports a "
        "carefully scoped empirical paper; a broad claim of robust state-of-the-art "
        "performance would need additional experiments.",
    ]),
    ("7. Conclusion", [
        "QD-HES composes reference-quality eligibility, motion-diversity budgets, and "
        "online error priorities in a hierarchical sampler for humanoid motion tracking. "
        "In the available experiments, its 50k validation endpoint improves over the "
        "quality-disabled counterpart, and its validation-selected 59k policy achieves "
        "the strongest aggregate final Test performance among the tested policies. "
        "The results also show the importance of distinguishing training budget, "
        "checkpoint selection, and final evaluation. Future work should establish "
        "multi-seed repeatability, matched selection budgets, and transfer beyond "
        "the present simulation setting.",
    ]),
]

REFERENCES = [
    "[1] Q. Liao, T. E. Truong, X. Huang, Y. Gao, G. Tevet, K. Sreenath, and C. K. Liu. "
    "BeyondMimic: From Motion Tracking to Versatile Humanoid Control via Guided Diffusion. "
    "arXiv:2508.08241, 2025, version 4. https://doi.org/10.48550/arXiv.2508.08241",
    "[2] K. Lee, S. Kim, Y. Lee, M. Park, H. Kim, D. Hwang, D. Kim, H. Lee, and J. Choo. "
    "PHUMA: Physically Reliable Humanoid Locomotion Dataset. arXiv:2510.26236, 2025; "
    "version 2 revised 2026. https://doi.org/10.48550/arXiv.2510.26236",
    "[3] Z. Chen, M. Ji, X. Cheng, X. Peng, X. B. Peng, and X. Wang. GMT: General Motion "
    "Tracking for Humanoid Whole-Body Control. arXiv:2506.14770, 2025. "
    "https://doi.org/10.48550/arXiv.2506.14770",
    "[4] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov. Proximal Policy "
    "Optimization Algorithms. arXiv:1707.06347, 2017. "
    "https://doi.org/10.48550/arXiv.1707.06347",
]
