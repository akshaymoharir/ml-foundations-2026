# Research Directions Building on VLDrive (arXiv:2511.06256, ICCV 2025)

Three directions, each with the motivating gap, literature verification, concrete approach, validation plan, and failure risk. Ranked at the end by (a) tractability for a single part-time PhD student and (b) likely impact.

**Verification legend used throughout:**
- ✅ **Verified** — I read the claim in the cited paper/abstract during this literature pass.
- ⚠️ **Inference** — my own reasoning or extrapolation; check before writing it into a proposal.

---

## Direction 1 — Q-VLDrive: Closed-Loop Quantization Study of Language-Grounded Driving MLLMs (Pruning × Quantization Interaction)

### Gap in VLDrive
VLDrive's efficiency claim is *parameter count* and nothing else (7B → 1.3B). ✅ Verified: the paper reports no latency, FLOPs, memory, energy, or embedded-hardware numbers, and never applies quantization. A 1.3B FP16 model is still ~2.6 GB of weights plus a KV-cache that grows with T frames × N_v tokens. Parameter reduction alone is not a deployment result.

### Is the gap open? (literature check)
- ✅ The VLA-for-AD survey (arXiv:2512.16760) states that meeting real-time constraints remains difficult and that sub-50 ms inference is an unmet requirement for safety-critical deployment. The field flags this as open.
- ✅ Quantization work for driving VLMs exists but is **not closed-loop control**: PeLiC-VLM (Advanced Engineering Informatics, 2026) quantizes attention in a driving VLM and reports 56/43 ms on Jetson AGX Orin, but for **visual question answering**, not for driving the vehicle. An on-board personalized motion-control system (arXiv:2411.11913) applies AWQ INT4 to Qwen-VL as an advisory/personalization module, not as the closed-loop planner evaluated by driving score.
- ✅ DyQ-VLA (arXiv:2603.07904) does temporal-dynamic quantization for **embodied manipulation VLAs** (~30.9% memory, ~99.5% task performance). Closest methodologically, wrong domain: no driving, no CARLA.
- ✅ A quantized VLM "semantic observer" for AVs (arXiv:2603.28888) runs NVFP4 Cosmos-Reason1-7B at ~500 ms, explicitly a 1–2 Hz *monitor beside* the control loop rather than inside it.
- ✅ RobotxR1 (arXiv:2505.03238) deploys a Q5-quantized Qwen2.5-3B on a Jetson Orin on a 1:10 car at ~8.3 s latency, which shows how immature quantized closed-loop control currently is.
- ⚠️ Inference from the above: **no systematic study exists of how weight/activation bit-width degrades closed-loop driving score in CARLA for a language-grounded driver, nor of how token pruning interacts with quantization.** I did not find one in this pass; a proposal should include a fresh search closer to submission.

### Concrete technical approach
1. Reproduce VLDrive-TinyLLaMA from the released code (github.com/ReaFly/VLDrive) and its LangAuto FP16 baseline numbers.
2. Apply PTQ ladders (GPTQ/AWQ weight-only at W8 and W4, then W8A8 SmoothQuant-style and W4A8) separately to the LM, the connector (Q-Former/MEFA MLPs), and the frozen visual encoder, to localize sensitivity.
3. **Core scientific question — pruning × quantization interaction:** CCDP keeps 30% of tokens, which concentrates information into fewer, denser activations. Hypothesis (⚠️ mine): this amplifies activation outliers and makes post-pruning activations *harder* to quantize than the unpruned baseline. Measure it with kurtosis/outlier statistics of connector outputs against retention ratio R ∈ {0.1, 0.3, 0.5, 1.0}. If it holds, the finding is general to compounding compression techniques, not specific to VLDrive.
4. Where degradation appears, apply mitigations: rotation-based outlier suppression (QuaRot/SpinQuant family), mixed-precision for the connector, and short QAT fine-tuning on the 25% mini-set VLDrive already uses for its ablations.
5. Deploy the best W4A8 configuration via TensorRT-LLM / llama.cpp on a Jetson Orin-class device. Report per-frame latency, memory, and energy alongside closed-loop DS. CARLA runs on a workstation streaming to the device, or measure model-only step latency on-device, keeping simulator cost separate from model cost.

### Experimental setup
- **Datasets/benchmarks:** LangAuto / -Short / -Tiny (direct comparability to VLDrive), 3 runs each as in the paper; optionally one Bench2Drive scenario subset for external validity.
- **Baselines:** FP16 VLDrive (upper bound); LMDrive-LiteLM FP16; naive round-to-nearest INT8/INT4 (lower bound); per-component quantized variants.
- **Metrics:** DS/RC/IS against bit-width (the closed-loop quantization curve, which is the headline plot); model-only latency (ms/frame) and memory on Orin; activation outlier statistics against R; Pareto frontier of DS against latency across {R, bit-width}.

### Main risk
The boring outcome: a 1.3B model may quantize to W4A8 with negligible DS loss and no interesting pruning interaction, which reduces the paper to an engineering report. Mitigation: the interaction study and the closed-loop-vs-open-loop degradation comparison still carry scientific weight even when headline degradation is small. Does open-loop waypoint L2 predict closed-loop DS loss under quantization? ⚠️ Probably not, since errors compound. Secondary risk: CARLA closed-loop evaluation is slow and high-variance, so budget 3 runs × many configs, which comes to significant GPU-weeks.

---

## Direction 2 — Instruction- and Risk-Conditioned Token Pruning with Safety-Weighted Reconstruction

### Gap in VLDrive
✅ Verified from the paper: CCDP's keep/prune probabilities come from visual features only (Eqs. 1–4, MLPs over F_i and its global average, with no instruction embedding as input). Tokens are discarded *before* the LM sees the instruction, so evidence that only matters under a specific command ("turn left at the intersection" → left-side tokens) can be pruned unrecoverably. DDIA re-weights attention downstream but cannot restore a discarded token. ✅ The paper also contains no analysis of *what* gets pruned: no breakdown of pruned-token overlap with pedestrians, cyclists, or lead vehicles, and no per-infraction-type reporting. The L2 reconstruction loss (Eq. 13) is ⚠️ by construction dominated by large regions and nearly blind to small VRUs.

### Is the gap open? (literature check)
- ✅ SparseVLM (arXiv:2410.04417) establishes text-guided visual token pruning, but training-free and evaluated on open-loop image/video understanding rather than control.
- ✅ FastDriveVLA (arXiv:2507.23318) reports that both visual-similarity and visual-text-attention pruning perform poorly in autonomous driving scenarios, and proposes reconstruction-based *foreground* pruning instead. That is a genuine open controversy: text guidance helps in VQA and appears to hurt in driving. ⚠️ I did not verify FastDriveVLA's evaluation protocol (likely open-loop; confirm), and neither it nor SparseVLM has been tested in **closed-loop** language-grounded driving, where the navigation instruction is far more behaviorally binding than a VQA prompt.
- ✅ ST-Prune (arXiv:2604.19145) does training-free spatio-temporal pruning for driving VLMs, and ✅ a history-conditioned pruning framework exists for vision-language *navigation* (arXiv:2603.06480). Both are adjacent domains, and neither trains an instruction-conditioned pruner jointly with a closed-loop driving objective.
- ⚠️ Inference: **learned, instruction- and risk-conditioned pruning, trained end-to-end with a closed-loop driving objective and analyzed per safety scenario, appears unclaimed.** The SparseVLM-vs-FastDriveVLA controversy is itself publishable material to resolve: is text guidance harmful in driving as such, or only when done training-free through attention heuristics?

### Concrete technical approach
1. **Instruction conditioning:** modify Eq. 3 so the pruning MLP input becomes [G_i ; L_i ; e_instr], where e_instr is a pooled instruction embedding taken from the LM's own embedding layer, so no new encoder is needed. Everything else in CCDP (Gumbel-Softmax, L_prun) stays unchanged, which gives a clean single-variable ablation against stock VLDrive.
2. **Risk prior:** compute a cheap geometric risk map from the LiDAR channel already in the input (per-token distance / time-to-collision proxy to detected agents) and feed it as a second conditioning signal. Deterministic, no learned detector needed.
3. **Safety-weighted reconstruction:** reweight L_rec (Eq. 13) with per-token weights w_j from the risk map, so failing to preserve information about a nearby agent costs more than losing sky or facade texture. VLDrive already showed reconstruction loss correlates with trajectory loss (r = 0.65 ✅), so sharpening the reconstruction prior toward safety-relevant content is a targeted edit rather than a speculative one.
4. **Diagnostic experiment first (cheap, week-one):** probe stock VLDrive counterfactually with the same frame and a swapped instruction, to show the mask is instruction-invariant (⚠️ expected by construction), and compute overlap statistics of pruned tokens against agent bounding boxes. This motivates the method and stands alone as an analysis contribution if the method underdelivers.

### Experimental setup
- **Datasets/benchmarks:** LangAuto suite (comparability); Bench2Drive 220-route protocol (✅ arXiv:2406.03877, 44 interactive scenarios, per-ability disentangled evaluation) for the safety-scenario breakdown the pruning analysis needs.
- **Baselines:** stock VLDrive/CCDP; CCDP + SparseVLM-style training-free text guidance, which isolates "learned vs. training-free" from "text vs. no text"; FastDriveVLA-style foreground reconstruction; random and structural-pooling pruning at matched R.
- **Metrics:** DS/RC/IS overall; **per-infraction-type counts** (pedestrian/vehicle collisions against route deviations); pruned-token-over-VRU overlap rate; mask-instruction mutual information; performance at aggressive retention (R = 0.1), where conditioning should matter most.

### Main risk
The FastDriveVLA finding may generalize. Instruction text in this benchmark is mostly coarse route guidance ("turn left ahead"), geometric saliency may already capture everything useful, and conditioning then adds parameters and noise for ≈0 DS. Mitigation: the risk-prior and safety-weighted-reconstruction half of the contribution does not depend on instruction conditioning and targets the collision failure mode VLDrive itself identifies as dominant. A rigorous negative result on instruction conditioning in closed loop, backed by the diagnostic analysis, still resolves a live disagreement in the literature. Second risk: this direction requires retraining, unlike Direction 1. VLDrive trains only the connector plus the 1.3B LM with a frozen encoder, and its own ablations use a 25% data subset, so single-node training is plausible (⚠️ verify VRAM needs against the released code; a 24 GB card is likely tight and cloud A100 time should be budgeted).

---

## Direction 3 — Streaming Temporal Memory Compression: Replacing MEFA's Uniform Average

### Gap in VLDrive
✅ Verified from the paper: MEFA compresses the entire Z = 10-frame memory bank with a *uniform average*, B_i^avg = Avg(B_i) (Eq. 5), before an MLP mixes it with the current frame. Averaging destroys temporal ordering, so velocity, acceleration, and the approach/recede direction of other agents disappear. Those are exactly the cues the paper's own motivation ("infer intentions from historical trajectory") says matter. Two of their results corroborate this: ✅ capacity Z = 20 performs *worse* than Z = 10 (DS 38.3 vs 39.8, so averaging over longer horizons dilutes rather than helps), and ✅ VLDrive's improvement margin is smallest on the long-route benchmark (+7.6% on LangAuto against +15–17% on Tiny/Short), which is ⚠️ consistent with temporal modeling being the residual bottleneck at long horizons, though not proof of it.

### Is the gap open? (literature check)
- ✅ ORION (arXiv:2503.19755) uses a QT-Former with a memory bank and history *queries* for long-horizon context in closed-loop Bench2Drive driving. Richer than averaging, but still a query-attention design with no explicit recurrent state and no fixed-budget streaming guarantee.
- ✅ The VLA-for-AD survey (arXiv:2512.16760) names streaming token compression and adaptive visual encoders as promising but immature directions for meeting real-time constraints, so the field flags this open.
- ✅ Training-free spatio-temporal compression exists (ST-Prune, arXiv:2604.19145; history-conditioned pruning for VLN, arXiv:2603.06480), but a training-free method cannot reshape what the memory *stores*, only what it discards from it.
- ⚠️ Inference: a **learned, constant-budget recurrent memory (state-space or token-merging) for the connector of a language-grounded closed-loop driver, ablated directly against averaging and query-attention memories**, appears unclaimed. This one carries the highest scooping risk of the three, since temporal compression is a crowded and fast-moving area. Re-verify immediately before the proposal.

### Concrete technical approach
1. Replace Eqs. 5–6 with a constant-size learned memory. Two candidates to ablate against each other: (a) a small state-space/GRU-style recurrent update M_t = f(M_{t−1}, F_t) over pooled frame descriptors, which is O(1) per step and keeps explicit ordering; (b) temporal token merging, which maintains a fixed budget of K memory tokens and merges the most similar ones (bipartite soft matching, ToMe-style, ⚠️ my transfer of an image-domain technique) as new frames arrive, so old frames decay by consolidation rather than truncation.
2. Feed the memory state into MEFA exactly where B_i^avg currently enters and keep the rest of VLDrive frozen. That gives a surgical, single-module comparison across {average, ORION-style history queries, recurrent, token-merging} at matched parameter and token budgets.
3. Add a lightweight temporal-order probe (an auxiliary loss, or an offline probe predicting ego/agent displacement direction from the memory state) to check that the new memory encodes dynamics rather than acting as a fancier average.
4. Sweep effective horizon (Z = 10/20/40). This is the key plot: averaging degrades with Z (already shown at Z = 20 ✅), and a good streaming memory should be flat or improving.

### Experimental setup
- **Datasets/benchmarks:** LangAuto long track (>500 m) is the primary target, since that is where the temporal bottleneck hypothesis bites; -Short/-Tiny act as controls and should show little change, which is a useful sanity check; optionally Bench2Drive scenarios with strong temporal structure (cut-in, overtaking, ✅ listed among its 44 scenarios).
- **Baselines:** stock MEFA (average); no-memory (Z = 0); ORION-style history queries; concatenation upper bound (all Z frames' tokens, no compression), which is expensive but bounds achievable performance.
- **Metrics:** DS/RC/IS on long routes; DS against horizon Z; temporal-probe accuracy; per-step connector latency and memory, which is the streaming-efficiency selling point.

### Main risk
Long-route closed-loop DS has high variance (±2–4 points across the tables in VLDrive itself ✅), so a real but modest gain may not separate from noise in 3 runs. Budget 5+ runs and consider per-scenario success rates (Bench2Drive style) as a lower-variance endpoint. The deeper risk: the LM's own KV-cache over concatenated frame tokens may already capture most of the usable temporal signal, which would make the connector memory redundant. The concatenation upper-bound baseline tests this cheaply and should run *first*, before any architecture work. If the upper bound does not sit meaningfully above stock MEFA, kill the direction early.

---

## Ranking

### (a) Tractability for a single (part-time) PhD student
1. **Direction 1 (Q-VLDrive)** — the core PTQ study needs no retraining and builds directly on released code. The expensive resource is CARLA evaluation time, which is parallelizable and schedulable. It also sits squarely on an existing model-compression/quantization thesis focus, so the tooling (GPTQ/AWQ, TensorRT) amortizes across the PhD.
2. **Direction 3 (streaming memory)** — one surgical module swap, with a cheap kill-switch experiment (the concatenation upper bound) before committing. Needs connector retraining, but on the 25% mini-set protocol VLDrive already validated.
3. **Direction 2 (conditioned pruning)** — the largest engineering surface: retraining, a risk-map pipeline, per-scenario infrastructure on two benchmarks, and the most adversarial related-work landscape to position against.

### (b) Likely impact
1. **Direction 2** — resolves a live contradiction in the literature (text-guided pruning helps in VQA and reportedly fails in driving), introduces safety-aware compression, meaning pruning that knows what it must not forget, and produces the per-scenario safety analysis the field lacks. Best fit for a top vision/robotics venue, and informative even as a rigorous negative result.
2. **Direction 1** — the first closed-loop quantization curve for language-grounded driving, plus a compounding-compression interaction finding, would be widely cited by the deployment-focused community, and the survey-documented real-time gap guarantees an audience. The novelty ceiling is lower, since it composes known techniques.
3. **Direction 3** — solid and well-motivated by VLDrive's own numbers, but temporal token compression is crowded and the expected effect size (long-route DS) fights benchmark variance.

### Recommendation to bring to the advisor
Lead with **Direction 1** as the first paper: fast, on-thesis, low risk, and it produces reusable infrastructure in a working VLDrive reproduction and a CARLA evaluation harness. **Direction 2** follows on, reusing that exact infrastructure and carrying the higher-impact scientific question. Direction 3 is the backup, gated on its cheap upper-bound experiment. The two-paper arc, from "how far can we compress a closed-loop driving MLLM" to "what must compression never throw away," also reads as a coherent thesis chapter pair.

---

## Verification summary

**Checked against sources in this pass:** all VLDrive claims (equations, ablations, tables, from the paper itself); SparseVLM's text-guided training-free framing and open-loop scope; FastDriveVLA's claim that similarity/text-attention pruning underperforms in driving; ST-Prune's and the VLN pruning paper's existence and scope; Bench2Drive's protocol (44 scenarios, 220 routes, multi-ability evaluation); ORION's QT-Former memory design and Bench2Drive results; the VLA survey's sub-50 ms deployment-gap and streaming-compression statements; DyQ-VLA, PeLiC-VLM, the AWQ on-board VLM, the NVFP4 semantic observer, and RobotxR1's quantized-Orin deployment (each from abstracts/excerpts, to be read in full before citing in a proposal).

**My inference, not verified:** that *no* closed-loop quantization study of a language-grounded driver exists (absence of evidence from searches, not proof); FastDriveVLA's exact evaluation protocol; the pruning-amplifies-activation-outliers hypothesis; the instruction-invariance of CCDP masks, which the equations imply but nobody has tested empirically; the attribution of VLDrive's weaker long-route margin to temporal averaging; VRAM feasibility of retraining on a single 24 GB GPU; and the ToMe-style merging transfer to temporal memory. Re-verify every ⚠️ item near proposal time, since this area moves monthly.
