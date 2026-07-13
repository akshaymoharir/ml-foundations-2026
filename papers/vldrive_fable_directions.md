# Research Directions Building on VLDrive (arXiv:2511.06256, ICCV 2025)

Three directions, each with the motivating gap, literature verification, concrete approach, validation plan, and failure risk. Ranked at the end by (a) tractability for a single part-time PhD student and (b) likely impact.

**Verification legend used throughout:**
- ✅ **Verified** — I read the claim in the cited paper/abstract during this literature pass.
- ⚠️ **Inference** — my own reasoning or extrapolation; check before writing it into a proposal.

---

## Direction 1 — Q-VLDrive: Closed-Loop Quantization Study of Language-Grounded Driving MLLMs (Pruning × Quantization Interaction)

### Gap in VLDrive
VLDrive's entire efficiency claim is *parameter count* (7B → 1.3B). ✅ Verified: the paper reports no latency, FLOPs, memory, energy, or embedded-hardware numbers, and never applies quantization. A 1.3B FP16 model is still ~2.6 GB of weights plus a KV-cache growing with T frames × N_v tokens — parameter reduction alone is not a deployment result.

### Is the gap open? (literature check)
- ✅ The VLA-for-AD survey (arXiv:2512.16760) states that meeting real-time constraints remains difficult and that sub-50 ms inference is an unmet requirement for safety-critical deployment — i.e., the field itself flags this as open.
- ✅ Quantization work for driving VLMs exists but is **not closed-loop control**: PeLiC-VLM (Advanced Engineering Informatics, 2026) quantizes attention in a driving VLM and reports 56/43 ms on Jetson AGX Orin — but for **visual question answering**, not driving the vehicle. An on-board personalized motion-control system (arXiv:2411.11913) applies AWQ INT4 to Qwen-VL — but as an advisory/personalization module, not the closed-loop planner evaluated by driving score.
- ✅ DyQ-VLA (arXiv:2603.07904) does temporal-dynamic quantization for **embodied manipulation VLAs** (~30.9% memory, ~99.5% task performance) — closest methodologically, wrong domain; no driving, no CARLA.
- ✅ A quantized VLM "semantic observer" for AVs (arXiv:2603.28888) runs NVFP4 Cosmos-Reason1-7B at ~500 ms — explicitly a 1–2 Hz *monitor beside* the control loop, not in it.
- ✅ RobotxR1 (arXiv:2505.03238) deploys a Q5-quantized Qwen2.5-3B on a Jetson Orin on a 1:10 car with ~8.3 s latency — demonstrating how immature quantized closed-loop control currently is.
- ⚠️ Inference from the above: **no systematic study exists of how weight/activation bit-width degrades closed-loop driving score in CARLA for a language-grounded driver, nor of how token pruning interacts with quantization.** I did not find one in this pass; a proposal should include a fresh search closer to submission.

### Concrete technical approach
1. Reproduce VLDrive-TinyLLaMA from the released code (github.com/ReaFly/VLDrive) and its LangAuto FP16 baseline numbers.
2. Apply PTQ ladders — GPTQ/AWQ weight-only (W8, W4), then W8A8 (SmoothQuant-style) and W4A8 — separately to the LM, the connector (Q-Former/MEFA MLPs), and the frozen visual encoder, to localize sensitivity.
3. **Core scientific question — pruning × quantization interaction:** CCDP keeps 30% of tokens, concentrating information into fewer, denser activations. Hypothesis (⚠️ mine): this amplifies activation outliers, making post-pruning activations *harder* to quantize than the unpruned baseline — measurable via kurtosis/outlier statistics of connector outputs vs. retention ratio R ∈ {0.1, 0.3, 0.5, 1.0}. If confirmed, this is a general finding about compounding compression techniques, not just a VLDrive result.
4. Mitigations where degradation appears: rotation-based outlier suppression (QuaRot/SpinQuant family), mixed-precision for the connector, and short QAT fine-tuning on the 25% mini-set VLDrive already uses for ablations.
5. Deploy the best W4A8 configuration via TensorRT-LLM / llama.cpp on a Jetson Orin-class device; report per-frame latency, memory, and energy alongside closed-loop DS (CARLA runs on a workstation streaming to the device, or measure model-only step latency on-device — clearly separating simulator cost from model cost).

### Experimental setup
- **Datasets/benchmarks:** LangAuto / -Short / -Tiny (direct comparability to VLDrive), 3 runs each as in the paper; optionally one Bench2Drive scenario subset for external validity.
- **Baselines:** FP16 VLDrive (upper bound); LMDrive-LiteLM FP16; naive round-to-nearest INT8/INT4 (lower bound); per-component quantized variants.
- **Metrics:** DS/RC/IS vs. bit-width (the "closed-loop quantization curve" — the headline plot); model-only latency (ms/frame) and memory on Orin; activation outlier statistics vs. R; Pareto frontier of DS vs. latency across {R, bit-width}.

### Main risk
The boring outcome: a 1.3B model may quantize to W4A8 with negligible DS loss and no interesting pruning interaction, reducing the paper to an engineering report. Mitigation: the interaction study and the closed-loop-vs-open-loop degradation comparison (does open-loop waypoint L2 predict closed-loop DS loss under quantization? ⚠️ likely not — compounding errors) carry scientific weight even if headline degradation is small. Secondary risk: CARLA closed-loop evaluation is slow and high-variance; budget ~3 runs × many configs = significant GPU-weeks.

---

## Direction 2 — Instruction- and Risk-Conditioned Token Pruning with Safety-Weighted Reconstruction

### Gap in VLDrive
✅ Verified from the paper: CCDP's keep/prune probabilities are computed from visual features only (Eqs. 1–4 — MLPs over F_i and its global average; the instruction embedding is not an input). Tokens are discarded *before* the LM sees the instruction, so evidence relevant only under a specific command ("turn left at the intersection" → left-side tokens) can be pruned unrecoverably; DDIA re-weights attention downstream but cannot restore discarded tokens. Additionally, ✅ the paper contains no analysis of *what* gets pruned — no breakdown of pruned-token overlap with pedestrians/cyclists/lead vehicles, and no per-infraction-type reporting; the L2 reconstruction loss (Eq. 13) is, ⚠️ by construction, dominated by large regions and nearly blind to small VRUs.

### Is the gap open? (literature check)
- ✅ SparseVLM (arXiv:2410.04417) establishes text-guided visual token pruning — but training-free, and evaluated on open-loop image/video understanding, not control.
- ✅ FastDriveVLA (arXiv:2507.23318) reports that both visual-similarity and visual-text-attention pruning "have shown poor performance in autonomous driving scenarios" and proposes reconstruction-based *foreground* pruning instead. This creates a genuine open controversy: text guidance helps in VQA, appears to hurt in driving — but ⚠️ I did not verify FastDriveVLA's evaluation protocol (likely open-loop; confirm), and neither it nor SparseVLM has been tested in **closed-loop** language-grounded driving, where the navigation instruction is far more behaviorally binding than a VQA prompt.
- ✅ ST-Prune (arXiv:2604.19145) does training-free spatio-temporal pruning for driving VLMs; ✅ a history-conditioned pruning framework exists for vision-language *navigation* (arXiv:2603.06480) — adjacent domains, neither trains an instruction-conditioned pruner jointly with a closed-loop driving objective.
- ⚠️ Inference: **learned, instruction- and risk-conditioned pruning trained end-to-end with a closed-loop driving objective and analyzed per safety scenario appears unclaimed.** The controversy above (SparseVLM vs. FastDriveVLA) is itself publishable material to resolve: is text guidance harmful in driving per se, or only when done training-free via attention heuristics?

### Concrete technical approach
1. **Instruction conditioning:** modify Eq. 3 so the pruning MLP input becomes [G_i ; L_i ; e_instr], where e_instr is a pooled instruction embedding (from the LM's own embedding layer — no new encoder). Everything else in CCDP (Gumbel-Softmax, L_prun) is unchanged, giving a clean single-variable ablation against stock VLDrive.
2. **Risk prior:** compute a cheap geometric risk map from the LiDAR channel already in the input (per-token distance / time-to-collision proxy to detected agents) and feed it as a second conditioning signal — deterministic, no learned detector needed.
3. **Safety-weighted reconstruction:** reweight L_rec (Eq. 13) with per-token weights w_j from the risk map, so failing to preserve information about nearby agents costs more than losing sky/facade texture. Since VLDrive already showed reconstruction loss correlates with trajectory loss (r = 0.65 ✅), sharpening the reconstruction prior toward safety-relevant content is a targeted, well-motivated edit.
4. **Diagnostic experiment first (cheap, week-one):** counterfactual probing of stock VLDrive — same frame, swapped instruction — to show the mask is instruction-invariant (⚠️ expected by construction), plus overlap statistics of pruned tokens vs. agent bounding boxes. This motivates the method *and* is a standalone analysis contribution if the method underdelivers.

### Experimental setup
- **Datasets/benchmarks:** LangAuto suite (comparability); Bench2Drive 220-route protocol (✅ arXiv:2406.03877 — 44 interactive scenarios, per-ability disentangled evaluation) for the safety-scenario breakdown the pruning analysis needs.
- **Baselines:** stock VLDrive/CCDP; CCDP + SparseVLM-style training-free text guidance (to isolate "learned vs. training-free" from "text vs. no text"); FastDriveVLA-style foreground reconstruction; random and structural-pooling pruning at matched R.
- **Metrics:** DS/RC/IS overall; **per-infraction-type counts** (pedestrian/vehicle collisions vs. route deviations); pruned-token-over-VRU overlap rate; mask-instruction mutual information; performance at aggressive retention (R = 0.1) where conditioning should matter most.

### Main risk
The FastDriveVLA finding may generalize: instruction text in this benchmark is mostly coarse route guidance ("turn left ahead"), and geometric saliency may already capture everything useful — conditioning adds parameters and noise for ≈0 DS. Mitigation: the risk-prior + safety-weighted-reconstruction half of the contribution is independent of instruction conditioning and directly targets the collision failure mode VLDrive itself identifies as dominant; and a rigorous negative result on instruction conditioning in closed loop, with the diagnostic analysis, still resolves a live disagreement in the literature. Second risk: retraining is required (unlike Direction 1) — though VLDrive trains only the connector + 1.3B LM with a frozen encoder, and its own ablations use a 25% data subset, so single-node training is plausible (⚠️ verify VRAM needs against the released code; a 24 GB card is likely tight and cloud A100 time should be budgeted).

---

## Direction 3 — Streaming Temporal Memory Compression: Replacing MEFA's Uniform Average

### Gap in VLDrive
✅ Verified from the paper: MEFA compresses the entire Z = 10-frame memory bank with a *uniform average*, B_i^avg = Avg(B_i) (Eq. 5), before an MLP mixes it with the current frame. Averaging destroys temporal ordering — velocity, acceleration, and approach/recede direction of other agents are exactly the cues the paper's own motivation ("infer intentions from historical trajectory") says matter. Two corroborating observations from their results: ✅ capacity Z = 20 performs *worse* than Z = 10 (DS 38.3 vs 39.8 — averaging over longer horizons dilutes rather than helps), and ✅ VLDrive's improvement margin is smallest precisely on the long-route benchmark (+7.6% on LangAuto vs +15–17% on Tiny/Short), ⚠️ consistent with (though not proof of) temporal modeling being the residual bottleneck at long horizons.

### Is the gap open? (literature check)
- ✅ ORION (arXiv:2503.19755) uses a QT-Former with a memory bank and history *queries* for long-horizon context in closed-loop Bench2Drive driving — richer than averaging, but still a query-attention design without an explicit recurrent state or fixed-budget streaming guarantee.
- ✅ The VLA-for-AD survey (arXiv:2512.16760) names streaming token compression and adaptive visual encoders as promising but immature directions for meeting real-time constraints — the field flags this open.
- ✅ Training-free spatio-temporal compression exists (ST-Prune, arXiv:2604.19145; history-conditioned pruning for VLN, arXiv:2603.06480) — but training-free methods can't reshape what the memory *stores*, only what they discard from it.
- ⚠️ Inference: a **learned, constant-budget recurrent memory (state-space / token-merging) for the connector of a language-grounded closed-loop driver, ablated directly against averaging and query-attention memories**, appears unclaimed. This one has the highest scooping risk of the three — temporal compression is a crowded, fast-moving area; re-verify immediately before proposal.

### Concrete technical approach
1. Replace Eq. 5–6 with a constant-size learned memory. Two candidates to ablate against each other: (a) a small state-space/GRU-style recurrent update M_t = f(M_{t−1}, F_t) over pooled frame descriptors — O(1) per step, explicit ordering; (b) temporal token merging — maintain a fixed budget of K memory tokens, merging the most similar (bipartite soft matching, ToMe-style ⚠️ my transfer of an image-domain technique) as new frames arrive, so old frames decay by consolidation rather than truncation.
2. Feed the memory state into MEFA exactly where B_i^avg currently enters, keeping the rest of VLDrive frozen — a surgical, single-module comparison: {average, ORION-style history queries, recurrent, token-merging} at matched parameter and token budgets.
3. Add a lightweight temporal-order probe (auxiliary loss or offline probe predicting ego/agent displacement direction from the memory state) to verify the new memory actually encodes dynamics, not just a fancier average.
4. Sweep effective horizon (Z = 10/20/40) — the key plot: averaging degrades with Z (already shown at Z = 20 ✅), a good streaming memory should be flat or improving.

### Experimental setup
- **Datasets/benchmarks:** LangAuto long track (>500 m) is the primary target since that's where the temporal bottleneck hypothesis bites; -Short/-Tiny as controls (should show little change — a useful sanity check); optionally Bench2Drive scenarios with strong temporal structure (cut-in, overtaking ✅ listed among its 44 scenarios).
- **Baselines:** stock MEFA (average); no-memory (Z = 0); ORION-style history queries; concatenation upper bound (all Z frames' tokens, no compression — expensive but bounds achievable performance).
- **Metrics:** DS/RC/IS on long routes; DS vs. horizon Z; temporal-probe accuracy; per-step connector latency/memory (the streaming-efficiency selling point).

### Main risk
Long-route closed-loop DS has high variance (±2–4 points across the tables in VLDrive itself ✅), so a real but modest gain may not separate from noise in 3 runs — budget 5+ runs and consider per-scenario success rates (Bench2Drive style) as a lower-variance endpoint. Deeper risk: the LM's own KV-cache over concatenated frame tokens may already capture most usable temporal signal, making the connector memory redundant — the concatenation upper-bound baseline tests this cheaply *first*, before any architecture work; if the upper bound isn't meaningfully above stock MEFA, kill the direction early.

---

## Ranking

### (a) Tractability for a single (part-time) PhD student
1. **Direction 1 (Q-VLDrive)** — no retraining for the core PTQ study; builds directly on released code; the expensive resource is CARLA evaluation time, which is parallelizable and schedulable. Also sits squarely on an existing model-compression/quantization thesis focus, so the tooling (GPTQ/AWQ, TensorRT) amortizes across the PhD.
2. **Direction 3 (streaming memory)** — one surgical module swap with a cheap kill-switch experiment (the concatenation upper bound) before committing; requires connector retraining but on the 25% mini-set protocol VLDrive already validated.
3. **Direction 2 (conditioned pruning)** — highest engineering surface: retraining, a risk-map pipeline, per-scenario infrastructure on two benchmarks, and the most adversarial related-work landscape to position against.

### (b) Likely impact
1. **Direction 2** — resolves a live contradiction in the literature (text-guided pruning: helps in VQA, reportedly fails in driving), introduces safety-aware compression (pruning that knows what it must not forget), and produces the per-scenario safety analysis the field lacks. Best fit for a top vision/robotics venue; informative even as a rigorous negative result.
2. **Direction 1** — first closed-loop quantization curve for language-grounded driving + a compounding-compression interaction finding would be widely cited by the deployment-focused community, and the survey-documented real-time gap guarantees an audience. Slightly narrower novelty ceiling since it composes known techniques.
3. **Direction 3** — solid and well-motivated by VLDrive's own numbers, but temporal token compression is crowded and the expected effect size (long-route DS) fights benchmark variance.

### Recommendation to bring to the advisor
Lead with **Direction 1** as the first paper (fast, on-thesis, low risk, produces reusable infrastructure: a working VLDrive reproduction + CARLA evaluation harness), with **Direction 2** as the follow-on that reuses that exact infrastructure and carries the higher-impact scientific question. Direction 3 is the backup, gated on its cheap upper-bound experiment. The two-paper arc — "how far can we compress a closed-loop driving MLLM" → "what must compression never throw away" — also reads as a coherent thesis chapter pair.

---

## Verification summary

**Checked against sources in this pass:** all VLDrive claims (equations, ablations, tables — from the paper itself); SparseVLM's text-guided training-free framing and open-loop scope; FastDriveVLA's claim that similarity/text-attention pruning underperforms in driving; ST-Prune's and the VLN pruning paper's existence and scope; Bench2Drive's protocol (44 scenarios, 220 routes, multi-ability evaluation); ORION's QT-Former memory design and Bench2Drive results; the VLA survey's sub-50 ms deployment-gap and streaming-compression statements; DyQ-VLA, PeLiC-VLM, the AWQ on-board VLM, the NVFP4 semantic observer, and RobotxR1's quantized-Orin deployment (each from abstracts/excerpts — read in full before citing in a proposal).

**My inference, not verified:** that *no* closed-loop quantization study of a language-grounded driver exists (absence of evidence from searches, not proof); FastDriveVLA's exact evaluation protocol; the pruning-amplifies-activation-outliers hypothesis; the instruction-invariance of CCDP masks (expected from the equations, untested empirically); the attribution of VLDrive's weaker long-route margin to temporal averaging; VRAM feasibility of retraining on a single 24 GB GPU; and the ToMe-style merging transfer to temporal memory. Every ⚠️ item above should be re-verified near proposal time — this area moves monthly.
