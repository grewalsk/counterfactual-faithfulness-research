# Stage 38 to next mathematical decision

**Decision:** run a small, development-only **Stage 38.1 coefficient-matching audit**. Do **not** open Stage 39, planning, or a larger hybrid model yet.

**Evidence base:** source-bound Stage 38 run `ceb85af5b4b9`; commit `a7ed07e2e79bc4da77e022f7765239b260bff35c`; locked PushT panel of 6,144 rows; independent recomputation from the result bundle; primary literature reviewed through 2026-08-23.

**Notation:** “S-PSCD” denotes the Stage 38 full semigroup predictive-state closure distillation model. “Overshooting” denotes its latent-only semigroup control. “RH-PSCD” below means the proposed risk-sensitive hybrid predictive-state closure model; it is a hypothesis, not an established method.

## 1. Executive decision

Run **Stage 38.1**, whose first and mandatory tier is a coefficient-matched comparison between full S-PSCD and latent overshooting on construction/model-selection/calibration data only. The current Stage 38 comparison is informative but not a fair causal ablation: after component-weight normalization, the full model's semigroup latent coefficient is `0.45 lambda`, whereas the latent-only control's is `1.00 lambda`. Because both use the same outer `lambda`, overshooting receives exactly `1/0.45 = 2.222...` times the latent-consistency pressure. It is therefore simultaneously:

- a **confounded component ablation**, because removing carrier and physical losses also changes the latent coefficient;
- a **conservative benchmark** for full S-PSCD, because the control is given greater pressure on the component it retains;
- not evidence that carrier and physical closure terms have zero value at equal latent pressure.

The coefficient-matched overshooting weights are unambiguous: use `lambda_os = 0.45 lambda_full`, hence **0.90 for JEPA-WM** and **0.45 for DINO-WM** under the Stage 38 selected full-model weights. Preserve the current mass-matched overshooting run as a sensitivity control, but do not call it the causal ablation.

Do not simply adopt overshooting as a successful final method. It matches or beats S-PSCD's mean, but it also fails the locked tail/closure criteria. Do not add CVaR, a contact gate, or a Koopman layer before repairing the control: each would add a second explanation to an unresolved comparison. If, and only if, full S-PSCD beats the coefficient-matched control on both frozen representations under the preregistered development gates in Section 8, proceed within Stage 38.1 to an event/reset diagnostic. That second tier tests whether explicit hybrid structure removes the concentrated `post_contact -> contact` failure. Stage 39 is reserved for a fresh confirmation after both tiers pass.

**Confidence:** 0.90 that coefficient matching is the correct immediate action; 0.75 that contact-event/reset structure is the leading next scientific hypothesis; below 0.50 that the present carrier contains enough label-free information to make RH-PSCD work. The principal uncertainty is identifiability: the rows show a repeatable hybrid-transition signature, but they do not distinguish guard-timing error, reset-map error, inherited carrier/readout insufficiency, and macro-step aliasing.

## 2. Evidence and integrity audit

### 2.1 Source and bundle integrity

The checkout was clean on `codex/stage34-predictive-fiber-abstraction`, and `HEAD` exactly matched the certified source commit:

```text
a7ed07e2e79bc4da77e022f7765239b260bff35c
```

The result manifest contained 176 entries. Independent verification found no missing files, hash mismatches, unlisted result files, or bad `.sha256` sidecars. The three source-bound files also matched their recorded hashes and byte counts exactly:

| File | SHA-256 prefix | Bytes |
|---|---:|---:|
| `notebooks/38_cross_model_pscd_confirmation.ipynb` | `3d14b792845bb491...` | 317,738 |
| `notebooks/build_stage38_cross_model_pscd_notebook.py` | `5c230358154afcff...` | 79,561 |
| `src/cf_faithfulness/stage38_cross_model_pscd.py` | `5e9ea6712e9260c...` | 16,793 |

The notebook reported `SOURCE_BOUND_EXECUTION_VERIFIED`, ran through the locked closure header, and certified that evaluation statistics were unread at model freeze. Frozen checkpoint parameter hashes did not change; construction, selection, calibration, and evaluation pools were disjoint; JEPA/DINO and seed panels were matched; planning remained sealed. The evaluation certificate covered 25 fitted models (24 learned model/variant/seed cells plus the simulator), and each closure model had 1,910,286 trainable parameters. `FAILURE_TRACE.txt` was `NONE`. Simulator preflight passed with physical error `.112643` and gain `.915614`; the locked simulator control passed with physical error `.062613` and gain `.971598` (95% interval `[.9682,.9749]`). The run used Python 3.13.15, PyTorch 2.11, CUDA 12.8, and an RTX PRO 6000 Blackwell. These checks support a **scientific failure**, not a pipeline failure.

One metadata defect should be corrected before any future report generation: `stage38_decision.json` contains a claim-boundary field that names JEPA-WM and DINO-WM as `representations_confirmed` even though both panels failed and `closure_confirmed=false`. The contradiction is wording in decision metadata; it does not alter the locked measurements.

### 2.2 Exact implemented objective

Let the frozen carrier be `z_t`, the four-step history encoder be

\[
s_t=E_\theta(z_{t-3:t}),
\]

and let the learned residual, three-expert mixture transition and readouts be

\[
\widehat s_{t+1}=F_\theta(s_t,a_t),\qquad
\widehat z_t=D_z(s_t),\qquad
\widehat x_t=D_x(s_t).
\]

Carrier, action, and physical targets are normalized. The physical training target is the output of the native construction-only grounded ridge readout, not simulator ground truth. With mean-squared normalized component losses, the implementation is:

\[
L_{1}=0.45L^{(1)}_z+0.25L^{(1)}_x+0.25L^{(1)}_s,
\]

where the direct-state target in `L_s^(1)` is `stopgrad(E(z_{t-2:t+1}))`. Free running begins from the single warm-up anchor at index 3 and recursively rolls through the remaining word:

\[
L_{\mathrm{fr}}=0.45L^{\mathrm{fr}}_z+0.25L^{\mathrm{fr}}_x+0.20L^{\mathrm{fr}}_s.
\]

Its time-step component losses are means over rollout positions. The free-run outer weight is zero for the one-step method and one otherwise.

For every eligible anchor and horizon `h in {2,4,8}`, including the cold anchor, the code compares an `h`-step composition against a stop-gradient directly encoded future state. Pair-level losses are averaged equally. For supplied component weights `(w_z,w_x,w_s)`, the semigroup term is

\[
L_{\mathrm{sg}}=
\frac{w_zL^{\mathrm{sg}}_z+w_xL^{\mathrm{sg}}_x+w_sL^{\mathrm{sg}}_s}
{w_z+w_x+w_s},
\qquad
L=L_1+\mathbf 1_{\mathrm{free}}L_{\mathrm{fr}}+\lambda L_{\mathrm{sg}}.
\]

The full weights are `(0.35,0.20,0.45)` and the overshooting weights are `(0,0,1)`. They already sum to one, so the effective latent coefficients are exactly `0.45 lambda` and `1.00 lambda`. The 2.222 factor is not an approximation due to batching; it follows algebraically from the implemented normalization.

The locked physical score is endpoint-only normalized path error after the warm-up. The locked semigroup score is a path average. These quantities answer different questions and should not be expected to rank models identically.

### 2.3 Locked panel and independent recomputation

The panel has 6,144 rows:

\[
2\ \text{models}\times 3\ \text{training seeds}\times
32\ \text{trajectory families}\times4\ \text{records}\times8\ \text{words}.
\]

There are 1,024 base cases per model/seed and no duplicated model-seed-case rows. The independent unit for broad physical diversity is at most the 32 trajectory families, not 6,144 rows. Words, records, and seeds are crossed/repeated observations. All reported aggregate values checked by the audit agreed to a maximum absolute discrepancy of `4.44e-16`.

The pooled descriptive comparison is:

| Model | Method | Mean | Median | p95 | CVaR95 | Cat. rate | Max |
|---|---|---:|---:|---:|---:|---:|---:|
| DINO | Native | .255402 | .168409 | .781356 | 1.343898 | .038086 | 4.209299 |
| DINO | One-step | .510768 | .392706 | 1.304803 | 1.885146 | .097005 | 4.852985 |
| DINO | PSCD | .179071 | .109206 | .581407 | 1.051412 | .022461 | 3.809470 |
| DINO | Overshooting | **.169181** | **.099606** | **.534263** | **1.031815** | **.021810** | **3.527399** |
| DINO | S-PSCD | .173954 | .105256 | .535330 | 1.034952 | .022135 | 3.694094 |
| DINO | Wrong history | .255076 | .168169 | .802781 | 1.170037 | .030273 | 3.795927 |
| JEPA | Native | .155129 | .080860 | .512644 | 1.073011 | .022461 | 5.484285 |
| JEPA | One-step | .654984 | .489553 | 1.682776 | 2.178638 | .201823 | 5.525719 |
| JEPA | PSCD | .127820 | .051977 | .623257 | 1.026100 | .017253 | 4.180871 |
| JEPA | Overshooting | .119062 | **.048604** | **.487460** | .954171 | .017253 | 4.095049 |
| JEPA | S-PSCD | **.118520** | .049538 | .543974 | **.952211** | **.012695** | **4.097801** |
| JEPA | Wrong history | .200903 | .109451 | .720008 | 1.215138 | .026367 | 4.299836 |

This table makes two facts coexist. First, recurrent predictive state helps substantially relative to native, one-step, and wrong-history controls. Second, the full component set does not establish an advantage over the stronger latent-only objective, and every candidate has an unacceptable heavy tail.

### 2.4 Paired uncertainty and specificity

I resampled training seeds and trajectory families hierarchically while preserving paired methods within each sampled unit. The following are S-PSCD minus comparator mean differences; negative is better for S-PSCD:

| Model | Comparator | Mean difference | Hierarchical 95% interval | S-PSCD row win rate |
|---|---|---:|---:|---:|
| DINO | Native | -.081449 | [-.087879, -.075407] | .783 |
| DINO | One-step | -.336815 | [-.409111, -.245854] | .929 |
| DINO | PSCD | -.005117 | [-.007907, -.001856] | .524 |
| DINO | Overshooting | **+.004773** | **[+.000392, +.008718]** | .399 |
| DINO | Wrong history | -.081123 | [-.089944, -.071671] | .740 |
| JEPA | Native | -.036609 | [-.041738, -.031864] | .739 |
| JEPA | One-step | -.536464 | [-.617405, -.460495] | .981 |
| JEPA | PSCD | -.009300 | [-.015422, -.004862] | .598 |
| JEPA | Overshooting | -.000542 | [-.004656, +.004153] | .494 |
| JEPA | Wrong history | -.082383 | [-.094499, -.070154] | .808 |

Against overshooting, S-PSCD is worse for DINO and indistinguishable for JEPA. The protocol-style physical ratios remain within the 1.05 cap for every seed, but semigroup ratios fail in five of six model-seed cells:

| Model | Seed | Physical S/OS | Semigroup S/OS |
|---|---:|---:|---:|
| DINO | 3801 | 1.000930 | 1.138934 |
| DINO | 3802 | 1.037580 | 1.041194 |
| DINO | 3803 | 1.046969 | 1.246716 |
| JEPA | 3801 | .963752 | 1.165836 |
| JEPA | 3802 | 1.034106 | 1.112824 |
| JEPA | 3803 | .990065 | 1.199002 |

The tail differences, with the same hierarchical resampling, are:

| Model | S-PSCD minus | p95 delta (95% interval) | CVaR95 delta (95% interval) | Cat. delta (95% interval) |
|---|---|---|---|---|
| DINO | Native | -.2460 [-.3391, -.1535] | -.3089 [-.3854, -.2301] | -.01595 [-.02474, -.00684] |
| DINO | PSCD | -.0461 [-.1100, +.0402] | -.0165 [-.0525, +.0185] | -.00033 [-.00586, +.00586] |
| DINO | Overshooting | +.0011 [-.0746, +.0547] | +.0031 [-.0579, +.0577] | +.00033 [-.01074, +.01204] |
| JEPA | Native | +.0313 [-.0412, +.1190] | -.1208 [-.2090, -.0415] | -.00977 [-.01530, -.00423] |
| JEPA | PSCD | -.0793 [-.1912, -.0127] | -.0739 [-.1291, -.0111] | -.00456 [-.01042, -.00033] |
| JEPA | Overshooting | +.0565 [-.0354, +.0909] | -.0020 [-.0359, +.0293] | -.00456 [-.01139, +.00098] |

Thus S-PSCD improves parts of the tail relative to native and PSCD, but not reliably relative to overshooting. The locked rejection is not contradicted by the independent audit.

The global decision `jepa_pscd_confirmation_failed` is internally consistent with the measurements: both model panels failed overshooting noninferiority and horizon/mode/tail closure requirements, while the simulator controls passed. Planning was therefore correctly left sealed.

### 2.5 Required S-PSCD stratified audit

All entries below are `mean / median / p95 / CVaR95 / catastrophic rate / maximum`.

**By seed**

| Model, seed | Six statistics |
|---|---|
| DINO 3801 | .172861 / .109758 / .526645 / .998451 / .017578 / 3.343049 |
| DINO 3802 | .173338 / .099456 / .521459 / 1.033677 / .019531 / 3.571027 |
| DINO 3803 | .175662 / .105520 / .557716 / 1.052556 / .029297 / 3.694094 |
| JEPA 3801 | .117274 / .048412 / .525354 / .951806 / .012695 / 3.989307 |
| JEPA 3802 | .120302 / .048347 / .548205 / .955760 / .012695 / 4.097801 |
| JEPA 3803 | .117985 / .051372 / .550837 / .932869 / .012695 / 3.993742 |

**By word length**

| Model, length | Six statistics |
|---|---|
| DINO, 9 | .138603 / .098702 / .301937 / .649613 / .006510 / 1.297951 |
| DINO, 10 | .215395 / .097814 / .988444 / 1.463536 / .046875 / 3.694094 |
| DINO, 11 | .187013 / .101307 / .755671 / 1.101933 / .035156 / 1.630393 |
| DINO, 12 | .154804 / .122377 / .397443 / .517437 / 0 / .925296 |
| JEPA, 9 | .072181 / .034280 / .275285 / .443547 / 0 / .723256 |
| JEPA, 10 | .167590 / .059727 / .941485 / 1.504280 / .041667 / 4.097801 |
| JEPA, 11 | .130811 / .043992 / .710844 / .835642 / .002604 / 1.072209 |
| JEPA, 12 | .103498 / .066494 / .362315 / .631916 / .006510 / 1.302836 |

Length alone is not monotone; length 10 is worst because the word identities and events at that length are difficult.

**By exact word**

| Model | Word | Mean | Median | p95 | CVaR95 | Cat. | Max |
|---|---|---:|---:|---:|---:|---:|---:|
| DINO | `ACACADCDACBD` | .197887 | .154581 | .419370 | .578137 | 0 | .925296 |
| DINO | `ADAABCDCB` | .166323 | .111632 | .306541 | .886043 | .013021 | 1.297951 |
| DINO | `ADCBBDABDDAD` | .111720 | .083683 | .242019 | .413405 | 0 | .596167 |
| DINO | `BADBBABDABC` | .102565 | .061320 | .382423 | .449448 | 0 | .567297 |
| DINO | `BBCBDAACAD` | .274757 | .180819 | .989706 | 1.477735 | .046875 | 3.694094 |
| DINO | `BDACBAACBDB` | .271460 | .153421 | 1.130015 | 1.276433 | .070312 | 1.630393 |
| DINO | `BDDBCDCACC` | .156033 | .049366 | .974290 | 1.425111 | .046875 | 2.323185 |
| DINO | `DDCCDAABD` | .110882 | .073257 | .277207 | .392694 | 0 | .584285 |
| JEPA | `ACACADCDACBD` | .140702 | .076065 | .543283 | .818756 | .013021 | 1.302836 |
| JEPA | `ADAABCDCB` | .079736 | .034737 | .408337 | .553088 | 0 | .723256 |
| JEPA | `ADCBBDABDDAD` | .066295 | .048688 | .194981 | .292878 | 0 | .385052 |
| JEPA | `BADBBABDABC` | .086680 | .020517 | .428801 | .588086 | .005208 | 1.072209 |
| JEPA | `BBCBDAACAD` | .196963 | .078150 | .892675 | 1.463200 | .028646 | 4.097801 |
| JEPA | `BDACBAACBDB` | .174941 | .069623 | .798172 | .879317 | 0 | .995291 |
| JEPA | `BDDBCDCACC` | .138217 | .029238 | 1.048587 | 1.506232 | .054688 | 2.390110 |
| JEPA | `DDCCDAABD` | .064627 | .033369 | .238856 | .301382 | 0 | .407417 |

**By initial mode**

| Model, initial mode | Six statistics |
|---|---|
| DINO contact | .081398 / .077790 / .159694 / .187393 / 0 / .291001 |
| DINO free | .122036 / .101546 / .247384 / .301444 / 0 / .420527 |
| DINO post-contact | .386848 / .235515 / 1.140596 / 1.576281 / .088542 / 3.694094 |
| DINO pre-contact | .105533 / .089755 / .224464 / .288780 / 0 / .394897 |
| JEPA contact | .043633 / .037278 / .093358 / .126844 / 0 / .225912 |
| JEPA free | .083103 / .044146 / .259463 / .457201 / 0 / .988066 |
| JEPA post-contact | .293139 / .116661 / 1.001054 / 1.534101 / .050781 / 4.097801 |
| JEPA pre-contact | .054206 / .036093 / .176343 / .212935 / 0 / .362948 |

**By terminal mode**

| Model, terminal mode | n | Six statistics |
|---|---:|---|
| DINO contact | 657 | .366707 / .184478 / 1.163277 / 1.653354 / .103501 / 3.694094 |
| DINO post-contact | 2,415 | .121515 / .096198 / .286793 / .396752 / 0 / .925296 |
| JEPA contact | 657 | .333578 / .197260 / 1.003994 / 1.604421 / .051750 / 4.097801 |
| JEPA post-contact | 2,415 | .060014 / .040858 / .159005 / .264739 / .002070 / 1.302836 |

**By initial-to-terminal mode pair**

| Model | Mode pair | n | Mean | Median | p95 | CVaR95 | Cat. | Max |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| DINO | contact -> contact | 168 | .074403 | .075332 | .132505 | .177042 | 0 | .273197 |
| DINO | contact -> post-contact | 600 | .083357 | .078216 | .161425 | .187965 | 0 | .291001 |
| DINO | free -> contact | 75 | .085514 | .083679 | .164126 | .168519 | 0 | .170734 |
| DINO | free -> post-contact | 693 | .125988 | .105450 | .251717 | .307319 | 0 | .420527 |
| DINO | post-contact -> contact | 399 | **.552330** | .389562 | **1.270423** | **1.938042** | **.170426** | 3.694094 |
| DINO | post-contact -> post-contact | 369 | .207912 | .166980 | .451112 | .604982 | 0 | .925296 |
| DINO | pre-contact -> contact | 15 | .108904 | .098806 | .144180 | .156795 | 0 | .156795 |
| DINO | pre-contact -> post-contact | 753 | .105466 | .088925 | .227062 | .290446 | 0 | .394897 |
| JEPA | contact -> contact | 168 | .044945 | .043208 | .089191 | .100933 | 0 | .110113 |
| JEPA | contact -> post-contact | 600 | .043265 | .034976 | .095149 | .134353 | 0 | .225912 |
| JEPA | free -> contact | 75 | .228365 | .030404 | .708844 | .825120 | 0 | .988066 |
| JEPA | free -> post-contact | 693 | .067382 | .044433 | .177870 | .238326 | 0 | .422125 |
| JEPA | post-contact -> contact | 399 | **.479857** | .303574 | **1.160556** | **1.954660** | **.085213** | 4.097801 |
| JEPA | post-contact -> post-contact | 369 | .091241 | .060330 | .275475 | .584119 | .013550 | 1.302836 |
| JEPA | pre-contact -> contact | 15 | .201309 | .197811 | .327130 | .362948 | 0 | .362948 |
| JEPA | pre-contact -> post-contact | 753 | .051275 | .035399 | .162044 | .195908 | 0 | .297605 |

These are descriptive conditional statistics because the mode labels are simulator-derived post hoc. They are not permissible oracle inputs at locked inference.

## 3. Diagnosis

### 3.1 Best-supported mechanism

The failure is a **structured event-transition failure, inherited partly from the carrier/native grounding, that ordinary latent composition does not expose**. The strongest evidence is:

1. Every one of DINO's 68 catastrophic S-PSCD rows is `post_contact -> contact`; 34 of JEPA's 39 are in that cell, with the other five `post_contact -> post_contact`.
2. Mode pair explains far more variance than word alone. One-way effect fractions are `.387`/.355 for mode pair, `.071`/.040 for word, and only `.028`/.026 for physical trajectory ID (DINO/JEPA). A mode-plus-word additive model explains `.467`/.399; adding their interaction raises this to `.578`/.480. The action word matters chiefly through the event itinerary it induces.
3. The failure is stable across optimization seeds: seed-mean standard deviations are only `.00150` and `.00158`; mean rank correlations across seeds are `.934` and `.957`. Fourteen DINO and nine JEPA base cases are catastrophic in all three seeds.
4. It recurs across frozen representation families. Of 33 base units catastrophic for at least one model/seed, 13 occur in both JEPA and DINO. Cross-model unit-mean Pearson correlation is `.913` and Spearman correlation is `.748`.
5. The worst common case—trajectory 62679, record 4226793, word `BBCBDAACAD`—has S-PSCD physical NMSE about `3.34–3.69` for DINO and `3.99–4.10` for JEPA, while its S-PSCD semigroup NMSE is only `.06–.08`. Low measured latent composition defect is therefore compatible with catastrophic physical event behavior.
6. Physical S-PSCD error remains correlated with native error (Spearman `.727` DINO, `.745` JEPA), while its association with its own semigroup score is only `.360`/.519. Of S-PSCD catastrophes, 88.2% for DINO and 64.1% for JEPA are already catastrophic under the native endpoint readout. S-PSCD repairs some native catastrophes but cannot erase an inherited ambiguity merely by becoming self-consistent.

This pattern is compatible with a guard-crossing timing or reset defect. Near contact, a small geometric state error can change whether a guard is crossed, when it is crossed, or which reset applies; that discrete itinerary change can dominate later continuous error. “Saltation defect” is a useful mechanistic hypothesis, but the current CSV has only coarse initial/terminal labels, not the event times and pre/post states needed to identify an actual saltation matrix. The evidence supports **hybrid-transition localization**, not yet the stronger claim that saltation sensitivity is the proven cause.

### 3.2 Competing explanations

| Explanation | Assessment from Stage 38 |
|---|---|
| Optimization variance | Largely disfavored by stable seed means and ranks. It cannot explain the shared worst cases. |
| A few trajectory families | Disfavored as the sole cause: trajectory ID explains about 2.6–2.8% marginal variance, and the top three families contain only 28–31% of catastrophic rows. Cluster dependence still matters for inference. |
| Specific word algebra | Partly supported. Words and word-by-mode interactions matter, but length is non-monotone and word effects do not replace event-mode effects. |
| Guard timing / reset error | Leading structural hypothesis. It predicts exactly the observed localized amplification, but needs event-resolved diagnostics. |
| Readout error | Plausible and inseparable at present. The physical target is itself a native grounded readout, and low latent semigroup error can coexist with high physical error. |
| Carrier-state insufficiency | Plausible. Cross-model recurrence and inherited native catastrophes suggest missing contact-relevant information or non-Markov finite histories. An oracle event ceiling can test this. |
| Smooth-transition misspecification | Possible but less specific. Low-error within-mode cells and localized event failure argue against it as the primary explanation. |
| Frameskip / macro-step aliasing | Plausible: an event can occur inside a macro transition without being represented as an explicit state. The locked rows cannot distinguish it from carrier insufficiency. |

### 3.3 What is not known

Stage 38 does not show that the four-carrier history is a sufficient predictive state, that its learned state is minimal, that its mixture components correspond to physical modes, that contact labels are inferable without simulator supervision, or that a reset model will improve the tail. It also does not isolate the contribution of carrier/physical semigroup losses because of the coefficient mismatch. Finally, 32 evaluation trajectory families are too few to interpret a row-level 95% tail estimate as a precise population certificate.

## 4. Mathematical reframing

### 4.1 Predictive-state quotient

Let a history be `h_t=(o_0,a_0,...,o_t)` and let `A*` be the set of finite admissible action words. Choose the relevant future random object `Y^u(h)`—for this project, future physical observations and/or task values under word `u`. Define predictive equivalence

\[
h\sim_{\mathcal U,\mathcal Y}h'
\quad\Longleftrightarrow\quad
\mathcal L(Y^u\mid h)=\mathcal L(Y^u\mid h')
\quad\text{for every }u\in\mathcal U\subseteq A^*.
\]

With deterministic targets, equality of laws can be replaced by equality of the relevant futures. A predictive state is a measurable representative of the quotient `H/~`; it is sufficient when every relevant future conditional factors through it. This is the controlled analogue of predictive state representations and causal states. The original PSR construction represents state by predictions of future tests rather than a latent physical state ([Littman, Sutton, and Singh, 2001](https://proceedings.neurips.cc/paper/2001/hash/1e4d36177d71bbb3558e43af9577d70e-Abstract.html)); causal states are equivalence classes of histories with the same conditional future distribution ([Shalizi and Crutchfield, 2001](https://csc.ucdavis.edu/~cmg/compmech/pubs/cmppss.htm)). Controlled bisimulation gives a related quotient by equating reward and transition behavior, with quantitative metrics for approximation ([Ferns, Panangaden, and Precup, 2004](https://mlanthology.org/uai/2004/ferns2004uai-metrics/)).

Stage 38 replaces the full history with `E(z_{t-3:t})` and tests a finite set of words and horizons. It can support finite-bank approximate predictive sufficiency for a chosen observation metric. It cannot establish global sufficiency, equality of full future laws, quotient minimality, or a causal state. A lower-dimensional statistic could perform equally well, and two histories indistinguishable on the tested bank can differ on an untested contact word.

**Definition 1 (finite-family approximate predictive closure).** Given state encoder `E`, word family `U`, relevant readout `D`, metric `d_Y`, and distribution `mu` on histories and words, `(E,F,D)` is `(epsilon,p)`-predictively closed when

\[
\Pr_{(h,u)\sim\mu}
\left[d_Y\{D(F_u(E(h))),Y^u(h)\}>\epsilon\right]\le p,
\]

and it is uniformly `epsilon`-closed on the family when the displayed error is at most `epsilon` for every supported `(h,u)`. A meaningful certificate must state `U`, `mu`, the observation/task class, and whether the claim is mean, quantile, or uniform. Stage 38 estimates several finite-family risks; it does not meet its own uniform/tail-oriented gates.

### 4.2 Action words: monoid action and controlled cocycle

Let `u=a_1...a_m` be executed left to right and define

\[
\Phi_u=F_{a_m}\circ\cdots\circ F_{a_1},\qquad \Phi_\epsilon=I.
\]

If `uv` means “execute `u`, then `v`,” then

\[
\Phi_{uv}=\Phi_v\circ\Phi_u.
\]

This is a right action, equivalently a homomorphism from the opposite free monoid. If the protocol writes the concatenated execution as `vu`, its defect is

\[
\Delta(u,v;s)=d_S\!\left(\Phi_{vu}(s),\Phi_v(\Phi_u(s))\right),
\]

with that convention stated explicitly. Calling this only a “time semigroup” hides the action sequence. A controlled cocycle is the more precise object: for an infinite action stream `omega` and left shift `theta`,

\[
\varphi(n+m,\omega,s)
=\varphi\!\left(m,\theta^n\omega,\varphi(n,\omega,s)\right).
\]

**Definition 2 (approximate action/cocycle closure).** For a state distribution `nu`, supported word-pair family `C`, and state metric `d_S`, the learned maps are `(epsilon,p)` action-closed when

\[
\Pr_{(s,u,v)\sim\nu\times C}[\Delta(u,v;s)>\epsilon]\le p.
\]

They are uniformly `epsilon` action-closed when the supremum of the defect on the supported set is at most `epsilon`. The cocycle version replaces the two-word defect by the corresponding `n,m,omega` identity. This definition concerns internal state composition; predictive closure additionally requires correct relevant readouts as in Definition 1.

This is the standard cocycle structure of random/controlled dynamical systems rather than an autonomous one-parameter semigroup (see Arnold's treatment of cocycles in [Random Dynamical Systems](https://link.springer.com/book/10.1007/978-3-662-12878-7)). S-PSCD's direct-vs-composed loss is best interpreted as a sampled approximate cocycle/action defect. Because its “direct” state is produced by the same encoder, it measures internal consistency, not physical correctness by itself.

### 4.3 Hybrid action closure

Let the augmented state be `(q,s)` with discrete mode `q in Q`. Within a mode,

\[
s^+=F_{q,a}(s)
\]

until a guard `G_{q\to r}(s,a)=0` is crossed, at which point

\[
(q,s^-)\mapsto(r,R_{q\to r}(s^-,a)).
\]

A learned model may use a calibrated event probability `pi_{q->r}` or a soft gate, but locked inference receives only the carrier history and action—not oracle contact labels. For a realized hybrid itinerary, word evolution alternates within-mode maps and resets:

\[
\widehat\Phi_u=\widehat F_{q_m,u_m}\circ
\widehat R_{q_{m-1}\to q_m}\circ\cdots\circ
\widehat R_{q_0\to q_1}\circ\widehat F_{q_0,u_0}.
\]

The mode-indexed operators and resets form a path/category-like action: resets are generally noninvertible, so “groupoid” is too strong. On the fully augmented state, ordinary monoid/cocycle composition still holds exactly; the scientific gain is a **factorization of its defect by event itinerary**, not replacement of algebraic composition.

**Definition 3 (approximate hybrid predictive closure).** On a prespecified distribution of histories and words, a hybrid state `(q,s)` is `(epsilon_flow, epsilon_event, epsilon_reset, epsilon_readout; p)`-closed when, outside an event of probability at most `p`, (i) every realized within-mode segment has transition/action defect at most `epsilon_flow`, (ii) the inferred event identity and time are correct to the declared discrete/time tolerance `epsilon_event`, (iii) every selected reset has state defect at most `epsilon_reset`, and (iv) the resulting relevant readout has error at most `epsilon_readout`. A risk-qualified version additionally bounds a stated tail functional over independent sampling clusters. This definition makes the itinerary, tolerance, word family, and observation class part of the claim; none may be changed after locked evaluation.

Define component defects on a fixed development distribution:

- **flow defect** `delta_flow`: mismatch of within-mode transition or within-mode cocycle composition;
- **event defect** `delta_event`: probability/calibration/timing loss for the guard or mode transition;
- **reset defect** `delta_reset`: post-event state mismatch conditional on the correct event and pre-event state;
- **readout defect** `delta_readout`: physical/carrier error conditional on the correct latent state;
- **tail-risk defect** `delta_tail(alpha)=CVaR_alpha(ell)-r_alpha` (or an upper confidence bound minus its threshold), which is a distributional risk functional, not an additive state-transition error.

This decomposition is more falsifiable than a generic hybrid mixture: it predicts which conditional diagnostic must improve and separates representation, event, reset, and decoder failures.

### 4.4 Saltation, Koopman, and stability

For a time-invariant guard with normal `n=nabla G` and reset `R`, first-order perturbations across an event are propagated by the saltation matrix

\[
\Xi=DR+
\frac{(f^+-DRf^-)n^\top}{n^\top f^-}.
\]

Near grazing, `n^T f^-` can be small, making `||Xi||` large: a small pre-event or timing perturbation can become a large post-event error. Saltation matrices are the standard first-order sensitivity object for hybrid events; see the modern tutorial by [Kong et al.](https://arxiv.org/abs/2306.06862) and the differentiability analysis of [Burden et al.](https://arxiv.org/abs/1407.1775). This mechanism explains why an average smooth loss can miss rare contact failures. It remains a hypothesis here until event-resolved states and timing are logged.

A controlled Koopman formulation would use action/mode-indexed composition operators

\[
(K_{q,a}f)(s)=f(F_{q,a}(s)),\qquad
(K^R_{q\to r}f)(s)=f(R_{q\to r}(s)).
\]

Controlled Koopman and DMD-with-control methods are established ([Proctor, Brunton, and Kutz, 2016](https://epubs.siam.org/doi/10.1137/15M1013857); [Korda and Mezić, 2018](https://doi.org/10.1016/j.automatica.2018.03.046)). A finite-dimensional linear invariant observable subspace is a strong restriction, and exact finite closure is generally exceptional ([Brunton et al., 2016](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0150171)). The current nonlinear `E,F,D` can be called a learned observable coordinate system only loosely; it is not evidence of a Koopman-invariant subspace. Replacing it with Koopman learning now would test a different function-class hypothesis without first resolving the known control confound.

Contraction, incremental stability, Lyapunov, and input-to-state stability provide the right language for turning local defects into rollout bounds. Within-mode contraction alone is insufficient when a reset has large saltation gain. A useful later certificate would bound products over hybrid cycles—for example, reset gain times intervening flow contraction—not just a global Jacobian average. Classical contraction analysis is due to [Lohmiller and Slotine (1998)](https://doi.org/10.1016/S0005-1098(98)00019-3). It can strengthen a successful hybrid model but cannot manufacture a missing event state.

## 5. Candidate theorem/propositions

### Proposition 1: hybrid rollout-error bound

**Statement.** Fix an action word and suppose true and learned systems follow the same finite itinerary of `N` stages, each stage being either a smooth segment or a reset. Let `e_i` be state error before stage `i`. Assume:

1. each true smooth map is `rho_i`-Lipschitz on the relevant tube and its learned approximation has uniform defect at most `epsilon_i`;
2. each true reset is `kappa_j`-Lipschitz and its learned reset error is at most `eta_j` when the event is correctly selected;
3. a wrong or mistimed event at reset `j` contributes an additional bounded discrepancy `B_j M_j`, where `M_j in {0,1}` and `E[M_j] <= p_{e,j}`;
4. the physical readout is `L_D`-Lipschitz with approximation defect `delta_D`.

Index all stages in execution order and set

\[
L_i=\begin{cases}
\rho_i,&i\text{ smooth},\\
\kappa_i,&i\text{ reset}.
\end{cases}
\]

and

\[
b_i=\begin{cases}
\epsilon_i,&i\text{ smooth},\\
\eta_i+B_iM_i,&i\text{ reset}.
\end{cases}
\]

Then

\[
\|e_N\|\le
\left(\prod_{i=0}^{N-1}L_i\right)\|e_0\|+
\sum_{i=0}^{N-1}b_i\prod_{r=i+1}^{N-1}L_r,
\]

and

\[
\|\widehat x_N-x_N\|\le L_D\|e_N\|+\delta_D.
\]

Taking expectations replaces each `M_i` by at most `p_e,i` under the bounded-cost assumption.

**Proof sketch.** For each stage, add and subtract the true map evaluated at the learned input. Lipschitzness and approximation error give `||e_{i+1}|| <= L_i||e_i||+b_i`. Repeated substitution proves the first inequality; applying the readout assumptions proves the second. This is a proposition under explicit uniform-tube and fixed-itinerary assumptions, not a distribution-free theorem about the trained network.

**Implications and limitations.** A rare event term can dominate if `B_i` or the following product of gains is large. Uniform flow contraction gives a horizon-independent bound only when reset-cycle gains are also controlled—for example, every recurrent hybrid cycle has product gain strictly below one. The proposition does not handle unbounded state spaces, an unbounded number of events, or endogenous itinerary changes without the explicit `B_i M_i` term. Near grazing, a saltation norm is a local candidate for `kappa_i` or `B_i`, but Stage 38 does not estimate it.

### Proposition 2: semigroup consistency cannot certify physical correctness

**Statement.** For any `M>0` and `p in (0,1)`, there is a hybrid controlled system and learned recursively compositional model with zero internal action-monoid defect and mean one-step squared error `pM^2`, but event-conditional squared error `M^2`.

**Construction.** Let the true scalar transition be identity except on a guard event `C` of probability `p`, where it resets by `x^+=x^-+M`. Let the learned transition be the identity for every action, and let its direct-state target be generated by the same identity representation. All learned word maps compose exactly, so its internal semigroup defect is zero. It errs by `M` exactly on `C`; the mean squared error is `pM^2`, which can be made arbitrarily small by taking `p` small while the conditional error remains `M^2` and can be arbitrarily large.

**Consequence.** Neither low mean one-step loss nor exact self-composition establishes tail-safe predictive closure near a rare guard. The Stage 38 common counterexample—very low semigroup score and very high physical score—is an empirical instance of the logical separation, not proof that this toy mechanism is the unique cause.

### Proposition 3: coefficient matching

**Statement.** With the implemented normalized semigroup loss, a full model `(w_z,w_x,w_s)=(.35,.20,.45)` at outer weight `lambda_f` and a latent-only control `(0,0,1)` have equal coefficients on `L_s^sg` if and only if `lambda_o=.45 lambda_f`.

**Proof.** The full normalized weights sum to one, so the latent coefficient is `.45 lambda_f`. The control's normalized latent weight is one, so its coefficient is `lambda_o`. Equate them. No statistical assumptions are required.

### Hybrid-closure conjecture to test

**Conjecture, not theorem.** If (i) contact-relevant event state is identifiable from frozen carrier history and action, (ii) the dominant Stage 38 error is event selection/timing or reset approximation rather than an irrecoverable readout ambiguity, and (iii) the hybrid model is capacity-controlled, then an event-factorized predictive transition should reduce `post_contact -> contact` mean and CVaR before any tail-risk objective is added.

This conjecture is falsified by a weak oracle-event ceiling, by an accurate label-free gate with no conditional physical gain, or by improvement that disappears under matched parameter count/supervision controls.

### Risk estimation with few trajectory families

The Rockafellar–Uryasev representation is implementable:

\[
\operatorname{CVaR}_\alpha(\ell)=
\min_\tau\left[\tau+\frac{1}{1-\alpha}\,\mathbb E(\ell-\tau)_+\right]
\]

([Rockafellar and Uryasev, 2000](https://uryasev.github.io/publications/)). But a 95% empirical tail over 32 independent families contains only about 1.6 families; calibration has only 24. A transition-level CVaR would falsely inflate sample size and invite memorization of correlated words. If risk optimization is reached at all, train on a loss aggregated over **complete physical trajectory families**, balanced over event-conditioned groups, at a more stable development level such as `alpha=.8` or `.9`; retain `CVaR95` as a prespecified evaluation metric with hierarchical family/seed uncertainty. Distributionally robust group objectives may be more stable than empirical 95% CVaR ([Duchi and Namkoong, 2018](https://arxiv.org/abs/1810.08750)). Conformal risk control requires exchangeable fresh examples and a prespecified loss and is not a cure for only 24–32 heterogeneous clusters ([Angelopoulos et al., 2022](https://arxiv.org/abs/2208.02814)). Few-cluster bootstrap uncertainty must also be treated cautiously ([MacKinnon and Webb, 2017](https://onlinelibrary.wiley.com/doi/10.1111/ectj.12107)).

## 6. Related-work and novelty matrix

| Mathematical idea | Closest prior work | Already known | Proposed new contribution | Novelty risk |
|---|---|---|---|---|
| Predictive-state quotient | [PSRs](https://proceedings.neurips.cc/paper/2001/hash/1e4d36177d71bbb3558e43af9577d70e-Abstract.html); [causal states](https://csc.ucdavis.edu/~cmg/compmech/pubs/cmppss.htm) | Histories can be quotiented by equality of controlled future predictions. | A frozen-world-model, finite-word empirical certificate tied to explicit failure controls. | High if merely renamed PSR sufficiency. |
| Controlled equivalence / bisimulation | [Ferns et al.](https://mlanthology.org/uai/2004/ferns2004uai-metrics/) | Metrics can quantify approximate behavioral equivalence in controlled systems. | Connect learned carrier histories, task-relevant readouts, and recursive word closure to a falsifiable quotient test. | Medium-high; must state how the metric/certificate differs. |
| Action-monoid / cocycle closure | [Arnold](https://link.springer.com/book/10.1007/978-3-662-12878-7) and standard controlled dynamics | Word-indexed maps obey a cocycle/action composition law. | Component-resolved empirical defects and coefficient-identifiable controls for learned predictive states. | Medium; algebra itself is standard. |
| Latent overshooting | [PlaNet / latent overshooting](https://proceedings.mlr.press/v97/hafner19a.html) | Multi-step latent consistency is established in recurrent world models. | Demonstrate when latent-only consistency is insufficient for grounded physical tail closure. | High unless the failure/certificate is the contribution. |
| Hybrid modes, guards, resets | [Neural Hybrid Automata](https://papers.nips.cc/paper/2021/hash/5291822d0636dc429e80e953c58b6a76-Abstract.html); [ContactNets](https://proceedings.mlr.press/v155/pfrommer21a.html) | Learned modes, event dynamics, and contact-structured models already exist. | A label-free hybrid predictive-state cocycle with separate flow/event/reset/readout defects on frozen representations. | Very high if reduced to a gate or mixture of experts. Recent hybrid world models make this crowded. |
| Saltation sensitivity | [Kong et al.](https://arxiv.org/abs/2306.06862); [Burden et al.](https://arxiv.org/abs/1407.1775) | Guard timing and reset sensitivity can amplify perturbations. | Empirically connect low internal closure defect to shared, event-localized catastrophic physical error, then test it with an event-resolved bound. | Medium if an actual estimator/bound is validated; high if only explanatory language. |
| Piecewise controlled Koopman | [DMDc](https://epubs.siam.org/doi/10.1137/15M1013857); [Korda–Mezić](https://doi.org/10.1016/j.automatica.2018.03.046); [hybrid Koopman](https://arxiv.org/abs/2006.12427) | Action-indexed observables and piecewise/reset operators are known. | Possibly a finite observable certificate for hybrid predictive equivalence. | High and premature; finite invariant subspaces are restrictive. |
| Contraction / incremental stability | [Lohmiller–Slotine](https://doi.org/10.1016/S0005-1098(98)00019-3) | Local contraction and stability convert perturbation bounds into rollout bounds. | Hybrid-cycle bound combining flow contraction, reset gain, event error, and readout error. | Medium if non-vacuous and empirically estimated; otherwise decorative. |
| CVaR / DRO / risk-sensitive control | [Rockafellar–Uryasev](https://uryasev.github.io/publications/); [Duchi–Namkoong](https://arxiv.org/abs/1810.08750); [risk-averse model-based control](https://pmc.ncbi.nlm.nih.gov/articles/PMC7990789/) | Tail objectives and risk-aware planning are established. | Cluster-aware family-level risk training after structural identifiability is shown. | Very high if novelty is “add CVaR.” |
| Quantile / entropic objectives | [Implicit Quantile Networks](https://proceedings.mlr.press/v80/dabney18a.html); [entropic-risk MDPs](https://proceedings.mlr.press/v206/lin-hau23a.html) | Distributional quantiles and entropic risk are standard. | No unique contribution presently identified. | Very high. |
| Signatures / Chen identity | [Chevyrev and Kormilitzin tutorial](https://jmlr.org/papers/volume22/20-620/20-620.pdf) | Path signatures compose via Chen-type identities. | None needed for the present finite action-word question. | Decorative: known words already compose, while contact resets are the missing structure. |
| Marked point processes / neural jump models | [Neural Jump SDEs](https://proceedings.neurips.cc/paper_files/paper/2019/file/59b1deff341edb0b76ace57820cef237-Paper.pdf) | Stochastic event times and marks can be learned. | Could model uncertain contact timing if the data become asynchronous/stochastic. | Overbuilt for the current discrete deterministic benchmark. |
| Operator-valued kernels | [Kadri et al.](https://www.jmlr.org/beta/papers/v17/11-315.html) | Vector/functional response operators are standard. | No diagnostic advantage established here. | Decorative function-class change. |

Recent work already combines hybrid gates, experts, simulators, and long-horizon/contact world models—for example [PRISM-WM](https://arxiv.org/abs/2512.08411), [Hybrid Neural World Models](https://arxiv.org/abs/2605.28317), and [simulator-informed hybrid latent state learning](https://ojs.aaai.org/index.php/AAAI/article/view/29075). Consequently, “hybrid world model,” “contact gate,” “mixture of experts,” “overshooting,” or “CVaR” cannot be the novelty claim.

The narrow contribution this project could uniquely own is:

> **A hybrid predictive-state action-cocycle closure certificate for frozen representations:** a coefficient-identifiable comparison, an event-itinerary decomposition into flow/event/reset/readout/tail defects, a non-vacuous rollout bound, and cross-representation evidence that low internal composition defect can coexist with shared catastrophic physical event error—followed by a label-free intervention that specifically repairs that defect.

Without the coefficient-matched causal result, validated hybrid intervention, and theorem/certificate, that contribution collapses into known ingredients and should be treated as non-novel.

## 7. Ranked method candidates

| Rank | Method | Expected information gain | Explain Stage 38? | Falsifiability | Compute | Novelty | Main confound |
|---:|---|---|---|---|---|---|---|
| 1 | **B. Coefficient-matched control repair** | Very high | Directly resolves the failed full-vs-overshoot inference | Very high: one algebraic contrast and fixed gates | Low | Necessary, not itself novel | A null result does not identify the physical mechanism |
| 2 | **C. Hybrid/event-reset state, conditional on rank 1** | High | Yes; directly targets the observed mode transition | High with oracle ceiling, label-free gate, and matched controls | Medium | Potentially medium only in the proposed certificate combination | Extra parameters and simulator event supervision |
| 3 | **G. Theory work in parallel** | Medium-high | Organizes the failure and yields sharp tests | High if bounds are estimable | Low | Potentially medium | Theory can become post-hoc decoration without intervention |
| 4 | **F. Predictive equivalence/bisimulation reframing** | Medium | Explains what “closure” should mean | Medium | Low | Medium as a certificate, low as terminology | Tested word bank may be far from quotient sufficiency |
| 5 | **A. Adopt latent overshooting** | Medium | Describes the current empirical ranking | High | None | Low | Current control is stronger and still fails the tail |
| 6 | **D. Tail-risk optimization** | Medium-low now | Treats symptoms, not state/event structure | Medium | Medium | Low | Few clusters; can memorize known hard families |
| 7 | **E. Controlled Koopman/operator learning** | Low-medium | Only indirectly | Medium | High | Low-medium | New function class and invariant-subspace assumptions |
| 8 | **H. Direct cross-environment confirmation** | Low now | No; transports unresolved confounds | High but wasteful | Very high | Potentially useful later | Negative result uninterpretable; positive result still confounded |
| 9 | **I. Abandon immediately** | Low today | Ends rather than explains | High | None | N/A | Premature before the cheap coefficient repair |

Rank 9 becomes rank 1 under any kill condition in Section 8. “Theory in parallel” means formal analysis and diagnostic definitions only; it does not delay the cheap control repair or authorize a larger experimental stage.

## 8. Minimal next experiment

### 8.1 Scope and data

Name the development audit **Stage 38.1**, not Stage 39. It must never read or tune against `locked_closure_rows.csv` or any Stage 38 evaluation aggregate. Use only the original Stage 38 construction, model-selection, and calibration pools:

- construction trajectory IDs `56000–57999`;
- model-selection IDs `58000–59999`;
- calibration IDs `60000–61999`;
- the corresponding already-separated action-word banks.

The Stage 38 locked pool `62000–65999` is now historical confirmation evidence and may be used only in this report, never as development feedback. Because Stage 38 used calibration data during final fitting, Stage 38.1 models must be refit with construction-only training and calibration-only scoring (or with a predeclared internal construction split if early stopping is unavoidable). Model-selection data may select among the already specified `lambda` choices, but may not set reporting thresholds.

Frozen JEPA-WM and DINO-WM checkpoint bytes remain unchanged. Reuse their construction/selection/calibration carrier artifacts, deterministic provenance machinery, grounded readout definitions, normalization specification, and architecture code after verifying hashes. Do not reuse trained Stage 38 closure-model weights for the causal comparison.

### 8.2 Tier A: the actual minimum

Train the following matched models for each representation and at least two development seeds:

| Variant | Semigroup component weights | Outer weight | Purpose |
|---|---|---:|---|
| Ordinary PSCD | none | 0 | reference |
| Current mass-matched overshooting | `(0,0,1)` | JEPA 2.0; DINO 1.0 | retain current sensitivity comparison |
| **Coefficient-matched overshooting** | `(0,0,1)` | **JEPA .90; DINO .45** | valid latent-pressure control |
| Full S-PSCD | `(.35,.20,.45)` | JEPA 2.0; DINO 1.0 | treatment |

For the full-vs-coefficient-matched comparison, hold architecture, parameter count, initialization stream, batch order, optimizer, learning rate, 320 epochs, history length, horizons, and data exactly matched. Log the raw `L_z^sg`, `L_x^sg`, and `L_s^sg`, their normalized component weights, outer weights, realized batch counts, and gradients/norms separately; reporting only `L_sg` is insufficient.

Use two seeds as a screening boundary, with a third seed **precommitted before looking at Tier A outcomes** and executed only if promotion is otherwise possible. This group-sequential rule saves compute without allowing seed shopping. The confirmatory requirement remains all three seeds.

### 8.3 Tier B: conditional event/reset diagnostic

Only if Tier A promotes, fit an event-factorized transition **without a tail loss first**. Keep the existing 256-dimensional state and base transition. Let `c_t=[s_t,a_t]`; use a gate `G: d_s+d_a -> 32 -> 1` and jump `J: d_s+d_a -> 32 -> d_s`, each with one SiLU hidden layer:

\[
g_t=\sigma(G_\phi(s_t,a_t)),\quad
s^-_{t+1}=F_{\mathrm{smooth}}(s_t,a_t),\quad
s_{t+1}=s^-_{t+1}+g_tJ_{\mathrm{event}}(s_t,a_t).
\]

The initial model should add this small gate/jump only, not a broader mixture. Evaluate four controls:

The no-tail supervised diagnostic loss is exactly

\[
L_{\mathrm{H0}}=L_1+L_{\mathrm{fr}}+\lambda_{\mathrm{full}}L_{\mathrm{sg}}
+0.10L_{\mathrm{event}},
\]

where `L_sg` retains full weights `(.35,.20,.45)` and `L_event` is class-balanced binary cross-entropy for whether at least one contact transition occurs inside the next simulator macro-step. Balance positive and negative construction examples to weight `.5` each; do not tune the `.10` coefficient. Apply the same hybrid transition recursively in free-run and semigroup terms. Log the unweighted event loss and every unweighted closure component before forming the total. The parameter-matched smooth control uses the same low-rank jump MLP as an unconditional residual and retains a scalar prediction head that is not fed back, so its trainable parameter count is identical; the shuffled-label control retains the complete hybrid graph.

1. **oracle-event ceiling:** train the jump/reset on construction event labels, then replace the predicted gate with the true macro-step event indicator during calibration rollouts to test whether correct event knowledge can help; this is diagnostic and cannot be the locked method;
2. **label-free inference model:** event labels may supervise construction training, but validation/evaluation gates consume only frozen carrier history and action;
3. **parameter-matched smooth control:** add an equally sized residual MLP without event factorization;
4. **supervision control:** shuffled event labels with the same optimization path.

Simulator contact labels lower scientific purity: they inject privileged dynamics information absent from the frozen representation and can turn the result into simulator-supervised adaptation rather than a representation-closure method. The preferred label-free alternative is a latent event variable trained from change-point evidence in carrier/physical prediction residuals, with a sparsity/duration prior and no contact labels. However, use that only after the supervised oracle ceiling establishes that an event intervention has useful headroom. If oracle contact is required at locked inference, reject RH-PSCD for the intended claim.

Only if the no-tail hybrid passes structural gates may one compare it with a family-risk objective. Use family-aggregated loss `ell_j` over a complete trajectory family (mean normalized physical path loss over all of its records, words, and rollout positions) and balanced event strata:

\[
L_{\mathrm{risk},\alpha}(\tau)=
\tau+\frac{1}{1-\alpha}\frac1B\sum_{j=1}^{B}(\ell_j-\tau)_+,
\]

with `alpha=.9`, a trainable scalar `tau`, and fixed outer coefficient `eta=.10`. Thus the conditional risk variant is `L_H0 + .10 L_risk,0.9`; it changes no architecture. Never compute the training CVaR over transitions or the known locked failure rows. Do not optimize over seeds; seeds quantify training variability. Report event-conditioned metrics but do not let the rarest group receive unlimited weight without a predeclared cap.

### 8.4 Metrics and preregistered promotion gates

Compute paired metrics per representation, seed, trajectory family, record, and word. Use a hierarchical bootstrap that resamples trajectory families and seeds, preserving all paired method/word/record rows. Because the number of clusters is small, show family-level effects and leave-one-family-out sensitivity in addition to intervals.

Tier A promotes only if all conditions hold on untouched calibration data:

1. **Coefficient specificity:** full S-PSCD improves mean endpoint physical NMSE by at least 5% relative to coefficient-matched overshooting for both JEPA and DINO, and the one-sided 95% hierarchical interval for the paired relative gain is above zero.
2. **No tradeoff:** the full model is noninferior within 5% on physical p95, CVaR95, and catastrophic rate in both models.
3. **Correct-history specificity:** it improves mean physical NMSE by at least 5% over wrong history with a positive lower interval in both models.
4. **Absolute viability:** mean `<=.25`, p95 `<=.35`, catastrophic rate `<=.02`, and the Stage 38 horizon/mode gates pass for every executed seed. CVaR95 is reported with an upper interval; due to few clusters it is not used as a standalone absolute certificate.
5. **Seed stability:** after the precommitted third seed, all three seed signs agree in both representations.

If Tier A promotes, Tier B's no-tail label-free hybrid must additionally satisfy:

1. at least 25% reduction in `post_contact -> contact` mean and CVaR95 relative to both full S-PSCD and the parameter-matched smooth control;
2. at least 10% reduction in overall p95 without worsening overall mean by more than 2%;
3. catastrophic rate `<=.02` and no new mode-pair catastrophic cluster;
4. event AUROC `>=.80`, Brier skill improvement `>=10%` over the event base-rate predictor, expected calibration error `<=.05`, and stable results across three seeds and both frozen models;
5. materially better outcomes than shuffled-label control, with the same supervision and parameter budget;
6. label-free locked inference: no oracle contact or future physical truth in the gate;
7. exact parameter count and source of every supervision signal reported.

The oracle-event ceiling must reduce `post_contact -> contact` mean or CVaR95 by at least 25% and overall p95 by at least 10%; otherwise the event/reset hypothesis lacks useful headroom and Tier B stops. A risk-sensitive Tier B extension may promote only if it improves family-level CVaR95 by at least 15% over the no-tail hybrid without violating any mean, event, or specificity gate.

These thresholds are development decisions and must be frozen before execution. They are intentionally stronger than “best mean,” because the proposed mechanism is specifically about event-localized tail closure.

### 8.5 Compute estimate

Stage 38's complete run took 2,132.85 seconds (`0.592` device-hours) on an RTX PRO 6000 Blackwell, although the artifact does not isolate every training component. With carriers and frozen readouts reused, Tier A should require roughly **1–2 GPU-hours** on the same class of GPU, including the precommitted third-seed boundary; reserve **2–4 additional GPU-hours** for the conditional hybrid/control tier. A conservative full Stage 38.1 reservation is **4–6 GPU-hours** on that device class, or roughly **6–10 hours on an L4-class device**, subject to a dry-run timing check. No GPU should be spent on risk training unless the structural hybrid clears its gates.

### 8.6 Kill criteria

Stop or sharply reframe the line when any applicable condition occurs:

- If full S-PSCD does not achieve the Tier A 5% coefficient-matched gain on **both** representations, kill the claim that its carrier/physical semigroup components add value. Use latent overshooting as the engineering baseline, but do not call recursive physical closure solved.
- If the oracle-event ceiling fails the 25% conditional or 10% overall-tail improvement, kill the saltation/event-reset intervention hypothesis for this representation and sampling rate.
- If the oracle ceiling works but the label-free gate fails calibration or the shuffled-label/capacity controls match it, conclude that contact-relevant event state is not identified by the frozen carrier/history. Stop post-hoc closure or begin a separately named representation-learning project.
- If improvement requires simulator labels at locked inference, claim only a simulator-supervised adapter, not closure of a frozen world-model representation.
- If any promoted hybrid still violates p95/catastrophic gates for either representation or any final seed, do not open Stage 39 or planning.
- If the proposed contribution reduces after review to a contact gate, mixture of experts, CVaR, or overshooting without a new certificate/bound and intervention result, abandon or reframe it because those components are established prior art.

## 9. Fresh confirmation roadmap

Stage 39 exists only if Stage 38.1 passes all applicable gates. Before running it, freeze a source-bound protocol, hypotheses, directions, thresholds, word-generation rules, cluster inference, and failure semantics. Use unused trajectory-ID pools and action-word banks, JEPA and DINO, and three final seeds. No threshold or architecture changes may occur after locked evaluation begins. Planning remains sealed until closure passes.

Prefer a second contact environment or a deliberately different contact geometry as an external-validity panel. It should change the guard/reset geometry rather than merely resample PushT states. The primary claim should require consistent signs in both representation families and no catastrophic concentration in the new event transitions.

**May be reused after hash verification:** frozen JEPA/DINO checkpoint bytes; source-bound execution and validation infrastructure; architecture definitions; the preregistered coefficient convention; fixed loss equations; frozen metric/gate definitions; deterministic audit code. The selected Stage 38 full-model `lambda` values may be carried forward only as prespecified protocol constants, not retuned from locked evidence.

**Must be regenerated for confirmation:** trajectory truth and IDs; action-word banks; construction/selection/calibration/evaluation carriers for new trajectories; grounded projections/readouts and normalization statistics; all closure-model weights; event labels used in construction; learned event calibrators; parameter-matched and shuffled controls; bootstrap draws; evaluation certificates; and all result hashes. Do not reuse Stage 38 evaluation rows, failure-specific words selected from them, trained closure models, or post-hoc cutoffs. The fresh evaluation pool must remain unread until every model and gate is frozen.

## 10. ICLR assessment

### What can currently be claimed

The current evidence supports a careful negative/diagnostic claim: on two frozen representation families, recurrent predictive-state distillation improves average physical recursion relative to native, one-step, and wrong-history controls, but full S-PSCD does not beat a stronger latent-only overshooting control and fails prespecified tail/mode closure gates. Catastrophes are seed-stable, cross-model correlated, and concentrated in `post_contact -> contact`; low internal semigroup error does not guarantee physical event accuracy. The result is unusually well source-bound and auditable.

It does **not** support representation confirmation, planning readiness, predictive-state minimality, a causal benefit from carrier/physical semigroup components, or a proven saltation mechanism. It also does not establish RH-PSCD.

### What a successful next result would support

A coefficient-matched full-model win would isolate the value of grounded semigroup components. A capacity- and supervision-controlled, label-free hybrid intervention that selectively reduces the event tail across JEPA and DINO would support the proposed defect decomposition and turn the Stage 38 negative result into a causal scientific finding. A validated rollout bound with measured flow/reset/event quantities would elevate the work beyond another hybrid architecture.

### What is still required for an ICLR-level paper

An ICLR-level submission would need:

- the Stage 38.1 coefficient-matched causal comparison and all negative controls;
- event-resolved logging that distinguishes guard detection, timing, reset, and readout error;
- a non-vacuous theorem/certificate with empirically estimated quantities, not just saltation terminology;
- label-free locked inference and a clear accounting of simulator supervision;
- fresh Stage 39 confirmation with unused pools, three final seeds, both frozen models, and preferably different contact geometry/environment;
- family-level uncertainty that respects clustered dependence, plus sensitivity to the small number of families;
- comparisons to strong hybrid, overshooting, and risk-aware world-model baselines;
- a claim whose novelty is the hybrid predictive-state cocycle certificate and its falsified/validated mechanism, not a gate, expert mixture, CVaR, or contact model.

Until those exist, the strongest form is a rigorous internal decision report or a negative-results/mechanistic workshop paper, not a full conference claim of solved recursive closure.

## 11. Final recommendation

Run Stage 38.1 and nothing larger: first refit a coefficient-matched latent-overshooting control (`lambda=.90` for JEPA, `.45` for DINO) against full S-PSCD on construction/model-selection/calibration data with matched seeds, architecture, initialization, epochs, and exact component logging; kill the extra-component claim unless full S-PSCD clears the preregistered 5% two-model gain and tail noninferiority gates. Only after that win, use an oracle-event ceiling followed by a small label-free, parameter-controlled event/reset transition without tail loss to test the `post_contact -> contact` mechanism; add family-level risk optimization only if the structural intervention already works. Do not open Stage 39, planning, a Koopman replacement, or cross-environment confirmation until both stages pass, and abandon or reframe the line if the result cannot be distinguished from established overshooting, hybrid gating, or CVaR prior art.
