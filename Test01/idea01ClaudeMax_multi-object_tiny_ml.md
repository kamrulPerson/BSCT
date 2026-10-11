## Step 2: Landscape (about 20 searches, Oct 2026)

**Occupied or saturated (avoid)**
- Malaria smears with calibrated parasitemia on-device: an Aug 2026 preprint (arXiv 2608.08566) already ships calibrated slide-level soft counting, bootstrap CIs and a human-in-the-loop interface.
- Crisis-image authenticity benchmarks: Forged Calamity (arXiv 2606.18554) has 30k images (24k synthetic, four diffusion models) and shows up to 50% accuracy loss on unseen generators or disaster types. RA-Bench (arXiv 2608.14391) adds 17,886 crisis videos and finds social dissemination makes detection harder.
- INT8 × calibration in medical imaging: Ozdemir & Li benchmark FP32/FP16/INT8 PTQ/QAT on three public datasets and eight backbones. A 2026 review (arXiv 2602.00910) already urges reporting ECE/NLL/rare-class sensitivity after quantization.
- Membership inference × compression: quantization-based and pruning-based MIA defences already exist.
- Flood segmentation: domain-generalization on Sen1Floods11 (JURSE 2025) and foundation-model benchmarks (EGU 2026).
- UAV tiny-person detection with light YOLO variants: FCML-YOLO (Remote Sensing 2026), Drones 2025.
- Bangladeshi vehicle detection: ABTD (8,605 images, 2026), ANNA, Vehicle-BD. These are detection sets only.
- Label-free performance monitoring of medical models: arXiv 2507.22776, arXiv 2207.05796.
- Conformal detection: SeqCRC (arXiv 2505.24038).
- Codec-aware tracking is partly explored (arXiv 2202.00892, arXiv 2404.04848), so I dropped it.

**Under-explored (basis for the ideas)**
- MOT privacy is done with noise or anonymisation, not biometric-style revocability/unlinkability.
- MOT attacks are plentiful but dedicated defences are thin.
- Medical forgery detection concentrates on CT/MRI.
- Selective prediction under INT8 is untouched.
- Security of the full compress chain for medical edge models is untouched.

"Not found" means not found in my searches. Run a Scopus/IEEE Xplore/Scholar check before committing.

## Step 3: Ideas

### 1. KeyTrack: revocable, unlinkable appearance embeddings for privacy-preserving MOT
- **Problem / who benefits:** CCTV analytics stores ReID embeddings and track logs. A leak or cross-camera reuse enables long-term linking of people or vehicles. Residents and operators who need flow statistics without persistent identity benefit.
- **Gap:**
  - TSDCRF (arXiv 2603.13667) evaluates on MOT16/17, Cityscapes and KITTI with noise-based protection, scored by KL-shift and tracking RMSE.
  - A WACV 2024 ReID framework protects the gallery with Gaussian noise.
  - Pedestrian anonymisation (arXiv 2207.11677) works at image level.
  - Face template protection is judged on revocability, unlinkability and irreversibility (ISO/IEC 24745/30136), and FaceLinkGen (arXiv 2602.02914) shows identity linkage can survive naive protection.
  - Not found: these criteria applied to tracker appearance features or session-scoped keys.
- **Novelty:** Treat tracker embeddings as biometric templates. A key-seeded transform rotates per session and is trained against a linkage adversary. Association works within a key epoch, while cross-epoch and cross-camera embeddings are unlinkable and non-invertible. A motion-only handover bridges key boundaries. The output is a measured utility–unlinkability frontier.
- **Method:**
  1. Define the threat model: log-leak adversary doing linkage, inversion, external-gallery ID, and key compromise.
  2. Freeze detector and ReID, and insert transform T_k (key-seeded orthogonal map plus a small learned nonlinear head).
  3. Train the head with a within-epoch triplet loss, cross-epoch indistinguishability, and adversarial linker/inverter.
  4. Add a rotation schedule with motion-only bridging.
  5. Report ISO-style privacy metrics and tracking utility.
- **Data / baselines:** MOT17 and MOT20 for utility; Market-1501 (plus MSMT17 if access allows) for attackers. Baselines are unprotected, Gaussian/DP gallery, TSDCRF (code released), blur/pixelate, plain random rotation, and a SlerpFace-style transform.
- **Metrics / ablations:**
  - Utility: HOTA, IDF1, IDSW, AssA versus rotation interval.
  - Privacy: SSWL-style system linkability, linkage Rank-1/mAP/AUC (Procrustes, learned, motion-only), inversion SSIM/LPIPS, key-compromise.
  - Ablate: adversarial training, embedding dimension, number of keys the attacker knows, handover, tracker (BoT-SORT/StrongSORT/DeepSORT/OC-SORT+ReID).
- **Effort:** Under 1 GPU-day on a T4 (cache embeddings once, small adversaries), about 8–10 person-weeks. **F4 · N4 · Q1 4**
- **Risk → fix:** Reviewers will say "orthogonal transform is trivial; rotation schemes fall to alignment attacks." Fix with adaptive linkers, adversarial training, honest failure analysis, and pedestrians plus vehicles.
- **Journals:** IEEE TIFS, TCSVT, T-ITS; Pattern Recognition; Computers & Security.

### 2. TrackGuard: tracker-agnostic association-layer defence against ID-switch/transfer attacks
- **Problem / who benefits:** Persistent ID manipulation breaks identity continuity in traffic and public-safety trackers. Traffic analytics operators and AV perception teams benefit.
- **Gap:**
  - BankTweak (IJCAI 2025) causes persistent ID switches via feature banks, up to 11.8× stronger than earlier attacks.
  - AdvTraj (arXiv 2512.01934) is a physical ID-transfer attack in CARLA and only discusses countermeasures.
  - FADE (arXiv 2604.00452) tabulates generic input-level defences such as Gaussian noise.
  - Not found: a dedicated association-layer defence with adaptive-attack evaluation.
- **Novelty:** An integrity monitor over three evidence channels: motion residual, appearance-bank outlier score, and occlusion/geometry plausibility. A conformal-calibrated flag triggers a tentative-ID policy (deferred commit, motion-only fallback). This turns persistent swaps into transient ones at a controlled false-flag rate on clean data.
- **Method:**
  1. Build an attack suite on MOT17/20: BankTweak-style feature-bank attack, a detection-level attack, and a digital trajectory ID-transfer replay.
  2. Compute monitor features per assignment.
  3. Calibrate the threshold by conformal risk control on clean splits.
  4. Add the tentative-ID policy.
  5. Run adaptive attacks where the attacker knows the monitor.
- **Data / baselines:** MOT17, MOT20, optionally DanceTrack. Baselines are undefended, noise/JPEG/smoothing, adversarially trained ReID, and association-parameter hardening (the BankTweak paper notes older attacks yield to distance-parameter tweaks).
- **Metrics / ablations:** Attack success rate, IDSW, IDF1/HOTA/AssA under attack, clean false-flag rate, latency. Ablate channels, delay k, α, bank size, adaptive strength.
- **Effort:** 2–3 GPU-days on a T4, about 10–12 person-weeks. **F3 · N4 · Q1 4**
- **Risk → fix:** Attack reproduction cost and adaptive-attack rigor. Limit scope to two attacks plus one trajectory attack, release code, and report the clean-performance cost.
- **Journals:** IEEE TIFS, T-ITS, TCSVT; Pattern Recognition; Computers & Security.

### 3. RP-QAT: rank-preserving evidential QAT for reliable abstention in INT8 diagnostic models
- **Problem / who benefits:** Phone and low-power-PC diagnostics must refer uncertain cases to a clinician. INT8 can silently scramble the uncertainty ranking. Low-resource clinics benefit.
- **Gap:**
  - INT8 work covers discrimination and calibration.
  - EV-AT (arXiv 2607.03075) targets adversarial selective classification without compression.
  - Galil & El-Yaniv (NeurIPS 2021) show tiny perturbations can cripple uncertainty estimates without hurting accuracy.
  - Not found: selective risk under quantization plus perturbation.
- **Novelty:** Distill the uncertainty ranking (evidence ordering), not just logits, from the FP32 teacher into the INT8 student. Add a differentiable selective-risk (AURC-style) surrogate. Evaluate the joint compression × perturbation setting.
- **Method:**
  1. Benchmark FP32→INT8/INT4 PTQ/QAT on AURC, risk@coverage, ECE.
  2. Add the RP-QAT loss.
  3. Stress with PGD and corruptions.
  4. Compare with SelectiveNet, MC-Dropout, temperature scaling, conformal abstention, and EV-AT-style training if code exists.
  5. Deploy via ONNX/TFLite on CPU.
- **Data:** NIH ChestX-ray14 (+ Shenzhen/Montgomery for cross-site), Camelyon17-WILDS, APTOS/EyePACS/Messidor-2.
- **Metrics / ablations:** AURC, selective risk at 80/90/95% coverage, error-detection AUROC, ECE/NLL, sensitivity@specificity, latency/size. Ablate bit-width, PTQ vs QAT, loss terms, backbone, calibration-set size.
- **Effort:** Under 3 GPU-days on a T4, about 6–8 person-weeks. Your existing EDL/KD/ECE code saves 1–2 weeks. **F5 · N3 · Q1 3**
- **Risk → fix:** "Incremental (QAT + selective prediction)." Fix with a clear new objective, a simple bound on rank flips under quantization noise, three modalities, and the joint-attack evaluation.
- **Journals:** IEEE JBHI; Pattern Recognition; Computers in Biology and Medicine; Medical Image Analysis (stretch).

### 4. Radiograph authenticity under messaging-app channels
- **Problem / who benefits:** Remote reads often use photos or screenshots shared via messaging apps (assumption, verify). Recompression erases forensic traces, and fake or edited radiographs enable fraud or sabotage. Telemedicine providers and insurers benefit.
- **Gap:**
  - M3Dsynth covers CT nodule injection/removal.
  - Back-in-Time Diffusion (ACM TIST 2025) targets CT/MRI tumour injection/removal.
  - arXiv 2603.01878 targets CT generators.
  - A Radiology study (via press coverage) found radiologists and leading AI models could not reliably spot synthetic X-rays.
  - Not found: 2D-radiograph detectors trained for messaging-channel degradation under edge constraints.
- **Novelty:** Channel-in-the-loop forensics. Train with a stochastic messaging-channel simulator (resize, JPEG quality, subsampling, screen recapture) plus a real-only anomaly regulariser. Ship a tiny detector with calibrated abstention.
- **Method:**
  1. Build real sets from public CXR data and fake sets from RoentGen plus a second generator and inpainting edits.
  2. Calibrate the channel simulator from a few real round-trips.
  3. Train a tiny residual + frequency CNN.
  4. Evaluate leave-one-generator-out, leave-one-dataset-out, and unseen-channel shift.
  5. Calibrate abstention.
- **Baselines:** CLIP linear-probe detectors, a CT-forgery detector adapted to CXR, a Back-in-Time-style detector (if reproducible).
- **Metrics / ablations:** AUROC/AP and TPR@1%FPR per generator and quality level, ECE, latency. Ablate simulator, residual branch, anomaly loss, model size.
- **Effort:** Generating 5–10k fakes on a T4 is roughly 15–30 GPU-hours (estimate), about 8–10 person-weeks. **F3 · N3 · Q1 3**
- **Risk → fix:** The AI-image-detection field is crowded and dataset building is heavy. Fix with leave-one-generator-out protocol, the channel formulation, and an ethics/dual-use note.
- **Journals:** ACM TIST; IEEE TIFS; Computers in Biology and Medicine; Computers & Security.

### 5. Compress-chain backdoor audit for edge diagnostic models (teacher → student → INT8)
- **Problem / who benefits:** Clinics download pretrained models and datasets and compress them for phones, so a poisoned stage can ship in the final artifact. Low-resource deployers benefit.
- **Gap:**
  - Ma & Mitra (arXiv 2504.21323): poisoned distillation data backdoors the student even with a clean teacher.
  - Learning the Wrong Lessons (arXiv 2303.05593) inserts trojans via unlabelled-data KD.
  - RoS-KD (arXiv 2210.08388) studies adversarial robustness of KD in medical imaging.
  - Not found: the full KD→prune→INT8 chain in medical imaging with an audit on the deployable model.
- **Novelty:** Measure backdoor persistence and emergence per compression stage, and propose a label-light audit on the final model (stage-wise trigger-probe consistency plus calibration drift).
- **Data / baselines:** MedMNIST v2 subsets plus an NIH ChestX-ray14 subset. Baselines are Neural Cleanse, STRIP, fine-pruning.
- **Metrics:** Attack success rate and clean AUC per stage, detection AUROC, CPU cost.
- **Effort:** About 6–8 person-weeks. **F4 · N2 · Q1 2**
- **Risk:** This is mostly "known attack, new domain."
- **Journals:** Computers & Security; Computers in Biology and Medicine.

### 6. Crisis-image authenticity under social dissemination (tiny, generator-conditional abstention)
- **Problem / who benefits:** Fake flood and cyclone images mislead responders and fact-checkers. Responders and fact-checkers benefit.
- **Gap:** Forged Calamity and RA-Bench (above) cover benchmarks and videos. Not found: an image-level lightweight detector with dissemination robustness and generator-conditional abstention.
- **Novelty:** Treat each new generator as a domain shift, with conformal abstention per generator family.
- **Data:** The consolidated crisis-image benchmark (CrisisMMD + AIDR + DMD) for real images; fakes from Forged Calamity if released, otherwise open diffusion models.
- **Effort:** About 8 person-weeks. **F3 · N2 · Q1 2**
- **Risk:** Method-level AIGI detection is crowded, so reviewers will call it a tweak.
- **Journals:** ACM TOMM; Expert Systems with Applications; IEEE TIFS (stretch).

## Step 4: Ranking

| Rank | Idea | Feas. | Nov. | Impact | F×N×I | Q1 |
|---|---|---|---|---|---|---|
| 1 | KeyTrack | 4 | 4 | 4 | 64 | 4 |
| 2 | TrackGuard | 3 | 4 | 4 | 48 | 4 |
| 3 | RP-QAT | 5 | 3 | 3 | 45 | 3 |
| 4 | Radiograph authenticity | 3 | 3 | 4 | 36 | 3 |
| 5 | Compress-chain backdoor audit | 4 | 2 | 3 | 24 | 2 |
| 5 | Crisis authenticity | 3 | 2 | 4 | 24 | 2 |

Impact = real-world relevance and Global South fit. All scores are my judgement.

**Top 2**
1. **KeyTrack** has the best risk-adjusted profile.
   - Compute is light, data is public, and it fits your multi-object and security interests.
   - Novelty is a cross-domain formulation (biometric template protection applied to MOT) that is cheap to prove.
   - The adaptive-attack evaluation is the paper, so budget time for it.
2. **TrackGuard** has the highest security impact.
   - It shares the MOT17/20, tracker and ReID stack with #1, so setup is paid once.
   - The hard part is attack reproduction and adaptive evaluation.

Healthcare fallback: #3 RP-QAT is the lowest execution risk, but it reads as incremental unless the rank-preserving objective and joint-attack evaluation are strong. Do not run two papers in 3 months. I would not spend the time on #5 or #6.

## Flags
- Journal quartiles and scope are from general knowledge and not re-checked against JCR/SJR.
- Access terms are unverified for MSMT17, VeRi-776, CheXpert and RoentGen weights.
- M3Dsynth is cited from a 2024 SigPort record, and I did not confirm its venue.
- The Radiology study is known to me only through secondary coverage, so verify the primary paper.
- The "Impact" and compute figures are estimates.

Pick one (or ask for a swap) and I'll run Step 5.
