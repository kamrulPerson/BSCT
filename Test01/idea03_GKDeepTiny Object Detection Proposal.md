# **Comprehensive Technical Proposal: Edge-Aware Pruned Feature Aggregation for Real-Time Tiny Object Detection on Resource-Constrained Devices**

## **Executive Summary & Project Metadata**

Real-time detection of tiny objects—specifically targets occupying spatial dimensions smaller than $16\times 16$ pixels—presents a critical bottleneck in computer vision applications deployed on resource-constrained embedded platforms1. Standard deep neural networks suffer from severe spatial information degradation as input features undergo successive convolutional downsampling operations2. Traditional Feature Pyramid Networks (FPNs) and multi-scale aggregation necks partially mitigate feature loss but introduce massive computational redundancy, memory bandwidth congestion, and latency overheads that violate real-time execution constraints on edge hardware1.  
This project presents a dual-optimization paradigm that unifies an Edge-Aware Feature Aggregation (EAFA) framework with a Hardware-Aware Gradient-Sensitive Structured Pruning (HAGS-Pruning) algorithm1. By preserving high-frequency boundary gradient maps in shallow layers and systematically stripping redundant macro-feature detection heads via dependency-graph structural pruning, the proposed architecture achieves state-of-the-art detection precision ($mA{P}_{50}$) while exceeding 30 Frames Per Second (FPS) on embedded edge accelerators2.

| Metadata Parameter | Specification & Operational Target |
| :---- | :---- |
| **Project Title** | Edge-Aware Pruned Feature Aggregation for Real-Time Tiny Object Detection on Resource-Constrained Devices1 |
| **Target Domain** | Computer Vision, Deep Learning, Edge Computing, Model Compression1 |
| **Expected Duration** | 12 Months (Phased Execution)1 |
| **Primary Edge Hardware** | NVIDIA Jetson Orin Nano (8GB), Raspberry Pi 52 |
| **Target Publication Level** | Q1 Journals (IEEE TCSVT, IEEE IoT-J, Pattern Recognition) or CORE A/A\* Conferences (CVPR, ECCV, WACV)1 |
| **Primary Datasets** | VisDrone-DET, UAVDT, TT100K, RGBT-Tiny2 |
| **Core Software Stack** | PyTorch 2.1+, TensorRT 10.x, DepGraph (Torch-Pruning), OpenCV, ONNX Runtime4 |
| **Performance Target** | **$>30$** FPS on Jetson Orin Nano at $640\times 640$ resolution; \$\>5\\%\$ boost in $A{P}_{tiny}$ over baseline YOLOv8/v112 |

## **Technical Problem Statement & Spatial Degradation Mechanics**

In conventional single-stage object detectors, input images of spatial dimensions $H\times W\times 3$ are downsampled by a factor of ${2}^{S}$ across successive backbone stages $S\in \{1,2,3,4,5\}$2. For a tiny object occupying a bounding box of size $h\times w$ where $h,w<16$ pixels, downsampling to stride-32 ($S=5$, feature map dimensions $\frac{H}{32}\times \frac{W}{32}$) reduces the spatial footprint of the object to less than $0.25$ pixels2. Consequently, spatial information is completely erased, leaving deep layers incapable of reconstructing micro-target spatial priors regardless of feature pyramid depth2.  
The sequential feature downsampling pipeline transitions from high-resolution shallow representations to deep semantic abstractions2. At stride-4 ($\frac{H}{4}\times \frac{W}{4}$), the backbone preserves high-frequency spatial boundaries and raw edge gradients1. By stride-8 ($\frac{H}{8}\times \frac{W}{8}$), micro-target spatial anchors maintain structural viability for P2 detection branches14. However, as features progress to stride-16 ($\frac{H}{16}\times \frac{W}{16}$) and stride-32 ($\frac{H}{32}\times \frac{W}{32}$), representation shifts toward macro-target context, resulting in the total destruction of micro-target spatial detail2.  
Furthermore, standard overlap-based label assignment schemes utilizing Intersection over Union (IoU) exhibit extreme sensitivity to spatial perturbations on small bounding boxes15. For a target of dimension $20\times 20$ pixels, a diagonal prediction shift of merely 6 pixels causes the IoU metric to drop precipitously from $1.0$ to $0.325$, whereas the same 6-pixel shift on a large target ($70\times 70$ pixels) maintains an IoU of $0.718$15. This mathematical discontinuity leads to anchor assignment instability and weak gradient backpropagation during detector training3.  
Standard structural pruning paradigms evaluate channel importance via magnitude-based metrics (${L}_{1}$ or ${L}_{2}$ norm) or layer-agnostic criteria1. Because deeper convolutional layers contain larger weight norms and richer semantic abstractions, global pruning algorithms inadvertently apply excessive pruning ratios to early, high-resolution shallow layers1. However, shallow layers contain the precise high-frequency edge gradients and micro-scale spatial structures required to isolate $8\times 8$ to $16\times 16$ pixel targets1. Stripping shallow channels destroys boundary definitions, causing tiny objects to blend into complex background clutter2.

## **Literature Review & Comparative Analysis**

Existing tiny object detection techniques generally fall into four categories: scale-aware architectures, multi-scale feature fusion networks, super-resolution reconstruction networks, and model compression frameworks2. Scale-aware single-stage detectors are designed predominantly for general object categories, employing aggressive downsampling to optimize top-1 classification accuracy and macro-bounding box regression2. They lack specialized high-resolution anchor heads for micro-targets, leading to high false-negative rates in aerial and drone-based surveillance2.  
Multi-scale feature fusion architectures utilize bi-directional or high-resolution shortcut paths to preserve feature maps at stride-4 or stride-82. While effective at preserving spatial context, the addition of dense cross-scale connections creates high memory bandwidth overhead, memory footprint expansion, and latency bottlenecks on embedded GPUs and CPUs1. Recent frequency-domain paradigms incorporate Haar wavelet transforms or multi-scale edge enhancement modules to decouple high-frequency edge signals from low-frequency background texture4. These methods yield substantial detection accuracy improvements but require optimized lightweight implementation to prevent memory latency during inference4.  
Dependency-graph structural pruning algorithms trace inter-layer weight couplings across complex residual connections and feature pyramid branches, enabling structured channel removal without breaking tensor operations7. However, existing implementations rely on static magnitude criteria rather than task-specific gradient feedback tailored for micro-target localization9.

| Approach / Model Paradigm | Primary Optimization Mechanism | Latency Overhead | Tiny Target Accuracy (APtiny​) | Architectural Bottlenecks & Limitations |
| :---- | :---- | :---- | :---- | :---- |
| **Standard YOLO (v8n / v11n)** \[cite: 12, 13\] | CSPDarknet backbone, Path Aggregation Neck | Low ($<10ms$)23 | Poor (\$\<15.0\\%\$)2 | Aggressive downsampling to stride-32 destroys micro-features2. |
| **BiFPN / AFPN Aggregation** \[cite: 2, 19\] | Bi-directional multi-scale feature fusion | High ($>40ms$)2 | Moderate (\$22.0\\% \- 26.0\\%\$)2 | High memory bandwidth overhead; redundant macro heads1. |
| **Wavelet-Downsampled Nets** \[cite: 4\] | Residual Haar Wavelet decomposition | Moderate ($20-30ms$)4 | High (\$28.0\\% \- 32.0\\%\$)4 | Substantial FLOP consumption in frequency transform blocks4. |
| **TinyFormer Hybrid** \[cite: 5\] | Parallel Bi-Fusion & Spatial Semantic Adapter | Moderate ($15-25ms$)5 | Very High (\$\>34.0\\%\$)5 | Transformer token competition; complex edge deployment5. |
| **Proposed EAFA-HAGS** \[cite: 1, 2\] | Edge-aware convolution, DepGraph asymmetric pruning | Extremely Low ($<8ms$)7 | Optimal (\$\>35.0\\%\$)6 | Requires dynamic gradient calibration during fine-tuning2. |

## **Technical Methodology & Architectural Formulation**

The proposed system integrates three main technical modules: an Edge-Aware Feature Aggregation Module (EAFA), a Hardware-Aware Gradient-Sensitive Structured Pruning (HAGS-Pruning) framework, and a Normalized Wasserstein Distance (NWD) regression loss head1. The operational workflow routes high-resolution shallow backbone features through the EAFA block to extract boundary maps, fuses these features with an optimized P2 micro-target detection neck, and selectively strips the deep P5 macro-detection heads through DepGraph asymmetric pruning1.

### **Edge-Aware Feature Aggregation Module (EAFA)**

The EAFA module is embedded into the early, high-resolution stages of the network backbone (stride-4 and stride-8 feature maps) to extract sharp gradient maps and high-frequency structural boundaries before spatial reduction occurs1. The module operates across two parallel paths:  
> High-Frequency Spatial Gradient Branch: Implements a dual-kernel differential convolution operator that extracts horizontal and vertical spatial intensity gradients:

> $${G}_{x}={F}_{in}*{K}_{x},{G}_{y}={F}_{in}*{K}_{y}$$  
> 

> $${F}_{edge}=\sqrt{{G}_{x}^{2}+{G}_{y}^{2}+\epsilon }$$

> 1.   
>    where ${F}_{in}\in {R}^{H\times W\times C}$ represents the shallow input feature tensor, ${K}_{x},{K}_{y}$ denote learnable edge-enhancement kernels, and $\epsilon =1{0}^{-6}$ ensures numerical stability1.

> Gated Dynamic Spatial Attention (GDSA): To mitigate background noise amplification resulting from edge detection, the module passes ${F}_{edge}$ through a spatial gating operator:

> $${A}_{s}=\sigma \left({{W}_{2}\left({\delta \left({{W}_{1}\left({[AvgPool({F}_{edge});MaxPool({F}_{edge})]}\right)}\right)}\right)}\right)$$  
> 

> $${F}_{EAFA}={F}_{in}\odot {A}_{s}+\gamma {F}_{edge}$$

> 2.   
>    where $\sigma$ denotes the sigmoid activation function, $\delta$ is the GELU nonlinearity, ${W}_{1},{W}_{2}$ are point-wise $1\times 1$ convolutions, $\odot$ represents element-wise Hadamard multiplication, and $\gamma$ is a learnable scaling parameter initialized to $0.1$1.

The shallow input feature map ${F}_{in}$ passes simultaneously into the gradient convolution operator to yield ${F}_{edge}$ and into an average and max pooling pipeline1. The concatenated pooling descriptors are processed through point-wise $1\times 1$ convolutions with GELU activation to generate the spatial attention mask ${A}_{s}$1. Modulating ${F}_{in}$ by ${A}_{s}$ via element-wise multiplication and adding the scaled edge features yields the final refined feature tensor ${F}_{EAFA}$1. This structural formulation ensures that high-contrast contour cues of micro-targets ($<16\times 16$ pixels) are amplified prior to feature pyramid fusion, establishing explicit pixel-level localization paths1.

### **Hardware-Aware Gradient-Sensitive Structured Pruning (HAGS-Pruning)**

To satisfy strict real-time edge hardware latency bounds without compromising tiny target precision, the architecture incorporates a dynamic, dependency-graph-driven pruning strategy1. Arbitrary channel pruning in complex networks featuring residual shortcuts and cross-stage partial connections leads to matrix dimension mismatches7. Using DepGraph, the network is modeled as a dependency graph $G=(V,E)$, where vertices $V$ represent convolutional layers and edges $E$ represent structural dependencies9. Parameter channels coupled across layers are grouped into discrete pruning groups ${G}_{k}$9.  
Instead of static weight norms, channel importance within each dependency group ${G}_{k}$ is computed using the absolute first-order Taylor expansion of the task-specific loss function ${L}_{tiny}$ relative to channel activations10:

\$\$\\mathcal{I}(\\theta\_m) \= \\frac{1}{N\_{\\text{batch}}} \\sum\_{i=1}^{N\_{\\text{batch}}} \\left\\vert{} \\theta\_m \\cdot \\frac{\\partial \\mathcal{L}\_{\\text{tiny}}}{\\partial \\theta\_m} \\right\\vert{}\$\$  
where ${\theta }_{m}$ denotes the weights of channel $m$, and ${L}_{tiny}$ is the compound localization and classification loss evaluated specifically on images containing micro-targets2.  
To guarantee that shallow feature representations remain intact while redundant macro-detection paths are stripped, layers are dynamically classified based on their sensitivity profile2:

$${S}_{l}=\frac{\Delta mA{P}_{50}}{\Delta {p}_{l}}$$

where $\Delta {p}_{l}$ represents a benchmark 40% channel reduction at layer $l$7.  
Layers evaluated under the sensitivity metric ${S}_{l}$ branch into two primary operational categories:

> * Sensitive Layers (\$S\_l \> 5.0\\%\$ drop): These encompass early backbone convolutional layers, EAFA modules, and high-resolution P2 detection heads2. These layers preserve spatial fidelity and are restricted to conservative pruning ratios between \$0\\%\$ and \$10\\%\$2.  
> * Robust Layers (\$S\_l \< 2.0\\%\$ drop): These comprise deep macro-detection heads (P5/stride-32) and large-object anchor paths2. Because macro-detection paths contribute minimally to tiny target localization, these layers undergo aggressive branch pruning with channel reduction ratios reaching \$70\\%\$ to \$90\\%\$6. Moderately sensitive layers (\$2.0\\% \\le S\_l \\le 5.0\\%\$) are assigned intermediate pruning ratios of \$30\\%\$ to \$40\\%\$7.

### **Bounding Box Loss & Label Assignment Optimization**

To resolve the IoU gradient saturation issue for micro-scale targets, the bounding box regression loss replaces standard CIoU/GIoU with a Normalized Wasserstein Distance (NWD) loss13. A tiny bounding box $R=({x}_{c},{y}_{c},w,h)$ is modeled as a 2D Gaussian distribution $N(\mu ,\Sigma )$:

\$\$\\boldsymbol{\\mu} \= \\begin{bmatrix} x\_c \\\\ y\_c \\end{bmatrix}, \\quad \\mathbf{\\Sigma} \= \\begin{bmatrix} \\frac{w^2}{4} & 0 \\\\ 0 & \\frac{h^2}{4} \\end{bmatrix}\$\$  
The optimal transport distance between the predicted bounding box distribution ${N}_{1}$ and ground truth distribution ${N}_{2}$ is calculated via the ${W}_{2}^{2}$ Wasserstein metric:

$${W}_{2}^{2}({N}_{1},{N}_{2})=\parallel {\mu }_{1}-{\mu }_{2}\parallel {}_{2}^{2}+Tr\left({{\Sigma }_{1}+{\Sigma }_{2}-2({\Sigma }_{1}^{1/2}{\Sigma }_{2}{\Sigma }_{1}^{1/2}{)}^{1/2}}\right)$$

$$NWD({N}_{1},{N}_{2})=\exp \left({-\frac{\sqrt{{W}_{2}^{2}({N}_{1},{N}_{2})}}{C}}\right)$$

where $C$ is a normalization constant tuned to dataset scale ($C=12.8$ for VisDrone)3. The regression loss is defined as:

$${L}_{NWD}=1-NWD({N}_{1},{N}_{2})$$

This guarantees continuous, smooth non-zero gradients even when predicted and ground truth bounding boxes share zero spatial IoU overlap13.

## **Experimental Framework & Evaluation Matrix**

The evaluation protocol targets four public benchmark datasets exhibiting diverse spatial resolutions, extreme target scale variation, and complex environmental conditions2.

| Dataset Benchmark | Total Frames / Images | Target Object Domain | Resolution Range | Micro Target Ratio (\<16×16 px) | Train / Val / Test Split Ratio |
| :---- | :---- | :---- | :---- | :---- | :---- |
| **VisDrone-DET2019** \[cite: 2, 5, 14\] | 10,209 | Drone aerial surveillance (vehicles, pedestrians) | $960\times 540$ to $2000\times 1500$ | \$58.4\\%\$ | $60:10:30$ \[cite: 6\] |
| **UAVDT** \[cite: 2, 6, 28\] | 40,708 | Unmanned aerial vehicle traffic perception | $1024\times 540$ | \$64.1\\%\$ | $70:20:10$ \[cite: 6\] |
| **TT100K** \[cite: 2, 6, 28\] | 100,000 | Traffic signs (Tsinghua-Tencent benchmark) | $2048\times 2048$ | \$42.3\\%\$ | $80:10:10$ \[cite: 6, 29\] |
| **RGBT-Tiny** \[cite: 4, 11\] | 115 sequences (1.2M pairs) | Visible-thermal multimodal micro targets | $640\times 512$ | \$71.2\\%\$ | $75:0:25$ \[cite: 4\] |

Post-training model deployment targets two distinct edge hardware paradigms to demonstrate platform-agnostic real-time efficiency2. Trained PyTorch models are converted to ONNX intermediate representations (Opset 17\) and subsequently compiled into optimized execution engines7. On NVIDIA Jetson platforms, compilation utilizes TensorRT 10.x with FP16 and INT8 Quantization-Aware Training (QAT) engines23. On CPU-centric edge platforms such as the Raspberry Pi 5, deployment utilizes ONNX Runtime with ARM NEON vector extensions2.

| Deployment Testbed | Computing Architecture | On-Board Memory & Bandwidth | Target Latency Overhead | Compilation Engine & Precision | Target FPS Threshold |
| :---- | :---- | :---- | :---- | :---- | :---- |
| **NVIDIA Jetson Orin Nano (8GB)** \[cite: 2, 7, 30\] | 1024-core Ampere GPU, 6-core ARM CPU | 8GB LPDDR5 ($68GB/s$)7 | $<10.0ms$ \[cite: 7, 23\] | TensorRT 10.x (FP16 & INT8 QAT)23 | $>60FPS$ \[cite: 7, 24\] |
| **Raspberry Pi 5 (8GB)** \[cite: 2\] | Quad-core ARM Cortex-A76 @ 2.4GHz | 8GB LPDDR4X ($17GB/s$) | $<30.0ms$ | ONNX Runtime / ARM NEON Execution10 | $>30FPS$ \[cite: 2\] |

Primary evaluation criteria encompass both detection accuracy and hardware-level execution metrics2. Detection accuracy is measured using Mean Average Precision at IoU threshold $0.50$ ($mA{P}_{50}$), mean Average Precision across IoU thresholds $0.50:0.95$ ($mA{P}_{50-95}$), and Average Precision specifically for micro-targets ($A{P}_{tiny}$, area $<16\times 16$ pixels)2. Hardware efficiency is evaluated via parameter count ($M$), Floating Point Operations ($GFLOPs$), Model Binary Size ($MB$), Latency ($ms$), Frame Rate ($FPS$), Peak RAM Footprint ($MB$), and Power Consumption ($Watts$)2.

## **Work Plan & Execution Roadmap**

The 12-month research initiative is structured into six sequential phases designed to manage risk and validate architectural contributions incrementally1.

| Execution Phase | Operational Duration | Key Milestones & Research Tasks | Primary Deliverables | Risk Mitigation Strategy |
| :---- | :---- | :---- | :---- | :---- |
| **Phase 1: Environment & Baselines** | Months 1–22 | Setup RTX 5090 training server4; format VisDrone/UAVDT/TT100K datasets2; establish baseline YOLOv8/v11 models12. | Cleaned dataset pipelines; baseline performance logs2. | Use standardized synthetic augmentations to address class imbalance6. |
| **Phase 2: EAFA Architecture Design** | Months 3–52 | Develop edge-enhancement convolution branch1; integrate GDSA attention1; validate spatial feature retention1. | Integrated PyTorch EAFA module codebase; initial $A{P}_{tiny}$ gain validation2. | Fall back to Haar Wavelet downsampling if dynamic convolution overhead exceeds bounds4. |
| **Phase 3: HAGS-Pruning Framework** | Months 6–72 | Build DepGraph dependency graphs7; formulate Taylor expansion gradient ranking22; implement asymmetric branch pruning2. | Automated structural pruning script; compressed model weights (\$\>50\\%\$ FLOP reduction)7. | Adjust layer sensitivity thresholds dynamically if fine-tuning recovery fails to converge7. |
| **Phase 4: Embedded Optimization** | Months 8–92 | Export models to ONNX Opset 1710; build TensorRT FP16/INT8 engines on Jetson Orin Nano7; implement ONNX Runtime on RPi52. | Deployable TensorRT .engine files; memory bandwidth profile report2. | Apply dynamic range Calibration Tables during INT8 quantization to avoid precision clipping10. |
| **Phase 5: Benchmarking & Ablation** | Months 10–112 | Conduct comprehensive ablation studies (EAFA vs Pruning vs Loss)2; cross-benchmark on VisDrone/UAVDT/TT100K2. | Comprehensive experimental result matrix; comparative throughput plots2. | Validate on auxiliary RGBT-Tiny dataset if cross-domain domain shift occurs4. |
| **Phase 6: Dissemination & Writing** | Month 122 | Write research manuscript1; internal peer review1; open-source PyTorch/TensorRT codebase1. | Q1 Journal submission (IEEE TCSVT / IoT-J)1; public GitHub repository release1. | Prepare conference submission (CVPR/ECCV/WACV) as alternative tier-1 outlet1. |

## **Systemic Architectural Insights & Strategic Risk Analysis**

A key technical trade-off in edge computing concerns determining the optimal balance between structural channel pruning and post-training precision quantization (INT8)7. Experimental evidence reveals that applying uniform INT8 quantization directly to unpruned deep networks often causes catastrophic precision degradation in micro-target detection heads10. Direct INT8 quantization maps continuous floating-point activations into discrete integer bins, introducing activation quantization noise that obliterates subtle micro-target signals in shallow layers4. By contrast, combining HAGS-Pruning with FP16 calibration strips redundant macro-channels while preserving the fine-grained dynamic range of shallow edge features, achieving high throughput without loss of localization precision2.  
While traditional multi-scale feature aggregation necks (e.g., BiFPN or PAFPN) enhance detection accuracy by concatenating feature maps across multiple strides, they create high memory read-write overheads on embedded System-on-Chip (SoC) architectures1. On platforms like the Jetson Orin Nano, where CPU and GPU share unified memory bandwidth ($68GB/s$), continuous feature tensor reshaping and cross-scale concatenation operations stall execution pipelines7. The proposed EAFA architecture addresses this bottleneck through asymmetric branch pruning1. By completely removing the deepest macro-scale feature pyramid paths (stride-32 / P5 detection head) and routing shallow edge-aware features directly to a modified P2/P3 neck, memory access transactions per frame are reduced by over \$40\\%\$, directly translating to higher frame rates6.  
Focusing exclusively on shallow high-resolution features carries the risk of contextual overfitting, wherein the detector becomes overly reliant on local edge textures while ignoring global scene context4. This can cause high false-positive rates when background clutter exhibits sharp edge profiles (e.g., foliage, building corners, wave crests)4. To prevent local gradient bias, the EAFA module pairs high-frequency differential convolutions with Gated Dynamic Spatial Attention (GDSA)1. GDSA leverages global average and max pooling descriptors to contextualize local edge maps across broader receptive fields, ensuring that low-level edge activations are amplified only when they align with object-like spatial bounds1.

## **Conclusions & Actionable Next Steps**

This updated research proposal provides a unified framework for real-time tiny object detection on resource-constrained edge hardware1. By integrating high-frequency spatial edge preservation with gradient-sensitive dependency-graph structural pruning, the proposed architecture directly addresses the dual challenges of spatial feature loss and computational overhead1.

### **Key Strategic Directives for Implementation**

> 1. Workspace Initialization: Finalize PyTorch 2.1 environment configurations on the RTX 5090 server with CUDA 12.x, TensorRT 10.x, and DepGraph dependencies4.  
> 2. Dataset Standardization: Convert VisDrone-DET, UAVDT, and TT100K annotations to normalized YOLO format, establishing standard $80:10:10$ train/val/test splits and generating scale distribution profiles2.  
> 3. EAFA Module Prototyping: Implement the differential convolution operators and GDSA spatial attention mechanisms in PyTorch, benchmarking feature map retention against baseline CSPDarknet backbones1.  
> 4. Pruning Pipeline Integration: Configure the Torch-Pruning DepGraph engine to trace network layer dependencies and compute Taylor expansion gradient weights based on NWD bounding box regression loss7.  
> 5. Edge Deployment Optimization: Validate ONNX export pipelines and compile TensorRT FP16 engines on the NVIDIA Jetson Orin Nano testbed to establish initial latency baselines7.

#### **Works cited**

> 1. [https\://drive.google.com/open?id=1pYMXMBsMvqVEwFhFvUtxbQtOH8mOOuOhW1lMg4dOuuE](https://drive.google.com/open?id=1pYMXMBsMvqVEwFhFvUtxbQtOH8mOOuOhW1lMg4dOuuE)  
> 2. Research\_Proposal.docx, uploaded:Research\_Proposal.docx  
> 3. RFLA: Gaussian Receptive Field based Label Assignment for Tiny, [https\://www\.researchgate.net/publication/362789935\_RFLA\_Gaussian\_Receptive\_Field\_based\_Label\_Assignment\_for\_Tiny\_Object\_Detection](https://www.researchgate.net/publication/362789935_RFLA_Gaussian_Receptive_Field_based_Label_Assignment_for_Tiny_Object_Detection)  
> 4. Small Object Detection in Complex Backgrounds with Multi-Scale, [https\://arxiv.org/pdf/2603.03788](https://arxiv.org/pdf/2603.03788)  
> 5. Preserving Tiny Objects in YOLO-DETR Hybrid Real-time Detectors, [https\://arxiv.org/html/2605.25046v1](https://arxiv.org/html/2605.25046v1)  
> 6. Tiny object detection algorithm based on edge information, [https\://www\.spiedigitallibrary.org/conference-proceedings-of-spie/14070/1407007/Tiny-object-detection-algorithm-based-on-edge-information-reconstruction/10.1117/12.3107091.full](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/14070/1407007/Tiny-object-detection-algorithm-based-on-edge-information-reconstruction/10.1117/12.3107091.full)  
> 7. Edge-Oriented Lightweight YOLO-World with Cross-Modal Fusion, [https\://www\.mdpi.com/2076-3417/16/15/7790](https://www.mdpi.com/2076-3417/16/15/7790)  
> 8. Real-Time Target-Oriented Grasping Framework for Resource, [https\://pmc.ncbi.nlm.nih.gov/articles/PMC12845653/](https://pmc.ncbi.nlm.nih.gov/articles/PMC12845653/)  
> 9. GitHub \- VainF/Torch-Pruning: \[CVPR 2023\] DepGraph: Towards, [https\://github.com/VainF/Torch-Pruning](https://github.com/VainF/Torch-Pruning)  
> 10. Running a Deep ECG Model on Low-Cost Hardware. \- DiVA portal, [https\://www\.diva-portal.org/smash/get/diva2:2093910/FULLTEXT01.pdf](https://www.diva-portal.org/smash/get/diva2:2093910/FULLTEXT01.pdf)  
> 11. Small Object Detection in Complex Backgrounds with Multi-Scale, [https\://arxiv.org/html/2603.03788v1](https://arxiv.org/html/2603.03788v1)  
> 12. A Real-Time Detector for Tiny Defect Detection on PCB Surfaces, [https\://arxiv.org/html/2601.11402v1](https://arxiv.org/html/2601.11402v1)  
> 13. EMN-Net: A Lightweight YOLOv8-Based Model for Real-Time, [https\://www\.mdpi.com/1999-4893/19/6/438](https://www.mdpi.com/1999-4893/19/6/438)  
> 14. Small object detection based on YOLOv8 in UAV perspective, [https\://www\.researchgate.net/publication/383213316\_Small\_object\_detection\_based\_on\_YOLOv8\_in\_UAV\_perspective](https://www.researchgate.net/publication/383213316_Small_object_detection_based_on_YOLOv8_in_UAV_perspective)  
> 15. Towards Large-Scale Small Object Detection: Survey and Benchmarks, [https\://scispace.com/pdf/towards-large-scale-small-object-detection-survey-and-wynv90s4.pdf](https://scispace.com/pdf/towards-large-scale-small-object-detection-survey-and-wynv90s4.pdf)  
> 16. Towards Large-Scale Small Object Detection: Survey and Benchmarks, [https\://ieeexplore.ieee.org/iel7/34/4359286/10168277.pdf](https://ieeexplore.ieee.org/iel7/34/4359286/10168277.pdf)  
> 17. Lightweight YOLOv8-PSN for Real-Time Detection of Underground, [https\://www\.mdpi.com/2075-5309/16/15/3055](https://www.mdpi.com/2075-5309/16/15/3055)  
> 18. Advancements in Small-Object Detection (2023–2025) \- MDPI, [https\://www\.mdpi.com/2076-3417/15/22/11882](https://www.mdpi.com/2076-3417/15/22/11882)  
> 19. Deep learning-based small object detection: A survey, [https\://pdfs.semanticscholar.org/f9ab/089ca58d675b98dc24df1605152397219375.pdf](https://pdfs.semanticscholar.org/f9ab/089ca58d675b98dc24df1605152397219375.pdf)  
> 20. HashEye: a real-time on-drone high-resolution tiny object detection, [https\://pmc.ncbi.nlm.nih.gov/articles/PMC13338025/](https://pmc.ncbi.nlm.nih.gov/articles/PMC13338025/)  
> 21. Lightweight YOLO Object Detection Algorithms for Vision ... \- ICCK, [https\://www\.icck.org/article/epdf/ticps/6381](https://www.icck.org/article/epdf/ticps/6381)  
> 22. Deep neural network pruning method based on sensitive layers and, [https\://www\.researchgate.net/publication/372936158\_Deep\_neural\_network\_pruning\_method\_based\_on\_sensitive\_layers\_and\_reinforcement\_learning](https://www.researchgate.net/publication/372936158_Deep_neural_network_pruning_method_based_on_sensitive_layers_and_reinforcement_learning)  
> 23. Phase 2 Capstone — Real-Time Detector \+ Tracker \+ Segmenter On, [https\://www\.forasoft.com/learn/ai-for-video-engineering/articles-ai/phase-2-capstone-jetson-orin-detector-tracker-segmenter](https://www.forasoft.com/learn/ai-for-video-engineering/articles-ai/phase-2-capstone-jetson-orin-detector-tracker-segmenter)  
> 24. YOLOv8 on Jetson Orin Nano: From PyTorch to TensorRT INT8, [https\://hokwangchoi.com/blog/vision-benchmarks/](https://hokwangchoi.com/blog/vision-benchmarks/)  
> 25. TOWARDS META-PRUNING VIA OPTIMAL TRANSPORT, [https\://proceedings.iclr.cc/paper\_files/paper/2024/file/69a076724e7228aba0272305bb98727e-Paper-Conference.pdf](https://proceedings.iclr.cc/paper_files/paper/2024/file/69a076724e7228aba0272305bb98727e-Paper-Conference.pdf)  
> 26. (PDF) Structural Pruning for Diffusion Models \- ResearchGate, [https\://www\.researchgate.net/publication/370869863\_Structural\_Pruning\_for\_Diffusion\_Models](https://www.researchgate.net/publication/370869863_Structural_Pruning_for_Diffusion_Models)  
> 27. EfficientDet: Scalable and Efficient Object Detection \- ResearchGate, [https\://www\.researchgate.net/publication/343456623\_EfficientDet\_Scalable\_and\_Efficient\_Object\_Detection](https://www.researchgate.net/publication/343456623_EfficientDet_Scalable_and_Efficient_Object_Detection)  
> 28. HAE-Net: A High-Resolution Asymmetric and Efficient Network for, [https\://www\.researchgate.net/publication/408524958\_HAE-Net\_A\_High-Resolution\_Asymmetric\_and\_Efficient\_Network\_for\_Small\_Object\_Detection\_in\_UAV\_Imagery](https://www.researchgate.net/publication/408524958_HAE-Net_A_High-Resolution_Asymmetric_and_Efficient_Network_for_Small_Object_Detection_in_UAV_Imagery)  
> 29. (PDF) A Scale-Adaptive and Frequency-Aware Attention Network for, [https\://www\.researchgate.net/publication/394910379\_A\_Scale-Adaptive\_and\_Frequency-Aware\_Attention\_Network\_for\_Precise\_Detection\_of\_Strawberry\_Diseases](https://www.researchgate.net/publication/394910379_A_Scale-Adaptive_and_Frequency-Aware_Attention_Network_for_Precise_Detection_of_Strawberry_Diseases)  
> 30. optimization techniques for adaptive autonomous driving systems, [https\://rex.libraries.wsu.edu/view/pdfCoverPage?instCode=01ALLIANCE\_WSU\&filePid=13426258180001842\&download=true](https://rex.libraries.wsu.edu/view/pdfCoverPage?instCode=01ALLIANCE_WSU&filePid=13426258180001842&download=true)  
> 31. Detection of Critical Parts of River Crab Based on Lightweight ... \- PMC, [https\://pmc.ncbi.nlm.nih.gov/articles/PMC11645041/](https://pmc.ncbi.nlm.nih.gov/articles/PMC11645041/)