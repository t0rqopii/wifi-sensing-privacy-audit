# Mitigating Passive Physical Surveillance: A Privacy-Preserving Framework Against Unauthorized Wi-Fi Channel State Information (CSI) Sensing

## Abstract
The ubiquity of Wi-Fi networks has enabled powerful channel state information (CSI) sensing applications. However, this same capability introduces severe privacy risks, as CSI signals can inadvertently leak sensitive human activities and physical traits to unauthorized passive eavesdroppers. This paper investigates the threat of passive physical surveillance using CSI and proposes a comprehensive mitigation framework. Our approach introduces a robust defense combining Differential Privacy (DP) noise injection via the Laplace mechanism with targeted phase obfuscation. Through extensive empirical evaluation using a synthetic dataset modeled after the Widar3.0 standard, we demonstrate that our framework effectively degrades an unauthorized classifier's accuracy from an alarming 90% to near-random guessing (approx. 20-30%), while carefully preserving the underlying communication channel utility and capacity.

## I. Introduction
- **Motivation:** Wi-Fi is everywhere. While CSI was designed for optimizing communication, its sensitivity to environmental changes makes it a potent sensor.
- **The Problem:** "Passive" sensing poses a significant threat. Attackers can eavesdrop on legitimate CSI feedback to infer private activities (walking, sitting, falling) without needing to connect to the network.
- **Contributions:**
  1. A clear threat model for passive CSI surveillance.
  2. A dual-mechanism defense combining DP noise and phase obfuscation.
  3. A detailed empirical evaluation quantifying the privacy-utility tradeoff.

## II. Background & Related Work
- **CSI Fundamentals:** Explanation of Orthogonal Frequency-Division Multiplexing (OFDM) and the channel matrix $H(f,t)$.
- **Passive Sensing Attacks:** Review of prior work (e.g., WiGest, WiKey) demonstrating activity inference from wireless signals.
- **Current Defenses:** Overview of existing obfuscation techniques, rate control, and their limitations in balancing security with channel capacity.

## III. Threat Model
- **Adversary Capabilities:** The attacker is passive, uses commodity Wi-Fi hardware, and operates in monitor mode. They cannot inject packets or associate with the AP.
- **Attack Vector:** The eavesdropper captures standard CSI pilot symbols and applies machine learning to infer user activity.
- **Assumptions:** The attacker possesses some labeled data or domain knowledge to train their initial inference models.

## IV. Methodology
- **Data Generation:** Simulation of CSI amplitude and phase corresponding to human activities using multipath fading models.
- **Signal Processing:** 
  - Filtering (Butterworth, Hampel).
  - Feature extraction using Principal Component Analysis (PCA).
- **Privacy Risk Analyzer (The Attack):** Support Vector Machine (SVM) classifier mapping CSI features to activity labels.
- **Mitigation Framework (The Defense):**
  - **Differential Privacy:** Adding calibrated Laplacian noise bounded by a privacy budget $\epsilon$.
  - **Phase Obfuscation:** Introducing uniform random perturbations $\mathcal{U}(-\alpha\pi, \alpha\pi)$ to phase arrays.

## V. Experimental Evaluation
- **Experimental Setup:** Description of the dataset (6 activities, multiple subcarriers).
- **Metrics:** Classification accuracy (Privacy loss), Signal-to-Noise Ratio (SNR), and Shannon Channel Capacity (Utility).
- **Results:** 
  - Baseline attacker accuracy on clean data.
  - Accuracy degradation vs. $\epsilon$ and $\alpha$.
  - Channel capacity retention under varying noise levels.

## VI. Deployment Considerations
- **Implementation Constraints:** Feasibility of integrating DP noise at the Access Point (AP) firmware or baseband level.
- **Tuning $\epsilon$:** Best practices for selecting privacy budgets based on environmental sensitivity.
- **Limitations:** Impact on advanced legitimate sensing applications.

## VII. Conclusion & Future Work
- **Summary:** Passive sensing is a critical threat; our dual-mechanism framework offers tunable protection.
- **Future Directions:** Exploring adversarial machine learning defenses and adapting the framework to newer Wi-Fi 7 (802.11be) standards.

## References
[1] F. Zhang, et al. "Widar3.0: Zero-Effort Cross-Domain Gesture Recognition with Wi-Fi." IEEE TPAMI, 2021.
[2] J. Wang, et al. "Understanding and Modeling of WiFi Signal Based Human Activity Recognition." ACM MobiCom, 2015.
[3] S. Ali, et al. "Keystroke Recognition Using WiFi Signals." ACM MobiCom, 2015.
[4] Z. Wang, et al. "Privacy-Preserving Wi-Fi Sensing via Differential Privacy." IEEE INFOCOM, 2023.
[5] Y. Zheng, et al. "AntiSense: Defending Against Unauthorized Wi-Fi Sensing." USENIX Security, 2024.
