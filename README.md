# Mitigating Passive Physical Surveillance: CSI Privacy Framework

This project contains the software implementation for the research paper: "Mitigating Passive Physical Surveillance: A Privacy-Preserving Framework Against Unauthorized Wi-Fi Channel State Information (CSI) Sensing."

## Project Overview

Wi-Fi Channel State Information (CSI) can inadvertently leak highly sensitive personal information, allowing unauthorized observers to perform passive physical surveillance (e.g., detecting human presence, activities, or postures) without the user's knowledge.

This framework provides:
1. **Synthetic CSI Data Generation:** Models realistic CSI data following standard datasets (like Widar3.0).
2. **Privacy Risk Analyzer:** Simulates an attacker using signal processing and Machine Learning (PCA + SVM) to extract activities from CSI.
3. **Defensive Mitigation Engine:** Injects Differential Privacy (DP) noise and obfuscates phase to protect user privacy.
4. **Evaluation Module:** Compares the attacker's accuracy before and after mitigation, preserving communication utility (SNR and channel capacity).

## Architecture

*   `src/core/csi_generator.py`: Generates synthetic data simulating human activities affecting Wi-Fi channels.
*   `src/core/signal_processor.py`: Filters noise and extracts PCA features.
*   `src/core/privacy_risk_analyzer.py`: Trains an SVM classifier on extracted features.
*   `src/core/mitigation_engine.py`: Applies privacy-preserving defenses (Laplace DP noise, phase randomization).
*   `src/evaluation/evaluator.py`: Orchestrates the experiments and evaluation metrics.
*   `src/utils/`: Data loaders, metrics, and visualization utilities.

## Setup and Installation

1. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run tests to ensure everything is working:
   ```bash
   pytest tests/
   ```

## Usage

Use `main.py` to run the various steps of the pipeline:

```bash
# Run the full pipeline (generate, analyze, mitigate, evaluate)
python main.py --mode full

# Run specific modules
python main.py --mode generate
python main.py --mode analyze
python main.py --mode mitigate
python main.py --mode evaluate
```

Results and plots are saved in the `output/` directory.

## Disclaimer

This project is created strictly for academic research, educational purposes, and privacy auditing regarding wireless signal propagation. It relies entirely on simulated environment data and open-source public benchmark datasets. It does not contain any functionality for unauthorized over-the-air interception, active network disruption, or wireless exploitation.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
