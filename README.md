Inductive Relational Learning under Temporal Collusion Drift

This repository presents a scalable inductive Graph Neural Network (GraphSAGE) framework for fraud detection under:

Temporal non-stationarity

Extreme class imbalance

Structural collusion persistence

Resource-constrained GPU environments

🔬 Key Contributions

Strict inductive temporal split (no validation leakage)

Manual mini-batch 1-hop subgraph sampling

Persistent collusion simulation framework

Feature-enhanced relational modeling under imbalance

Scalable training on 6.3M transactions (8.6M nodes)

📊 Experimental Findings
Model	Configuration	PR-AUC
GraphSAGE	Structure-only	~0.50
GraphSAGE	Feature-enhanced	~0.998

Structure-only message passing fails under temporal drift.
Exposure-aware node representations significantly improve continuation detection.

📁 Dataset

This project uses the PaySim dataset available on Kaggle:

https://www.kaggle.com/datasets/mtalaltariq/paysim-data

Due to licensing and size constraints, the dataset is not included in this repository.

🏗 System Overview

See figures/ for pipeline and experimental design diagrams.

Full technical manuscript available in:

manuscript/fraud_gnn_report.pdf

⚙️ Implementation Notes

Custom adjacency dictionary construction

Manual subgraph sampling (no torch-sparse dependency)

Inductive GraphSAGE architecture

Weighted BCE loss for extreme imbalance
