JC Enrichment Network Studio

An interactive tool for transforming enrichment analysis results into network-based biological insight.

👉 Live App:
https://jc-enrichment-network-studio232-390287836436.europe-west1.run.app/
🚀 Features
🔹 Bipartite Enrichment Network
Gene ↔ pathway relationships
Weighted by enrichment significance
Interactive Plotly visualization
🔹 Diffusion-Based Prioritization (Main Graph / Bipartite)
Random-walk / PageRank-style scoring
Three ranking modes:
Evidence-weighted
Connectivity-weighted
Balanced
Separate pathway and gene rankings
🔹 Top Candidate Identification
Combines:
statistical significance
network centrality
connectivity
Produces actionable follow-up targets
🔹 Pathway Projection Network
Pathway ↔ pathway relationships
Based on shared genes
Reveals biological modules and themes
🔹 Projection Diffusion
Identifies central biological processes
Highlights coherent pathway clusters
🔹 Consensus Candidates
Integrates bipartite + projection signals
Produces final prioritized pathway list
🔹 Export Options
PNG (via Plotly UI)
SVG / PDF (publication-ready)
CSV tables for all analyses
🧪 Input Format

Long-format CSV:

gene	pathway
GeneA	Pathway1
GeneB	Pathway1
GeneB	Pathway2

Optional columns:

adjusted p-value
score / weight
⚙️ Local Setup
in bash

git clone <repo>
cd Network_Vis_App
pip install -r requirements.txt
python app.py

Then open:
http://localhost:8050

With Docker:
in bash
docker build -t enrichment-network .
docker run -p 8050:8050 enrichment-network

🌐 Deployment

Deployed on Google Cloud Run for scalable, containerized execution.

.

🧠 Why this tool?

Standard enrichment analysis answers:

“What is statistically enriched?”

This tool answers:

“What is biologically important, connected, and worth following up?”

📌 Use Cases
RNA-seq / scRNA-seq pathway interpretation
GWAS enrichment exploration
Functional genomics analysis
Hypothesis generation


🧱 Tech Stack
Python
Dash
Plotly
NetworkX
SciPy
