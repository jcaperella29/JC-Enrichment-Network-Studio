# JC Enrichment Network Studio

A lightweight Dash app for exploring **gene ↔ enrichment term/pathway** relationships as an interactive **bipartite network**.

Upload a long-format CSV (one row per membership edge), map the item/group columns, build a graph, and then:
- visually explore the network
- filter by degree / weight
- view summary stats + top nodes
- export nodes/edges CSVs for downstream analysis

---

## What this app does

**Input:** long-format enrichment membership table  
**Output:** an interactive network visualization + stats tables + exportable nodes/edges

### Key features
- **Bipartite network view** (Groups/Terms on one side, Genes on the other)
- **Optional weighted edges** (e.g., adjusted p-value transformed to `-log10(padj)`)
- **Filters**
  - search by gene/term name
  - minimum node degree
  - minimum edge weight
  - maximum number of groups shown
  - layout mode (bipartite vs force-directed)
- **Stats tab**
  - node/edge counts, components, largest component
  - top groups and top genes by degree
- **Export**
  - download `nodes.csv` and `edges.csv` after graph build

---

## Repository structure

Network_Vis_App/
├── app.py
├── requirements.txt
├── Dockerfile.txt
├── TEST.csv
└── enrichr_terms_genes_test.csv


- **app.py** — the Dash app
- **requirements.txt** — Python dependencies
- **Dockerfile.txt** — example Docker build file (rename to `Dockerfile` if you want to build directly)
- **TEST.csv / enrichr_terms_genes_test.csv** — sample data you can upload to confirm everything works

---

## Input CSV format

Your **raw upload must be long-format**: **one row per edge**.

Minimum required columns:
- **Item column** (gene) — e.g. `TP53`
- **Group column** (term/pathway) — e.g. `p53 signaling pathway`

Optional:
- **Weight column** (e.g., `adj_p`, `padj`, `FDR`, etc.)

Example:

| gene | term | adj_p |
|------|------|-------|
| TP53 | p53 signaling pathway | 1.2e-06 |
| MDM2 | p53 signaling pathway | 3.9e-04 |
| CDKN1A | Cell cycle arrest | 2.0e-05 |

### Weight handling (important)
If you provide an adjusted p-value column, the app converts it to:

**weight_plot = -log10(adj_p)**

So:
- smaller p-values → **bigger** weights → **thicker/more prominent** edges (when enabled)
- filtering with “minimum edge weight” behaves intuitively

---

## Run locally

### 1) Create an environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows


2) Install dependencies

pip install -r requirements.txt

3) Launch

python app.py

Then open:

http://localhost:8050

Run with Docker (optional)
Your repo includes Dockerfile.txt. If you want to build directly, rename it:
mv Dockerfile.txt Dockerfile
Then build + run:
docker build -t jc-enrichment-network-studio .
docker run -p 8050:8050 jc-enrichment-network-studio
Open:

http://localhost:8050
How to use

Upload a CSV in long format.

Pick:

Item column (genes)

Group column (terms/pathways)

Optional Weight column

Click “Preprocess / Build graph”

Use the sidebar filters to explore:

search

minimum degree

weight threshold

layout mode

Go to Stats tab for rankings

Export nodes/edges via Download buttons

Outputs

After building the graph, you can download:

nodes.csv

Includes:

node id

label

node_type (item or group)

any stored attributes

edges.csv

Includes:

source

target

weight_raw (original)

weight_plot (transformed -log10(padj) when applicable)

Notes / troubleshooting

If your CSV loads but dropdowns are empty:

confirm it’s valid CSV

confirm the file has headers

If weights look “backwards”:

ensure you supplied adjusted p-values (smaller is better)

the app will convert to -log10(p), so higher = stronger

Very large graphs (10k+ edges) can feel heavy in the browser. Use:

max groups

minimum degree

search filtering
