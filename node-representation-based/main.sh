Datasets=("Celegans" "USAir" "NS" "Yeast" "Power" "arxiv" "PPI_subgraph")
Models=("gcn" "sage")
# Loop through datasets,models
for Y in "${Models[@]}"; do
    for X in "${Datasets[@]}"; do
        python main_results.py "$X" "$Y"
    done
done