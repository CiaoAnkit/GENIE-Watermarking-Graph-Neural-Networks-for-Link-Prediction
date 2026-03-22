Datasets=("Celegans" "USAir" "NS" "Yeast" "Power" "arxiv" "PPI_subgraph")
# Loop through X
for X in "${Datasets[@]}"; do
    python weight_quantization.py "$X"
done
echo "The results are written onto quantization_results.txt file"