Datasets=("Celegans" "USAir" "NS" "Yeast" "Power")
# Loop through X
for X in "${Datasets[@]}"; do
    python main.py "$X"
done