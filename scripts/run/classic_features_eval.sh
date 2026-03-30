INPUT_FOLDER="feature_extraction"

python -m src.scdino.classic.run_knn "$INPUT_FOLDER/features.csv" --train-fraction 0.8 --seed 42