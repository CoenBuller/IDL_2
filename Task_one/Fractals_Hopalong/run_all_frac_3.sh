#!/bin/bash

source /vol/home/s4949404/miniconda3/etc/profile.d/conda.sh
conda activate IDL2

# === SETTINGS ===
NOTEBOOKS=(
    "Fractal_set_3.ipynb"
    "fire_subsets_Fractal_set_3.ipynb"
    "vividis_subsets_Fractal_set_3.ipynb"
)

LOGDIR="logs_set3"
OUTFILE="training_status_set3.out"
mkdir -p "$LOGDIR"

echo "🚀 Starting long training for SET 3..." | tee "$OUTFILE"
echo "Logs saved in: $LOGDIR" | tee -a "$OUTFILE"
echo "" | tee -a "$OUTFILE"

START_TIME=$(date +%s)

# === RUN NOTEBOOKS ONE BY ONE ===
for nb in "${NOTEBOOKS[@]}"; do
    TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
    LOGFILE="$LOGDIR/${nb%.ipynb}_$TIMESTAMP.log"
    DONEFILE="$LOGDIR/${nb%.ipynb}_${TIMESTAMP}_FINISHED.txt"

    echo "-----------------------------------------" | tee -a "$OUTFILE"
    echo "▶ Executing notebook: $nb" | tee -a "$OUTFILE"
    echo "📄 Log: $LOGFILE" | tee -a "$OUTFILE"
    echo "-----------------------------------------" | tee -a "$OUTFILE"

    # Execute notebook
    jupyter nbconvert --to notebook --execute "$nb" --inplace \
        --log-level=INFO &> "$LOGFILE"

    # Check exit status
    if [ $? -eq 0 ]; then
        echo "✅ Finished: $nb" | tee -a "$OUTFILE"
        echo "Notebook finished at $(date)" > "$DONEFILE"
    else
        echo "❌ ERROR during execution: $nb" | tee -a "$OUTFILE"
        echo "Check log: $LOGFILE" | tee -a "$OUTFILE"
        exit 1
    fi

    # Progress message with elapsed time
    NOW=$(date +%s)
    ELAPSED=$(( (NOW - START_TIME) / 3600 ))
    echo "⏳ Total elapsed time so far: ~${ELAPSED} hours" | tee -a "$OUTFILE"
    echo "" | tee -a "$OUTFILE"
done

echo "🎉 ALL NOTEBOOKS (SET 3) COMPLETED SUCCESSFULLY!" | tee -a "$OUTFILE"

