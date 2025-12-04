#!/bin/bash

# === SETTINGS ===
NOTEBOOKS=(
    "all_fractals_dataset_B.ipynb"
    "cliford_fractals_dataset_B.ipynb"
    "de_jong_fractals_dataset_B.ipynb"
    "dream_fractals_dataset_B.ipynb"
    "hop_fractals_dataset_B.ipynb"
)

LOGDIR="logs"
OUTFILE="training_status.out"
mkdir -p "$LOGDIR"

echo "🚀 Starting long training..." | tee "$OUTFILE"
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
    echo "⏳ Total elapsed time: ~${ELAPSED} hours" | tee -a "$OUTFILE"
    echo "" | tee -a "$OUTFILE"
done

echo "🎉 ALL NOTEBOOKS COMPLETED SUCCESSFULLY!" | tee -a "$OUTFILE"
