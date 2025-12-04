#!/bin/bash

source /vol/home/s4949404/miniconda3/etc/profile.d/conda.sh
conda activate IDL2

# === SETTINGS ===
NOTEBOOKS=(
    "prova1.ipynb"
    "prova2.ipynb"
)

LOGDIR="logs_prova"
OUTFILE="training_status_prova.out"
mkdir -p "$LOGDIR"

echo "🚀 Starting PROVA notebook execution..." | tee "$OUTFILE"
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

    # Check status
    if [ $? -eq 0 ]; then
        echo "✅ Finished: $nb" | tee -a "$OUTFILE"
        echo "Notebook finished at $(date)" > "$DONEFILE"
    else
        echo "❌ ERROR during execution: $nb" | tee -a "$OUTFILE"
        echo "Check log: $LOGFILE" | tee -a "$OUTFILE"
        exit 1
    fi

    # Time feedback
    NOW=$(date +%s)
    ELAPSED=$(( (NOW - START_TIME) / 3600 ))
    echo "⏳ Elapsed: ~${ELAPSED} hours" | tee -a "$OUTFILE"
    echo "" | tee -a "$OUTFILE"
done

echo "🎉 PROVA COMPLETED SUCCESSFULLY!" | tee -a "$OUTFILE"
