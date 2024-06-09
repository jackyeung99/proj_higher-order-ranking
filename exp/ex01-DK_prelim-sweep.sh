#! /usr/bin/env bash
#
# Script to run FR's C code for multi-reactive rankings in a variety of synthetic settings.
#
# @author: Daniel Kaiser
# @created: 2024-05-30
# ----------------------------------------------------------------------
# Source progress bar; thanks to https://github.com/roddhjav/progressbar
source "../bin/progressbar.sh" || exit 1

# GLOBALS
## Pathing
SIMULATION="../bin/bt_model.out"  # primary executable
PATH_OUT="../bin/"  # location of raw output files
PATH_RES="../res/"  # location of processed output file(s)

## Parameters
N=1000
K_MAX=10
GAME_PLAYER_RATIOS=(1 10 20 30 40 50 60 70 80 90 100)
NUM_REPS=10

## Hyperparameters
VERBOSE=false
# ~~~~~~~~~~
# FUNCTIONS
function prepare_header() {
    if [ ! -f "$PATH_OUT/_headers.txt" ]; then
        echo "player,true,HO,HOL,bin" > "$PATH_OUT/_headers.txt"
    fi
    return $?  # propogate errors
}

function clean_files() {
    rm *TMP*
    rm scores_*.txt
}

function simulation() {
    # Args (ordered): K1, K2, RATIO, MODEL, REP
    # ---
    # Process args
    M=$(echo "$N*$3" | bc)

    # Run simulation
    if $VERBOSE; then
        "$SIMULATION" "$N" "$M" "$1" "$2" "$3" "$4"
    else
        "$SIMULATION" "$N" "$M" "$1" "$2" "$3" "$4" > /dev/null
    fi

    # Combine outputs
    cat "scores_gt.txt" | tr " " "," > "TMP"  # combine player ids and gt with , delimiter
    paste -d, "TMP" <(awk '{print $2}' "scores_HO.txt") <(awk '{print $2}' "scores_HOL.txt") <(awk '{print $2}' "scores_bin.txt") > "TMP2"  # build body of results
    cat "$PATH_OUT/_headers.txt" "TMP2" > "$PATH_RES/scores_N-${N}_M-${M}_K1-${1}_K2-${2}_model-${4}_rep-${5}.txt"  # add header, save to results directory

    return 0
}
# ~~~~~~~~~
# MAIN
function main() {
    # Prepare one-time output processing
    prepare_header

    # Factorial-design sweep over parameters
    for REP in $(seq $NUM_REPS); do
        for RATIO in "${GAME_PLAYER_RATIOS[@]}"; do
            echo $RATIO
            for K1 in $(seq 2 $K_MAX); do
                for K2 in $(seq $K1 $K_MAX); do
                    for MODEL in 0 1; do
                        # Run simulation and safely save output
                        simulation $K1 $K2 $RATIO $MODEL $REP
                    done
                done
            done
        done
        # progressbar "ex01" $REP $NUM_REPS
    done

    clean_files

    # Exit code
    return 0
}

main
