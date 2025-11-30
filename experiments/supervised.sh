#!/bin/bash
# filepath: /Users/sakethkoona/Documents/Work/ProjectsnShit/MLBlackjackProject/gambling-ml/experiments/supervised.sh

# Script to run supervised learning experiments with different parameter combinations
# Make sure to run this from the gambling-ml/ directory

echo "Starting supervised learning experiments..."
echo "Timestamp: $(date)"
echo "========================================="

# Configuration
N_SAMPLES=50000
N_DECKS=4
PYTHON_SCRIPT="scripts/supervised.py"

# Check if script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: $PYTHON_SCRIPT not found"
    echo "Please run this script from the gambling-ml/ directory"
    echo "Current directory: $(pwd)"
    exit 1
fi

# Check if we're in the right directory structure
if [ ! -d "scripts" ] || [ ! -d "envs" ]; then
    echo "Error: Expected directory structure not found"
    echo "Please run this script from the gambling-ml/ directory"
    echo "Expected structure:"
    echo "  gambling-ml/"
    echo "    ├── scripts/"
    echo "    ├── envs/"
    echo "    ├── logs/     (will be created if needed)"
    echo "    └── plots/    (will be created if needed)"
    exit 1
fi

# Arrays of parameters to test
MODEL_TYPES=("logistic" "random_forest")
POLICIES=("simple" "stochastic")

# Function to run a single experiment
run_experiment() {
    local model_type=$1
    local policy=$2
    local experiment_name="${model_type}_${policy}"
    
    echo ""
    echo "Running experiment: $experiment_name"
    echo "Model: $model_type, Policy: $policy"
    echo "Samples: $N_SAMPLES, Decks: $N_DECKS"
    echo "-----------------------------------------"
    
    # Run the experiment
    python3 "$PYTHON_SCRIPT" \
        --model_type "$model_type" \
        --policy "$policy" \
        --n_samples "$N_SAMPLES" \
        --n_decks "$N_DECKS"
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo "✅ Experiment $experiment_name completed successfully"
    else
        echo "❌ Experiment $experiment_name failed with exit code $exit_code"
        return $exit_code
    fi
}

# Function to run experiments without plots (faster)
run_experiment_no_plots() {
    local model_type=$1
    local policy=$2
    local experiment_name="${model_type}_${policy}"
    
    echo ""
    echo "Running experiment (no plots): $experiment_name"
    echo "Model: $model_type, Policy: $policy"
    echo "Samples: $N_SAMPLES, Decks: $N_DECKS"
    echo "-----------------------------------------"
    
    # Run the experiment without plots
    python3 "$PYTHON_SCRIPT" \
        --model_type "$model_type" \
        --policy "$policy" \
        --n_samples "$N_SAMPLES" \
        --n_decks "$N_DECKS" \
        --no_plots
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo "✅ Experiment $experiment_name completed successfully"
    else
        echo "❌ Experiment $experiment_name failed with exit code $exit_code"
        return $exit_code
    fi
}

# Parse command line arguments
SKIP_PLOTS=false
QUICK_MODE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --no-plots)
            SKIP_PLOTS=true
            shift
            ;;
        --quick)
            QUICK_MODE=true
            N_SAMPLES=10000
            shift
            ;;
        --samples)
            N_SAMPLES="$2"
            shift 2
            ;;
        --decks)
            N_DECKS="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --no-plots    Skip generating plots (faster execution)"
            echo "  --quick       Quick mode (10k samples instead of 50k)"
            echo "  --samples N   Number of training samples (default: 50000)"
            echo "  --decks N     Number of decks (default: 4)"
            echo "  -h, --help    Show this help message"
            echo ""
            echo "This script will run experiments with all combinations of:"
            echo "  Model types: logistic, random_forest"
            echo "  Policies: simple, stochastic"
            echo ""
            echo "Run from the gambling-ml/ directory"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Display configuration
echo "Configuration:"
echo "  Training samples: $N_SAMPLES"
echo "  Number of decks: $N_DECKS"
echo "  Skip plots: $SKIP_PLOTS"
echo "  Quick mode: $QUICK_MODE"
echo "  Working directory: $(pwd)"
echo ""

# Check and create directories only if they don't exist
if [ ! -d "logs" ]; then
    echo "Creating logs/ directory..."
    mkdir -p logs
else
    echo "Using existing logs/ directory"
fi

if [ ! -d "plots" ]; then
    echo "Creating plots/ directory..."
    mkdir -p plots
else
    echo "Using existing plots/ directory"
fi

echo ""

# Initialize counters
total_experiments=0
successful_experiments=0
failed_experiments=0

# Run all combinations
for model_type in "${MODEL_TYPES[@]}"; do
    for policy in "${POLICIES[@]}"; do
        total_experiments=$((total_experiments + 1))
        
        if [ "$SKIP_PLOTS" = true ]; then
            if run_experiment_no_plots "$model_type" "$policy"; then
                successful_experiments=$((successful_experiments + 1))
            else
                failed_experiments=$((failed_experiments + 1))
            fi
        else
            if run_experiment "$model_type" "$policy"; then
                successful_experiments=$((successful_experiments + 1))
            else
                failed_experiments=$((failed_experiments + 1))
            fi
        fi
        
        # Add a small delay between experiments
        sleep 1
    done
done

# Summary
echo ""
echo "========================================="
echo "Experiment Summary:"
echo "Total experiments: $total_experiments"
echo "Successful: $successful_experiments"
echo "Failed: $failed_experiments"
echo "Completion time: $(date)"

if [ $failed_experiments -gt 0 ]; then
    echo "❌ Some experiments failed. Check the output above for details."
    exit 1
else
    echo "✅ All experiments completed successfully!"
    
    # Show where results are saved
    echo ""
    echo "Results saved to:"
    echo "  JSON logs: $(pwd)/logs/"
    if [ "$SKIP_PLOTS" = false ]; then
        echo "  Plots: $(pwd)/plots/"
    fi
    
    # Show what was created
    echo ""
    echo "Files created:"
    if [ -d "logs" ]; then
        echo "  Logs:"
        ls -1 logs/ | tail -4 | sed 's/^/    /'
    fi
    if [ -d "plots" ] && [ "$SKIP_PLOTS" = false ]; then
        echo "  Plot directories:"
        ls -1 plots/ | tail -4 | sed 's/^/    /'
    fi
fi