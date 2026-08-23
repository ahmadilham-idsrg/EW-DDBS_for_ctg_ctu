#!/usr/bin/env bash
# Full reproduction from scratch. Expect roughly 6 hours on a single T4 GPU,
# most of it in the CTU-UHB runs. Each seed is a separate invocation so that a
# dropped session costs one run rather than all of them.
set -euo pipefail
cd "$(dirname "$0")/../src"

echo "== UCI cohort, three seeds =="
python run_uci.py --seeds 42   --outdir ../results/results_uci     --folds 5
python run_uci.py --seeds 7    --outdir ../results/results_uci_s2  --folds 5
python run_uci.py --seeds 2024 --outdir ../results/results_uci_s3  --folds 5

echo "== UCI feature-ordering permutation control (seed 42) =="
python run_uci.py --permute-features --seeds 42 \
       --outdir ../results/results_uci_perm --folds 5

echo "== CTU-UHB cohort, three seeds =="
python run_ctu_uhb.py --seeds 42   --outdir ../results/results_ctu    --folds 10
python run_ctu_uhb.py --seeds 7    --outdir ../results/results_ctu_s2 --folds 10
python run_ctu_uhb.py --seeds 2024 --outdir ../results/results_ctu_s3 --folds 10

echo "== CTU-UHB permutation control and determinism audit (seed 42) =="
python run_ctu_uhb.py --permute-features --seeds 42 \
       --outdir ../results/results_ctu_perm --folds 10
python run_ctu_uhb.py --seeds 42 --outdir ../results/results_ctu_run2 --folds 10

echo "== Merge and analyse =="
python merge_runs.py --out ../results/results_uci_all \
       ../results/results_uci ../results/results_uci_s2 ../results/results_uci_s3
python merge_runs.py --out ../results/results_ctu_all \
       ../results/results_ctu ../results/results_ctu_s2 ../results/results_ctu_s3
python analyze_results.py --dir ../results/results_uci_all --ref "EWDDBS (Entropy+Safe+Tomek)"
python analyze_results.py --dir ../results/results_ctu_all --ref "EWDDBS (Entropy+Safe+Tomek)"

echo "== Regenerate every number quoted in the manuscript =="
python ../scripts/verify_paper_numbers.py --root ../results
