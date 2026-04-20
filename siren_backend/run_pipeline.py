"""
SIREN Pipeline Runner — Executes all stages in order.

Execution order:
    1. data_generator    → Generate synthetic 100K orders + 1K merchants
    2. layer1_denoising  → FOR signal de-noising
    3. layer2_enrichment → External signal enrichment
    4. layer3_bias       → Per-merchant bias correction
    5. model_training    → Train Baseline XGB, SIREN XGB, SIREN LGBM
    6. visualizations    → Generate all 6 charts
    7. accuracy_report   → Generate and save accuracy report
    8. FastAPI server     → Start the backend (optional, via --serve flag)

Usage:
    python run_pipeline.py           # Run pipeline without starting server
    python run_pipeline.py --serve   # Run pipeline then start server
"""

import sys
import os
import time
import argparse

# Ensure the script's directory is in the path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)


def run_step(step_num: int, total: int, name: str, func):
    """
    Execute a single pipeline step with timing and error handling.
    
    Args:
        step_num: Current step number.
        total: Total number of steps.
        name: Human-readable step name.
        func: Callable to execute.
    """
    print(f"\n{'━' * 60}")
    print(f"  STEP {step_num}/{total}: {name}")
    print(f"{'━' * 60}")
    
    start = time.time()
    try:
        result = func()
        elapsed = time.time() - start
        print(f"\n  ✅ {name} completed in {elapsed:.1f}s")
        return result
    except Exception as e:
        elapsed = time.time() - start
        print(f"\n  ❌ {name} FAILED after {elapsed:.1f}s: {e}")
        raise


def main():
    """Run the complete SIREN pipeline end-to-end."""
    parser = argparse.ArgumentParser(description="SIREN Pipeline Runner")
    parser.add_argument("--serve", action="store_true", help="Start FastAPI server after pipeline")
    parser.add_argument("--skip-data", action="store_true", help="Skip data generation if data exists")
    args = parser.parse_args()
    
    total_start = time.time()
    
    print("=" * 63)
    print("   SIREN PIPELINE -- FULL EXECUTION")
    print("   Signal-Informed Restaurant ETA Network")
    print("=" * 63)

    
    # Check if we can skip data generation
    data_exists = (
        os.path.exists(os.path.join(BASE_DIR, "data", "orders_100k.csv"))
        and os.path.exists(os.path.join(BASE_DIR, "data", "merchants_1k.csv"))
    )
    
    total_steps = 7
    
    # Step 1: Data Generation
    if args.skip_data and data_exists:
        print(f"\n  ⏭  Skipping data generation (--skip-data, data exists)")
    else:
        from data_generator import main as gen_main
        run_step(1, total_steps, "Data Generation", gen_main)
    
    # Step 2: Layer 1 — De-noising
    from layer1_denoising import main as l1_main
    run_step(2, total_steps, "Layer 1 — FOR Signal De-noising", l1_main)
    
    # Step 3: Layer 2 — Enrichment
    from layer2_enrichment import main as l2_main
    run_step(3, total_steps, "Layer 2 — External Signal Enrichment", l2_main)
    
    # Step 4: Layer 3 — Bias Correction
    from layer3_bias_correction import main as l3_main
    run_step(4, total_steps, "Layer 3 — Per-Merchant Bias Correction", l3_main)
    
    # Step 5: Model Training
    from model_training import main as mt_main
    run_step(5, total_steps, "Model Training (3 models)", mt_main)
    
    # Step 6: Visualizations
    from visualizations import main as viz_main
    run_step(6, total_steps, "Visualizations (6 charts)", viz_main)
    
    # Step 7: Accuracy Report
    from accuracy_report import main as ar_main
    run_step(7, total_steps, "Accuracy Report", ar_main)
    
    # Summary
    total_elapsed = time.time() - total_start
    print(f"\n{'=' * 60}")
    print(f"  🎉 SIREN PIPELINE COMPLETE — Total time: {total_elapsed:.1f}s")
    print(f"{'=' * 60}")
    
    # List outputs
    print("\n  Generated files:")
    for subdir in ["data", "models", "outputs"]:
        dirpath = os.path.join(BASE_DIR, subdir)
        if os.path.exists(dirpath):
            for f in sorted(os.listdir(dirpath)):
                fpath = os.path.join(dirpath, f)
                size = os.path.getsize(fpath)
                if size > 1e6:
                    size_str = f"{size/1e6:.1f} MB"
                elif size > 1e3:
                    size_str = f"{size/1e3:.1f} KB"
                else:
                    size_str = f"{size} B"
                print(f"    {subdir}/{f:.<40s} {size_str}")
    
    # Start server if requested
    if args.serve:
        print(f"\n{'━' * 60}")
        print("  Starting FastAPI server on http://0.0.0.0:8000")
        print(f"{'━' * 60}")
        import uvicorn
        uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
