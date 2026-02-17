from mlstudy.trading.backtest.mean_reversion.sweep.sweep_results_reader import SweepResultsReader

if __name__ == "__main__":
    run_dir = r"C:\Users\yihu0\code\david\mlstudy\data\mr_backtest\out"
    results = SweepResultsReader.load_sweep_run(run_dir)
    print(results)