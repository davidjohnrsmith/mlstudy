from mlstudy.trading.backtest.mean_reversion import load_sweep_run

if __name__ == "__main__":
    run_dir = r"C:\Users\yihu0\code\david\mlstudy\data\mr_backtest\out"
    results = load_sweep_run(run_dir)
    print(results)