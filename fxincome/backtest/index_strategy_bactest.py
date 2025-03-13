import backtrader as bt
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from fxincome import const, logger
from fxincome.backtest.index_strategy import IndexEnhancedStrategy, BondData
from fxincome.backtest.index_extreme import IndexExtremeStrategy


def plot_result(year, strategy_name="enhanced"):
    """
    Plot the result of a backtest for a specific year and strategy.

    Args:
        year (int): Year to plot
        strategy_name (str): Strategy name ('enhanced' or 'extreme')
    """
    result = pd.read_csv(
        const.INDEX_ENHANCEMENT.RESULT_PATH + f"{year}_{strategy_name}_result.csv",
        parse_dates=["DATE"],
    )
    fig = plt.figure(num=1, figsize=(15, 5))
    ax = fig.add_subplot(111, label="1")
    # 画出yield和base yield
    lns1 = ax.plot(result["DATE"], result["Yield"], color="r", label="Yield")
    lns2 = ax.plot(result["DATE"], result["BaseYield"], color="b", label="BaseYield")
    lns = lns1 + lns2
    ax.set_xlabel("Date")
    ax.set_ylabel("Yield")
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc=0, fontsize=10)
    plt.title(f"{year} {strategy_name.replace('_', ' ').title()} Strategy Yield")
    plt.tight_layout()
    plt.savefig(
        const.INDEX_ENHANCEMENT.RESULT_PATH + f"{year}_{strategy_name}_yield_plot.png"
    )
    plt.close()


def main():
    """
    Main function to run the backtest for multiple years and strategies.
    Configures parameters, runs backtests, records results, and generates visualizations.
    """
    # use a dataframe to record the return of each year
    year_start = 2017
    year_end = 2023
    result_df = pd.DataFrame(
        index=range(year_start, year_end + 1),
        columns=[
            "Yield",
            "Base Yield",
            "Excess Return",
            "Max Loss",
            "Base Max Loss",
            "Trade days",
        ],
    )
    low_percentile = 25
    high_percentile = 75
    extreme_low_percentile = (
        10  # Add extreme low percentile for IndexExtremeStrategy
    )
    extreme_high_percentile = (
        90  # Add extreme high percentile for IndexExtremeStrategy
    )

    # Choose strategy: 'enhanced' or 'extreme'
    strategy_name = "extreme"  # Change this to switch strategies

    # Expert mode settings for IndexExtremeStrategy
    expert_mode = True  # Set to True to use expert mode
    expert_signal = 0  # 0 for rates down, 1 for rates up

    # Visualization settings
    plot_results = True  # Whether to generate plots for each year

    for year in range(year_start, year_end + 1):
        all_bond = pd.read_excel(
            const.INDEX_ENHANCEMENT.CDB_INFO_PATH,
            parse_dates=["maturitydate", "carrydate", "issuedate"],
        )
        selected_bond = all_bond[
            (all_bond["maturitydate"] >= datetime.datetime(year, 12, 31))
        ]
        selected_bond = selected_bond[
            (selected_bond["maturitydate"] - selected_bond["carrydate"]).dt.days > 380
        ]
        # 对selected_bond中的每只券，分别从D:\data中读取数据，进行回测
        cerebro = bt.Cerebro()
        code_list = []
        base_code = ""
        for i, j in selected_bond.iterrows():
            code = j["windcode"][:-3]
            # 如果code最后两位为QF，则跳过
            if code[-2:] == "QF":
                continue
            # 如果code开头前两位为year的后两位且base_code为空，则将code赋值给base_code
            if code[:2] == str(year - 1)[-2:] and base_code == "":
                base_code = j["windcode"][:-3]
            code_list.append(code)
            input_file = const.INDEX_ENHANCEMENT.CDB_PATH + code + ".csv"
            price_df = pd.read_csv(input_file, parse_dates=["DATE"])
            price_df = price_df[price_df["DATE"] <= datetime.datetime(year, 12, 31)]
            price_df = price_df[price_df["DATE"] >= datetime.datetime(year, 1, 1)]
            data1 = BondData(dataname=price_df, nocase=True)
            cerebro.adddata(data1, name=code)
        # 读取base_code的数据
        input_file = const.INDEX_ENHANCEMENT.CDB_PATH + base_code + ".csv"
        price_df = pd.read_csv(input_file, parse_dates=["DATE"])
        price_df = price_df[price_df["DATE"] <= datetime.datetime(year, 12, 31)]
        price_df = price_df[price_df["DATE"] >= datetime.datetime(year, 1, 1)]
        each_result_df = pd.DataFrame(
            index=price_df["DATE"],
            columns=code_list,
        )

        # Set the selected strategy
        if strategy_name == "enhanced":
            # Use the original IndexEnhancedStrategy
            cerebro.addstrategy(
                IndexEnhancedStrategy,
                year=year,
                base_code=base_code,
                code_list=code_list,
                each_result_df=each_result_df,
                low_percentile=low_percentile,
                high_percentile=high_percentile,
            )
            strategy_class = IndexEnhancedStrategy
        else:
            # Use the new IndexExtremeStrategy
            cerebro.addstrategy(
                IndexExtremeStrategy,
                year=year,
                base_code=base_code,
                code_list=code_list,
                each_result_df=each_result_df,
                low_percentile=low_percentile,
                high_percentile=high_percentile,
                extreme_low_percentile=extreme_low_percentile,
                extreme_high_percentile=extreme_high_percentile,
                expert_mode=expert_mode,
                expert_signal=expert_signal,
            )
            strategy_class = IndexExtremeStrategy

        cerebro.broker.set_cash(strategy_class.INIT_CASH)
        # cerebro.broker.set_slippage_perc(perc=0.0001)
        logger.info(f"Running {year} with {strategy_name} strategy")
        strategies = cerebro.run()
        logger.info(
            f"PROFIT: {(cerebro.broker.get_value() - strategy_class.INIT_CASH) / 10000:.2f}"
        )
        result_df.loc[year, "Yield"] = (
            (cerebro.broker.get_value() - strategy_class.INIT_CASH)
            / strategies[0].CASH_AVAILABLE
            * 100
        )
        # 删除求和后为0的列
        strategies[0].result.to_csv(
            const.INDEX_ENHANCEMENT.RESULT_PATH + f"{year}_{strategy_name}_result.csv",
            index=False,
        )
        result_df.loc[year, "Base Yield"] = strategies[0].last_yield_base * 100
        result_df.loc[year, "Excess Return"] = (
            result_df.loc[year, "Yield"] - result_df.loc[year, "Base Yield"]
        )
        result_df.loc[year, "Max Loss"] = strategies[0].result["Yield"].min()
        result_df.loc[year, "Base Max Loss"] = strategies[0].result["BaseYield"].min()
        result_df.loc[year, "Trade days"] = strategies[0].numbers_tradays

        # Plot results if enabled
        if plot_results:
            plot_result(year, strategy_name)

    # Calculate the average return of all years
    result_df.loc["Average"] = result_df.mean()
    result_df.to_csv(
        const.INDEX_ENHANCEMENT.RESULT_PATH + f"{strategy_name}_all_result.csv",
        index=True,
    )

    # Print summary of results
    print("\nSummary of Results:")
    print(f"Strategy: {strategy_name.replace('_', ' ').title()}")
    if strategy_name == "extreme" and expert_mode:
        print(f"Expert Mode: {'Rates Down' if expert_signal == 0 else 'Rates Up'}")
    print("\nYearly Performance:")
    for year in range(year_start, year_end + 1):
        print(
            f"{year}: Yield={result_df.loc[year, 'Yield']:.2f}%, Base={result_df.loc[year, 'Base Yield']:.2f}%, Excess={result_df.loc[year, 'Excess Return']:.2f}%"
        )
    print(
        f"\nAverage: Yield={result_df.loc['Average', 'Yield']:.2f}%, Base={result_df.loc['Average', 'Base Yield']:.2f}%, Excess={result_df.loc['Average', 'Excess Return']:.2f}%"
    )


if __name__ == "__main__":
    main()