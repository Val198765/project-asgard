# Project Asgard 🏛️

Project Asgard is an open-source framework designed to help students and enthusiasts backtest equity index strategies while avoiding common industry pitfalls. It is developed as part of a project by an upcoming new mandate of the **John Molson Trading League (JMTL)**. At its heart is **Gungnir**, a precision backtesting engine built to ensure that quantitative strategies are grounded in reality.

## ⚠️ Disclaimer & Legal Notice
**THIS SOFTWARE IS FOR EDUCATIONAL PURPOSES ONLY.**

- **NOT INVESTMENT ADVICE**: Project Asgard and the Gungnir engine are tools for learning and experimentation. Nothing in this project constitutes financial, investment, or trading advice.
- **NO LIABILITY**: The author is not responsible for any financial losses, damages, or legal issues resulting from the use or misuse of this software.
- **TRADING RISK**: Do not use this software for actual trading or managing real capital. Backtested results are historical and do not guarantee future performance.
- **Jurisdiction**: This project is provided "as is" without warranty of any kind. Any disputes shall be governed by the laws of the Province of Quebec and the laws of Canada applicable therein, with the courts of Montreal, Quebec, having exclusive jurisdiction, without prejudice to international law.

## 🎯 Purpose
Backtesting is often plagued by "silent killers" that lead to unrealistic results. Project Asgard is built to explicitly address:
- **Look-ahead Bias**: Ensuring data is only used if it was available at the time of the trade.
- **Data Leaks**: Maintaining strict separation between selection data and testing data.
- **Trading Fees**: Incorporating the cost of execution to ensure strategies are viable in the real world.

## 🛠 Current Capabilities
- **Custom Index Construction**: Tools to define portfolios and calculate weights.
- **Equity Backtesting (Gungnir Engine)**: High-performance backtesting against historical price data.
- **Performance Metrics**: Generation of detailed reports including skewness, kurtosis, and growth prints.

### Data Sources
Currently, the project utilizes:
- **Theta Data**: For high-quality equities and options data.
- **Massive**: For equity data processing.

## 🚀 Roadmap
We are actively looking to expand the framework to include:
- **Full Share Purchase**: Moving from fractional to realistic share-lot backtesting.
- **Synthetic Replication**: Using options and futures to backtest the cost and efficiency of replicating index returns synthetically.

## 🤝 Call for Collaborators
This project was initiated by **Valentino Magniette-Bosseboeuf** as part of a student-led effort. While the core logic is functional, it is still in a rough, early-stage version. I am looking for passionate students and quant enthusiasts to:

Feel free to reach out via [LinkedIn](https://www.linkedin.com/in/valentino-mb/) for collaboration or questions.
- **Audit the Logic**: Help find bugs or mathematical errors in the weighting and backtesting engines.
- **Improve Efficiency**: Optimize the data pipeline.
- **Expand Strategies**: Implement new index selection criteria.

If you are interested in quantitative finance and want to help build a tool that helps other students avoid common mistakes, please open an issue or submit a pull request!

## 📜 License
Project Asgard is open-source. To ensure that the project and its improvements remain accessible to all students and enthusiasts, this project utilizes a reciprocal (copyleft) licensing model. 

**All contributions to this project must be made under the same open-source license**, ensuring that any enhancements or derivatives also remain open for the community. This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## 📦 Installation & Requirements
Ensure you have Python 3.10+ installed.

```bash
pip install pandas numpy scipy matplotlib plotly weasyprint thetadata
```

## 📁 Project Structure
- `Master_Backtest.py`: The primary execution engine (**Gungnir**).
- `indices_scripts/`: Logic for defining specific index portfolios.
- `util_scripts/`: Utility functions for weight computation, price loading, and indexing.
- `indices_data/`: (User Provided) Storage for stock prices and weights.
- `benchmark_data/`: (User Provided) Benchmark CSVs.
- `backtest_results/`: Output folder for generated reports and analysis.
