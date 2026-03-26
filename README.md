
## Dev/Creator
**tubakhxn**

## What is this project?
This project simulates and animates a "Monte Carlo universe" of possible stock price paths using Geometric Brownian Motion (GBM). It visually demonstrates the range of possible future outcomes for a stock price, as modeled by stochastic processes in quantitative finance.

## How does it work?
- Simulates 1000+ stock price paths over 252 trading days using the GBM model.
- Colors each path by its final value (blue for low, red for high) to show the distribution of outcomes.
- Animates the evolution of all paths, the mean path, and the probability cone (1 standard deviation) over time.
- Displays a side panel with the histogram and probability density function (PDF) of the final values.
- All dependencies are auto-installed on first run.

## Usage
Run the main script directly in your Python environment:

    python main.py

The script will:
1. Install dependencies (numpy, matplotlib, plotly, scikit-learn, scipy)
2. Simulate stock price paths
3. Animate and display the results

No manual setup required.

## Relevant Wikipedia Links
- [Monte Carlo method](https://en.wikipedia.org/wiki/Monte_Carlo_method)
- [Geometric Brownian motion](https://en.wikipedia.org/wiki/Geometric_Brownian_motion)
- [Stochastic process](https://en.wikipedia.org/wiki/Stochastic_process)
- [Value at risk](https://en.wikipedia.org/wiki/Value_at_risk)
