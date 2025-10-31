# manipulation_sim_streamlit.py
"""
Streamlit app: Educational simulation of manipulative quoting in a low-liquidity option market.

- The "algo" posts wide passive quotes (bid, ask).
- When a human posts an order (e.g., buy @ 21), the algo can step in as buyer (e.g., bid 22),
  accumulate inventory, push price upward and later sell to the human when the market reaches
  a threshold (e.g., 20% above fair price), then revert to passive quotes.
- This demonstrates how a naive buyer can end up with an unfavorable fill and a loss.

USAGE:
    pip install streamlit numpy pandas plotly
    streamlit run manipulation_sim_streamlit.py

DISCLAIMER:
    Educational only. No live trading or exchange connectivity. Do NOT use to manipulate markets.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="Educational Manipulation Simulator", layout="wide")

st.title("🔬 Educational — Low-Liquidity Option Manipulation Simulator")
st.markdown(
    """
This is an **educational simulation** only. It demonstrates how a manipulative quoting
pattern (an algorithm that posts on both sides and later sells to a naive buyer) can
cause a retail trader to buy above fair value and incur a loss.

**DO NOT** use this code to connect to any exchange or live-trade.
"""
)

# -----------------------------
# Sidebar controls (simulation parameters)
# -----------------------------
st.sidebar.header("Simulation Parameters")

fair_price = st.sidebar.number_input("Fair price (reference)", value=40.0, step=1.0, format="%.2f")
passive_bid = st.sidebar.number_input("Algo passive BID (initial)", value=20.0, step=0.5, format="%.2f")
passive_ask_before = st.sidebar.number_input("Algo passive ASK (initial/high)", value=100.0, step=0.5, format="%.2f")
human_order_price = st.sidebar.number_input("Human order price (submitted)", value=21.0, step=0.5, format="%.2f")
human_order_size = st.sidebar.number_input("Human order size (contracts)", min_value=1, value=10, step=1)
algo_step_buy_size = st.sidebar.number_input("Algo accumulation size when stepping in", min_value=1, value=10, step=1)
threshold_pct = st.sidebar.slider("Sell trigger: % above fair (threshold)", 5, 100, 20) / 100.0
simulation_steps = st.sidebar.slider("Simulation time steps", 6, 60, 12)
human_cash_start = st.sidebar.number_input("Human starting cash", value=10000.0, step=100.0, format="%.2f")
random_seed = st.sidebar.number_input("Random seed (for repeatability)", value=42, step=1)

st.sidebar.markdown("---")
st.sidebar.caption("This simulation is simplified for educational/detection purposes.")

# -----------------------------
# Run simulation button
# -----------------------------
run_sim = st.button("▶️ Run Simulation")

# -----------------------------
# Simulation function
# -----------------------------
def run_simulation(
    fair_price,
    passive_bid,
    passive_ask_initial,
    human_order_price,
    human_order_size,
    algo_step_buy_size,
    threshold_pct,
    steps,
    human_cash_start,
    seed=42,
):
    np.random.seed(seed)
    times = list(range(steps))

    # Initialize quotes
    current_bid = passive_bid
    current_ask = passive_ask_initial
    bid_history = []
    ask_history = []
    mid_history = []

    # Execution record
    executions = []

    # Inventories / cash
    human_inventory = 0
    human_cash = human_cash_start
    algo_inventory = 0
    algo_cash = 0.0

    threshold_price = fair_price * (1 + threshold_pct)

    human_placed_order = False
    human_filled_by_algo = False

    # We'll simulate that human places order at step t_place (choose near early)
    t_place = max(1, steps // 4)  # e.g., 1/4th into the simulation

    for t in times:
        bid_history.append(current_bid)
        ask_history.append(current_ask)
        mid_history.append((current_bid + current_ask) / 2.0)

        # Human places order at t_place
        if t == t_place:
            human_placed_order = True
            executions.append({
                "time": t,
                "actor": "human",
                "side": "submit_order",
                "price": human_order_price,
                "size": human_order_size,
                "note": "human posted a buy order (not yet filled)"
            })

            # Algo reacts by stepping up its bid above the human order and buys (accumulates)
            if human_order_price > current_bid:
                step_bid = human_order_price + 1.0  # e.g., 22
                # Algo buys into dark/other liquidity (simulated)
                algo_inventory += algo_step_buy_size
                algo_cash -= algo_step_buy_size * step_bid
                current_bid = step_bid
                executions.append({
                    "time": t,
                    "actor": "algo",
                    "side": "buy",
                    "price": step_bid,
                    "size": algo_step_buy_size,
                    "note": "algo stepped up bid and accumulated"
                })

        # If algo has inventory, simulate upward pressure (price creep)
        if algo_inventory > 0:
            # ramp factor: smaller as we approach threshold
            distance_to_threshold = max(0.01, (threshold_price - current_ask))
            step = max(0.2, 0.05 * distance_to_threshold) + np.random.uniform(0.0, 1.2)
            prev_bid = current_bid
            prev_ask = current_ask
            current_bid = current_bid + step * 0.6
            current_ask = current_ask + step * 1.0

        # Trigger: if ask >= threshold and human still wants to buy and not filled
        if human_placed_order and (not human_filled_by_algo):
            if current_ask >= threshold_price:
                # Algo posts a sell to fill naive human at the threshold price
                sell_price = threshold_price
                sell_size = human_order_size
                total_cost = sell_price * sell_size
                # If human has funds, fill them
                if human_cash >= total_cost:
                    human_inventory += sell_size
                    human_cash -= total_cost
                    executions.append({
                        "time": t,
                        "actor": "human",
                        "side": "buy",
                        "price": sell_price,
                        "size": sell_size,
                        "note": "human filled by algo at elevated price"
                    })
                    # Algo sells from its inventory
                    sold_from_algo = min(algo_inventory, sell_size)
                    algo_inventory -= sold_from_algo
                    algo_cash += sold_from_algo * sell_price
                    executions.append({
                        "time": t,
                        "actor": "algo",
                        "side": "sell",
                        "price": sell_price,
                        "size": sold_from_algo,
                        "note": "algo sold accumulated inventory to human"
                    })

                    human_filled_by_algo = True

                    # After selling, algo reverts to passive wide quotes
                    current_bid = passive_bid
                    current_ask = passive_ask_initial

        # Random occasional passive liquidity appears (tiny)
        if np.random.rand() < 0.08:
            # small change to current_ask (simulate noise liquidity)
            current_ask += np.random.uniform(-2.0, 3.0)
            current_ask = max(current_bid + 0.5, current_ask)

    # Build results DataFrame
    exec_df = pd.DataFrame(executions)

    # Compute human metrics
    human_cost_basis = human_cash_start - human_cash
    human_notional_at_fair = human_inventory * fair_price
    human_unrealized_pnl = human_notional_at_fair - human_cost_basis

    summary = {
        "human_inventory": human_inventory,
        "human_cash_remaining": human_cash,
        "human_cost_basis": human_cost_basis,
        "human_notional_at_fair": human_notional_at_fair,
        "human_unrealized_pnl": human_unrealized_pnl,
        "algo_inventory_remaining": algo_inventory,
        "algo_cash": algo_cash,
        "threshold_price": threshold_price,
    }

    price_history = pd.DataFrame({
        "time": times,
        "bid": bid_history,
        "ask": ask_history,
        "mid": mid_history
    })

    return exec_df, price_history, summary

# -----------------------------
# Run and display results
# -----------------------------
if run_sim:
    with st.spinner("Running simulation..."):
        exec_df, price_history, summary = run_simulation(
            fair_price=fair_price,
            passive_bid=passive_bid,
            passive_ask_initial=passive_ask_before,
            human_order_price=human_order_price,
            human_order_size=int(human_order_size),
            algo_step_buy_size=int(algo_step_buy_size),
            threshold_pct=threshold_pct,
            steps=int(simulation_steps),
            human_cash_start=float(human_cash_start),
            seed=int(random_seed),
        )

    st.success("Simulation finished")

    # Left: price chart, Right: executions and summary
    col1, col2 = st.columns([2, 1])

    # Price chart
    with col1:
        st.subheader("Price History (bid / ask / mid)")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=price_history["time"], y=price_history["bid"], mode="lines+markers", name="Bid"))
        fig.add_trace(go.Scatter(x=price_history["time"], y=price_history["ask"], mode="lines+markers", name="Ask"))
        fig.add_trace(go.Scatter(x=price_history["time"], y=price_history["mid"], mode="lines", name="Mid"))
        fig.add_hline(y=fair_price, line_dash="dot", annotation_text="Fair price", annotation_position="bottom left")
        fig.add_hline(y=summary["threshold_price"], line_dash="dash", line_color="red", annotation_text="Threshold", annotation_position="top left")
        fig.update_layout(height=420, xaxis_title="Time step", yaxis_title="Price")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Price History Table")
        st.dataframe(price_history.style.format({"bid": "{:.2f}", "ask": "{:.2f}", "mid": "{:.2f}"}), height=200)

    # Executions & summary
    with col2:
        st.subheader("Executions")
        if exec_df.empty:
            st.info("No executions occurred. Try changing parameters (e.g., make algo_step_buy_size > 0 or reduce passive_ask).")
        else:
            # show chronological table
            st.dataframe(exec_df.sort_values("time").reset_index(drop=True).style.format({"price": "{:.2f}"}), height=320)

            # allow CSV download of executions
            csv = exec_df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download executions CSV", csv, file_name="executions.csv", mime="text/csv")

        st.markdown("---")
        st.subheader("Human Trader Summary (mark-to-fair-price)")
        st.write(f"Fair price (reference): **{fair_price:.2f}**")
        st.write(f"Threshold (sell trigger): **{summary['threshold_price']:.2f}**")
        st.metric("Human inventory", f"{summary['human_inventory']}")
        st.metric("Human cash remaining", f"{summary['human_cash_remaining']:.2f}")
        st.metric("Human cost basis (spent)", f"{summary['human_cost_basis']:.2f}")
        st.metric("Human notional @ fair", f"{summary['human_notional_at_fair']:.2f}")
        pnl_str = f"{summary['human_unrealized_pnl']:.2f}"
        if summary['human_unrealized_pnl'] < 0:
            st.error(f"Unrealized PnL (MTM @ fair): {pnl_str} (loss)")
        else:
            st.success(f"Unrealized PnL (MTM @ fair): {pnl_str}")

    st.markdown("---")
    st.caption("This simulation is educational only. It simulates how a manipulative pattern can cause a naive buyer to buy at an elevated price in a low-liquidity market.")
else:
    st.info("Set parameters on the left and click ▶️ Run Simulation to begin.")
