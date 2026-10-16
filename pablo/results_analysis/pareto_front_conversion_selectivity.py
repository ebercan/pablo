"""Plot conversion/selectivity datapoints with explicit color/shape legends.

Uses all datapoints with recorded results from the Experimental Design sheet and
draws an interactive Plotly scatter where:
- x-axis is Conversion
- y-axis is Selectivity
- color encodes Catalyst
- marker symbol encodes Solvent
- marker size encodes Temperature
- marker border is black and thicker when Base is used (Base != "no")

The figure is written to ``pareto_front_conversion_selectivity.html`` next to this
script.

Usage:
    python -m pablo.results_analysis.pareto_front_conversion_selectivity
"""

from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


EXCEL_FILE = Path(__file__).resolve().parents[1] / "nhc_doe.xlsx"
OUTPUT_HTML = Path(__file__).with_name("pareto_front_conversion_selectivity.html")

SYMBOL_SEQUENCE = [
    "square",
    "circle",
    "diamond",
    "cross",
    "x",
    "triangle-up",
    "triangle-down",
    "triangle-left",
    "triangle-right",
    "pentagon",
    "hexagon",
    "star",
    "hourglass",
    "bowtie",
]


def base_used(value: object) -> bool:
    """Return whether a base was used for a row.

    Args:
        value: Cell value from the Base column.

    Returns:
        True when base is used and False when the value is missing or equals
        "no" (case-insensitive).
    """
    if pd.isna(value):
        return False
    if isinstance(value, str) and value.strip().lower() == "no":
        return False
    return True


def build_symbol_map(values: Iterable[str]) -> Dict[str, str]:
    """Build a deterministic solvent-to-symbol mapping.

    Args:
        values: Solvent labels present in result rows.

    Returns:
        A dict mapping each unique solvent label to a Plotly marker symbol.
    """
    unique_values = sorted(set(values))
    symbol_map: Dict[str, str] = {}
    for i, value in enumerate(unique_values):
        symbol_map[value] = SYMBOL_SEQUENCE[i % len(SYMBOL_SEQUENCE)]
    return symbol_map


def build_color_map(values: Iterable[str]) -> Dict[str, str]:
    """Build a deterministic catalyst-to-color mapping.

    Args:
        values: Catalyst labels present in result rows.

    Returns:
        A dict mapping each unique catalyst label to a color string.
    """
    unique_values = sorted(set(values))
    palette = (
        px.colors.qualitative.Plotly
        + px.colors.qualitative.D3
        + px.colors.qualitative.G10
        + px.colors.qualitative.T10
        + px.colors.qualitative.Alphabet
    )
    color_map: Dict[str, str] = {}
    for i, value in enumerate(unique_values):
        color_map[value] = palette[i % len(palette)]
    return color_map


def scale_marker_sizes(temperature: pd.Series) -> List[float]:
    """Scale temperature values to marker sizes.

    Args:
        temperature: Numeric temperature column.

    Returns:
        A list of marker sizes in pixels.
    """
    min_size = 8.0
    max_size = 22.0
    t_min = float(temperature.min())
    t_max = float(temperature.max())
    if t_max == t_min:
        return [0.5 * (min_size + max_size)] * len(temperature)
    scaled = min_size + (temperature - t_min) * (max_size - min_size) / (t_max - t_min)
    return scaled.tolist()


def compute_pareto_front(df: pd.DataFrame) -> pd.DataFrame:
    """Compute nondominated points for maximizing conversion and selectivity.

    Args:
        df: Dataframe containing Conversion and Selectivity columns.

    Returns:
        Dataframe with nondominated points sorted by Conversion.
    """
    pairs = df[["Conversion", "Selectivity"]].reset_index(drop=True)
    keep = [True] * len(pairs)
    for i in range(len(pairs)):
        conv_i = float(pairs.at[i, "Conversion"])
        sel_i = float(pairs.at[i, "Selectivity"])
        for j in range(len(pairs)):
            if i == j:
                continue
            conv_j = float(pairs.at[j, "Conversion"])
            sel_j = float(pairs.at[j, "Selectivity"])
            dominates = (conv_j >= conv_i and sel_j >= sel_i) and (
                conv_j > conv_i or sel_j > sel_i
            )
            if dominates:
                keep[i] = False
                break

    front = pairs.loc[keep].drop_duplicates().sort_values(["Conversion", "Selectivity"])
    return front.reset_index(drop=True)


def add_separate_legends(
    fig: go.Figure,
    catalyst_color_map: Dict[str, str],
    solvent_symbol_map: Dict[str, str],
) -> None:
    """Add separate legend groups for catalyst colors and solvent shapes.

    Args:
        fig: Figure object to mutate.
        catalyst_color_map: Mapping from catalyst label to color.
        solvent_symbol_map: Mapping from solvent label to symbol.
    """
    catalyst_labels = sorted(catalyst_color_map)
    for i, label in enumerate(catalyst_labels):
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                name=str(label),
                legendgroup="catalyst",
                legendgrouptitle_text="Catalyst (color)" if i == 0 else None,
                marker={
                    "symbol": "square",
                    "size": 10,
                    "color": catalyst_color_map[label],
                    "line": {"color": "black", "width": 0.0},
                },
                hoverinfo="skip",
                showlegend=True,
            )
        )

    solvent_labels = sorted(solvent_symbol_map)
    for i, label in enumerate(solvent_labels):
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                name=str(label),
                legendgroup="solvent",
                legendgrouptitle_text="Solvent (shape)" if i == 0 else None,
                marker={
                    "symbol": solvent_symbol_map[label],
                    "size": 10,
                    "color": "rgba(120,120,120,0.9)",
                    "line": {"color": "black", "width": 0.0},
                },
                hoverinfo="skip",
                showlegend=True,
            )
        )


def main() -> None:
    """Create and save the interactive conversion/selectivity scatter plot."""
    df = pd.read_excel(EXCEL_FILE, sheet_name="Experimental Design")

    # Ignore trailing design rows that have no measured results.
    df = df.dropna(subset=["Yield", "Conversion", "Selectivity"], how="all")
    df = df.dropna(subset=["Conversion", "Selectivity"]).copy()

    df["Temperature"] = pd.to_numeric(df["Temperature"], errors="coerce")
    df = df.dropna(subset=["Temperature"]).copy()

    df["Base_Used"] = df["Base"].apply(base_used)
    df["Catalyst_Label"] = df["Catalyst"].astype(str)
    df["Solvent_Label"] = df["Solvent"].astype(str)
    df["Base_Label"] = df["Base"].astype(str)

    symbol_map = build_symbol_map(df["Solvent_Label"].tolist())
    color_map = build_color_map(df["Catalyst_Label"].tolist())

    marker_sizes = scale_marker_sizes(df["Temperature"])
    marker_symbols = [symbol_map[label] for label in df["Solvent_Label"]]
    marker_colors = [color_map[label] for label in df["Catalyst_Label"]]
    marker_border_width = [2.0 if used else 0.0 for used in df["Base_Used"]]

    custom_columns = [
        "Base_Used",
        "Batch",
        "Catalyst",
        "Solvent",
        "Base_Label",
        "Temperature",
        "Time",
        "Formaldehyde Ratio in Eq.",
        "Catalyst Loading in Molar Ratio",
        "Yield",
        "Conversion",
        "Selectivity",
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["Conversion"],
            y=df["Selectivity"],
            mode="markers",
            showlegend=False,
            customdata=df[custom_columns].to_numpy(),
            marker={
                "size": marker_sizes,
                "symbol": marker_symbols,
                "color": marker_colors,
                "opacity": 0.9,
                "line": {
                    "color": "black",
                    "width": marker_border_width,
                },
            },
            hovertemplate=(
                "Batch: %{customdata[1]}<br>"
                "Catalyst: %{customdata[2]}<br>"
                "Solvent: %{customdata[3]}<br>"
                "Base: %{customdata[4]}<br>"
                "Base used: %{customdata[0]}<br>"
                "Temperature: %{customdata[5]}<br>"
                "Time: %{customdata[6]}<br>"
                "Formaldehyde Ratio in Eq.: %{customdata[7]}<br>"
                "Catalyst Loading in Molar Ratio: %{customdata[8]}<br>"
                "Yield: %{customdata[9]}<br>"
                "Conversion: %{customdata[10]}<br>"
                "Selectivity: %{customdata[11]}<extra></extra>"
            ),
        )
    )

    pareto_front = compute_pareto_front(df)
    fig.add_trace(
        go.Scatter(
            x=pareto_front["Conversion"],
            y=pareto_front["Selectivity"],
            mode="lines",
            showlegend=False,
            hoverinfo="skip",
            line={
                "color": "rgba(150,150,150,0.28)",
                "width": 1.0,
                "dash": "dash",
            },
        )
    )

    add_separate_legends(fig, color_map, symbol_map)

    fig.update_layout(
        title="Conversion vs Selectivity (All Datapoints)",
        xaxis_title="Conversion",
        yaxis_title="Selectivity",
        template="plotly_white",
        legend={"tracegroupgap": 8},
    )

    fig.write_html(OUTPUT_HTML)
    fig.show()


if __name__ == "__main__":
    main()
