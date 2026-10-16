"""Parse the NHC dataset Excel workbook: variable definitions and available reaction agents.

The per-batch agent availability feeds the plan-ahead orchestrator.
"""

import pandas as pd
from pymoo.core.variable import Binary, Choice, Integer, Real


def parse_variable_definitions(variable_definitions_df):
    """Parse the 'Parameter Definition' sheet into pymoo variable objects.

    Args:
        variable_definitions_df: Transposed DataFrame with 'Type' and 'Values'
            columns, indexed by variable name.

    Returns:
        Dict mapping variable names to pymoo variable objects (Real, Choice,
        Integer, or Binary).

    Raises:
        ValueError: If a variable's type is not Real, Choice, Integer or Binary;
            if a Real variable does not have exactly two bounds; or if an Integer
            variable does not have exactly two integer-valued bounds.
    """
    variable_definitions = {}
    validtypes = {"Real", "Choice", "Integer", "Binary"}

    for row in variable_definitions_df.iterrows():
        variable_name = row[0]
        var_type = row[1]["Type"]

        if var_type not in validtypes:
            raise ValueError(
                f"Invalid type '{var_type}' for variable '{variable_name}'"
            )

        values = row[1]["Values"]

        if var_type == "Real":
            bounds = tuple(v for v in values if v != "")
            if len(bounds) != 2:
                raise ValueError(
                    f"Real type for '{variable_name}' must have two bounds"
                )
            variable_definitions[variable_name] = Real(bounds=bounds)

        elif var_type == "Choice":
            choices = [v for v in values if v != ""]
            variable_definitions[variable_name] = Choice(options=choices)

        elif var_type == "Integer":
            bounds = tuple(v for v in values if v != "")
            if len(bounds) != 2 or any(v != int(v) for v in bounds):
                raise ValueError(
                    f"Integer type for '{variable_name}' must have two integer bounds"
                )
            variable_definitions[variable_name] = Integer(
                bounds=(int(bounds[0]), int(bounds[1]))
            )

        elif var_type == "Binary":
            variable_definitions[variable_name] = Binary()

    return variable_definitions


def read_data(excel_file, *, device=None, dtype=None):
    """Read available reaction agents and variable definitions from an Excel workbook.

    Args:
        excel_file: Path to the NHC dataset Excel workbook.
        device: Unused, kept for backward compatibility.
        dtype: Unused, kept for backward compatibility.

    Returns:
        Tuple of two elements:
            - ``available_reaction_agents``: Dict with keys ``"Catalyst"``,
              ``"Solvent"`` and ``"Base"``, each a list of four lists of agent
              names, one per planned batch (Next, Subsequent, Following, Fourth
              Batch columns marked ``"y"`` in the "Available ..." sheets).
            - ``variable_definitions``: Dict of pymoo variable definitions, keyed
              by variable name with any ``" in ..."`` unit suffix removed.
    """
    variable_definitions = pd.read_excel(excel_file, sheet_name="Parameter Definition")

    variable_definitions = variable_definitions.transpose()
    variable_definitions.columns = variable_definitions.iloc[0]
    variable_definitions = variable_definitions[1:]
    variable_definitions["Values"] = variable_definitions.iloc[:, 1:].apply(
        lambda x: tuple(x.dropna()), axis=1
    )
    variable_definitions = variable_definitions[["Type", "Values"]]

    variable_definitions = parse_variable_definitions(variable_definitions)
    variable_definitions = {
        (key.split(" in")[0] if " in" in key else key): value
        for key, value in variable_definitions.items()
    }

    available_catalysts = pd.read_excel(excel_file, sheet_name="Available Catalysts")

    next_batch_catalysts = available_catalysts.loc[
        available_catalysts["Next Batch"] == "y", "Catalyst"
    ].tolist()
    subsequent_batch_catalysts = available_catalysts.loc[
        available_catalysts["Subsequent Batch"] == "y", "Catalyst"
    ].tolist()
    following_batch_catalysts = available_catalysts.loc[
        available_catalysts["Following Batch"] == "y", "Catalyst"
    ].tolist()
    fourth_batch_catalysts = available_catalysts.loc[
        available_catalysts["Fourth Batch"] == "y", "Catalyst"
    ].tolist()
    available_catalysts = [
        next_batch_catalysts,
        subsequent_batch_catalysts,
        following_batch_catalysts,
        fourth_batch_catalysts,
    ]

    available_solvents = pd.read_excel(excel_file, sheet_name="Available Solvents")

    next_batch_solvents = available_solvents.loc[
        available_solvents["Next Batch"] == "y", "Solvent"
    ].tolist()
    subsequent_batch_solvents = available_solvents.loc[
        available_solvents["Subsequent Batch"] == "y", "Solvent"
    ].tolist()
    following_batch_solvents = available_solvents.loc[
        available_solvents["Following Batch"] == "y", "Solvent"
    ].tolist()
    fourth_batch_solvents = available_solvents.loc[
        available_solvents["Fourth Batch"] == "y", "Solvent"
    ].tolist()
    available_solvents = [
        next_batch_solvents,
        subsequent_batch_solvents,
        following_batch_solvents,
        fourth_batch_solvents,
    ]

    available_bases = pd.read_excel(excel_file, sheet_name="Available Bases")

    next_batch_bases = available_bases.loc[
        available_bases["Next Batch"] == "y", "Base"
    ].tolist()
    subsequent_batch_bases = available_bases.loc[
        available_bases["Subsequent Batch"] == "y", "Base"
    ].tolist()
    following_batch_bases = available_bases.loc[
        available_bases["Following Batch"] == "y", "Base"
    ].tolist()
    fourth_batch_bases = available_bases.loc[
        available_bases["Fourth Batch"] == "y", "Base"
    ].tolist()
    available_bases = [
        next_batch_bases,
        subsequent_batch_bases,
        following_batch_bases,
        fourth_batch_bases,
    ]

    available_reaction_agents = {
        "Catalyst": available_catalysts,
        "Solvent": available_solvents,
        "Base": available_bases,
    }

    return (
        available_reaction_agents,
        variable_definitions,
    )
