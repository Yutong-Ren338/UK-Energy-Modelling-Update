import pandas as pd


def rolling_mean_circular(df: pd.DataFrame, column: str, window_size: int) -> pd.Series:
    """Apply a rolling mean with circular padding to a DataFrame column.

    This function applies circular padding by concatenating the end of the data
    to the beginning and the beginning to the end, then applies a centered
    rolling mean to smooth the data while maintaining the original length.

    Args:
        df: DataFrame containing the data.
        column: Column name to apply the rolling mean on.
        window_size: Size of the rolling window. Should be an odd number for proper centering.

    Returns:
        Series with the smoothed values, same length as the original data.
    """
    half_window = window_size // 2

    # Check if the column has pint units and store them
    units = None
    data = df[column]

    if "pint" in str(df[column].dtype):
        units = df[column].pint.units
        data = df[column].pint.magnitude

    # Apply rolling mean with circular padding
    data_padded = pd.concat([data.tail(half_window), data, data.head(half_window)]).reset_index(drop=True)
    result = data_padded.rolling(window=window_size, center=True).mean().iloc[half_window:-half_window]
    result.index = df.index

    # Convert back to pint quantity if original had units
    if units is not None:
        result = result.astype(f"pint[{units}]")

    return result
