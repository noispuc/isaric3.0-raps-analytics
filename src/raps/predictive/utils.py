import pandas as pd
import plotly.graph_objs as go

def fig_forest_plot(
        df,
        dictionary=None,
        title='Forest Plot',
        labels=['Study', 'OddsRatio', 'LowerCI', 'UpperCI'],
        graph_id='forest-plot',
        graph_label='',
        graph_about='',
        only_display=False):
    """
    Generates an interactive Forest Plot using Plotly.

    Parameters:
    - df (pd.DataFrame): DataFrame containing regression results.
    - dictionary (dict, optional): Not used in current implementation.
    - title (str): Title of the plot.
    - labels (list): List of column names in df: [study, estimate, lower CI, upper CI].
    - graph_id (str): Optional ID for the graph.
    - graph_label (str): Optional label for the graph.
    - graph_about (str): Optional description of the graph.
    - only_display (bool): If True, returns the figure for display.

    Returns:
    - fig (plotly.graph_objs.Figure): Forest Plot figure object.
    """

    # Sort values by estimate for better visual ordering
    df = df.sort_values(by=labels[1], ascending=True)

    # Validate required columns
    if not set(labels).issubset(df.columns):
        print(df.columns)
        error_str = f'DataFrame must contain the following columns: {labels}'
        raise ValueError(error_str)

    # Create traces for point estimates
    traces = [
        go.Scatter(
            x=df[labels[1]],
            y=df[labels[0]],
            mode='markers',
            name='Estimate',
            marker=dict(color='blue', size=10)
        )
    ]

    # Add confidence interval lines
    for _, row in df.iterrows():
        traces.append(
            go.Scatter(
                x=[row[labels[2]], row[labels[3]]],
                y=[row[labels[0]], row[labels[0]]],
                mode='lines',
                showlegend=False,
                line=dict(color='blue', width=2)
            )
        )

    # Layout configuration
    layout = go.Layout(
        title=title,
        xaxis=dict(title='Coefficient'),
        yaxis=dict(
            title='',
            automargin=True,
            tickmode='array',
            tickvals=df[labels[0]].tolist(),
            ticktext=df[labels[0]].tolist()
        ),
        shapes=[
            dict(
                type='line',
                x0=1, y0=-0.5,
                x1=1, y1=len(df[labels[0]]) - 0.5,
                line=dict(color='red', width=2)
            )
        ],
        margin=dict(l=100, r=100, t=100, b=50),
        height=600
    )

    return go.Figure(data=traces, layout=layout)
