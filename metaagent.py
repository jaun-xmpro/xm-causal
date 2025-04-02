import json
import pandas as pd
from dowhy import CausalModel
import networkx as nx
import pathlib


def on_create(data: dict) -> dict:
    """
    Initialize the causal analysis setup with predefined data, graph, and config.
    
    Returns:
        dict: Status of initialization
    """
    try:
        return {"status": "initialized"}
    except Exception as e:
        return {"status": f"Error {e}"}


def on_receive(data: dict) -> dict:
    """
    Perform causal inference analysis with provided or default graph, config, and data.
    
    Args:
        data (dict): Payload of the task containing dataset, graph_edges, method, treatment, and outcome
    
    Returns:
        dict: Causal analysis results or error message
    """
    try:
        # Extract arguments from the input dictionary
        analysis_data_string = data.get('dataset', None)
        analysis_graph_edges_string = data.get('graph_edges', None)
        analysis_method = data.get('method', None)
        analysis_treatments = data.get('treatment', None)
        analysis_outcomes = data.get('outcome', None)

        # Load the dataset: try JSON first, then CSV file if JSON fails
        try:
            analysis_data = json.loads(analysis_data_string)
        except (json.JSONDecodeError, TypeError):
            analysis_data_path = pathlib.Path(analysis_data_string)
            if analysis_data_path.exists():
                analysis_data = pd.read_csv(analysis_data_path)
            else:
                raise ValueError("Invalid data path or data format.")
        if analysis_data is None:
            raise ValueError("No data provided for analysis.")
        
        # Parse the graph edges from the JSON string
        if analysis_graph_edges_string is not None:
            analysis_graph_edges = json.loads(analysis_graph_edges_string)
        else:
            raise ValueError("No graph edges provided for analysis.")
        
        # Create a directed graph from the edges
        analysis_graph = nx.DiGraph(analysis_graph_edges)
        
        # List to store results for all treatment-outcome pairs
        effect_data = []

        # Iterate over each treatment-outcome pair
        for treatment in analysis_treatments:
            for outcome in analysis_outcomes:
                try:
                    # Create a new CausalModel for each treatment-outcome pair
                    causal_model = CausalModel(
                        data=analysis_data,
                        graph=analysis_graph,
                        treatment=treatment,
                        outcome=outcome
                    )
                    
                    # Identify the causal effect
                    identified_estimand = causal_model.identify_effect(proceed_when_unidentified=True)
                    
                    # Estimate the causal effect
                    causal_estimate = causal_model.estimate_effect(
                        identified_estimand,
                        method_name=analysis_method,
                        control_value=0,
                        treatment_value=1
                    )
                    
                    # Store the results
                    effect_data.append({
                        "treatment": treatment,
                        "outcome": outcome,
                        "estimate": causal_estimate.value,
                        "method": analysis_method,
                        "interpretation": str(causal_estimate)
                    })
                except Exception as e:
                    print(f"Error estimating effect for treatment {treatment} and outcome {outcome}: {e}")
                    effect_data.append({
                        "treatment": treatment,
                        "outcome": outcome,
                        "error": str(e)
                    })

        # Compile the results into a dictionary
        result = {
            "causal_effects": effect_data,
            "graph_edges": analysis_graph_edges,
            "treatments": analysis_treatments,
            "outcomes": analysis_outcomes
        }
        
        return {"result": json.dumps(result)}
    except Exception as e:
        return {"status": f"Error {e}"}


def on_destroy() -> dict:
    """
    Clean up the global dataset, graph, and config.
    
    Returns:
        dict: Status of destruction
    """
    return {"status": "destroyed"}


# Example usage
if __name__ == "__main__":
    # Define the causal method without a trailing comma
    method_name = "backdoor.linear_regression"

    # Define the graph edges as a list of tuples
    edges = [
        # Air intake system
        ('operating_altitude', 'air_filter_pressure'),
        ('air_filter_pressure', 'egt_turbo_inlet'),
        ('air_filter_pressure', 'fuel_consumption'),
        
        # Primary mechanical relationships
        ('payload_weight', 'engine_load'),
        ('haul_road_gradient', 'engine_load'),
        ('engine_load', 'engine_rpm'),
        ('engine_load', 'fuel_consumption'),
        ('engine_rpm', 'vehicle_speed'),
        ('engine_rpm', 'air_filter_pressure'),
        
        # Environmental influences
        ('operating_altitude', 'engine_load'),
        ('ambient_temp', 'engine_coolant_temp'),
        ('ambient_temp', 'egt_turbo_inlet'),
        
        # Fuel and combustion chain
        ('fuel_consumption', 'egt_turbo_inlet'),
        ('engine_load', 'egt_turbo_inlet'),
        
        # Temperature cascade through exhaust system
        ('egt_turbo_inlet', 'egt_turbo_outlet'),
        ('egt_turbo_outlet', 'egt_stack'),
        
        # Cooling system relationships
        ('engine_coolant_temp', 'egt_turbo_inlet'),
        ('engine_rpm', 'engine_coolant_temp')
    ]

    # Convert edges to a JSON-serializable format (list of lists) and then to a JSON string
    edges_json = json.dumps([[u, v] for u, v in edges])

    # Define treatments and outcomes as lists
    treatments = ['air_filter_pressure', 'engine_coolant_temp', 'engine_load', 
                  'ambient_temp', 'engine_rpm', 'fuel_consumption']
    outcomes = ['egt_turbo_inlet', 'egt_turbo_outlet', 'egt_stack']

    # Initialize the model
    print("on_create (empty):", on_create({}))

    # Send task with arguments
    print("on_receive (task):", on_receive({
        'dataset': 'data.csv',
        'graph_edges': edges_json,
        'method': method_name,
        'treatment': treatments,
        'outcome': outcomes
    }))

    # Clean up
    print("on_destroy:", on_destroy())