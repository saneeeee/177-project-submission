from pyvis.network import Network
from typing import List, Dict
from collections import defaultdict
import webbrowser
import os
import tempfile


def create_transaction_graph(transactions: List[Dict], output_path: str = None) -> str:
    """
    Create an interactive visualization of transaction patterns.
    Returns the path to the generated HTML file.
    """
    # Create network
    net = Network(notebook=False, height="750px", width="100%", bgcolor="#ffffff", 
                 font_color="#000000")
    net.force_atlas_2based()
    
    # Track total volume between addresses
    edge_volumes = defaultdict(float)
    for tx in transactions:
        key = (tx['from_addr'], tx['to_addr'])
        edge_volumes[key] += tx['token_amount']

    # Track total volume per address
    address_volumes = defaultdict(float)
    for tx in transactions:
        address_volumes[tx['from_addr']] += tx['token_amount']
        address_volumes[tx['to_addr']] += tx['token_amount']

    # Add nodes
    addresses = set()
    for tx in transactions:
        addresses.add(tx['from_addr'])
        addresses.add(tx['to_addr'])

    for addr in addresses:
        # Use shortened address for display
        label = f"{addr[:6]}...{addr[-4:]}"
        volume = address_volumes[addr]
        # Scale node size based on volume
        size = min(50, 10 + (volume / max(address_volumes.values())) * 40)
        net.add_node(addr, label=label, title=f"Volume: {volume:.2f}", size=size)

    # Add edges
    for (from_addr, to_addr), volume in edge_volumes.items():
        # Scale edge width based on volume
        width = 1 + (volume / max(edge_volumes.values())) * 5
        net.add_edge(from_addr, to_addr, value=width, 
                    title=f"Volume: {volume:.2f}",
                    color='#0000FF')

    # Save and open
    if output_path is None:
        # Create temporary file
        temp_dir = tempfile.gettempdir()
        output_path = os.path.join(temp_dir, "transaction_graph.html")
    
    net.save_graph(output_path)
    webbrowser.open('file://' + os.path.abspath(output_path))
    
    return output_path
