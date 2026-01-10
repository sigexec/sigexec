"""Directed Acyclic Graph for signal processing chain."""

from typing import List, Dict, Set
from .block import ProcessingBlock
from .data import SignalData


class DAG:
    """
    Directed Acyclic Graph for organizing and executing processing blocks.
    
    The DAG manages the execution order of processing blocks and ensures
    that dependencies are resolved correctly.
    """
    
    def __init__(self):
        """Initialize an empty DAG."""
        self.blocks: List[ProcessingBlock] = []
    
    def add_block(self, block: ProcessingBlock):
        """
        Add a processing block to the DAG.
        
        Args:
            block: The processing block to add
        """
        if block not in self.blocks:
            self.blocks.append(block)
    
    def add_chain(self, *blocks: ProcessingBlock):
        """
        Add a chain of blocks to the DAG and connect them sequentially.
        
        Args:
            *blocks: Variable number of processing blocks
        """
        for i, block in enumerate(blocks):
            self.add_block(block)
            if i > 0:
                blocks[i-1].connect(block)
    
    def topological_sort(self) -> List[ProcessingBlock]:
        """
        Perform a topological sort on the blocks.
        
        Returns:
            List of blocks in execution order
        """
        # Build adjacency list and in-degree count
        in_degree: Dict[ProcessingBlock, int] = {block: 0 for block in self.blocks}
        
        for block in self.blocks:
            for output in block.outputs:
                if output in in_degree:
                    in_degree[output] += 1
        
        # Find all blocks with no incoming edges
        queue = [block for block in self.blocks if in_degree[block] == 0]
        sorted_blocks = []
        
        while queue:
            block = queue.pop(0)
            sorted_blocks.append(block)
            
            for output in block.outputs:
                if output in in_degree:
                    in_degree[output] -= 1
                    if in_degree[output] == 0:
                        queue.append(output)
        
        # Check for cycles
        if len(sorted_blocks) != len(self.blocks):
            raise ValueError("Graph contains a cycle!")
        
        return sorted_blocks
    
    def execute(self, initial_data: SignalData, start_block: ProcessingBlock = None) -> SignalData:
        """
        Execute the DAG starting from a specific block or the first block.
        
        Args:
            initial_data: The initial signal data
            start_block: Optional starting block (defaults to first block)
            
        Returns:
            The final processed signal data
        """
        if start_block is None:
            if not self.blocks:
                raise ValueError("No blocks in DAG")
            start_block = self.blocks[0]
        
        # For a simple chain, just execute sequentially
        current_data = initial_data
        current_block = start_block
        
        while current_block:
            current_data = current_block.process(current_data)
            
            # Move to next block
            if current_block.outputs:
                current_block = current_block.outputs[0]
            else:
                break
        
        return current_data
    
    def __repr__(self):
        """Return string representation of the DAG."""
        return f"DAG(blocks={len(self.blocks)})"
