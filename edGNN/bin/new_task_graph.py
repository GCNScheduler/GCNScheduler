from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from wfcommons.wfchef.wfchef_abstract_recipe import WfChefWorkflowRecipe
from wfcommons.wfchef.recipes.cycles.recipe import CyclesRecipe
from wfcommons.wfchef.recipes.montage import MontageRecipe
from wfcommons.wfchef.recipes.seismology import SeismologyRecipe
from wfcommons.wfchef.recipes.blast import BlastRecipe
from wfcommons.wfchef.recipes.bwa import BwaRecipe
from wfcommons.wfchef.recipes.epigenomics import EpigenomicsRecipe
from wfcommons.wfchef.recipes.srasearch import SrasearchRecipe
from wfcommons.wfchef.recipes.genome import GenomeRecipe
from wfcommons.wfchef.recipes.soykb import SoykbRecipe
from wfcommons.common.workflow import Workflow
import pathlib
import json
import numpy as np
import tempfile
import torch

RECIPES: Dict[str, WfChefWorkflowRecipe] = {
    "montage": MontageRecipe,
    "cycles": CyclesRecipe,
    "seismology": SeismologyRecipe,
    "blast": BlastRecipe,
    "bwa": BwaRecipe,
    "epigenomics": EpigenomicsRecipe,
    "srasearch": SrasearchRecipe,
    "genome": GenomeRecipe,
    "soykb": SoykbRecipe
}

thisdir = pathlib.Path(__file__).resolve().parent

@dataclass
class Task():
    name: str
    children: List[str]
    runtime: float
    outputs: List[Tuple[str, int]]
    inputs: List[Tuple[str, int]]
    task_id: int

def workflow_to_forward_graph(workflow: List[Dict[str, Any]]) -> Tuple[Dict[int, List[int]], torch.Tensor, Dict[Tuple[int, int], int]]:
    output_providers: Dict[str, Task] = {}
    tasks: Dict[str, Task] = {}
    for i, task_data in enumerate(workflow):
        task = Task(
            name=task_data["name"],
            children=task_data["children"],
            runtime=task_data["runtime"],
            outputs= [
                (f["name"], f["size"]) 
                for f in task_data["files"] 
                if f["link"] == "output"
            ],
            inputs=[
                (f["name"], f["size"]) 
                for f in task_data["files"] 
                if f["link"] == "input"
            ],
            task_id=i
        )
        tasks[task.name] = task
        for output_name, _ in task.outputs:
            output_providers[output_name] = task

    comp = np.zeros(len(tasks))
    edge_weights = {}
    forward_graph: Dict[int, List[int]] = {}
    for _, task in tasks.items():
        comp[task.task_id] = task.runtime
        forward_graph[task.task_id] = [
            tasks[child_name].task_id for child_name in task.children
        ]
        for input_name, input_size in task.inputs:
            if input_name not in output_providers:
                continue
            parent_task = output_providers[input_name]
            edge = (parent_task.task_id, task.task_id)
            edge_weights.setdefault(edge, 0)
            edge_weights[edge] += input_size

    comp_torch = torch.Tensor([comp.tolist()])
    return forward_graph, comp_torch, edge_weights

def get_graph(recipe: str, num_tasks: int) -> Dict[int, List[int]]:
    if recipe not in RECIPES:
        raise ValueError(f"Invalid recipe '{recipe}'. Valid recipes are: {list(RECIPES.keys())}")


    abstract_workflow = RECIPES[recipe].from_num_tasks(num_tasks=num_tasks).build_workflow()
    with tempfile.NamedTemporaryFile() as tf:
        abstract_workflow.write_json(tf.name)
        workflow_instance = json.loads(pathlib.Path(tf.name).read_text())
        
    return workflow_to_forward_graph(workflow_instance["workflow"]["tasks"])
    
if __name__ == "__main__":
    forward_graph, comp, edge_weights = get_graph("montage", 200)

    # Forward Graph dictionary parents -> [child_1, child_2, ...]
    print(forward_graph)

    # Comp Tensor
    print(comp)

    # Edge weights (amount of data sent across task dependecies (parent, child) -> # bytes
    print(edge_weights)
