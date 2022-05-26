
# GCNScheduler

We consider the classical problem of scheduling task graphs corresponding to complex applications on distributed computing systems. A number of heuristics have been previously proposed to optimize task scheduling with respect to metrics such as makespan and throughput. However, they tend to be slow to run, particularly for larger problem instances, limiting their applicability in more dynamic systems. Motivated by the goal of solving these problems more rapidly, we propose, for the first time, a graph convolutional network-based scheduler (GCNScheduler). By carefully integrating an inter-task data dependency structure with network settings into an input graph and feeding it to an appropriate GCN, the GCNScheduler can efficiently schedule tasks of complex applications for a given objective. We evaluate our scheme with baselines through simulations. We show that not only can our scheme quickly and efficiently learn from existing scheduling schemes, but also it can easily be applied to large-scale settings where current scheduling schemes fail to handle. We show that it achieves better makespan than the classic HEFT algorithm, and almost the same throughput as throughput-oriented HEFT (TP-HEFT), while providing several orders of magnitude faster scheduling times in both cases. 



## Installation

Create a virtual environment:

```sh
python3 -m venv venv
```

Activate the virtual environment:

```sh
source venv/bin/activate
```

Install the package as editable as well as  any dependencies:

```sh
pip3 install -e .
```

Add the repository path to your `PYTHONPATH`:

```sh
export PYTHONPATH="<PATH>/edGNN:$PYTHONPATH"
```

## Running GCNScheduler:

```sh
cd edGNN/bin
run_model --dataset aifb  --config_fpath ../core/models/config_files/config_edGNN_node_class.json  --data_path ../preprocessed_graphs/aifb/ --n-epochs 40 --weight-decay 0 --lr 0.001
```
