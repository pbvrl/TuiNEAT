### Naming conventions

`conns` for the connections of the whole population, or multiple networks
`cs` for a single network's connections
`c` is for a single connection

`nodes` for the nodes of the whole population, or multiple networks
`nds` for a single network's nodes
`n` for a single node

`innovs` for the innovation numbers of the whole population
`inns` for the innovation numbers of a single network
`inn` for a single innovation number

`species` for all species
`sp` for a single species

`reps` for all representatives
`rep` for a single representative

`parents` for all parent pairs
`prts` for one parent pair

`dists` for each species members' distance to its representative
`dsts` for one species members' distance to its representative
`distances` for one individual's distance to all representatives

`params` for the parameters for forward passing all individuals
`prms` for the parameters for forward passing one individual
Exception to this: The `best_params`  that implements evojax's interface.

`toposorted_idxs` refers to the indices of `nodes` sorted such that each network is topologically sorted.
`toposrtd_idxs` refers to the toposort of one network

`is_**` indicates a boolean mask

`input` and `output` as in `is_input` refers to the input/output nodes at the beginning/end of one network.
`in` and `out` as in `in_topo` refers to the in and out nodes of one connection.

### Special comment types

`FULLFILLS`: Satisfies an interface/method/format from an external library when othwerise it wouldn't be there or have that naming.
`TODO`
`WARNING`: Either something is not supported, or something is a (one-shot)-vibe-coded module instead of carefully designed clean code.
