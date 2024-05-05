# cs5220_final_project

## Useful commands

Generate build files:
`cmake -DCMAKE_BUILD_TYPE=Release ..`

Allocate a GPU node:
`salloc -N 1 -q interactive -t 01:00:00 --constraint gpu -G 1 --account=m4341`

Run serial implementation: `./serial`

Run parallel implementation: `./gpu`

Run parallel iterative implementation: `./gpu_iterative`
