# Course Task: Memory-Efficient Transformer Training Techniques

- I don't have a GPU locally, so I had to setup things on Athena, it took a while
- The setup works now - training complete on CUDA-enabled GPU
- the lab-related experiments are work in progress!!

- added random seed so runs are reproducible and (more importantly) comparable (in terms of PPL for example especially after just onen epoch) 

- installed flash att. had to use no cache for hpc

- implementation of the flashattention block 

- run into OOM with Flash Attention for max batch size - will fix tomorrow 