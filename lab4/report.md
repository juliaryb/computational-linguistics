# Course Task: Memory-Efficient Transformer Training Techniques
## The experiments are implemented in run_baseline.py (in the main repo folder)

- I don't have a GPU locally, so I had to setup things on Athena, it took a while
- The setup works now - training complete on CUDA-enabled GPU
- the lab-related experiments are work in progress!!

- added random seed so runs are reproducible and (more importantly) comparable (in terms of PPL for example especially after just onen epoch) 

- installed flash att. had to use no cache for hpc

- implementation of the flashattention block 

- run into OOM with Flash Attention for max batch size - will fix tomorrow 

- I am a bit confused by what we're measinring in the 20 "warmup" runs - if we're taking the max then I guess it makes sense bcs we are taking into consideraton all the needed startups of kernels etc. and the amount they need, but then taking mean time? shouldn;t it also be max time and then maybe if we knew how many runs are needed for warmup, we'd average after later runs to know the average stp time 

"For each configuration, memory usage was measured over 20 consecutive training steps, and the maximum observed peak memory was reported, as this determines whether a configuration can fit on the GPU without out-of-memory errors. Training speed was measured as the mean step time over the same 20 steps, providing an estimate of steady-state throughput while amortizing one-time startup overheads such as kernel initialization and autotuning."

```
- so adding the for loop in batch size profiling helped overcome oom issues - for just one run it was too optimistic - changing to do 5 runs made it error-proof at least for the couple of runs that were run 