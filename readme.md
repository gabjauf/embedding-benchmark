# Benchmarking embedding libraries in NodeJS

A Proof Of Concept benchmark of different libraries

## Why ?

When trying to make an app with embeddings involved, I quickly noticed a bottleneck during the indexation process with faiss. After investigation, it was clearly the embedding that were taking too long. Even though it improved when changing to transformer.js, it was still slow, even crashing the process (Promise.all is a bad idea when indexing a large pdf).

## Setup
Benchmark consists in 4 concurrent queries for the same text that may or may not be parallelized under the form of a worker thread in NodeJS.

## Run
Download the models first:
[mxbai-embed-large-v1](https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1/resolve/main/onnx/model_quantized.onnx)
[mxbai-embed-large-v1.Q8_0.gguf](https://huggingface.co/ChristianAzinn/mxbai-embed-large-v1-gguf/resolve/main/mxbai-embed-large-v1.Q8_0.gguf)

Library benchmark: `npm run bench`
Transformer benchmark: `npm run bench:transformer`

## Results

Measurements were done using [mitata](https://github.com/evanwashere/mitata).
Results are for CPU and source can be found inside `bench-embedder.mts`

```
cpu: Intel(R) Core(TM) i9-9880H CPU @ 2.30GHz
runtime: node 22.11.0 (x64-darwin)

benchmark              avg (min … max) p75   p99    (min … top 1%)
-------------------------------------- -------------------------------
pool onnx               823.07 ms/iter 855.44 ms          █           
                  (698.89 ms … 1.07 s) 899.73 ms █▁▁█▁▁█▁███▁█▁▁▁█▁▁█▁
pool transformer.js     806.19 ms/iter 823.71 ms          ██   █      
               (726.32 ms … 916.92 ms) 861.81 ms █▁█▁▁▁▁▁▁██▁███▁▁▁▁▁▁
onnx Node               402.81 ms/iter 421.11 ms     █                
               (347.28 ms … 495.16 ms) 454.64 ms █▁█▁███▁▁▁█▁▁██▁▁▁█▁▁
onnx Web                   5.74 s/iter    5.78 s █   █                
                     (5.51 s … 6.05 s)    5.98 s █▁▁▁█▁▁█▁████▁▁▁█▁▁▁▁
transformer.js          390.88 ms/iter 411.28 ms    ▃          █      
               (347.19 ms … 448.54 ms) 439.21 ms ▆▁▆█▆▁▁▆▁▁▁▆▁▁█▁▁▁▁▁▁
LlamaCPP                   1.55 s/iter    1.71 s  █                   
                     (1.36 s … 1.90 s)    1.76 s █████▁▁▁█▁▁▁▁▁█▁▁▁██▁
Ollama                  950.06 ms/iter 998.83 ms █           █      █ 
                  (805.64 ms … 1.07 s)    1.01 s █▁▁▁▁▁▁▁▁▁▁██▁▁▁████▁

                        ┌                                            ┐
                           ┌┬
              pool onnx    ││
                           └┴
                           ╷┬
    pool transformer.js    ├│
                           ╵┴
                        ┬┐
              onnx Node ││
                        ┴┘
                                                                 ╷┌┬ ╷
               onnx Web                                          ├┤│─┤
                                                                 ╵└┴ ╵
                        ┬┐
         transformer.js ││
                        ┴┘
                                ┌─┬┐
               LlamaCPP         │ ││
                                └─┴┘
                            ╷┬
                 Ollama     ├│
                            ╵┴
                        └                                            ┘
                        347.19 ms            3.16 s             5.98 s
```

We have a clear winner here, being transformer.js 🤗

What we can learn here:

- Making many workers is useless (and gives worst results with every kind of ONNXRuntime based libraries)
- Onnx web on nodejs using wasm has catastrophic performance and cannot be used in a worker
- LlamaCPP is often praised for performance, but this is not clearly visible in this benchmark (even though the model should be quantized in Q8). Also note that adding or not threads does not impact benchmark results

### Further notes
While transformer.js is the best in this benchmark, the following results from transformer benchmark (`bench-transformer.mts`) show that parallelization does not happen inside transformers and onnx-runtime libraries, with runtime for 4 concurrent tasks being 4x one task:

```
cpu: Intel(R) Core(TM) i9-9880H CPU @ 2.30GHz
runtime: node 22.11.0 (x64-darwin)

benchmark              avg (min … max) p75   p99    (min … top 1%)
-------------------------------------- -------------------------------
transformer.js          490.19 ms/iter 499.43 ms █      █ ▃           
               (360.43 ms … 762.94 ms) 685.80 ms █▁▁▁▁▆▆█▁█▁▁▁▁▁▁▁▁▁▁▁
1 only transformer.js   103.51 ms/iter 112.59 ms     █                
                (82.60 ms … 128.13 ms) 120.24 ms █▁▁▁██▁▁█▁█▁██▁▁██▁▁▁
onnx Node               445.00 ms/iter 474.24 ms        █          █  
               (402.72 ms … 490.41 ms) 481.32 ms █▁▁█▁████▁▁▁▁▁█▁▁▁█▁▁
1 only onnx node        111.94 ms/iter 118.47 ms                █     
                (88.00 ms … 129.16 ms) 122.40 ms ▆▁▁▁▁▆▁▁▆▁▁▁▁▆▁█▆▁▆▆▁

                        ┌                                            ┐
                                             ┌────────┬┐             ╷
         transformer.js                      │        │├─────────────┤
                                             └────────┴┘             ╵
                        ╷┌┬╷
  1 only transformer.js ├┤│┤
                        ╵└┴╵
                                                ╷┌─┬─┐╷
              onnx Node                         ├┤ │ ├┤
                                                ╵└─┴─┘╵
                        ╷┌┬┐
       1 only onnx node ├┤││
                        ╵└┴┘
                        └                                            ┘
                        82.60 ms          384.20 ms          685.80 ms
```

It probably means that optimizations could still happen, even though we need to have in mind that given the size of LLMs, these libraries probably optimized the RAM usage way more than the CPU efficiency.

## Limitations

Please note that this is a JS benchmark on CPU. Even though the libraries are coded in Cpp, it is probably not relevant for other languages given that the bindings or wasm can give a lot of variation in the performance.
