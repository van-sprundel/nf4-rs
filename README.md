Was reading https://ngrok.com/blog/quantization and felt like trying this out in Rust. 
Sometimes you gotta kill time :P

NF4 is a data structure that's optimized for quantization of LLMs. A few key take-aways:
- It does not contain Infinity or NaN, as these values are not important for LLMs
- The 16 values below are the CDF of N(0,1) (normal distr.), divided by 16. 
  - the normal distribution has the most density near zero, so adjacent quantiles near zero are close together (giving fine resolution)
  - quantiles in the tails are far apart (coarser resolution where there's less data).
    - if you know how a float distribution looks like (there's an example in the article), then you know this doesnt match well against gaussian data.

Running it against Qwen2-0.5B results in some success (make sure you run in release):

```sh
WEIGHTS
block size     mean abs      p99 abs    mean rel%     p99 rel%  compression
        32     0.026668     0.077339       17.71%       83.80%         6.4x
        64     0.027191     0.080459       19.15%       84.04%         7.1x
       128     0.026923     0.080752       20.59%       84.40%         7.5x
       256     0.027560     0.080545       22.09%       84.85%         7.8x

BIASES
block size     mean abs      p99 abs    mean rel%     p99 rel%  compression
        32     0.156922     0.800379       43.28%      100.00%         6.4x
        64     0.202169     0.854703       53.70%      100.00%         7.1x
       128     0.247598     0.895509       62.80%      100.00%         7.5x
       256     0.257528     0.923497       63.71%      100.00%         7.6x
```

Seems like all block sizes have a similar error rate. (~0.027), but compression does seem to improve on higher block sizes. p99 is also awful, but im not sure why.
