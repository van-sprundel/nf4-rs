Was reading https://ngrok.com/blog/quantization and felt like trying this out in Rust. 
Sometimes you gotta kill time :P

NF4 is a data structure that's optimized for quantization of LLMs. A few key take-aways:
- It does not contain Infinity or NaN, as these values are not important for LLMs
- The 16 values below are the CDF of N(0,1) (normal distr.), divided by 16. 
  - the normal distribution has the most density near zero, so adjacent quantiles near zero are close together (giving fine resolution)
  - quantiles in the tails are far apart (coarser resolution where there's less data).
    - if you know how a float distribution looks like (there's an example in the article), then you know this doesnt match well against gaussian data.

<img width="693" height="220" alt="image" src="https://github.com/user-attachments/assets/0870a778-da2b-4ac2-bef2-1a88d38f0774" />
