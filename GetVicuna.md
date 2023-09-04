First, download Vicuna’s **delta weight** from these repositories: [7B](https://huggingface.co/lmsys/vicuna-7b-v1.1) and [13B](https://huggingface.co/lmsys/vicuna-13b-v1.1). this can be done by

```
git lfs install
git clone https://huggingface.co/lmsys/vicuna-7b-v1.1  # larger, need at least 24G gpu memory
# or
git clone https://huggingface.co/lmsys/vicuna-7b-delta-v1.1  # smaller, need 12G gpu memory
```
**Note that this is not directly the working weight**, but the difference between the working weight and the original weight of LLAMA-13B. (Due to LLAMA’s rules, we cannot distribute the weight of LLAMA.)

Then, you need to obtain the original LLAMA-7B or LLAMA-13B weights in the HuggingFace format either following the instruction provided by HuggingFace [here](https://huggingface.co/docs/transformers/main/model_doc/llama) or from the Internet.

When these two weights are ready, we can use tools from Vicuna’s team to create the real working weight.
```
pip install git+https://github.com/lm-sys/FastChat.git@v0.1.10

python -m fastchat.model.apply_delta --base /path/to/llama-13bOR7b-hf/  --target /path/to/save/vicuna/weight/  --delta /path/to/vicuna-13bOR7b-delta-v1.1/
```
