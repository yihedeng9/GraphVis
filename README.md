# GraphVis


## Install
Install Package
```Shell
conda create -n graphvis python=3.10 -y
conda activate graphvis
pip install --upgrade pip  
pip install -e .
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```

## Instruction
1. Generate training data for visual graph comprehension
```Shell
python generate_data.py
```
2. Run the shell script
```Shell
bash scripts/finetune_lora.sh
```


## Acknowledgement

- [LLaVA](https://github.com/haotian-liu/LLaVA)
