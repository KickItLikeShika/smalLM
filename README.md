# SmallLM

Training, Evaluating and Infering a Small LM from scratch.

Weights: https://huggingface.co/KickItLikeShika/smalLM-v0.1


# DDP Training Run
```
torchrun --standalone --nproc_per_node=1 main.py
```

# Inference
Everything needed to infer the model will be at the `inference` directory.
```
python infer.py "Text to complete"
```