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
![Screenshot 2024-10-28 at 2 18 10 PM](https://github.com/user-attachments/assets/c99afdfb-747f-4689-92f8-eecf4a1b522d)

# Acknowledgement
A lot of the work done here is inspired by Andrej Karpthy's work.
