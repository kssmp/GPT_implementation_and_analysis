# Implementation and Optimization of the GPT-2 Model 
Name : Saket Koppineedi

Enrollment no. : BT20ECE114

Email Id : saketkoppineedi@gmail.com

---

## Overview of the Assignment

Embarking on this challenge with enthusiasm, I aimed to demonstrate my comprehension of the Transformer architecture. Throughout this endeavor, I sought to grasp and execute various aspects of the assigned tasks, working towards creating a functional model. Drawing inspiration from a diverse array of sources, I have provided references at the end of this document. I've elucidated the code for each task under specific headings, detailing my approach and the challenges faced while tackling the intricacies of this complex model.

// Please refer to the results file for all indexed outcomes \\\

# Task 1 | GPT-2 Model & Checkpoints

## Model Architecture & Approach
The GPT language model is realized as a neural network using PyTorch. The architecture comprises key components:
- Token Embedding Layer: Transforms input tokens into continuous vector representations.
- Position Embedding Layer: Introduces positional information into the input tokens.
- Multi-Head Attention Mechanism: Utilizes multiple attention heads to capture diverse context aspects.
- FeedForward Neural Network: Implements a straightforward neural network structure for further processing.
- Layer Normalization: Applied at various stages for training stabilization.
- Linear Head: Generates output logits based on the processed input.

## Training
Training involves utilizing the AdamW optimizer with specified hyperparameters, such as batch size, block size, maximum iterations, evaluation interval, learning rate, device, evaluation iterations, number of embedding dimensions, number of attention heads, number of layers, and dropout rate. The training dataset is divided into 90% for training and 10% for validation. The training process entails estimating loss and updating model parameters through backpropagation.
![Screenshot 2023-12-16 at 5 32 54 PM](https://github.com/kssmp/GPT_implementation_and_analysis/assets/115448106/7633d6f9-4b94-4664-9972-ec941a908586)

## Results
Upon comparison with the [GPT2 125M]((https://github.com/kssmp/GPT_implementation_and_analysis/blob/main/Task_1/Evaluation_using_125M.py)) pretrained checkpoints, I recognized the computational constraints, with my system taking over 15 minutes to handle half a million parameters. Consequently, the generated outputs are not on par with the actual model.



# Task 2 | Transformer Architectural Modifications 

## Rotary Positional Embedding:

The introduction of Rotary Positional Embedding brings a distinctive geometric perspective to traditional positional embeddings. While enhancing the model's ability to capture intricate patterns, it increases the parameter count, necessitating careful consideration of the trade-off between representation richness and computational complexity. While coding, I encountered multiple tensor broadcasting errors that remain unresolved for now. Nevertheless, efforts to address these issues continue.

## Group Query Attention:

Group Query Attention optimizes parameter sharing within the attention mechanism, potentially reducing the overall model parameters. However, in our specific implementation, it's essential to note that there might be no noticeable change in parameters. This is contingent on the dataset's characteristics and the model's capacity to benefit from parameter sharing. Detailed insights into the impact on model efficiency, despite potential parameter stability, are available in the accompanying GitHub file.

## Sliding Window Attention:

The addition of Sliding Window Attention localizes the attention span, enhancing computational efficiency at the expense of reduced global context awareness. Users should evaluate the application's needs, balancing efficiency and the requirement for capturing long-range dependencies. Similar to Group Query Attention, the number of parameters might not exhibit significant changes due to dataset-specific factors. The associated GitHub file provides experimental details and results for a comprehensive understanding.


# Task 3 | Training Loop Implementation 

## Training Loop

Develop a training loop with the following specifications:

1. **Single GPU Training Loop**
2. **Distributed Data Parallel**
3. **Fully Sharded Data Parallel**

## Parallelization Strategies:
The script accommodates both Distributed Data Parallelism (DDP) and Fully Sharded Data Parallelism (FSDP) to optimize training across multiple GPUs. DDP initializes when multiple GPUs are available using DistributedDataParallel. For single GPU setups, no specific parallelization is applied. Additionally, FSDP initialization is included as a comment, enabling users to uncomment and utilize it as an alternative parallelization strategy.

## Training Loop: ##
The training loop iterates over the specified number of iterations (max_iters), periodically evaluating training and validation losses using the estimate_loss function. The model processes input batches (xb) to compute logits and loss. For DDP, the script performs an all-reduce operation on the loss for gradient synchronization across GPUs. Gradients are then zeroed, and backpropagation is conducted. The optimizer is stepped to update the model's parameters. The script also includes commented lines for FSDP, offering users flexibility to choose between DDP and FSDP based on their parallelization requirements.

## References
- [OpenAI repository](https://github.com/openai/gpt-2)
- [Andrej Karpathy](https://github.com/karpathy/nanoGPT)
- [Attention is all you need](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)
- [ChatGPT (ironically!!)](https://chat.openai.com/)
  
