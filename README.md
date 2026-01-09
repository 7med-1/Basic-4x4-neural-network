# Neural Network From Scratch in C
================================

Introduction
------------

This project is created to **simplify the mechanism of neural networks** and explain how they work.It is coded in **pure C** so it can be as basic as possible and easy to understand.

No machine learning libraries are used.Everything is implemented manually to show how neural networks work internally.

The goal of this project is **learning**, not performance.

What This Project Does
----------------------

The input is a **4 × 3 matrix** that contains only **0 or 1** values.This matrix is flattened into **12 input values**.

The neural network is trained **5000 times (epochs)** to classify the input as:

*   UP
    
*   MID
    
*   DOWN
    

The classification is based on **where most of the 1s appear**:

*   Top row → UP
    
*   Middle row → MID
    
*   Bottom row → DOWN
    

Neural Network Architecture
---------------------------

The network has three layers:

*   Input layer: 12 neurons
    
*   Hidden layer: 8 neurons
    
*   Output layer: 3 neurons
    

Structure:

Input (12) → Hidden (8, ReLU) → Output (3, Softmax)

Training Data
-------------

The dataset contains **9 training samples**:

*   3 samples labeled UP
    
*   3 samples labeled MID
    
*   3 samples labeled DOWN
    

Each sample is a flattened **4 × 3 matrix**.

Example input labeled as UP:

`   1 1 1 1  0 0 0 0  0 0 0 0   `

Flattened representation:

`   {1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0}   `

Weights and Biases
------------------

The model uses:

*   W1 and b1 for input → hidden layer
    
*   W2 and b2 for hidden → output layer
    

At the start, all weights and biases are initialized with **random values between -0.1 and 0.1**.

This is important because:

*   It breaks symmetry
    
*   It allows neurons to learn different features
    
*   Starting from zero would prevent learning
    

Random initialization function:

`   float randf()  {   return ((float)rand() / RAND_MAX) * 0.2f - 0.1f;  }   `

Forward Pass
------------

### Input → Hidden Layer

For each hidden neuron, we compute:

b1 + W1 \* input

Then we apply **ReLU activation**.

``   for (int i = 0; i < HIDDEN; i++)
    {
        b1[i] = randf();
        for (int j = 0; j < INPUT; j++)
            W1[i][j] = randf();
    }  ``

ReLU activation function:

`   float relu(float x)  {  return x > 0 ? x : 0;  }   `

### Hidden → Output Layer

The hidden layer output is reused to compute the output layer values:

b2 + W2 \* hidden

``   for (int i = 0; i < OUTPUT; i++)
    {
        b2[i] = 0;
        for (int j = 0; j < HIDDEN; j++)
            W2[i][j] = randf();
    }   ``

At this point, z contains **raw scores (logits)**.

Softmax
-------

Softmax converts logits into probabilities.

`` void softmax(float *z)
{
    float max = z[0];
    for (int i = 1; i < OUTPUT; i++)
        if (z[i] > max)
            max = z[i];
    float sum = 0;
    for (int i = 0; i < OUTPUT; i++)
    {
        z[i] = expf(z[i] - max);
        sum += z[i];
    }
    for (int i = 0; i < OUTPUT; i++)
        z[i] /= sum;
}  
``

After softmax:

*   All values are between 0 and 1
    
*   The sum equals 1
    
*   The highest value is the prediction
    

Training and Backpropagation
----------------------------

The model is trained using:

*   **Cross-entropy loss**
    
*   **Gradient descent**
    

### Output Gradient

The gradient for the output layer is:

`   dz[i] = z[i] - (i == train_y[n]);   `

This measures how wrong the prediction is.

### Update Output Layer

Weights and biases are updated to reduce the loss:

`   b2[i] -= LR * dz[i];  W2[i][j] -= LR * dz[i] * h[j];   `

### Hidden Layer Gradient

The error is propagated back to the hidden layer.ReLU derivative is applied.

`   dh[j] = h[j] > 0 ? sum : 0;   `

If the neuron is inactive, its gradient is zero.

### Update Input → Hidden Layer

`   b1[j] -= LR * dh[j];  W1[j][k] -= LR * dh[j] * train_x[n][k];   `

Training Loop
-------------

*   All training samples are processed
    
*   One full pass over the dataset is called **one epoch**
    
*   Training runs for **5000 epochs**
    
*   Each epoch improves the model gradually
    

Testing
-------

After training, a new **4 × 3 input matrix** is passed to the network.

The output is a probability distribution:

`   UP   : 0.92  MID  : 0.06  DOWN : 0.02   `

The class with the highest value is the final prediction.

Why This Project Exists
-----------------------

*   To understand neural networks from scratch
    
*   No frameworks
    
*   No hidden abstraction
    
*   Full visibility of the math
    
*   Beginner friendly
    

Notes
-----

* This code is not optimized

* This is not production-ready

* This project is for educational purposes only

*   This code is not optimized
    
*   This is not production-ready
    
*   This project is for educational purposes only


