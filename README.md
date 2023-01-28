# Introduction

This page contains a collation of resources and exercises on interpretability. The focus is on [`TransformerLens`](https://github.com/neelnanda-io/TransformerLens), a library maintained by Neel Nanda.

## About TransformerLens

From the description in Neel Nanda's repo:

> TransformerLens is a library for doing [mechanistic interpretability](https://distill.pub/2020/circuits/zoom-in/) of GPT-2 Style language models. The goal of mechanistic interpretability is to take a trained model and reverse engineer the algorithms the model learned during training from its weights. It is a fact about the world today that we have computer programs that can essentially speak English at a human level (GPT-3, PaLM, etc), yet we have no idea how they work nor how to write one ourselves. This offends me greatly, and I would like to solve this!
> 
> TransformerLens lets you load in an open source language model, like GPT-2, and exposes the internal activations of the model to you. You can cache any internal activation in the model, and add in functions to edit, remove or replace these activations as the model runs. The core design principle I've followed is to enable exploratory analysis. One of the most fun parts of mechanistic interpretability compared to normal ML is the extremely short feedback loops! The point of this library is to keep the gap between having an experiment idea and seeing the results as small as possible, to make it easy for **research to feel like play** and to enter a flow state. Part of what I aimed for is to make my experience of doing research easier and more fun, hopefully this transfers to you!

## Glossary

Neel recently released a [Mechanistic Interpretability Glossary](https://dynalist.io/d/n2ZWtnoYHrU1s4vnFSAQ519J), which I highly recommend you check out. It's a great resource for understanding the terminology and concepts that are used in the following pages. Don't worry if you don't understand much of the material right now; hopefully much of it will become clearer as you go through the exercises.

## Prerequisites

This material starts with a guided implementation of transformers, so you don't need to understand how they work before starting. However, there are a few things we do recommend:

### Linear algebra

This is probably the most important prerequisite. You should be comfortable with the following concepts:

- [Linear transformations](https://www.youtube.com/watch?v=kYB8IZa5AuE) - what they are, and why they matter
- How [http://mlwiki.org/index.php/Matrix-Matrix_Multiplication](matrix multiplication) works
- Basic matrix properties: rank, trace, determinant, transpose, inverse
- Bases, and basis transformations

[This video series](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab) by 3B1B provides a good overview of these core topics (although you can probably skip it if you already have a reasonably strong mathematical background).

### Neural Networks

It would be very helpful to understand the basics of what neural networks are, and how they work. The best introductory resources here are 3B1B's videos on neural networks:

* [But what is a neural network? | Chapter 1, Deep learning](https://www.youtube.com/watch?v=aircAruvnKk)
* [Gradient descent, how neural networks learn | Chapter 2, Deep learning](https://www.youtube.com/watch?v=IHZwWFHWa-w)
* [What is backpropagation really doing? | Chapter 3, Deep learning](https://www.youtube.com/watch?v=Ilg3gGewQ5U)

### Basic Python

It's important to be able to code at a reasonably proficient level in Python. As a rough guide, you should:

* Understand most (e.g. >75%) of the material [here](https://book.pythontips.com/en/latest/), up to and including chapter 21. Not all of this will be directly useful for these exercises, but reading through this should give you a rough idea of the kind of level that is expcted of you.
* Be comfortable with easy or medium [LeetCode problems](https://leetcode.com/).
* Know what vectorisation is, and how to use languages like NumPy or PyTorch to perform vectorised array operations.
    * In particular, these exercises are all based in PyTorch, so going through a tutorial like [this one](https://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html) might be a good idea (the tensors section is very important; most of the following sections would have diminishing returns from studying but might still be useful).

### Other topics

Here are a few other topics that would probably be useful to have some familiarity with. They are listed in approximately descending order of importance (and none of them are as important as the three sections above):

* Basic probability & statistics (e.g. normal and uniform distributions, independent random variables, estimators)
* Calculus (and how it relates to backpropagation and gradient descent)
* Information theory (e.g. what is cross entropy, and what does it mean for a predictive model to minimise cross entropy loss between its predictions and the true labels)
* Familiarity with other useful Python libraries (e.g. `einops` for rearranging tensors, `typing` for typechecking, `plotly` for interactive visualisations)
* Working with VSCode, and basic Git (this will be useful if you're doing these exercises from VSCode rather than from Colab)


The main prerequisite we assume in these pages is a working understanding of transformers. In particular, you are strongly recommended to go through Neel's [Colab & video tutorial](https://colab.research.google.com/github/neelnanda-io/Easy-Transformer/blob/clean-transformer-demo/Clean_Transformer_Demo_Template.ipynb#scrollTo=SKVxRKXVsgO6).

You are also strongly recommended to read the paper [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html), or watch Neel's walkthrough of the paper [here](https://www.youtube.com/watch?v=KV5gbOmHbjU). Much of this material will be re-covered in the following pages (e.g. as we take a deeper dive into induction heads in the first set of exercises), but it will still probably be useful.

Below are a collection of unstructured notes from Neel, to help better understand the Mathematical Frameworks paper. Several of these are also addressed in his glossary and video walkthrough. There's a lot here, so don't worry about understanding every last point!