# Positional Bias in AI Alignment Debates
## Introduction
* This repository is my capstone project for the [AI Alignment Course](https://aisafetyfundamentals.com/alignment/) offered by Blue Dot Impact, where I replicate some of the experiments described in the paper, [Debating with More Persuasive LLMs Leads to More Truthful Answers](https://arxiv.org/abs/2402.06782). 
* Here, I report that using **gpt-4o-mini-2024-07-18** as an AI judge in a debate setting leads to a positional bias in favor of the option presented second, and in favor of the first agent to speak.
  1. You can read more details about the experiments I conducted in this [report](report.md). 
  2. I have created a [web app](https://quality-data-debate-app.onrender.com/) where you can view the debate transcripts for all experiments I performed and get a sense of how different debate protocols work.

## Looking for an easier way to implement debate protocols for your research?
* I used [AutoGen](https://microsoft.github.io/autogen/stable/index.html) to create scaffolding in the form of prompts that guide LLM agents through each round of debate. [AutoGen](https://microsoft.github.io/autogen/stable/index.html) makes creation of this scaffolding much easier to implement, and my hope is that researchers in this area might use this code as a starting point for implementing their own experiments using an established framework, rather than starting from scratch. 
* It is worth noting that, as far as I can tell, using this scaffolding I was able to fully address "self-defeating" behavior, where a model concedes to the opposing side, as reported in [Debating with More Persuasive LLMs Leads to More Truthful Answers](https://arxiv.org/abs/2402.06782).
* While some of this may be due to the use of a more modern LLM model than the one used in the original paper, I myself saw frequent self-defeating behavior when I first started experimenting with debate protocols, and I believe that the scaffolding facilitated by AutoGen was instrumental in eliminating this behavior. 
* The implementations of the various debate protocols can be found in their corresponding files [here](debate-for-ai-alignment/src/debate_for_ai_alignment/pipelines/debate).  
* This project was created using a [Kedro](https://kedro.org/) project template, which I use to orchestrate the data pipelines for the project. I highly recommend using Kedro for your own projects, as it offers a standardized way to structure your code and data for all your data science projects.

## Running the Experiments in this Repository
* If you are familiar with Kedro, you can clone this repository and it should be relatively straightforward from there to run the experiments I conducted.
* If you are not familiar with Kedro, I will soon add a quick start guide to help you get started.
