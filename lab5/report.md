1. Install ollama and pull models:
    - Small standard model: ollama pull qwen2.5:1.5b
    - Large reasoning model: ollama pull deepseek-r1:7b

[under construciton :)]

# Understanding the assignment
1. prepare a task description (to be used for every prompt for a given task)
2. before prompting choose evaluation criteria for a given task (but ensuring the same scale/rubric for all tasks 0-3)
3. Development set for prompt engineering - to check whether the task descriptions (so prompts) work fine / to test their different versions and compare outputs - prompt engineering here is modyfying the prompt to observe the response and therefore choose wihch one worked "best"
- Might be that the conclusions here will be that if a prompt is short and vague the result is not great but if it's specific - like tone, approach, formatting instructions and/or length specification - the output is much better for a given task
4. Evaluation examples should not have been used in the prompt engineering phase for fair evaluation (scoring).
5. Running all experiments 


# Experimental seetup:
- generate outputs with the models using two/three different prompts for a given task
- compare outputs, save the prompt that behaved better to the json file
- save the tested prompts to `prompt_notes.md`
This is considered to be the prompt engineering documentation part. Afterwards,
- run `run_eval.py` to get outputs for all tasks
- using the metrics defined in `scoring.md`, assign scores to each output
- evaluate the results quantitavely and qualitatively

# init questions to answer in conclusions
Which tasks benefit most from few-shot?
Does CoT help the small model uniformly?
Where does the reasoning model outperform despite no CoT prompt?
Where does it not?
Did the reasoning model show more nuance?
Were there trade-offs (verbosity vs clarity)?


# prompt engineering comment
“Prompt templates were refined using a small development set of 2–3 examples per task type. Initial prompts were adjusted to improve clarity and output format consistency (e.g. explicitly requesting concise answers or step-by-step reasoning). The final prompts were then fixed and applied to unseen evaluation examples.”

Locally run:
```
ollama run qwen2.5:1.5b "Analyze the following ethical scenario. Identify the main ethical issues, discuss relevant principles, and provide a balanced response and/or solution to the problem. Keep the answer concise (4–6 sentences).

A company discovers a serious security flaw in its product, but disclosing it immediately could cause financial damage. What should the company do?"
```

It's non-trivial - (I suppose) for some tasks including more instructions ensured better behaviour - for some the difference in outputs varied probably mostly due to it being a different run rather than a better prompt:
- for ethical scenarios - when I included "keep the answer concise deepseek was returning answers in my opiinon too abstract / diffcult to understand without prior knowledge

# random observations:
- I like how ollama allows for querying models 
- this is very non-trivial since it involves a lot of personal judgment - I think biases can easily be encoded in the prompts, then in the evaluations themsevels - it is not obvious to be objective (because what is objectivity?) so especially for the ethical / common sense tasks the scoring can be biased

# How to run
```python
python scripts/run_eval.py --task prompts/ethical_reasoning.json
```

# Prompt engineering
The final prompts (used in evaluation) are defined in `prompts/`. During the **prompt engineering** phase one or two alternative prompts were tested (on dev examples) and the one that provided the "nicest" output was added to the .json files in `prompts/` and were later applied to the validation examples.

# TODO: add a description of what happens in `run_eval.py`
