# Evaluating Large Language Models with Diverse Prompting Strategies

The goal of this assignment is to systematically evaluate and compare open-source Large Language Models across diverse task types using different prompting techniques.

1. Installed ollama and pulled models:
    - Small standard model: ollama pull **qwen2.5:1.5b**
    - Large reasoning model: ollama pull **deepseek-r1:7b**


# Key assignment considerations
1. Prepare a task description (to be used for every prompt for a given task)
2. Before prompting choose evaluation criteria for a given task (but ensuring the same scale/rubric for all tasks 0-3)
3. Development set for prompt engineering - to check whether the task descriptions (so prompts) work fine / to test their different versions and compare outputs - prompt engineering here is modyfying the prompt to observe the response and therefore choose wihch one worked "best".
4. Evaluation examples should not have been used in the prompt engineering phase for fair evaluation (scoring).


# Experimental setup:
- from the terminal generate outputs with the models using two/three different prompts for a given task
- compare outputs, save the prompt that behaved better to the respective json file in `prompts/` 
- save the tested prompts to `prompt_notes.md`

This is considered to be the prompt engineering documentation part. Afterwards,
- run `run_eval.py` to get outputs for all tasks for the evaluation set
- using the metrics defined in `scoring.md`, assign scores to each output
- evaluate the results quantitatively and qualitatively

# Task descriptions and examples
The `prompts` folder contains .json files for each of the 10 evaluation tasks:

1. Instruction Following (IFEval-style)
2. Logical Reasoning
3. Creative Writing
4. Code Generation
5. Reading Comprehension
6. Common Sense Reasoning
7. Language Understanding & Ambiguity
8. Factual Knowledge & Retrieval
9. Mathematical Problem Solving
10. Ethical Reasoning & Nuance

Example task json file contents:
```json
{
  "task_name": "Ethical Reasoning & Nuance",
  "description": "Analyze the following ethical scenario. Identify the main ethical issues, discuss relevant principles, and provide a balanced response and/or solution to the problem. Keep the answer concise but not simplified - do not assume that the reader knows formal principles of ethics.",

  "dev_examples": [
    {
      "input": "A company discovers a serious security flaw in its product, but disclosing it immediately could cause financial damage. What should the company do?",
      "output": "The company faces a conflict between financial interests and responsibility to users. Ethical principles such as non-maleficence and transparency suggest that protecting users should take priority. While disclosure may cause short-term losses, withholding information risks greater harm and loss of trust. A responsible approach would involve informing users while working to mitigate the issue."
    },
    {
      "input": "A doctor must decide which of two patients receives a life-saving treatment when resources are limited. How should this decision be approached?",
      "output": "This scenario involves ethical tension between fairness and maximizing benefit. Principles such as justice, medical need, and likelihood of recovery are often considered. No choice is ethically neutral, so transparent criteria and institutional guidelines are important. The decision should aim to minimize harm while respecting professional ethics."
    }
  ],

  "eval_example": {
    "input": "A university offers scholarships exclusively for women in STEM fields to address historical underrepresentation. Some students argue this promotes fairness and inclusion, while others claim it is discriminatory against equally qualified men. How should this policy be evaluated ethically?"
  }
}
```

# Prompt engineering
A simplified pipeline of prompt engineering was used - I used just the smaller model and then only tested alternative prompts if the initial ones didn't provide satisfactory results. 

The final prompts (used in evaluation) are defined in the `prompts` folder. During the **prompt engineering** phase one or two alternative prompts were tested (on dev examples) and the one that provided the "nicest" output was added to the .json files in `prompts/` and were later applied to the validation examples.

How to run:
```
ollama run qwen2.5:1.5b "Analyze the following ethical scenario. Identify the main ethical issues, discuss relevant principles, and provide a balanced response and/or solution to the problem. Keep the answer concise (4–6 sentences).

A company discovers a serious security flaw in its product, but disclosing it immediately could cause financial damage. What should the company do?"
```

**Observations:**
- Prompt engineering (and documenting it) is a non-trivial task - for some tasks including more instructions ensured better behaviour, but sometimes the outputs proved worse than for an initial prompt with less constraints. Sometimes for "improved prompts" differences in generated outputs varied probably mostly due to it being a different (new) run rather than a better prompt:
- **for example:** for ethical scenarios - when I included "keep the answer concise" deepseek was returning answers that were (in my opinion) too abstract / difficult to understand without prior knowledge
- After reading the outputs I often had ideas for improved prompts - like for code I'd add "make sure it's efficient". 
- It's a difficult task because often we know what we expect from the output only once we've seen it - maybe that's how few-shot leanring could also be helpful to us - we have to imagine what the output we expect would be, so it might help us write a better prompt (e.g. describing how we want our code to be documented)


# Running the experiments
Once the final prompts for each task have been established in the respective .json files in `prompts` the `run_eval.py` script can be used. It implements a unified evaluation pipeline that takes a single task definition in JSON format and evaluates it across multiple models and prompting strategies. For each task, the script automatically constructs zero-shot, few-shot, and (where applicable) chain-of-thought prompts, queries the specified models via the Ollama API, and records both the generated responses and basic runtime metadata to `results` as `raw.jsonl`. This file can be later exported to csv to manually score models' outputs based on criteria defined in `scoring.md`.

How to run
```python
python scripts/run_eval.py --task prompts/ethical_reasoning.json
```

# Scoring
All tasks are scored using a 0–3 scale per criterion.

Score interpretation:
- 0 = incorrect, missing, or unusable
- 1 = partially correct, shallow, or flawed
- 2 = mostly correct, adequate
- 3 = fully correct, high quality

Each task defines one task-specific criterion in `scoring.md`. Using one criterion per task avoids evaluating properties that were not explicitly elicited by the prompt and ensures consistent, fast, and defensible scoring across models and prompting strategies (it also allows for completing this assignment quicker).

**Observations:**
It's really difficult to score some outputs - like the ethical reasoning task - the scoring itself can be biased, so can the response. Similarly - scoring creative writing can easily become subjective (what we like more). Therefore, to obtain meaningful benchmarks, the scoring schemes would have to be refined and precise. And the prompts especially for few-shot learning should be optimised to what exactly we would consider the "perfect output".

# Quantitative analysis 
Unfortunately due to hardware constraints querying deepseek sometimes returned an empty string. These entries were skipped in the evaluations (not calculated towards mean scores). A detailed report can be seen in the ipynb file `qualitative_quantitative_analysis.ipynb`. A summary of the results
can be found below:

### Missing deepseek outputs
The empty string outputs were logged for the following tasks: 
<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>task_name</th>
      <th>model</th>
      <th>strategy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4</th>
      <td>Code Generation</td>
      <td>deepseek-r1:7b</td>
      <td>few_shot</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Creative Writing</td>
      <td>deepseek-r1:7b</td>
      <td>zero_shot</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Ethical Reasoning &amp; Nuance</td>
      <td>deepseek-r1:7b</td>
      <td>zero_shot</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Ethical Reasoning &amp; Nuance</td>
      <td>deepseek-r1:7b</td>
      <td>few_shot</td>
    </tr>
    <tr>
      <th>29</th>
      <td>Instruction Following</td>
      <td>deepseek-r1:7b</td>
      <td>few_shot</td>
    </tr>
  </tbody>
</table>
</div>

- The ethics task used a larger number of tokens for the input as compared to other tasks, which could potentially be the problem here.
- The rest of the tasks could also be considered slightly more complex to the ones that did not return empty outputs. 

### Mean scores per model and prompting strategy across different tasks.
<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>model</th>
      <th>strategy</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>deepseek-r1:7b</td>
      <td>few_shot</td>
      <td>2.43</td>
    </tr>
    <tr>
      <th>1</th>
      <td>deepseek-r1:7b</td>
      <td>zero_shot</td>
      <td>2.50</td>
    </tr>
    <tr>
      <th>2</th>
      <td>qwen2.5:1.5b</td>
      <td>cot</td>
      <td>2.00</td>
    </tr>
    <tr>
      <th>3</th>
      <td>qwen2.5:1.5b</td>
      <td>few_shot</td>
      <td>2.40</td>
    </tr>
    <tr>
      <th>4</th>
      <td>qwen2.5:1.5b</td>
      <td>zero_shot</td>
      <td>2.10</td>
    </tr>
  </tbody>
</table>
</div>


- The scores indicate that Deepseek using the zero-shot prompting strategy performed the best under the defined metrics. However, the missing outputs mentioned above could be influencing the results. 
- For Qwen few-shot seems to significantly outperform the zero-shot and CoT strategies. This observation is consistent with the findings of *"Language Models are Few-Shot Learners" (Brown et al., 2020)*, which show that **providing a small number of input–output examples** enables models to infer the task structure directly from local context, effectively performing **in-context learning without any parameter updates**.

### Mean scores per model and task:
<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>task_name</th>
      <th>model</th>
      <th>mean_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Code Generation</td>
      <td>deepseek-r1:7b</td>
      <td>3.00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Code Generation</td>
      <td>qwen2.5:1.5b</td>
      <td>2.33</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Common Sense Reasoning</td>
      <td>deepseek-r1:7b</td>
      <td>2.50</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Common Sense Reasoning</td>
      <td>qwen2.5:1.5b</td>
      <td>1.67</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Creative Writing</td>
      <td>deepseek-r1:7b</td>
      <td>2.00</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Creative Writing</td>
      <td>qwen2.5:1.5b</td>
      <td>2.00</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Ethical Reasoning &amp; Nuance</td>
      <td>qwen2.5:1.5b</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Factual Knowledge &amp; Retrieval</td>
      <td>deepseek-r1:7b</td>
      <td>3.00</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Factual Knowledge &amp; Retrieval</td>
      <td>qwen2.5:1.5b</td>
      <td>3.00</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Instruction Following</td>
      <td>deepseek-r1:7b</td>
      <td>2.00</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Instruction Following</td>
      <td>qwen2.5:1.5b</td>
      <td>1.33</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Language Understanding &amp; Ambiguity</td>
      <td>deepseek-r1:7b</td>
      <td>1.50</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Language Understanding &amp; Ambiguity</td>
      <td>qwen2.5:1.5b</td>
      <td>1.67</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Logical Reasoning</td>
      <td>deepseek-r1:7b</td>
      <td>3.00</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Logical Reasoning</td>
      <td>qwen2.5:1.5b</td>
      <td>3.00</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Mathematical Problem Solving</td>
      <td>deepseek-r1:7b</td>
      <td>3.00</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Mathematical Problem Solving</td>
      <td>qwen2.5:1.5b</td>
      <td>3.00</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Reading Comprehension</td>
      <td>deepseek-r1:7b</td>
      <td>2.00</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Reading Comprehension</td>
      <td>qwen2.5:1.5b</td>
      <td>2.67</td>
    </tr>
  </tbody>
</table>
</div>
- The per-task results reveal differences in how model size and prompting strategy interact with task type. 
- DeepSeek consistently achieves the highest scores on tasks that require multi-step reasoning or precise internal consistency, such as logical reasoning, mathematical problem solving, and code generation.
- Ethical reasoning achieved the lowest score of all tasks (only for Qwen since the output was empty for Deepseek). This may be due to the evaluation being inherently subjective, but also reflects that ethical reasoning tasks may be sensitive to prompt phrasing.
- In addition, ethics-related tasks are particularly prone to inductive bias, as models may implicitly favor certain normative perspectives. 
- In this experiment, the generated outputs often appeared shallow relative to the complexity of the scenario or biased toward a single viewpoint, failing to explicitly consider alternative ethical principles or trade-offs.

### Final model comparison across strategies and tasks:
<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>model</th>
      <th>mean_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>deepseek-r1:7b</td>
      <td>2.47</td>
    </tr>
    <tr>
      <th>1</th>
      <td>qwen2.5:1.5b</td>
      <td>2.17</td>
    </tr>
  </tbody>
</table>
</div>

- The final score is in favour of the larger model. However, there is a trade-off of the performance we can get (for some tasks similar or even better for the smaller model) and the computation costs. 
- Therefore model choice is dependent on the needs for a given application
- Prompting strategies (like few-shot) can improve model's performance and might be enough of an improvement when using a smaller model as opposed to switching to a bigger (and more resource-demanding) model

# Final observations:
- I like the interface that ollama exposes for language models - it's easy to compare their behaviours because of the unified interface
- Different runs generate different prompts, so the evaluation might not be as accurate - maybe it would have to be averaged - a couple of runs per each task (like 5x zero-shot for for example creative_writing.json) 
- I feel like the evaluation process is non-trivial since it involves a lot of personal judgment - I think biases can easily be encoded in the prompts, then in the evaluations themsevels - It is not straightforward to be objective (because what is objectivity?) so especially for the ethical / common sense tasks the scoring can turn out to be biased