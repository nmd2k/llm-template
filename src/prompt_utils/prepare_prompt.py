import random
# ========== Breadth Prompt ==========
breadth_instruction = "I want you act as a Prompt Creator.\r\n\
Your goal is to draw inspiration from the #Given Prompt# to create a brand new prompt.\r\n\
This new prompt should belong to the same domain as the #Given Prompt# but be even more rare.\r\n\
The LENGTH and complexity of the #Created Prompt# should be similar to that of the #Given Prompt#.\r\n\
The #Created Prompt# must be reasonable and not contain any example, task input or code snippet.\r\n\
The prompt must be understood and responded by humans.\r\n\
'#Given Prompt#', '#Created Prompt#', 'given prompt' and 'created prompt' are not allowed to appear in #Created Prompt#\r\n"

def createBreadthPrompt(instruction):
    prompt = breadth_instruction
    prompt += "#Given Prompt#: \r\n {} \r\n".format(instruction)
    prompt += "#Created Prompt#: \r\n"
    return prompt

# ========== Depth Prompt ==========
depth_instruction = "I want you act as a Prompt Rewriter.\r\n \
Your objective is to rewrite a given prompt into a more complex version to make those famous AI systems (e.g., chatgpt and GPT4) a bit harder to handle.\r\n \
But the rewritten prompt must be reasonable and must be understood and responded by humans.\r\n \
Your rewriting cannot omit the non-text parts such as the table and code in #The Given Prompt#:. Also, please do not omit the input in #The Given Prompt#. \r\n \
You SHOULD complicate the given prompt using the following method: \r\n\
{} \r\n\
You should try your best not to make the #Rewritten Prompt# become verbose, #Rewritten Prompt# can only add 10 to 20 words into #The Given Prompt#. \r\n\
'#The Given Prompt#', '#Rewritten Prompt#', 'given prompt' and 'rewritten prompt' are not allowed to appear in #Rewritten Prompt#\r\n"

methods = {
    "constraints": "Please add one more constraints/requirements into '#The Given Prompt#'",
    "deepen": "If #The Given Prompt# contains inquiries about certain issues, the depth and breadth of the inquiry can be increased.",
    "concretize": "Please replace general concepts with more specific concepts.",
    "reasoning": "If #The Given Prompt# can be solved with just a few simple thinking processes, you can rewrite it to explicitly request multiple-step reasoning.",
}
def createDepthPrompt(instruction):
    method = random.choice(list(methods.values()))
    prompt = depth_instruction.format(method)
    prompt += "#The Given Prompt#: \r\n {} \r\n".format(instruction)
    prompt += "#Rewritten Prompt#:\r\n"
    return prompt

# ========== Multiple-answer prompt ==========
multi_answer_instruction = "I want you act as a Prompt Creator.\r\n\
Your objective is to rewrite a given prompt into a multiple-choices format question to challenge famous AI systems (e.g., chatgpt and GPT4).\r\n\
But the rewritten prompt must be reasonable, able to understood and responded by humans.\r\n\
The rewritten prompt should have at least four answers, which is detail, consise and suitable for multiple-choice format with only one being the correct answer.\r\n\
#Given Prompt#: \r\n\
{} \r\n\
#Rewriten Prompt#:\r\n"

def createMultiAnswerQAPrompt(instruction):
    return multi_answer_instruction.format(instruction)

# ========== True/False prompt ==========
truefalse_instruction = "I want you to act as a Prompt Creator.\r\n\
Your objective is to rewrite a given prompt into a context and list of true/false statements exam.\r\n\
But the rewritten context must be detailed, self-explanatory, and informative.\r\n\
The rewritten prompt should have at least four statements that express the context of the prompt.\r\n\
#Given Prompt#:\r\n\
{} \r\n\
#Rewriten Prompt#:\r\n"

def createTruefalsePrompt(instruction):
    return truefalse_instruction.format(instruction)