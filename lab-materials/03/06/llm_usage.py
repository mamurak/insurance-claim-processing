from os import environ

from langchain.chains import LLMChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.evaluation import load_evaluator
from langchain.llms import VLLMOpenAI
from langchain.prompts import PromptTemplate

from certs import prepare_certs


INFERENCE_SERVER_URL = environ.get('llm_endpoint_url')
MAX_NEW_TOKENS = 512
TOP_P = 0.95
TEMPERATURE = 0.01
PRESENCE_PENALTY = 1.03


def infer_with_template(input_text, template, inference_url=''):
    inference_url = inference_url or INFERENCE_SERVER_URL
    prepare_certs(inference_url)
    llm = VLLMOpenAI(
        openai_api_key="EMPTY",
        openai_api_base=f"{inference_url}/v1",
        model_name="/mnt/models/",
        max_tokens=MAX_NEW_TOKENS,
        top_p=TOP_P,
        temperature=TEMPERATURE,
        presence_penalty=PRESENCE_PENALTY,
        streaming=False,
        verbose=False,
    )

    PROMPT = PromptTemplate.from_template(template)

    llm_chain = LLMChain(llm=llm, prompt=PROMPT)

    return llm_chain.run(input_text)


def similarity_metric(predicted_text, reference_text):
    embedding_model = HuggingFaceEmbeddings()
    evaluator = load_evaluator("embedding_distance", embeddings=embedding_model)
    distance_score = evaluator.evaluate_strings(
        prediction=predicted_text, reference=reference_text
    )
    return 1-distance_score["score"]


if __name__ == '__main__':
    with open('example_text.txt') as f:
        input_text = f.read()

    with open('template.txt') as f:
        template = f.read()

    infer_with_template(input_text, template)