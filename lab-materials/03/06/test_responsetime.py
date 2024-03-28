from json import dump
from time import perf_counter

from llm_usage import infer_with_template


max_response_time = 3


def test_responsetime(inference_url=''):
    TEMPLATE = """<s>[INST] <<SYS>>
Answer below truthfully and in less than 10 words:
<</SYS>>
{silly_question}
[/INST]"""

    start = perf_counter()
    infer_with_template(
        "Who saw a saw saw a salsa?", TEMPLATE, inference_url=inference_url
    )
    response_time = perf_counter() - start

    assert (response_time <= max_response_time), \
        f"Response took {response_time} which is greater than {max_response_time}"

    print(f"Response time was OK at {response_time} seconds")

    with open("responsetime_result.json", "w") as f:
        dump({"response_time": response_time}, f)


if __name__ == '__main__':
    test_responsetime()