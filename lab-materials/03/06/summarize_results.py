from json import dump, load


def summarize_results():
    results = {}
    result_files = [
        'responsetime_result.json',
        'quality_result.json',
    ]
    for result_file in result_files:
        with open(result_file, 'r') as input_file:
            results.update(load(input_file))

    print(f'Aggregated results:\n{results}')

    with open("results.json", "w") as f:
        dump(results, f)


if __name__ == '__main__':
    summarize_results()