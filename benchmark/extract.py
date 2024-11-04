import os, json, gzip, argparse, pandas as pd

def get_argparser() -> argparse.ArgumentParser:
    """
    Get the configured argument parser
    """
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--output_filepath', '-o',
                        metavar='NAME',
                        dest='output_filepath',
                        default='results.json',
                        type=str,
                        help='Path to the output file')

    required = parser.add_argument_group('required arguments')
    required.add_argument('--results_directory', '-i',
                        metavar='PATH',
                        dest='results_directory',
                        required=True,
                        type=str,
                        help='Path to the directory containing the results of the benchmark')
    return parser

def read_gz_file(path):
    with gzip.open(path, 'rt') as f:
        return json.load(f)

if __name__ == '__main__':   

    # Read arg parameters
    parser = get_argparser()
    args = parser.parse_args()

    # Extract results
    results_df = pd.DataFrame(columns=['problem', 'prompt', 'prediction', 'passed'])
    for root, dirs, files in os.walk(args.results_directory):
        for file in files:
            if file.endswith('.results.json.gz'):
                filepath = os.path.join(root, file)
                data = read_gz_file(filepath)

                problem = data['name']
                prompt = data['prompt']
                tests = data['tests']

                result = data['results'][0]
                prediction = result['program'][len(prompt):-len(tests)]
                passed = True if result['status'] == 'OK' else False

                new_row = pd.DataFrame([
                    {'problem': problem, 
                     'prompt': prompt, 
                     'prediction': prediction, 
                     'passed': passed}
                    ])
                results_df = pd.concat([results_df, new_row], ignore_index=True)
    
    # Export results
    results_df['problem_num'] = results_df['problem'].str.extract('(\d+)').astype(int)
    results_df = results_df.sort_values(by='problem_num').drop(columns='problem_num')
    results_df.to_json(args.output_filepath, orient='records')
                