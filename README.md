# Julia Copilot - Fine-Tuning and Benchmarking

Students: Arnaud Fauconnet, Lorenzo Varese, Vittorio Perozzi

This repository focuses on fine-tuning and evaluating the Julia Copilot model using curated datasets and robust benchmarking techniques. It encompasses data preparation, clustering attempts, benchmarking evaluations, and training scripts for the model.

## Repository Structure

```
.
├── README.md                 
├── benchmark/                # Evaluation and benchmarking scripts and results
│   └── ...
├── clustering-attempt/       # Clustering experiments
│   ├── clustering_metrics2_20_all.png
│   ├── clustering_metrics2_20_without_keyword.png
│   └── report_clustering.md  
├── data/                     
│   ├── combined_projects.zip # Julia project and function datasets
│   └── julia.csv.gz          # CSV data related to each repository
├── doc/                      
│   └── requirements.pdf
├── requirements.txt          
└── src/                     
    ├── clone.py              # Script to clone and clean GitHub repositories
    ├── encode_data.py        # Data encoding script
    ├── parse_re.py           # Regex-based parsing utility
    ├── parse_ts.py           # Tree-sitter parsing utility
    ├── query.py              # Query generation and handling
    ├── stat_test.py          # Statistical tests for analysis
    └── train.py              # Training script for fine-tuning the model
```

## How to Run the Tool

This project uses GitHub repositories marked with Julia as the primary language. Repositories are cloned and cleaned (removing non-Julia files). Due to file size limitations, the pre-cloned and cleaned dataset (350 MB approx.) cannot be included directly in this repository. However, you can download the data directly from the provided server.

### Steps to Run

1. **Download Pre-cloned Data**  
   To avoid re-cloning repositories manually, download the preprocessed data using the following command (assuming you are in the root folder of the repository):

   ```bash
    scp <YOUR_USER>@gym.si.usi.ch:/home/SA24-G3/project2/data/repos.zip data/
    ```

2. **Install Dependencies**  
   Install the necessary dependencies by running:

   ```bash
   pip install -r requirements.txt
   ```

3. **Execute the Scripts**  
   - To clone and preprocess repositories: `python src/clone.py`
   - To train the model: `python src/train.py`
   - To benchmark the model: Use `benchmark/evaluate.sh`

## Training Parameters

| Parameter            | Default Value                     | Description                                                                                     |
|----------------------|-----------------------------------|-------------------------------------------------------------------------------------------------|
| `--model`            | `HuggingFaceTB/SmolLM-135M`      | The pre-trained model to use for training.                                                     |
| `--quantized`        | `False`                          | Flag to enable model quantization using BitsAndBytes.                                          |
| `--first-line`       | `False`                          | Extract only the first line of the documentation as input.                                     |
| `--frac-of-data`     | `1`                              | Fraction of the dataset to use for training. Use a value between 0 and 1 for partial datasets. |
| `--batch-size`       | `2`                              | Batch size for training. Increase this if your GPU supports larger batches.                    |
| `--encoded-data-root`| `data/encoded_data`              | Path to the encoded dataset.                                                                   |

### Example Usage

```bash
python train.py --model HuggingFaceTB/SmolLM-135M --quantized --frac-of-data 0.35 --batch-size 4
```

This command trains the model with quantization enabled, using 35% of the dataset, and a batch size of 4.