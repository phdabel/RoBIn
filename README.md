# RoBIn: A Transformer-based Model for Risk of Bias Inference with Machine Reading Comprehension

RoBIn - Risk of Bias Inference.

This work is under review at the Journal of Biomedical Informatics. A preliminary version is available as:

```bibtex
@misc{Dias+2024,
      title={RoBIn: A Transformer-Based Model For Risk Of Bias Inference With Machine Reading Comprehension}, 
      author={Abel Corrêa Dias and Viviane Pereira Moreira and João Luiz Dihl Comba},
      year={2024},
      eprint={2410.21495},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2410.21495}, 
}
```

## Data

The data used in this project is from the following sources:
- [PubMed](https://pubmed.ncbi.nlm.nih.gov/)
- [PubMed Central](https://www.ncbi.nlm.nih.gov/pmc/)
- [Cochrane](https://www.cochranelibrary.com/)

The RoBIn dataset is available in the `data` directory.


## Running Models
To run the models, you need to install the required packages. You can install the required packages by running the following command:
```bash
pip install -r requirements.txt
```

### SVM and LR Models

To run SVM and LR models, you need to run the following command:
```bash
python binary-svm-training.py
```

Results will be saved in the `save` directory.

### $RoBIn^{Ext}$

To run $RoBIn^{Ext}$ model, you need to run the following command:
```bash
python .\robin-training.py --model_name allenai/biomed_roberta_base --max_seq_length 512 --learning_rate 4e-5 --batch_size 16 --epochs 10 gradient_accumulation_steps 4 --l1_regularization 0.1
```

The model and results are saved in the `save` directory.
Check the `robin-training.py` file for more details on the arguments.

### $RoBIn^{Gen}$

To run $RoBIn^{Gen}$ model, you need to run the following command:
```bash
python genrobin_training.py --model_name "GanjinZero/biobart-base" --batch_size 4 --learning_rate 3e-5 --epochs 10 --max_seq_length 512 --data_folder data --dataset_name robqa --warmup_steps 0 --l1_regularization 0.1 --gradient_accumulation_steps 4
```
