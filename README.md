# GroundHog


This repo contains code for the paper: [GroundHog: Dialogue Generation using Multi-Grained Linguistic Input](https://aclanthology.org/2024.codi-1.14/)

### Installation

Works for the python3.7+ environment.

```bash
pip install requirements.txt
```

### Code structure 
1) ```GroundHog Prepare.ipynb``` contains data preprocessing and constructs trains/test splits based on the dialodues and liguistic parsing results.
2) ```GroundHog Train Infer Cycles.ipynb``` contains train runs using various combinations of the input and output features alongwith the infrence examples. Uses ```train_eval_scripts_v2.py``` script.
3) ```Base BART.ipynb``` contains the base BART training, including data preparation, train cycle (with token weights) and inference.
4) ```Quality Estimation.ipynb``` contains evaluation scripts for the selected metrics calculation (ROUGE- and BLEU-based).

Code in folders:
1) ```custom_bart_scripts``` - contains code for the BART training with token weights (see also https://github.com/alchernyavskiy/discourse_mpc_generation)
2) ```multienc_bart_scripts_v2``` contains scripts for the GroundHog training and inference:
- ```modeling_custom_bart.py``` contains the GroundHod architecture (see Figure Architecture). Modifies base ```modeling_bart.py``` adding the **BartMultiEncoderModel** class.
- ```run_summarization.py``` is used to run the training cycle for the GroundHog model.
- ```generation_utils_custom.py``` is used during the inference stage

**Note**: BartMultiEncoderModel class can be also used for other purposes.


### Model Architecture
The GroundHog architecture comprises individual BART encoders for each
input text, which are subsequently aggregated and used as input for the BART decoder. To reduce the dimensionality
of the inputs, a 1D convolutional layer is applied to all inputs except the main input. The shared embedding layer is
denoted by the gear icon. In addition, intermediate tensor dimensions are indicated (batchsize is denoted as bs).

![Architecture](https://github.com/alchernyavskiy/GroundHog/blob/main/architecture.png?raw=true)

 
### Citation
If you find this repository helpful, feel free to cite our publication:

```
@inproceedings{chernyavskiy-etal-2024-groundhog,
    title = "{G}round{H}og: Dialogue Generation using Multi-Grained Linguistic Input",
    author = "Chernyavskiy, Alexander  and
      Ostyakova, Lidiia  and
      Ilvovsky, Dmitry",
    editor = "Strube, Michael  and
      Braud, Chloe  and
      Hardmeier, Christian  and
      Li, Junyi Jessy  and
      Loaiciga, Sharid  and
      Zeldes, Amir  and
      Li, Chuyuan",
    booktitle = "Proceedings of the 5th Workshop on Computational Approaches to Discourse (CODI 2024)",
    month = mar,
    year = "2024",
    address = "St. Julians, Malta",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.codi-1.14",
    pages = "149--160",
    abstract = "Recent language models have significantly boosted conversational AI by enabling fast and cost-effective response generation in dialogue systems. However, dialogue systems based on neural generative approaches often lack truthfulness, reliability, and the ability to analyze the dialogue flow needed for smooth and consistent conversations with users. To address these issues, we introduce GroundHog, a modified BART architecture, to capture long multi-grained inputs gathered from various factual and linguistic sources, such as Abstract Meaning Representation, discourse relations, sentiment, and grounding information. For experiments, we present an automatically collected dataset from Reddit that includes multi-party conversations devoted to movies and TV series. The evaluation encompasses both automatic evaluation metrics and human evaluation. The obtained results demonstrate that using several linguistic inputs has the potential to enhance dialogue consistency, meaningfulness, and overall generation quality, even for automatically annotated data. We also provide an analysis that highlights the importance of individual linguistic features in interpreting the observed enhancements.",
}
```
