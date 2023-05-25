# python-ml-template

## Getting started

### Folder structure

```bash
.                                                                                       
├── data                      <- Data dirs should NOT be checked into git.              |
│   ├── interim               <- Intermediate data that has been transformed.           |
│   ├── processed             <- The final dataset for model training.                  |- Do NOT check data files into your repo!
│   ├── raw                   <- The original data.                                     |
│   └── results               <- Data after training the model, i.e. weights.           |
├── docs                      <- Documentation for your project (e.g. markdown files).
├── .gitignore                <- Ignore files that should not be commited to git.
├── LICENSE                   <- Add your name or replace with a license of your choice.
├── main.py                   <- This file will be called when running a slurm job.
├── models                    <- Trained models, model predictions or model summaries.
├── notebooks                 <- Jupyter notebooks. E.g. 1-mm-data-exploration.ipynb.
├── README.md                 <- README file for this git repo.
├── requirements.txt          <- Pip modules to be installed. See "pip install -r".
├── RUN_FIRST.sh              <- RUN THIS FIRST! uncaches data dirs.
├── setup.sh                  <- This file will be called before *main.py* when running a slurm job.
└── src                       <- Source code for this project.
    ├── __init__.py           <- Makes src a python module.
    ├── data                  <- Scripts to download or generate data.
    ├── models                <- Scripts to train models and make predictions.
    │   ├── train_model.py   
    │   └── predict_model.py 
    └── visualisation         <- scripts to create visualisations
        └──visualise.py
```

### RUN_FIRST.sh

Please run the following command before making any changes to the directory:

```bash
bash RUN_FIRST.sh
```

This script will untrack the data and output directories, so they don't get checked into the remote repository, and then delete itself.

### Data storage

As git should not be used to store large amounts of unchanging data, please do not check image files into the repository. Once you have run **RUN_FIRST.sh** You may use the *data* directory to locally store images you have downloaded from elsewhere.

***

# Editing this README

When you're ready to make this README your own, just edit this file and use the handy template below (or feel free to structure it however you want - this is just a starting point!). Thank you to [makeareadme.com](https://www.makeareadme.com/) for this template.

## Suggestions for a good README

Every project is different, so consider which of these sections apply to yours. The sections used in the template are suggestions for most open source projects. Also keep in mind that while a README can be too long and detailed, too long is better than too short. If you think your README is too long, consider utilizing another form of documentation rather than cutting out information.

## Name

Choose a self-explaining name for your project.

## Description

Let people know what your project can do specifically. Provide context and add a link to any reference visitors might be unfamiliar with. A list of Features or a Background subsection can also be added here. If there are alternatives to your project, this is a good place to list differentiating factors.

## Badges

On some READMEs, you may see small images that convey metadata, such as whether or not all the tests are passing for the project. You can use Shields to add some to your README. Many services also have instructions for adding a badge.

## Visuals

Depending on what you are making, it can be a good idea to include screenshots or even a video (you'll frequently see GIFs rather than actual videos). Tools like ttygif can help, but check out Asciinema for a more sophisticated method.

## Installation

Within a particular ecosystem, there may be a common way of installing things, such as using Yarn, NuGet, or Homebrew. However, consider the possibility that whoever is reading your README is a novice and would like more guidance. Listing specific steps helps remove ambiguity and gets people to using your project as quickly as possible. If it only runs in a specific context like a particular programming language version or operating system or has dependencies that have to be installed manually, also add a Requirements subsection.

## Usage

Use examples liberally, and show the expected output if you can. It's helpful to have inline the smallest example of usage that you can demonstrate, while providing links to more sophisticated examples if they are too long to reasonably include in the README.

## Support

Tell people where they can go to for help. It can be any combination of an issue tracker, a chat room, an email address, etc.

## Roadmap

If you have ideas for releases in the future, it is a good idea to list them in the README.

## Contributing

State if you are open to contributions and what your requirements are for accepting them.

For people who want to make changes to your project, it's helpful to have some documentation on how to get started. Perhaps there is a script that they should run or some environment variables that they need to set. Make these steps explicit. These instructions could also be useful to your future self.

You can also document commands to lint the code or run tests. These steps help to ensure high code quality and reduce the likelihood that the changes inadvertently break something. Having instructions for running tests is especially helpful if it requires external setup, such as starting a Selenium server for testing in a browser.

## Authors and acknowledgment

Show your appreciation to those who have contributed to the project.

## License

For open source projects, say how it is licensed.

## Project status

If you have run out of energy or time for your project, put a note at the top of the README saying that development has slowed down or stopped completely. Someone may choose to fork your project or volunteer to step in as a maintainer or owner, allowing your project to keep going. You can also make an explicit request for maintainers.
