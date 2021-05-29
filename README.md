# is-fake

Purpose of this repository is to store code and notebooks for my master's thesis.

Purpose of this project is to build a classifier that can distinguish fake news with high accuracy. 

### Detection

Detection was based on text features (NLP analysis). Also fake news propagation was analysed.

### Datasets

All datasets are gathered here:

https://drive.google.com/drive/folders/1AMYyQXlme7CtBtVzrs8_OOJeplNI_f5f?usp=sharing

### Thesis
The paper contains results collected according to my best knowledge in 2020/2021.

### Development

```bash
$ cd is-fake
$ pyenv install 3.8.0
$ pyenv virtualenv 3.8.0 is-fake
```

#### Configuring PyCharm to use the virtual environment
- Preferences (eg. MacOS: `⌘` + `,`)
- Project: `is-fake` / Project interpreter
- Click on the "gear" icon, and then select "Add"
- Virtualenv environment
- Existing environment
- Choose Python 3.8 (fake) that should me located in `~/.pyenv/versions/is-fake/bin/python3`


```bash
$ pip install --upgrade pip
$ pip install -r requirements.txt
```
