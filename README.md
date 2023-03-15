# NER_card_prediction

* 初始化
``python -m spacy init fill-config ./base_config.cfg ./config.cfg``

* 將pickel檔轉換成spacy檔案格式
``python preprocess.py``

* train model
``python -m spacy train .\config.cfg --output .\output\ --paths.train .\data\train.spacy --paths.dev .\data\test.spacy``
