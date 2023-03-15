# NER_card_prediction

實作流程：
1. 透過pytesseract辨識名片中的文字
2. 使用BIO方式標記訓練資料：
    * Name：NAME
    * Designation：DES
    * Organization：ORG
    * Phone Number：PHONE
    * Email Address：EMAIL
    * Website：WEB
3. 資料清整
4. 透過spacy訓練NER模型，整理非結構化資料
   * 初始化
   ``python -m spacy init fill-config ./base_config.cfg ./config.cfg``

   * 將pickel檔轉換成spacy檔案格式
   ``python preprocess.py``

   * train model
   ``python -m spacy train .\config.cfg --output .\output\ --paths.train .\data\train.spacy --paths.dev .\data\test.spacy``
