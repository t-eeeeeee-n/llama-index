FROM python:3.11

# アプリケーションディレクトリを作成する
WORKDIR /var/www/gpt

# pipのアップデート
RUN pip install --upgrade pip

# pipでインストールしたいモジュールをrequirements.txtに記述しておいて、
# コンテナ内でpipにインストールさせる
# requirements.txtの書き方は[pip freeze]コマンドから参考に出来る
COPY ./container/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# アプリケーションコードをコンテナにコピー
COPY ./src /var/www/gpt

#CMD [ "python", "main.py" ]
