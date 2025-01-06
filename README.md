# Persona Maker powered by TempestAI 🌪️

## 概要 📖

**Persona Maker**は、LLM（大規模言語モデル）を使用して、ターゲットユーザーのペルソナを生成するためのクールなStreamlitアプリケーションです。サービスのタイトルや要件、ターゲットの性別や年齢を入力するだけで、詳細なペルソナを自動生成します。

## 特徴 ✨

- **簡単な入力フォーム**: サービスタイトル、サービス要件、ターゲットの性別、ターゲットの年齢を入力するだけでOK！
- **詳細なペルソナ生成**: 名前、年齢、性別、身長、体重、出身地、職業、趣味、性格、年収などの詳細な情報を含むペルソナを生成。
- **LangChain統合**: LangChainを使用して、リアルで説得力のあるペルソナを生成。
- **クールなUI**: 見た目もクールで使いやすいインターフェース。

## インストール 🛠️

以下の手順に従って、ローカル環境にインストールしてください。

1. リポジトリをクローンします。

    ```bash
    git clone https://github.com/your-username/persona-tempestai.git
    cd persona-tempestai
    ```

2. 必要なパッケージをインストールします。

    ```bash
    pip install -r requirements.txt
    ```

3. [.env](http://_vscodecontentref_/0)ファイルを作成し、Google APIキーを設定します。

    ```plaintext
    GOOGLE_API_KEY=your_google_api_key
    ```

## 使い方 🚀

1. Streamlitアプリケーションを起動します。

    ```bash
    streamlit run main.py
    ```

2. ブラウザで表示されるフォームに必要な情報を入力し、「ペルソナ生成」ボタンをクリックします。

3. 生成されたペルソナが表示されます。

---

作成者: [KSW-1024](https://github.com/ksw-1024)
