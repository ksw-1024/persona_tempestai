import tiktoken
from langchain.prompts import ChatPromptTemplate

def CountToken(prompt_template: ChatPromptTemplate, input_data: dict) -> int:
    """
    LangChainのChatPromptTemplate型を受け取り、TikTokenでトークン数をカウントする関数。

    Args:
        prompt_template (ChatPromptTemplate): トークン数を計算する対象のプロンプトテンプレート。

    Returns:
        int: トークン数。
    """
    # ChatPromptTemplateを文字列に変換
    prompt_string = str(prompt_template.format(input_data))
    print(prompt_string)

    # TikTokenのエンコーダーを初期化 (cl100k_baseはOpenAIのモデルで使用される一般的なエンコーディング)
    encoder = tiktoken.get_encoding("cl100k_base")

    # トークン数を計算
    token_count = len(encoder.encode(prompt_string))

    return token_count