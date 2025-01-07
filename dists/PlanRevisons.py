import os
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAI
from langchain_ollama.llms import OllamaLLM

from pydantic import BaseModel, Field

from .GetToken import CountToken

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

model_gemini = GoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=1)
model_local  = OllamaLLM(
    model="qwen2-5-72b",
    temperature=1,
)

class Opinion(BaseModel):
    want_level: int = Field(description="サービスの魅力度レベル。0から10の間の整数値。")
    reason: str = Field(description="理由。100字以内。")

def SuggestBusinessPlan(service_concept, service_customer, service_description, service_revenue, persona_list, use_local):
    if use_local:
        model = model_local
    else:
        model = model_gemini
    
    output_parser = StrOutputParser()
    
    persona_summerize_prompt = ChatPromptTemplate.from_template(
        template="""次のユーザーの意見を500字程度にまとめてください。\nユーザーの意見: {persona}"""
    )
    
    persona_summerize_chain = persona_summerize_prompt | model | output_parser
    print(f"全員の意見まとめ 入力トークン数: {CountToken(persona_summerize_prompt, {'persona': "\n".join(persona_list)})}")
    persona_summerize = persona_summerize_chain.invoke({"persona": "\n".join(persona_list)})
    
    persona_remake_prompt = ChatPromptTemplate.from_template(
        template="""次のユーザーの意見の要約を元に、サービスを改良してください。
元のサービス要件の形式を必ず守りなさい。必要な文言のみ出力しなさい。
また、一番下には改良した部分を簡潔にまとめなさい。
---
改良した部分の記述形式

改良点:
* ここに改良した部分を記述
* ここに改良した部分を記述
* ここに改良した部分を記述

ユーザーの意見: {persona}

## サービス要件
### サービスコンセプト
{service_concept}

### ターゲット顧客
{service_customer}

### サービス説明
{survice_description}

### 収益モデル
{service_revenue}"""
    )
    
    chain = persona_remake_prompt | model | output_parser
    print(f"トークン数: {CountToken(persona_remake_prompt, {'persona': persona_summerize, 'service_concept': service_concept, 'service_customer': service_customer, 'survice_description': service_description, 'service_revenue': service_revenue})}")
    return_data = chain.invoke({"persona": persona_summerize, "service_concept": service_concept, "service_customer": service_customer, "survice_description": service_description, "service_revenue": service_revenue})
    return return_data