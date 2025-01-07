import os
import time
from dotenv import load_dotenv
from uuid import uuid4

from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_core.output_parsers import StrOutputParser

import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAI
from langchain_ollama.llms import OllamaLLM

from pydantic import BaseModel, Field

from .GetToken import CountToken

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

unique_id = uuid4().hex[0:8]
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")
os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGCHAIN_ENDPOINT")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

model_gemini = GoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=1)
model_local  = OllamaLLM(
    model="qwen2-5-72b",
    temperature=1,
)

class Character(BaseModel):
    # 基本的な属性情報（デモグラフィック変数）
    name: str = Field(description="氏名(フルネーム)")
    age: int = Field(description="年齢")
    gender: str = Field(description="性別")
    residence: str = Field(description="居住地")
    housing: str = Field(description="住居情報")
    job: str = Field(description="職業・役職")
    company_size: str = Field(description="会社規模")
    salary: str = Field(description="年収")
    educational_background: str = Field(description="学歴")
    family_structure: str = Field(description="家族構成")
    
    # 心理的特性（サイコグラフィック変数）
    values: str = Field(description="価値観・人生観")
    lifestyle: str = Field(description="ライフスタイル")
    hobbies: str = Field(description="趣味・嗜好")
    goals: str = Field(description="目標・理想")
    
    # 行動特性（ビヘイビア変数）
    purchasing_behavior: str = Field(description="購買行動")
    information_sources: str = Field(description="情報収集方法")
    devices: str = Field(description="使用デバイス")
    sns_usage: str = Field(description="SNS利用状況")
    daily_schedule: str = Field(description="日課・タイムスケジュール")
    
    # 現在の課題とニーズ
    concerns: str = Field(description="悩み")
    needs: str = Field(description="解決したいこと")
    
    # その他の詳細情報
    favorite_brands: str = Field(description="好きなブランドや商品")
    favorite_media: str = Field(description="よく見る映画・動画チャンネル")
    relationships: str = Field(description="人間関係")
    recent_events: str = Field(description="最近の出来事やエピソード")
    
class Opinion(BaseModel):
    want_level: int = Field(description="サービスの魅力度レベル。0から10の間の整数値。")
    reason: str = Field(description="理由。100字以内。")

def GenerateHumanModel(gender, age_range_start, age_range_end, use_local):
    if use_local:
        model = model_local
    else:
        model = model_gemini
    
    output_parser = PydanticOutputParser(pydantic_object=Character)
    format_instructions = output_parser.get_format_instructions()

    prompt = ChatPromptTemplate.from_template(
        template="""あなたは超次元的存在であり、人間を生み出す事ができる存在です。次の条件を満たした日本人の人間モデルを1人のみ生成してください。条件を守りながら、なるべく多種多様な氏名、職業、家族構成、社会的地位の人間を生み出しなさい。\n\n年齢: {age}代\n性別: {gender}\n\n{format_instructions}"""
    )
    
    prompt_with_format_instructions = prompt.partial(format_instructions=format_instructions)
    
    chain = prompt_with_format_instructions | model | output_parser
    print(f"トークン数: {CountToken(prompt_with_format_instructions, {"age": age_range_start + "〜" + age_range_end, "gender": gender})}")
    
    for _ in range(3):
        try:
            human_model = chain.invoke({"age": age_range_start + "〜" + age_range_end, "gender": gender})
            return human_model
        except Exception as e:
            print(f"人間モデルの生成に失敗しました。再試行します。エラー: {e}")
            time.sleep(5)
            
    return None

def GenerateComment(service_title, service_concept, service_customer, service_description, service_revenue, character_data: Character, use_local):
    
    if use_local:
        model = model_local
    else:
        model = model_gemini
        
    input_data = {
        "name": character_data.name,
        "age": character_data.age,
        "gender": character_data.gender,
        "residence": character_data.residence,
        "housing": character_data.housing,
        "job": character_data.job,
        "company_size": character_data.company_size,
        "salary": character_data.salary,
        "educational_background": character_data.educational_background,
        "family_structure": character_data.family_structure,
        "values": character_data.values,
        "lifestyle": character_data.lifestyle,
        "hobbies": character_data.hobbies,
        "goals": character_data.goals,
        "purchasing_behavior": character_data.purchasing_behavior,
        "information_sources": character_data.information_sources,
        "devices": character_data.devices,
        "sns_usage": character_data.sns_usage,
        "daily_schedule": character_data.daily_schedule,
        "concerns": character_data.concerns,
        "needs": character_data.needs,
        "favorite_brands": character_data.favorite_brands,
        "favorite_media": character_data.favorite_media,
        "relationships": character_data.relationships,
        "recent_events": character_data.recent_events,
        "service_title": service_title,
        "service_concept": service_concept,
        "service_customer": service_customer,
        "service_description": service_description,
        "service_revenue": service_revenue
    }
    
    positive_prompt = ChatPromptTemplate.from_template(
        template="""あなたは{name}です。プロフィールは以下の通りです。
名前: {name}
年齢: {age}歳
性別: {gender}
居住地: {residence}
住居情報: {housing}
職業・役職: {job}
会社規模: {company_size}
年収: {salary}
学歴: {educational_background}
家族構成: {family_structure}
価値観・人生観: {values}
ライフスタイル: {lifestyle}
趣味・嗜好: {hobbies}
目標・理想: {goals}
購買行動: {purchasing_behavior}
情報収集方法: {information_sources}
使用デバイス: {devices}
SNS利用状況: {sns_usage}
日課・タイムスケジュール: {daily_schedule}
悩み: {concerns}
解決したいこと: {needs}
好きなブランドや商品: {favorite_brands}
よく見る映画・動画チャンネル: {favorite_media}
人間関係: {relationships}
最近の出来事やエピソード: {recent_events}

あなたは{service_title}のユーザーです。サービスに関する感想を200字程度で述べてください。プロフィールを元に口調などを調整し、主観的な意見を交えつつ、出来る限り肯定的に書いてください。
ただし、要件以外についてのコメントは控えてください。

### 要件
{service_title}のコンセプト: {service_concept}
{service_title}の顧客像: {service_customer}
{service_title}のサービス内容: {service_description}
{service_title}の収益モデル: {service_revenue}"""
    )
    
    negative_prompt = ChatPromptTemplate.from_template(
        template="""あなたは{name}です。プロフィールは以下の通りです。
名前: {name}
年齢: {age}歳
性別: {gender}
居住地: {residence}
住居情報: {housing}
職業・役職: {job}
会社規模: {company_size}
年収: {salary}
学歴: {educational_background}
家族構成: {family_structure}
価値観・人生観: {values}
ライフスタイル: {lifestyle}
趣味・嗜好: {hobbies}
目標・理想: {goals}
購買行動: {purchasing_behavior}
情報収集方法: {information_sources}
使用デバイス: {devices}
SNS利用状況: {sns_usage}
日課・タイムスケジュール: {daily_schedule}
悩み: {concerns}
解決したいこと: {needs}
好きなブランドや商品: {favorite_brands}
よく見る映画・動画チャンネル: {favorite_media}
人間関係: {relationships}
最近の出来事やエピソード: {recent_events}

あなたは{service_title}のユーザーです。サービスに関する感想を200字程度で述べてください。プロフィールを元に口調などを調整し、主観的な意見を交えつつ、出来る限り否定的に書いてください。
ただし、要件以外についてのコメントは控えてください。

### 要件
{service_title}のコンセプト: {service_concept}
{service_title}の顧客像: {service_customer}
{service_title}のサービス内容: {service_description}
{service_title}の収益モデル: {service_revenue}"""
    )
    
    output_parser = StrOutputParser()
    
    positive_chain = positive_prompt | model | output_parser
    print(f"ポジティブプロンプト 入力トークン数: {CountToken(positive_prompt, input_data)}")
    positive_chain_output = positive_chain.invoke(input_data)
       
    negative_chain = negative_prompt | model | output_parser
    print(f"ネガティブプロンプト 入力トークン数: {CountToken(negative_prompt, input_data)}")
    negative_chain_output = negative_chain.invoke(input_data)
    
    synthesize_prompt = ChatPromptTemplate.from_template(
        template="""あなたは{name}です。プロフィールは以下の通りです。
名前: {name}
年齢: {age}歳
性別: {gender}
居住地: {residence}
住居情報: {housing}
職業・役職: {job}
会社規模: {company_size}
年収: {salary}
学歴: {educational_background}
家族構成: {family_structure}
価値観・人生観: {values}
ライフスタイル: {lifestyle}
趣味・嗜好: {hobbies}
目標・理想: {goals}
購買行動: {purchasing_behavior}
情報収集方法: {information_sources}
使用デバイス: {devices}
SNS利用状況: {sns_usage}
日課・タイムスケジュール: {daily_schedule}
悩み: {concerns}
解決したいこと: {needs}
好きなブランドや商品: {favorite_brands}
よく見る映画・動画チャンネル: {favorite_media}
人間関係: {relationships}
最近の出来事やエピソード: {recent_events}

あなたは{service_title}のユーザーです。このサービスに対して、偏った2つの感想を抱きました。この2つの感想を参考にして、より説得力のある意見を200字程度で作成してください。
プロフィールを元に、主観的な視点を含めてください。また、意見が肯定、否定のどちらかに偏っても構いません。
肯定的意見: {positive}
否定的意見: {negative}"""
    )
    
    synthesize_chain = synthesize_prompt | model | output_parser
    print(f"総合感想プロンプト 入力トークン数: {CountToken(synthesize_prompt, {**input_data, **{'positive': positive_chain_output, 'negative': negative_chain_output}})}")
    return_data = synthesize_chain.invoke({
        **input_data,
        **{
        "positive": positive_chain_output,
        "negative": negative_chain_output
        }
        })
    
    return return_data

def OpinionSummerizer(service_title, character_data: Character, opinion, use_local):
    
    if use_local:
        model = model_local
    else:
        model = model_gemini
    
    output_parser = PydanticOutputParser(pydantic_object=Opinion)
    format_instructions = output_parser.get_format_instructions()
    
    opinion_prompt = ChatPromptTemplate.from_template(
        template="""あなたは{name}です。プロフィールは以下の通りです。
名前: {name}
年齢: {age}歳
性別: {gender}
居住地: {residence}
住居情報: {housing}
職業・役職: {job}
会社規模: {company_size}
年収: {salary}
学歴: {educational_background}
家族構成: {family_structure}
価値観・人生観: {values}
ライフスタイル: {lifestyle}
趣味・嗜好: {hobbies}
目標・理想: {goals}
購買行動: {purchasing_behavior}
情報収集方法: {information_sources}
使用デバイス: {devices}
SNS利用状況: {sns_usage}
日課・タイムスケジュール: {daily_schedule}
悩み: {concerns}
解決したいこと: {needs}
好きなブランドや商品: {favorite_brands}
よく見る映画・動画チャンネル: {favorite_media}
人間関係: {relationships}
最近の出来事やエピソード: {recent_events}

あなたは{service_title}というサービスに対して、以下の感想を持っています。
この感想を元に、以下の要件を満たすような意見を作成してください。
感想: {opinion}

{format_instructions}"""
    )
    
    prompt_with_format_instructions = opinion_prompt.partial(format_instructions=format_instructions)
    print(f"まとめ 入力トークン数: {CountToken(prompt_with_format_instructions, {
                "name": character_data.name,
                "age": character_data.age,
                "gender": character_data.gender,
                "residence": character_data.residence,
                "housing": character_data.housing,
                "job": character_data.job,
                "company_size": character_data.company_size,
                "salary": character_data.salary,
                "educational_background": character_data.educational_background,
                "family_structure": character_data.family_structure,
                "values": character_data.values,
                "lifestyle": character_data.lifestyle,
                "hobbies": character_data.hobbies,
                "goals": character_data.goals,
                "purchasing_behavior": character_data.purchasing_behavior,
                "information_sources": character_data.information_sources,
                "devices": character_data.devices,
                "sns_usage": character_data.sns_usage,
                "daily_schedule": character_data.daily_schedule,
                "concerns": character_data.concerns,
                "needs": character_data.needs,
                "favorite_brands": character_data.favorite_brands,
                "favorite_media": character_data.favorite_media,
                "relationships": character_data.relationships,
                "recent_events": character_data.recent_events,
                "service_title": service_title,
                "opinion": opinion,
                "format_instructions": format_instructions
            })}")
    
    chain = prompt_with_format_instructions | model | output_parser
    for _ in range(3):
        try:
            opinion_data = chain.invoke({
                "name": character_data.name,
                "age": character_data.age,
                "gender": character_data.gender,
                "residence": character_data.residence,
                "housing": character_data.housing,
                "job": character_data.job,
                "company_size": character_data.company_size,
                "salary": character_data.salary,
                "educational_background": character_data.educational_background,
                "family_structure": character_data.family_structure,
                "values": character_data.values,
                "lifestyle": character_data.lifestyle,
                "hobbies": character_data.hobbies,
                "goals": character_data.goals,
                "purchasing_behavior": character_data.purchasing_behavior,
                "information_sources": character_data.information_sources,
                "devices": character_data.devices,
                "sns_usage": character_data.sns_usage,
                "daily_schedule": character_data.daily_schedule,
                "concerns": character_data.concerns,
                "needs": character_data.needs,
                "favorite_brands": character_data.favorite_brands,
                "favorite_media": character_data.favorite_media,
                "relationships": character_data.relationships,
                "recent_events": character_data.recent_events,
                "service_title": service_title,
                "opinion": opinion,
                "format_instructions": format_instructions
            })
            return opinion_data
        except Exception as e:
            print(f"有効なデータの生成に失敗しました。再試行します。エラー: {e}")
            time.sleep(5)
            
    return None