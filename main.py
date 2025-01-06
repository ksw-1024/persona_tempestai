import os
import time
from dotenv import load_dotenv

import pandas as pd

import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAI
from langchain.output_parsers import PydanticOutputParser
from langchain_core.output_parsers import StrOutputParser

import google.generativeai as genai
from langchain_ollama.llms import OllamaLLM

from pydantic import BaseModel, Field

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# model = GoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=1)
model  = OllamaLLM(
    model="llama-3-swallow-70b",
    temperature=1,
)

class Character(BaseModel):
    # 基本的な属性情報（デモグラフィック変数）
    name: str = Field(description="氏名(フルネーム)")
    age: int = Field(description="年齢")
    sex: str = Field(description="性別")
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

def generate_human_model(gender, age_range_start, age_range_end):
    
    output_parser = PydanticOutputParser(pydantic_object=Character)
    format_instructions = output_parser.get_format_instructions()

    prompt = ChatPromptTemplate.from_template(
        template="""
            あなたは超次元的存在であり、人間を生み出す事ができる存在です。次の条件を満たした日本人の人間モデルを1人のみ生成してください。条件を守りながら、なるべく多種多様な氏名、職業、家族構成、社会的地位の人間を生み出しなさい。\n\n年齢: {age}代\n性別: {gender}\n\n{format_instructions}
        """
    )
    
    prompt_with_format_instructions = prompt.partial(format_instructions=format_instructions)
    
    chain = prompt_with_format_instructions | model | output_parser
    for _ in range(3):
        try:
            human_model = chain.invoke({"age": age_range_start + "〜" + age_range_end + "代", "gender": gender})
            return human_model
        except Exception as e:
            st.warning(f"人間モデルの生成に失敗しました。再試行します。エラー: {e}")
            time.sleep(5)
            
    st.error("有効な人間モデルの生成に失敗しました。")
    return None
            
def generate_persona(service_title, service_data, character_data: Character):
    positive_prompt = ChatPromptTemplate.from_template(
        template="""
            あなたは{name}です。プロフィールは以下の通りです。
            名前: {name}
            年齢: {age}歳
            性別: {sex}
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
            
            あなたは{service_title}のユーザーです。サービスに関する感想を述べてください。口調なども含めて、自由に書いてください。出来る限り肯定的に書いてください。
            ただし、要件以外についてのコメントは控えてください。
            {service_title}の要件: {service_data}
        """
    )
    
    negative_prompt = ChatPromptTemplate.from_template(
        template="""
            あなたは{name}です。プロフィールは以下の通りです。
            名前: {name}
            年齢: {age}歳
            性別: {sex}
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
            
            あなたは{service_title}のユーザーです。サービスに関する感想を述べてください。口調なども含めて、自由に書いてください。出来る限り否定的に書いてください。
            ただし、要件以外についてのコメントは控えてください。
            {service_title}の要件: {service_data}
        """
    )
    
    output_parser = StrOutputParser()
    
    positive_chain = positive_prompt | model | output_parser
    positive_chain_output = positive_chain.invoke({
        "name": character_data.name,
        "age": character_data.age,
        "sex": character_data.sex,
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
        "service_data": service_data
    })
       
    negative_chain = negative_prompt | model | output_parser
    negative_chain_output = negative_chain.invoke({
        "name": character_data.name,
        "age": character_data.age,
        "sex": character_data.sex,
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
        "service_data": service_data
    })
    
    synthesize_prompt = ChatPromptTemplate.from_template(
        template="""
            あなたは{name}です。プロフィールは以下の通りです。
            名前: {name}
            年齢: {age}歳
            性別: {sex}
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
            
            あなたは{service_title}に対して、偏った2つの感想を抱きました。この2つの感想を総合して、より説得力のある意見を500字程度で作成してください。
            プロフィールを元に、主観的な視点を含めてください。また、意見が肯定、否定のどちらかに偏っても構いません。
            肯定的意見: {positive}
            否定的意見: {negative}
        """
    )
    
    synthesize_chain = synthesize_prompt | model | output_parser
    return_data = synthesize_chain.invoke({
        "name": character_data.name,
        "age": character_data.age,
        "sex": character_data.sex,
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
        "positive": positive_chain_output,
        "negative": negative_chain_output
        })
    return return_data

def remake_service(service_data, persona_list):
    
    output_parser = StrOutputParser()
    
    persona_summerize_prompt = ChatPromptTemplate.from_template(
        template="""
            次のユーザーの意見を500字程度にまとめてください。
            ユーザーの意見: {persona}
        """
    )
    
    persona_summerize_chain = persona_summerize_prompt | model | output_parser
    persona_summerize = persona_summerize_chain.invoke({"persona": "\n".join(persona_list)})
    
    persona_remake_prompt = ChatPromptTemplate.from_template(
        template="""
            次のユーザーの意見の要約を元に、サービスを改良してください。
            元のサービス要件の形式を必ず守りなさい。必要な文言のみ出力しなさい。
            ユーザーの意見: {persona}
            サービス要件: {service_data}
        """
    )
    
    chain = persona_remake_prompt | model | output_parser
    return_data = chain.invoke({"persona": persona_summerize, "service_data": service_data})
    return return_data

def opinion_maker(character_data: Character, opinion):
    output_parser = PydanticOutputParser(pydantic_object=Opinion)
    format_instructions = output_parser.get_format_instructions()
    
    opinion_prompt = ChatPromptTemplate.from_template(
        template="""
            あなたは{name}です。プロフィールは以下の通りです。
            名前: {name}
            年齢: {age}歳
            性別: {sex}
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
            
            {format_instructions}
        """
    )
    
    prompt_with_format_instructions = opinion_prompt.partial(format_instructions=format_instructions)
    
    chain = prompt_with_format_instructions | model | output_parser
    for _ in range(3):
        try:
            opinion_data = chain.invoke({
                "name": character_data.name,
                "age": character_data.age,
                "sex": character_data.sex,
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
            st.warning(f"有効なデータの生成に失敗しました。再試行します。エラー: {e}")
            time.sleep(5)
            
    st.error("有効なデータの生成に失敗しました。試行回数を変更し、再度実行してください。")
    return None

st.set_page_config(page_title="ペルソナ生成", layout="centered")

st.markdown("""
<style>
.big-title {
    font-size: 2.5em;
    color: #00adb5;
    text-align: center;
    margin-bottom: 0.5em;
}
.input-box {
    background-color: #f0f0f0;
    padding: 1em;
    border-radius: 8px;
    margin-bottom: 1em;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="big-title">ペルソナ作成</div>', unsafe_allow_html=True)

example_service_req = """1. サービス概要
目的: 家庭やオフィスに手軽に花を取り入れることで、生活空間の彩りと癒しを提供。
対象顧客: 
  - 花屋に行く時間がないが花を楽しみたい人。
  - 季節感やインテリアとして花を取り入れたい人。
  - ギフト用途で利用したい人。

2. サービス内容
配送プラン
- 配送頻度: 毎日、週1回、隔週、月1回など柔軟な選択肢を提供。
- 配送方法:
  - ポスト投函（小型ブーケ用）。
  - 宅配便（大きめの花束や特別なアレンジメント用）。
  - 手渡し配送（高級プラン向け）。

料金プラン
- 初心者向け: 小型ブーケで550円～1,000円/回（送料別）。
- 中級者向け: ボリュームがある花束で1,500円～2,500円/回。
- 高級プラン: 特別アレンジメントで3,000円以上。

カスタマイズオプション
- 花の種類や色味の選択（例: 季節感重視、特定色指定）。
- ギフトラッピングやメッセージカード。
- 環境配慮型（ロスフラワー活用）オプション。

付加価値サービス
- 花瓶や延命剤の提供。
- 花の種類や飾り方を紹介するガイド付き。
- サブスクリプション特典（割引、スキップ・解約自由）。"""

with st.form("persona_form"):
    service_title = st.text_input("サービスタイトル", value="フラデリ")
    service_req = st.text_area("サービス要件", value=example_service_req, height=400)
    gender = st.selectbox("ターゲットの性別", ["男性", "女性", "その他", "男女どちらでも"])
    
    col1, col2 = st.columns([1, 1])
    with col1:
        age_range_start = st.selectbox("ターゲットの年代（開始）", [str(i) for i in range(10, 101, 10)])
    with col2:
        age_range_end = st.selectbox("ターゲットの年代（終了）", [str(i) for i in range(10, 101, 10)])
        
    number_of_people = st.number_input("生成する人数", min_value=1, max_value=10, value=1)
    submitted = st.form_submit_button("ペルソナ生成")
    
if submitted:
    people_list = []
    persona_list = []
    opinion_list = []
    for i in range(number_of_people):
        person_model = generate_human_model(gender, age_range_start, age_range_end)
        if person_model is None:
            break
        people_list.append(person_model)
        with st.expander(f"生成された人間モデル: {person_model.name}", expanded=False):
            st.markdown(f"""
                * 名前: {person_model.name}
                * 年齢: {person_model.age}歳
                * 性別: {person_model.sex}
                * 居住地: {person_model.residence}
                * 住居情報: {person_model.housing}
                * 職業・役職: {person_model.job}
                * 会社規模: {person_model.company_size}
                * 年収: {person_model.salary}
                * 学歴: {person_model.educational_background}
                * 家族構成: {person_model.family_structure}
                * 価値観・人生観: {person_model.values}
                * ライフスタイル: {person_model.lifestyle}
                * 趣味・嗜好: {person_model.hobbies}
                * 目標・理想: {person_model.goals}
                * 購買行動: {person_model.purchasing_behavior}
                * 情報収集方法: {person_model.information_sources}
                * 使用デバイス: {person_model.devices}
                * SNS利用状況: {person_model.sns_usage}
                * 日課・タイムスケジュール: {person_model.daily_schedule}
                * 悩み: {person_model.concerns}
                * 解決したいこと: {person_model.needs}
                * 好きなブランドや商品: {person_model.favorite_brands}
                * よく見る映画・動画チャンネル: {person_model.favorite_media}
                * 人間関係: {person_model.relationships}
                * 最近の出来事やエピソード: {person_model.recent_events}
            """
            )
        persona_data = generate_persona(service_title, service_req, person_model)
        persona_list.append(persona_data)
        
        with st.expander(f"生成されたコメント", expanded=False):
            st.success(persona_data)
        
        opinion_data = opinion_maker(person_model, persona_data)
        opinion_list.append(opinion_data)
        st.markdown(f"""
            ## 意見生成完了
            ### 生成された意見
        """
        )
        st.markdown(f"""
            * サービスの需要レベル: {opinion_data.want_level}
            * 理由: {opinion_data.reason}""")
        st.write("------------")
        time.sleep(30)
        
    remake_survice_data = remake_service(service_req, persona_list)
    st.markdown(f"""
        ## サービス改良完了
        ### 改良されたサービス要件"""
    )
    st.write(remake_survice_data)
    
    st.write("------------")
    
    # データフレームの作成
    data = []
    for person, persona, opinion in zip(people_list, persona_list, opinion_list):
        data.append([person.name, person.age, person.sex, person.residence, person.housing, person.job, person.company_size, person.salary, person.educational_background, person.family_structure, person.values, person.lifestyle, person.hobbies, person.goals, person.purchasing_behavior, person.information_sources, person.devices, person.sns_usage, person.daily_schedule, person.concerns, person.needs, person.favorite_brands, person.favorite_media, person.relationships, person.recent_events, persona, opinion.want_level, opinion.reason])
        
    df = pd.DataFrame(data, columns=["名前", "年齢", "性別", "居住地", "住居情報", "職業・役職", "会社規模", "年収", "学歴", "家族構成", "価値観・人生観", "ライフスタイル", "趣味・嗜好", "目標・理想", "購買行動", "情報収集方法", "使用デバイス", "SNS利用状況", "日課・タイムスケジュール", "悩み", "解決したいこと", "好きなブランドや商品", "よく見る映画・動画チャンネル", "人間関係", "最近の出来事やエピソード", "生成されたコメント", "サービスの需要レベル", "理由"])
    
    # CSVファイルの出力
    csv_file = df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
    
    # ダウンロードボタンの作成
    st.download_button(
        label="CSVファイルのダウンロード",
        data=csv_file,
        file_name="persona_data.csv",
        mime="text/csv"
    )