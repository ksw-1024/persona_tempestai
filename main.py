from dotenv import load_dotenv
import os

import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.output_parsers import PydanticOutputParser
from langchain_core.output_parsers import StrOutputParser

import google.generativeai as genai
from langchain_openai import ChatOpenAI

from pydantic import BaseModel, Field

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0)
# model  = ChatOpenAI(
#     base_url="http://localhost:11434/v1",
#     model="llama-3.3-70b",
#     api_key="not required",
# ).bind(
#     response_format={"type": "json_object"},
# )

class Character(BaseModel):
    name: str = Field(description="名前")
    age: int = Field(description="年齢")
    sex: str = Field(description="性別")
    height: int = Field(description="身長")
    weight: int = Field(description="体重")
    place: str = Field(description="出身地")
    job: str = Field(description="職業")
    hobby: str = Field(description="趣味")
    personality: str = Field(description="性格")
    appearance: str = Field(description="外見")

def generate_human_model(gender, age_range):
    
    output_parser = PydanticOutputParser(pydantic_object=Character)
    format_instructions = output_parser.get_format_instructions()

    prompt = ChatPromptTemplate.from_template(
        template="""
            あなたは超次元的存在であり、人間を生み出す事ができる存在です。次の条件を満たした人間モデルを生成してください。\n\n年齢: {age}代\n性別: {gender}\n\n{format_instructions}
        """
    )
    
    prompt_with_format_instructions = prompt.partial(format_instructions=format_instructions)
    
    chain = prompt_with_format_instructions | model | output_parser
    human_model = chain.invoke({"age": ", ".join(age_range), "gender": gender})
    return human_model

def generate_persona(service_title, service_data, character_data: Character):
    prompt = ChatPromptTemplate.from_template(
        template="""
            あなたは{name}です。プロフィールは以下の通りです。
            名前: {name}
            年齢: {age}歳
            性別: {sex}
            身長: {height}cm
            体重: {weight}kg
            出身地: {place}
            職業: {job}
            趣味: {hobby}
            性格: {personality}
            外見: {appearance}
            
            あなたは{service_title}のユーザーです。サービスに関する感想を述べてください。口調なども含めて、自由に書いてください。出来る限り批判的に書いてください。
            {service_title}の要件: {service_data}
        """
    )
    
    output_parser = StrOutputParser()
    
    chain = prompt | model | output_parser
    return_data = chain.invoke({"name": character_data.name, "age": character_data.age, "sex": character_data.age, "height": character_data.height, "weight": character_data.weight, "place": character_data.place, "job": character_data.job, "hobby": character_data.hobby, "personality": character_data.personality, "appearance": character_data.appearance, "service_title": service_title, "service_data": service_data})
    return return_data

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

with st.form("persona_form"):
    service_title = st.text_input("サービスタイトル", value="うんたらサービス")
    service_req = st.text_area("サービス要件")
    gender = st.selectbox("ターゲットの性別", ["男性", "女性", "その他"])
    age_range = st.multiselect("ターゲットの年代", [str(i) for i in range(10, 101, 10)])
    number_of_people = st.number_input("生成する人数", min_value=1, max_value=10, value=1)
    submitted = st.form_submit_button("ペルソナ生成")
    if submitted:
        person_model = []
        for i in range(number_of_people):
            person_model.append(generate_human_model(gender, age_range))
        st.markdown(f"""
            ## 人間モデル生成完了
            ### 生成された人間モデル
            * 名前: {person_model[0].name}
            * 年齢: {person_model[0].age}歳
            * 性別: {person_model[0].sex}
            * 身長: {person_model[0].height}cm
            * 体重: {person_model[0].weight}kg
            * 出身地: {person_model[0].place}
            * 職業: {person_model[0].job}
            * 趣味: {person_model[0].hobby}
            * 性格: {person_model[0].personality}
            * 外見: {person_model[0].appearance}
        """
        )
        persona_data = generate_persona(service_title, service_req, person_model[0])
        print(persona_data)
        st.markdown(f"""
            ## ペルソナ生成完了
            ### 生成されたペルソナ
                
            {persona_data}
        """
        )