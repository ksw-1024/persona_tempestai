import os
import time
import pandas as pd

import streamlit as st

import plotly.express as px

from dists.GeneratePersona import GenerateHumanModel, GenerateComment, OpinionSummerizer
from dists.PlanRevisons import SuggestBusinessPlan

graph_data = []

def update_graph():
    if graph_data:
        names, levels = zip(*graph_data)
        fig = px.bar(x=names, y=levels, labels={"x": "名前", "y": "サービスに対して感じた魅力度"}, title="サービスの魅力度レベル")
        graph_placeholder.plotly_chart(fig)

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

example_service_title = "オトナの趣味ラボ"
example_service_concept = """中年男性向けに、趣味を通じて人生の充実感を提供するコミュニティ型サービス。"""
example_service_customer = """仕事や家庭で忙しいが、自分の時間を持ちたい中年男性。\n新しい趣味を始めたいが、何から始めればいいかわからない人。\n同じ趣味を持つ仲間と交流したい人。"""
example_service_description = """オトナの趣味ラボは、中年男性が新しい趣味を見つけたり、既存の趣味を深めたりすることで、日常に充実感と楽しみを提供するサービスです。オンラインとオフラインでのイベントやワークショップを通じて、さまざまな趣味（アウトドア、料理、DIY、写真、音楽など）を体験できます。また、専用アプリで趣味に関する情報共有や仲間との交流が可能です。初心者向けガイドや専門家によるアドバイスも充実しており、誰でも気軽に参加できます。"""
example_service_revenue = """基本プラン: 月額1,500円でオンラインコンテンツやコミュニティ利用権。\nプレミアムプラン: 月額3,500円でオンライン+オフラインイベント参加権。\n個別セッション: 専門家による1対1指導で1回5,000円～10,000円。"""

with st.form("persona_form"):
    service_title = st.text_input("サービスタイトル", value=example_service_title)
    service_concept = st.text_area("サービスコンセプト", value=example_service_concept)
    service_customer = st.text_area("顧客像", value=example_service_customer)
    service_description = st.text_area("サービス概要", value=example_service_description)
    service_revenue = st.text_area("収益モデル", value=example_service_revenue)
    
    gender = st.selectbox("ターゲットの性別", ["男性", "女性", "その他", "男女どちらでも"])
    
    col1, col2 = st.columns([1, 1])
    with col1:
        age_range_start = st.selectbox("ターゲットの年代（開始）", [str(i) for i in range(10, 101, 10)])
    with col2:
        age_range_end = st.selectbox("ターゲットの年代（終了）", [str(i) for i in range(10, 101, 10)])
        
    number_of_people = st.number_input("生成する人数", min_value=1, max_value=3, value=1)
    use_local = st.checkbox("ローカルモデルを使用する", value=False)
    submitted = st.form_submit_button("ペルソナ生成")
    
    graph_placeholder = st.empty()
    
if submitted:
    people_list = []
    persona_list = []
    opinion_list = []
    for i in range(number_of_people):
        person_model = GenerateHumanModel(gender, age_range_start, age_range_end, use_local)
        if person_model is None:
            st.error("有効な人間モデルの生成に失敗しました。")
            break
        people_list.append(person_model)
        with st.expander(f"生成された人間モデル: {person_model.name}", expanded=False):
            st.markdown(f"""
                * 名前: {person_model.name}
                * 年齢: {person_model.age}歳
                * 性別: {person_model.gender}
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
        persona_data = GenerateComment(service_title, service_concept, service_customer, service_description, service_revenue, person_model, use_local)
        persona_list.append(persona_data)
        
        with st.expander(f"生成されたコメント", expanded=False):
            st.success(persona_data)
        
        opinion_data = OpinionSummerizer(service_title, person_model, persona_data, use_local)
        
        if opinion_data is None:
            st.error("有効なデータの生成に失敗しました。試行回数を変更し、再度実行してください。")
            break
        
        opinion_list.append(opinion_data)
        st.markdown(f"""
            ## 意見生成完了
            ### 生成された意見"""
        )
        st.markdown(f"""
            * サービスの需要レベル: {opinion_data.want_level}
            * 理由: {opinion_data.reason}""")
        
        graph_data.append((person_model.name, opinion_data.want_level))
        update_graph()
        
        st.write("------------")
        if not use_local:
            waiting_text = "I am preparing to work on..."
            waiting_bar = st.progress(0, text=waiting_text)
            
            percent_complete = 0.0
            total_count = 300
            
            for _ in range(total_count):
                percent_complete += 1 / total_count
                if percent_complete <= 1:
                    waiting_bar.progress(percent_complete, text=waiting_text)
                time.sleep(0.1)
            
            waiting_bar.empty()
                    
    remake_survice_data = SuggestBusinessPlan(service_concept, service_customer, service_description, service_revenue, persona_list, use_local)
    st.markdown(f"""
        ## サービス改良完了
        ### 改良されたサービス要件"""
    )
    st.write("------------")
    
    st.write(remake_survice_data)
    
    st.write("------------")
    
    # データフレームの作成
    data = []
    for person, persona, opinion in zip(people_list, persona_list, opinion_list):
        data.append([person.name, person.age, person.gender, person.residence, person.housing, person.job, person.company_size, person.salary, person.educational_background, person.family_structure, person.values, person.lifestyle, person.hobbies, person.goals, person.purchasing_behavior, person.information_sources, person.devices, person.sns_usage, person.daily_schedule, person.concerns, person.needs, person.favorite_brands, person.favorite_media, person.relationships, person.recent_events, persona, opinion.want_level, opinion.reason])
        
    df = pd.DataFrame(data, columns=["名前", "年齢", "性別", "居住地", "住居情報", "職業・役職", "会社規模", "年収", "学歴", "家族構成", "価値観・人生観", "ライフスタイル", "趣味・嗜好", "目標・理想", "購買行動", "情報収集方法", "使用デバイス", "SNS利用状況", "日課・タイムスケジュール", "悩み", "解決したいこと", "好きなブランドや商品", "よく見る映画・動画チャンネル", "人間関係", "最近の出来事やエピソード", "生成されたコメント", "サービスの需要レベル", "理由"])
    
    # CSVファイルの出力
    csv_file = df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
    
    # ダウンロードボタンの作成
    st.download_button(
        label="ヒューマンモデルファイルのダウンロード",
        data=csv_file,
        file_name="persona_data.csv",
        mime="text/csv"
    )