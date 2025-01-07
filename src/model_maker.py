import os
import time
import pandas as pd

import streamlit as st

from dists.GeneratePersona import GenerateHumanModel

st.set_page_config(page_title="人間モデル生成", layout="centered")

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

st.markdown('<div class="big-title">人間モデル生成</div>', unsafe_allow_html=True)

with st.form("model_form"):
    gender = st.selectbox("性別", ["男性", "女性", "その他", "男女どちらでも"])
    
    col1, col2 = st.columns([1, 1])
    with col1:
        age_range_start = st.selectbox("年代（開始）", [str(i) for i in range(10, 101, 10)])
    with col2:
        age_range_end = st.selectbox("年代（終了）", [str(i) for i in range(10, 101, 10)])
        
    number_of_people = st.number_input("生成する人数", min_value=1, max_value=100, value=1)
    use_local = st.checkbox("ローカルモデルを使用する", value=True)
    submitted = st.form_submit_button("モデル生成")

if submitted:
    people_list = []
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
    
    # データフレームの作成
    data = []
    for person in people_list:
        data.append([person.name, person.age, person.sex, person.residence, person.housing, person.job, person.company_size, person.salary, person.educational_background, person.family_structure, person.values, person.lifestyle, person.hobbies, person.goals, person.purchasing_behavior, person.information_sources, person.devices, person.sns_usage, person.daily_schedule, person.concerns, person.needs, person.favorite_brands, person.favorite_media, person.relationships, person.recent_events])
        
    df = pd.DataFrame(data, columns=["名前", "年齢", "性別", "居住地", "住居情報", "職業・役職", "会社規模", "年収", "学歴", "家族構成", "価値観・人生観", "ライフスタイル", "趣味・嗜好", "目標・理想", "購買行動", "情報収集方法", "使用デバイス", "SNS利用状況", "日課・タイムスケジュール", "悩み", "解決したいこと", "好きなブランドや商品", "よく見る映画・動画チャンネル", "人間関係", "最近の出来事やエピソード"])
    
    # CSVファイルの出力
    csv_file = df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
    
    # ダウンロードボタンの作成
    st.download_button(
        label="CSVファイルのダウンロード",
        data=csv_file,
        file_name="human_model_data.csv",
        mime="text/csv"
    )