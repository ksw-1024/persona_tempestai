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
    use_local = st.checkbox("ローカルモデルを使用する", value=False)
    submitted = st.form_submit_button("モデル生成")

if submitted:
    people_list = []
    for _ in range(10):
        for i in range(1, 11):
            person_model = GenerateHumanModel("男性", str(10*i), str(10*i+10), use_local)
            if person_model is None:
                st.error("有効な人間モデルの生成に失敗しました。")
                break
            people_list.append(person_model)
            st.write(f"生成された人間モデル: {person_model.name}")
            
            person_model = GenerateHumanModel("女性", str(10*i), str(10*i+10), use_local)
            if person_model is None:
                st.error("有効な人間モデルの生成に失敗しました。")
                break
            people_list.append(person_model)
            st.write(f"生成された人間モデル: {person_model.name}")


    
    # データフレームの作成
    data = []
    for person in people_list:
        data.append([person.name, person.age, person.gender, person.residence, person.housing, person.job, person.company_size, person.salary, person.educational_background, person.family_structure, person.values, person.lifestyle, person.hobbies, person.goals, person.purchasing_behavior, person.information_sources, person.devices, person.sns_usage, person.daily_schedule, person.concerns, person.needs, person.favorite_brands, person.favorite_media, person.relationships, person.recent_events])
        
    df = pd.DataFrame(data, columns=["名前", "年齢", "性別", "居住地", "住居情報", "職業・役職", "会社規模", "年収", "学歴", "家族構成", "価値観・人生観", "ライフスタイル", "趣味・嗜好", "目標・理想", "購買行動", "情報収集方法", "使用デバイス", "SNS利用状況", "日課・タイムスケジュール", "悩み", "解決したいこと", "好きなブランドや商品", "よく見る映画・動画チャンネル", "人間関係", "最近の出来事やエピソード"])
    
    # CSVファイルの出力
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    csv_file_path = f"human_model_data_{timestamp}.csv"
    df.to_csv(csv_file_path, index=False, encoding="utf-8-sig")
    st.success(f"CSVファイルが {csv_file_path} に保存されました。")