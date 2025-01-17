import os
import time
import pandas as pd

import streamlit as st

import matplotlib.pyplot as plt
import japanize_matplotlib
import matplotlib.font_manager as fm

from dists.GeneratePersona import GenerateHumanModel, GenerateComment, OpinionSummerizer
from dists.PlanRevisons import SuggestBusinessPlan

current_dir = os.getcwd()
font_path = os.path.join(current_dir, "fonts", "NotoSansJP-VariableFont_wght.ttf")
font_prop = fm.FontProperties(fname=font_path)

graph_data = []

def update_graph(person, data):
    # 新しいデータを追加
    graph_data.append((person.name, data.want_level))
    
    # グラフをクリアして再描画
    fig, ax = plt.subplots()
    names, levels = zip(*graph_data)
    ax.bar(names, levels)
    ax.set_title("サービスの需要レベル", fontproperties=font_prop)
    ax.set_xlabel("人物名", fontproperties=font_prop)
    ax.set_ylabel("需要レベル", fontproperties=font_prop)
    graph_placeholder.pyplot(fig)

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
    use_local = st.checkbox("ローカルモデルを使用する", value=True)
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
        persona_data = GenerateComment(service_title, service_req, person_model, use_local)
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
            ### 生成された意見
        """
        )
        st.markdown(f"""
            * サービスの需要レベル: {opinion_data.want_level}
            * 理由: {opinion_data.reason}""")
        update_graph(person_model, opinion_data)
        st.write("------------")
        if not use_local:
            waiting_text = "I am preparing to work on..."
            waiting_bar = st.progress(0, text=waiting_text)
            
            percent_complete = 0.0
            total_count = 30
            
            for _ in range(total_count):
                percent_complete += 1 / total_count
                if percent_complete <= 1:
                    waiting_bar.progress(percent_complete, text=waiting_text)
                time.sleep(1)
            
            waiting_bar.empty()
                    
    remake_survice_data = SuggestBusinessPlan(service_req, persona_list, use_local)
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