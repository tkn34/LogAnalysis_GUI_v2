
#import
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import  StratifiedKFold
from sklearn.model_selection import  GridSearchCV
import lightgbm as lgb
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
# Streamlit
import streamlit as st
from PIL import Image
# import plotly.express as px
# LogAnalysis
sys.path.append(os.path.join(os.path.dirname("__file__"), "./src/"))
from data_preprocessing import DataPreprocessing
from data_preprocessing import FeatureExtraction
from data_preprocessing import RelatedInfo
import warnings
warnings.simplefilter('ignore')

from st_aggrid import AgGrid




def highlight_greaterthan(s, threshold, column):
    is_max = pd.Series(data=False, index=s.index)
    is_max[column] = s.loc[column] >= threshold
    return ['background-color: red' if is_max.any() else '' for v in is_max]




if __name__ == '__main__':
    # 初期設定
    st.set_page_config(layout="wide")
    st.title("Log Analysis for AI.")
    
    # リソース選択
    st.subheader('1. リソースの選択')
    st.caption('分析対象リソースを選択してください。')
    resource_list = ["BGL", "Apache"]
    selected_resource = st.multiselect('分析対象リソース', resource_list, default='BGL')
    st.session_state["selected_resource"] = selected_resource
    #resource = st.selectbox('分析対象リソース',('BGL', 'Apache'))
    
    # Feedbackの取得
    st.subheader("2. 過去の障害情報")
    st.caption('過去に発生した障害の一覧表です。')
    feedback = pd.read_csv("./data/feedback.csv", encoding="SJIS")
    feedback = feedback[(feedback['Resource'].isin(selected_resource))]
    display_feedback  = feedback[["message_time", "title", "service info", "message", "create user", "create datetime"]]
    st.session_state["feedback"] = feedback
    st.table(display_feedback)
    
    
    # 学習データの取得
    st.subheader("3. 分析実行")
    st.caption('データ取得/特徴量抽出/ログ類似度算出/を行います。')
    if len(selected_resource) != 0:
        file_name_dict = {}
        for resource in selected_resource:
            file = st.file_uploader(resource + "データの取得", type=['csv'])
            file_name_dict[resource] = file
        submit = st.button("データの表示")
        if submit == True:
            for resource, file in file_name_dict.items():
                st.markdown(f'{file.name} をアップロードしました.')
                data = pd.read_csv("./data/" + resource + "/" + file.name, encoding="SJIS")
                st.session_state[resource + "_data"] = data
                # データ表示
                display_data = st.session_state[resource + "_data"][["message_time", "message"]]
                # データ件数可視化
                plot_data = pd.DataFrame({"message": display_data["message"]})
                plot_data.index = pd.to_datetime(display_data["message_time"])
                res = plot_data.resample("1H").count()
                sns.lineplot(data=res["message"])
                fig = plt.show()
                
                # 表示
                st.write(" ---- + {} + ----".format(resource))
                col1, col2 = st.beta_columns(2)
                with col1:
                    st.write('ログデータ')
                    st.write(display_data)
                with col2:
                    st.write('ログデータ(可視化)')
                    st.set_option('deprecation.showPyplotGlobalUse', False)
                    st.pyplot(fig)
    st.write('')
    st.write('')
    st.write('')
    
    st.write("コサイン類似度の閾値")
    st.caption('過去の障害情報との関連度をもとに異常検知を行います。 0～100で設定し、値が大きいほど、過去の障害との関連が強いログのみ検知します。')
    cos_threshold = st.slider('閾値', min_value=0, max_value=100, step=1, value=75)
    st.write('')
    st.write('')
    st.write('')
    
    st.write("特徴量抽出方法の選択")
    st.caption('特徴量抽出方法を選択してください。TF-ILFもしくはTF-IDFから選択できます。')
    feature_type = st.selectbox('特徴量抽出方法',('TF-ILF', 'TF-IDF'))
    st.write('')
    st.write('')
    st.write('')
    
    exec_analysis = st.button('異常検知 実行')
    if exec_analysis == True:
        result_all = []
        for resource in st.session_state.selected_resource:
            st.write("--------------- + {} + -----------------".format(resource))
            feedback_resource = st.session_state.feedback[(st.session_state.feedback['Resource'].isin([resource]))]
            # 固有値の削除
            feedback_resource["message_after"] = DataPreprocessing(feedback_resource["message"].values)()
            st.session_state[resource + "_data"]["message_after"] = DataPreprocessing(st.session_state[resource + "_data"]["message"].values)()
            st.write("Remove Parameter   : Done.")
            
            # TF-IDF
            feedback_tmp = feedback_resource[["message_after"]]
            feedback_tmp["label"] = -1
            data_tmp = st.session_state[resource + "_data"][["message_after"]]
            data_tmp["label"] = 0
            st.write("Feature Extraction : Done.")
            data_all = pd.concat([data_tmp, feedback_tmp]).reset_index(drop=True)
            data_feature, _ = FeatureExtraction(data_all["message_after"].values, mode="train", fe_type=feature_type)()
            data_feature["label"] = data_all["label"]
            
            # Cosine Similar
            result = RelatedInfo(data_feature, feedback_resource, cos_threshold)()
            result = pd.concat([st.session_state[resource + "_data"][["message_time", "message"]], result], axis=1)
            result["Resource"] = resource
            result_all.append(result)
            st.write("Cosine Similar     : Done.")
        st.subheader("4. 分析結果")
        st.caption('分析結果を表示します.')
        if len(result_all) == 1:
            result_all = result_all[0].sort_values("message_time")
        else:
            result_all = pd.concat(result_all).sort_values("message_time").reset_index(drop=True)
        result = result_all.style.apply(highlight_greaterthan, threshold=1.0, column=['正常(0)/異常(1)'], axis=1)
        st.write(result)
        