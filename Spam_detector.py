import numpy as np
import streamlit as st
from joblib import load
from sklearn.feature_extraction.text import TfidfVectorizer

mail_detector=load("Mail_model.joblib")
feature_extraction=load("vectorizor.joblib")


if __name__=="__main__":

    st.title("Mail Spam detector")
    inputted_mail=st.text_input("Paste your mail content:")

    if st.button("Press for detection"):
        new_mail=[inputted_mail]
        converted_mail=feature_extraction.transform(new_mail)
        pred=mail_detector.predict(converted_mail)
        prediction=pred[0]

        if prediction==0:
            st.write("This mail is not a spam")
            st.success("thanks for using our service..")
        else:
            st.write("This mail is spam")
            st.success("thanks for using our service")


