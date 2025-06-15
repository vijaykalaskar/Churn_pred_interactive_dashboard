import streamlit as st

def fade_in_markdown(text):
    st.markdown(f"""
    <div style="
        animation: fadeIn 1s ease-in;
        -webkit-animation: fadeIn 1s ease-in;
    ">
        {text}
    </div>

    <style>
    @keyframes fadeIn {{
        from {{ opacity: 0; }}
        to {{ opacity: 1; }}
    }}
    </style>
    """, unsafe_allow_html=True)

def styled_header(title):
    fade_in_markdown(f"<h2 style='font-size: 36px; font-weight: bold; color: #1f77b4;'>{title}</h2>")

styled_header("🧩 Problem Statement")

fade_in_markdown("## Why This Matters")
st.write(
    "Every lost user means lost revenue. In the crowded fintech space, spotting "
    "customers who are about to churn—and winning them back—can make or break growth."
)

st.markdown("---")

fade_in_markdown("## Mission")
st.markdown("""
Build a working churn-prediction tool and dashboard demo that includes:

1. Generating churn-probability scores for each user.  
2. Delivering an interactive dashboard that visualizes those scores.
""")
