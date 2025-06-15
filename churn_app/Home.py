# import streamlit as st

# st.set_page_config(page_title="Stop the Churn", layout="wide")
# st.title("🧠 Data Science Project 2: Stop the Churn")
# st.markdown("---")

# st.write("Welcome to the **Stop the Churn** dashboard.")
# st.write("Use the sidebar to navigate through each project stage.")
###############

# import streamlit as st

# # Page setup
# st.set_page_config(page_title="Stop the Churn", layout="wide", page_icon="🧠")

# # Custom CSS with dark theme compatibility and animations
# st.markdown("""
#     <style>
#     body {
#         background-color: #000000;
#         color: #ffffff;
#     }

#     .main-title {
#         font-size: 3rem;
#         font-weight: 900;
#         color: #00FFFF;
#         text-align: center;
#         animation: fadeInDown 1.5s ease-in-out;
#     }

#     .subtext {
#         font-size: 1.25rem;
#         color: #CCCCCC;
#         text-align: center;
#         margin-top: -1rem;
#         margin-bottom: 1.5rem;
#         animation: fadeInUp 2s ease-in-out;
#     }

#     .welcome-box {
#         background: linear-gradient(135deg, #111111, #222222);
#         border-left: 6px solid #00FFFF;
#         padding: 1.5rem;
#         margin: 1rem auto;
#         border-radius: 12px;
#         box-shadow: 0 0 20px rgba(0,255,255,0.1);
#         animation: slideInLeft 1.2s ease-in-out;
#     }

#     .sidebar-instruction {
#         font-size: 1.1rem;
#         color: #AAAAAA;
#         font-style: italic;
#         text-align: center;
#         animation: fadeIn 2.5s ease-in-out;
#     }

#     @keyframes fadeInDown {
#         from { opacity: 0; transform: translateY(-30px); }
#         to { opacity: 1; transform: translateY(0); }
#     }

#     @keyframes fadeInUp {
#         from { opacity: 0; transform: translateY(30px); }
#         to { opacity: 1; transform: translateY(0); }
#     }

#     @keyframes slideInLeft {
#         from { opacity: 0; transform: translateX(-50px); }
#         to { opacity: 1; transform: translateX(0); }
#     }

#     @keyframes fadeIn {
#         from { opacity: 0; }
#         to { opacity: 1; }
#     }

#     footer {
#         visibility: hidden;
#     }
#     </style>
# """, unsafe_allow_html=True)

# # --- Rendered Layout ---
# st.markdown('<div class="main-title">🧠 Data Science Project 2: Stop the Churn</div>', unsafe_allow_html=True)
# st.markdown('<div class="subtext">A Business-Critical Customer Retention Dashboard</div>', unsafe_allow_html=True)
# st.markdown("---")

# st.markdown("""
# <div class="welcome-box">
#     <h3 style="color:#00FFFF;">Welcome to the Stop the Churn Dashboard 👋</h3>
#     <p style="color:#CCCCCC;">
#         Explore patterns behind customer churn using data science and machine learning.
#         Dive into exploratory analysis, model training, and actionable insights.
#     </p>
# </div>
# """, unsafe_allow_html=True)

# st.markdown('<div class="sidebar-instruction">👉 Use the sidebar to explore each phase of the project.</div>', unsafe_allow_html=True)

################

import streamlit as st

# Page configuration
st.set_page_config(page_title="Stop the Churn", layout="wide", page_icon="🧠")

# 🌸 Falling flower animation
st.markdown("""
    <script>
    const confettiContainer = document.createElement('div');
    confettiContainer.style.position = 'fixed';
    confettiContainer.style.top = '0';
    confettiContainer.style.left = '0';
    confettiContainer.style.width = '100%';
    confettiContainer.style.height = '100%';
    confettiContainer.style.zIndex = '9999';
    confettiContainer.style.pointerEvents = 'none';
    document.body.appendChild(confettiContainer);

    function createPetal() {
        const petal = document.createElement('div');
        petal.innerHTML = '🌸';
        petal.style.position = 'absolute';
        petal.style.fontSize = Math.random() * 20 + 10 + 'px';
        petal.style.top = '-50px';
        petal.style.left = Math.random() * window.innerWidth + 'px';
        petal.style.animation = 'fall ' + (Math.random() * 5 + 5) + 's linear forwards';
        confettiContainer.appendChild(petal);

        setTimeout(() => {
            confettiContainer.removeChild(petal);
        }, 10000);
    }

    setInterval(createPetal, 300);

    const style = document.createElement('style');
    style.innerHTML = `
        @keyframes fall {
            to {
                transform: translateY(100vh) rotate(360deg);
                opacity: 0;
            }
        }
    `;
    document.head.appendChild(style);
    </script>
""", unsafe_allow_html=True)

# 🌑 Dark theme styling + animations
st.markdown("""
    <style>
    body {
        background-color: #000000;
        color: #ffffff;
    }

    .main-title {
        font-size: 3rem;
        font-weight: 900;
        color: #00FFFF;
        text-align: center;
        animation: fadeInDown 1.5s ease-in-out;
    }

    .subtext {
        font-size: 1.25rem;
        color: #CCCCCC;
        text-align: center;
        margin-top: -1rem;
        margin-bottom: 1.5rem;
        animation: fadeInUp 2s ease-in-out;
    }

    .welcome-box {
        background: linear-gradient(135deg, #111111, #222222);
        border-left: 6px solid #00FFFF;
        padding: 1.5rem;
        margin: 1rem auto;
        border-radius: 12px;
        box-shadow: 0 0 20px rgba(0,255,255,0.1);
        animation: slideInLeft 1.2s ease-in-out;
    }

    .sidebar-instruction {
        font-size: 1.1rem;
        color: #AAAAAA;
        font-style: italic;
        text-align: center;
        animation: fadeIn 2.5s ease-in-out;
    }

    @keyframes fadeInDown {
        from { opacity: 0; transform: translateY(-30px); }
        to { opacity: 1; transform: translateY(0); }
    }

    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }

    @keyframes slideInLeft {
        from { opacity: 0; transform: translateX(-50px); }
        to { opacity: 1; transform: translateX(0); }
    }

    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }

    footer {
        visibility: hidden;
    }
    </style>
""", unsafe_allow_html=True)

# 🧠 Main title and intro
st.markdown('<div class="main-title">🧠 Data Science Project 2: Stop the Churn</div>', unsafe_allow_html=True)
st.markdown('<div class="subtext">A Business-Critical Customer Retention Dashboard</div>', unsafe_allow_html=True)
st.markdown("---")

# 📦 Welcome Box
st.markdown("""
<div class="welcome-box">
    <h3 style="color:#00FFFF;">Welcome to the Stop the Churn Dashboard 👋</h3>
    <p style="color:#CCCCCC;">
        Explore customer churn patterns using data science and machine learning.
        Dive into trends, model performance, and actionable business insights to reduce churn and retain valuable customers.
    </p>
</div>
""", unsafe_allow_html=True)

# 📚 Navigation Instruction
st.markdown('<div class="sidebar-instruction">👉 Use the sidebar to navigate through each phase of the project.</div>', unsafe_allow_html=True)
