import streamlit as st

st.set_page_config(layout="wide",page_title="Lithofacies Digitalization",page_icon="📈",)

col1, col2 = st.columns((1, 8))
col1.image('http://thepetro.cloud/wp-content/uploads/2023/05/Petrocloud.png',width=80)
col2.markdown("<h1 style='text-align: left; margin-bottom: 0;'>Lithofacies Digitizer</h1>", unsafe_allow_html=True)

st.markdown("<hr style='border-top: 2px solid ; margin-top: 0;'/>", unsafe_allow_html=True)

css = """
        <style>
        input[type="number"] {
            text-align: center;
        }
        </style>
        """

st.markdown(css, unsafe_allow_html=True)

st.markdown("<h2 style='text-align: left;'>Functionality Description</h2>", unsafe_allow_html=True)
st.markdown("""<p style='text-align: justify;'>The algorithm utilize DeepLabV3+ image segmentation architecture to digtizile lithofacies data in mudlog documents to digital format.</p>""", unsafe_allow_html=True)
st.markdown("""
        <ul>
        <li style="font-size: 20px;"><strong>Input<strong>: Images, Axes-Range</li>
        </ul>
        """, unsafe_allow_html=True)
