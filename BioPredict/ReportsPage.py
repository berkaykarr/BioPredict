import streamlit as st
import base64

st.markdown("""
    <style>
    .report-box {
        background-color: #f9f9ff;
        padding: 20px 30px;
        border-radius: 12px;
        margin-bottom: 25px;
        border-left: 5px solid #fbc02d;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.05);
    }
    .report-header {
        font-size: 20px;
        font-weight: bold;
        color: #222;
        margin-bottom: 8px;
    }
    .download-link {
        margin-top: 12px;
        display: inline-block;
        padding: 8px 14px;
        background-color: #fbc02d;
        color: white;
        border-radius: 6px;
        text-decoration: none;
        font-weight: 500;
        transition: background-color 0.3s;
    }
    .download-link:hover {
        background-color: #fef0ad;
    }
    </style>
""", unsafe_allow_html=True)
st.markdown("""
<h1 style="
    font-size: 42px;
    font-weight: 900;
    color: #dba608;
    text-align: center;
    margin-bottom: 0.1px;
    font-family: 'Arial Black', sans-serif;
">
📁 All Saved Medical Reports
</h1>
""", unsafe_allow_html=True)




if "all_reports" in st.session_state and st.session_state["all_reports"]:
    report_types = list(st.session_state["all_reports"].keys())
    selected_type = st.selectbox("🔍 Choose Report Type", report_types)

    selected_reports = st.session_state["all_reports"][selected_type]

    for i, report in enumerate(selected_reports, 1):
        st.markdown(f"""
        <div class="report-box">
            <div class="report-header">{i}. {report['user_name']} &nbsp; <span style='font-size:15px;color:#666;'>({report['timestamp']})</span></div>
        """, unsafe_allow_html=True)

        with st.expander("📄 View Report", expanded=False):
            st.components.v1.html(report["html"], height=600, scrolling=True)

        report_bytes = report["html"].encode("utf-8")
        b64 = base64.b64encode(report_bytes).decode()
        filename = f"{report['user_name']}_{selected_type}_report.html"
        href = f'<a class="download-link" href="data:text/html;base64,{b64}" download="{filename}">⬇️ Download as HTML</a>'
        st.markdown(href, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

else:
    st.info("No reports have been saved yet. Please generate a prediction to save a report.")
