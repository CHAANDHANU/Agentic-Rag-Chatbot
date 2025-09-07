import streamlit as st
import requests

if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("📂 AI Chatbot with Memory + File Upload")

# Sidebar for file upload
uploaded_file = st.file_uploader("Upload a file", type=["txt", "pdf", "docx"])

if uploaded_file:
    files = {"file": uploaded_file.getvalue()}
    res = requests.post("http://127.0.0.1:8000/upload_file/", files={"file": (uploaded_file.name, uploaded_file.getvalue())})
    if res.status_code == 200:
        st.success(f"✅ File {uploaded_file.name} uploaded successfully!")

# Display chat history
for role, msg in st.session_state.messages:
    with st.chat_message(role):
        st.markdown(msg)

# Chat input (can accept text + files together in chat)
if prompt := st.chat_input("Type a message or ask about uploaded files..."):
    # Save user message
    st.session_state.messages.append(("user", prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    # Send to backend
    res = requests.post("http://127.0.0.1:8000/chat/", json={"message": prompt})
    if res.status_code == 200:
        answer = res.json()["response"]
        st.session_state.messages.append(("assistant", answer))
        with st.chat_message("assistant"):
            st.markdown(answer)
    else:
        st.error("❌ Backend error")
