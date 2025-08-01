from uuid import uuid4

import requests
import streamlit as st


# --- AgentService: Handles backend API calls ---
class AgentClient:
    def __init__(self, base_url="http://localhost:8000/api"):
        self.upload_url = f"{base_url}/upload"
        self.chat_url = f"{base_url}/chat"

    def set_session_id(self, session_id):
        self.session_id = session_id

    def upload_file(self, file, password):
        files = {"file": (file.name, file, file.type)}
        data = {"password": password}
        return requests.post(
            self.upload_url,
            files=files,
            data=data,
            headers={"session_id": self.session_id},
        )

    def send_message(self, message):
        response = requests.post(
            self.chat_url,
            json={"message": message},
            headers={"session_id": self.session_id},
        )
        return response.json().get("reply", "No response.")


# --- ChatBot: Manages UI and state ---
class ChatBot:
    def __init__(self, agent: AgentClient):
        self.agent = agent
        st.set_page_config(page_title="CAS Analyzer", layout="centered")
        # st.title("🤖 CAS Analyzer")
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        if "file_uploaded" not in st.session_state:
            st.session_state.file_uploaded = False

    def handle_upload(self):
        # with st.container():
        # st.subheader("📄 Upload Encrypted File")
        with st.expander("🔐 Upload your CAS here before proceeding.", expanded=True):
            uploaded_file = st.file_uploader(
                "Choose a file",
                type=["txt", "pdf", "csv", "zip"],
                label_visibility="visible",
            )
            password = st.text_input("Enter password (if encrypted):", type="password")
            upload_clicked = st.button("Upload File")

            if upload_clicked:
                if not uploaded_file:
                    st.warning("Please select a file before uploading.")
                    return

                response = self.agent.upload_file(uploaded_file, password)
                if response.status_code == 200:
                    st.success("✅ File uploaded and decrypted successfully.")
                    st.session_state.file_uploaded = True
                    st.session_state.chat_history.append(
                        ("system", f"You uploaded {uploaded_file.name}.")
                    )
                    st.session_state.chat_history.append(
                        ("assistant", response.json().get("reply", ""))
                    )

                else:
                    st.error("❌ File upload or decryption failed.")

    def handle_chat(self):
        if not st.session_state.file_uploaded:
            st.info("Please upload and decrypt a file before starting the chat.")
            return  # ⛔ Prevent chat input from showing

        user_input = st.chat_input("Ask something...")

        if user_input:
            st.session_state.chat_history.append(("user", user_input))
            reply = self.agent.send_message(user_input)
            st.session_state.chat_history.append(("assistant", reply))

    def _handle_chat(self):
        # st.subheader("💬 Chat")
        user_input = st.chat_input("Ask something...")

        if user_input:
            st.session_state.chat_history.append(("user", user_input))
            reply = self.agent.send_message(user_input)
            st.session_state.chat_history.append(("assistant", reply))

    def render_history(self):
        for role, message in st.session_state.chat_history:
            with st.chat_message(role):
                st.markdown(message)


def main():

    agent = AgentClient()
    session_id = st.session_state.get("session_id") or str(uuid4())
    st.session_state["session_id"] = session_id
    agent.set_session_id(session_id)

    chatbot = ChatBot(agent)

    chatbot.handle_upload()
    chatbot.handle_chat()
    chatbot.render_history()


if __name__ == "__main__":
    main()
