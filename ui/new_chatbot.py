from datetime import datetime
from uuid import uuid4

import requests
import streamlit as st


# --- AgentService: Handles backend API calls ---
class AgentClient:
    def __init__(self, base_url="http://localhost:8000/api"):
        self.chat_url = f"{base_url}/chat"

    def set_session_id(self, session_id):
        self.session_id = session_id

    def send_message(self, message):
        response = requests.post(
            self.chat_url,
            json={"message": message},
            headers={"session_id": self.session_id},
        )
        return response.json().get("reply", "No response.")


# --- Q&A ChatBot: Manages UI and state for Q&A interface ---
class QAChatBot:
    def __init__(self, agent: AgentClient):
        self.agent = agent
        st.set_page_config(
            page_title="Q&A Assistant", layout="wide", initial_sidebar_state="expanded"
        )

        # Initialize session state
        if "qa_history" not in st.session_state:
            st.session_state.qa_history = []
        if "current_question" not in st.session_state:
            st.session_state.current_question = ""
        if "current_answer" not in st.session_state:
            st.session_state.current_answer = ""

    def add_to_history(self, question, answer):
        """Add Q&A pair to history with timestamp"""
        timestamp = datetime.now().strftime("%H:%M")
        qa_pair = {"timestamp": timestamp, "question": question, "answer": answer}
        st.session_state.qa_history.append(qa_pair)

    def handle_qa_interface(self):
        st.header("❓ Ask a Question")
        st.markdown(
            "**This is a Q&A interface - each question is independent and doesn't build on previous conversations.**"
        )

        # Use form for proper input handling
        with st.form("question_form", clear_on_submit=True):
            col1, col2 = st.columns([5, 1])
            with col1:
                question = st.text_input(
                    "",
                    placeholder="e.g. What did I order from Amazon last month? or List emails from my daughter's school",
                    key="question_input",
                )
            with col2:
                st.write("")  # Add spacing to align with input
                st.write("")  # Add more spacing
                submit_button = st.form_submit_button("➤", help="Send question")

        # Handle form submission
        if submit_button and question.strip():
            st.session_state.current_question = question
            with st.spinner("Getting answer..."):
                answer = self.agent.send_message(question)
                st.session_state.current_answer = answer
                self.add_to_history(question, answer)
            st.rerun()

    def render_current_qa(self):
        """Render the current question and answer prominently"""
        if st.session_state.current_question and st.session_state.current_answer:
            st.markdown("---")
            st.subheader("💬 Current Question & Answer")

            # Question
            with st.container():
                st.markdown("**Question:**")
                st.info(st.session_state.current_question)

            # Answer
            with st.container():
                st.markdown("**Answer:**")
                st.success(st.session_state.current_answer)

    def render_sidebar_history(self):
        """Render Q&A history in sidebar"""
        with st.sidebar:
            st.header("📚 Question History")

            if not st.session_state.qa_history:
                st.info("No questions asked yet.")
                return

            # Show history in reverse chronological order
            for i, qa in enumerate(reversed(st.session_state.qa_history)):
                with st.expander(
                    f"Q: {qa['question'][:50]}... ({qa['timestamp']})", expanded=False
                ):
                    st.markdown("**Question:**")
                    st.text(qa["question"])
                    st.markdown("**Answer:**")
                    st.text(qa["answer"])

                    # Add button to load this Q&A as current
                    if st.button(
                        f"Load Q&A #{len(st.session_state.qa_history) - i}",
                        key=f"load_{i}",
                    ):
                        st.session_state.current_question = qa["question"]
                        st.session_state.current_answer = qa["answer"]
                        st.rerun()

            # Clear history button
            if st.button("Clear All History", type="secondary"):
                st.session_state.qa_history = []
                st.session_state.current_question = ""
                st.session_state.current_answer = ""
                st.rerun()


def main():
    agent = AgentClient()
    session_id = st.session_state.get("session_id") or str(uuid4())
    st.session_state["session_id"] = session_id
    agent.set_session_id(session_id)

    qa_bot = QAChatBot(agent)

    # Main layout
    qa_bot.handle_qa_interface()
    qa_bot.render_current_qa()

    # Sidebar
    qa_bot.render_sidebar_history()


if __name__ == "__main__":
    main()
