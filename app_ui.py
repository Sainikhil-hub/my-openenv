import gradio as gr
import requests

API_URL = "http://127.0.0.1:7860"  # Same process: FastAPI + Gradio both on port 7860

def parse_obs(obs):
    types = {0.0: "Routine", 0.5: "Technical", 1.0: "Billing"}
    return {
        "Issue Type": types.get(obs[0], "Unknown"),
        "Complexity": round(obs[1] * 100, 1),
        "Frustration": round(obs[2] * 100, 1),
        "Turns Taken": int(obs[3] * 10)
    }

def generate_chat(action, reward, terminated, info):
    chat_log = []
    
    if action == 1:
        chat_log.append({"role": "assistant", "content": "Could you provide more specific details regarding your issue?"})
        chat_log.append({"role": "user", "content": "Sure, here is some more information... (Complexity dropped, but Frustration increased slightly)"})
    elif action == 2:
        chat_log.append({"role": "assistant", "content": "I'm escalating your ticket to a senior human specialist."})
        chat_log.append({"role": "user", "content": "Thank you, I will wait. (Issue Escalated. Large Cost Penalty)"})
    elif action == 0:
        if terminated and info.get("resolved"):
            chat_log.append({"role": "assistant", "content": "Here is the exact solution to your problem. Have a great day!"})
            chat_log.append({"role": "user", "content": "Wow, that completely fixed it! Thank you! (Issue Resolved. Huge Reward!)"})
        else:
            chat_log.append({"role": "assistant", "content": "Have you tried turning it off and on again?"})
            chat_log.append({"role": "user", "content": "That didn't help at all! This is ridiculous! (Failed attempt. Frustration spiked)"})

    return chat_log

def init_env():
    try:
        req = requests.post(f"{API_URL}/reset")
        req.raise_for_status()
        data = req.json()
        obs = data["observation"]
        parsed = parse_obs(obs)
        
        chat = [{"role": "assistant", "content": f"New Customer Connection Established.\nIssue Category: {parsed['Issue Type']}"}]
        
        return (
            parsed["Complexity"],
            parsed["Frustration"],
            "Agent Dashboard Active. Awaiting action.",
            chat,
            0.0, # Reset run score
            data
        )
    except Exception as e:
        return 0, 0, f"API Error: {e}", [], 0.0, None

def take_step(state, chat_history, current_score, action_idx):
    if not state:
        return init_env()[0:4] + (current_score, state,)
        
    try:
        req = requests.post(f"{API_URL}/step", json={"action": action_idx})
        req.raise_for_status()
        data = req.json()
        
        obs = data["observation"]
        parsed = parse_obs(obs)
        reward = float(data["reward"])
        terminated = data["terminated"]
        truncated = data["truncated"]
        
        new_score = current_score + reward
        
        new_chats = generate_chat(action_idx, reward, terminated, data["info"])
        for c in new_chats:
            chat_history.append(c)
            
        status_msg = f"Last Reward: {reward:.2f} | Total Session Score: {new_score:.2f}"
        
        if terminated:
            if data["info"].get("resolved"):
                status_msg = f"✅ SUCCESS: Issue Resolved! Final Score: {new_score:.2f}"
                chat_history.append({"role": "assistant", "content": "Session Ended (Resolved). Please Reset."})
            elif data["info"].get("escalated"):
                status_msg = f"⚠️ ESCALATED: Sent to Human. Final Score: {new_score:.2f}"
                chat_history.append({"role": "assistant", "content": "Session Ended (Escalated). Please Reset."})
        elif truncated:
            status_msg = f"❌ FAILED: Customer Left Angry! Final Score: {new_score:.2f}"
            chat_history.append({"role": "assistant", "content": "Session Ended (Timeout/Angry). Please Reset."})
            
        return (
            parsed["Complexity"],
            parsed["Frustration"],
            status_msg,
            chat_history,
            new_score,
            data if not (terminated or truncated) else None
        )
        
    except Exception as e:
        return 0, 0, str(e), chat_history, current_score, state


custom_css = """
.score-box { font-size: 24px; font-weight: bold; color: #4CAF50; text-align: center; padding: 10px; border: 2px solid #4CAF50; border-radius: 10px; margin-bottom: 20px;}
.header-text { text-align: center; margin-bottom: 5px; }
"""

with gr.Blocks(title="Smart Customer Support Dashboard") as demo:
    gr.Markdown("# 🎧 AI Customer Support Terminal", elem_classes=["header-text"])
    gr.Markdown("*Your goal is to maximize your total score. If an issue is too complex, asking questions lowers complexity but slowly raises frustration.*", elem_classes=["header-text"])
    
    state_store = gr.State(None)
    score_store = gr.State(0.0)
    
    with gr.Row():
        # LEFT COLUMN: THE CHAT
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="Live Customer Conversation", height=400)
            
            with gr.Row():
                btn_ask = gr.Button("❓ Ask for Info (-1 pt)", variant="secondary")
                btn_respond = gr.Button("🤖 Auto-Respond (Risk/Reward)", variant="primary")
            
            btn_escalate = gr.Button("👩‍💼 Escalate to Human (-5 pts)", variant="stop")
            
        # RIGHT COLUMN: ANALYTICS
        with gr.Column(scale=1):
            gr.Markdown("### 📊 Live Interaction Metrics")
            
            status_html = gr.Markdown("<div class='score-box'>Not Started</div>")
            
            complexity_bar = gr.Slider(0, 100, label="🧩 Issue Complexity (How hard is the fix?)", interactive=False)
            frustration_bar = gr.Slider(0, 100, label="😡 Customer Frustration (Will they leave?)", interactive=False)
            
            gr.Markdown("---")
            btn_reset = gr.Button("🔄 Connect New Customer")

    # Wire up buttons
    btn_reset.click(
        init_env, 
        outputs=[complexity_bar, frustration_bar, status_html, chatbot, score_store, state_store]
    )
    
    # helper for wrapper
    def wrapper(state, chat, score, action):
        return take_step(state, chat, score, action)

    btn_respond.click(
        lambda s, c, score: take_step(s, c, score, 0), 
        inputs=[state_store, chatbot, score_store], 
        outputs=[complexity_bar, frustration_bar, status_html, chatbot, score_store, state_store]
    )
    btn_ask.click(
        lambda s, c, score: take_step(s, c, score, 1), 
        inputs=[state_store, chatbot, score_store], 
        outputs=[complexity_bar, frustration_bar, status_html, chatbot, score_store, state_store]
    )
    btn_escalate.click(
        lambda s, c, score: take_step(s, c, score, 2), 
        inputs=[state_store, chatbot, score_store], 
        outputs=[complexity_bar, frustration_bar, status_html, chatbot, score_store, state_store]
    )

    demo.load(init_env, outputs=[complexity_bar, frustration_bar, status_html, chatbot, score_store, state_store])

# Gradio is mounted inside FastAPI via gr.mount_gradio_app() in app.py.
# Do NOT call demo.launch() here — it is served by uvicorn on port 7860.
