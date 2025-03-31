import google.generativeai as genai
import os
import sys
from dotenv import load_dotenv
from flask import Flask, render_template, jsonify, request

# --- Flask 应用初始化 ---
app = Flask(__name__)

# --- 加载环境变量 ---
print("尝试加载 .env 文件...")
load_dotenv()

# --- 全局变量存储 Gemini Chat Session ---
chat_session = None

# --- Gemini 配置 ---
def configure_gemini():
    """
    配置 Gemini 客户端并启动一个 Chat Session。
    如果成功，将 chat_session 设置为 ChatSession 对象。
    """
    global chat_session
    if chat_session:
        print("Chat session 已存在。")
        return True # 已配置

    # 检查代理环境变量
    http_proxy = os.getenv("HTTP_PROXY")
    https_proxy = os.getenv("HTTPS_PROXY")
    if http_proxy: print(f"检测到 HTTP_PROXY: {http_proxy}")
    if https_proxy: print(f"检测到 HTTPS_PROXY: {https_proxy}")

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("错误：在环境变量中找不到 GEMINI_API_KEY 或 OPENAI_API_KEY。", file=sys.stderr)
        return False

    try:
        print("正在配置 Gemini 客户端...")
        genai.configure(api_key=api_key)
        model_name = 'gemini-1.5-flash' # 或者 'gemini-pro', 'gemini-1.5-pro'
        model = genai.GenerativeModel(model_name)

        # 启动聊天会话，可以传入历史记录或让它从空开始
        # 也可以在这里添加 system_instruction (如果模型支持)
        chat_session = model.start_chat(history=[])
        # 示例 system instruction:
        # chat_session = model.start_chat(
        #     enable_automatic_function_calling=True, # 如果需要函数调用
        #     history=[],
        #     # system_instruction="你是一个乐于助人的 AI 助手。" # 注意：不是所有模型都很好地支持 system_instruction
        # )

        print(f"Gemini 配置成功，聊天会话已启动 (使用模型: {model_name})")
        return True
    except Exception as e:
        print(f"配置 Gemini 或启动聊天时出错: {e}", file=sys.stderr)
        chat_session = None
        return False

# --- Flask 路由定义 ---

@app.route('/')
def home():
    """渲染聊天页面 HTML"""
    return render_template('chat.html')

@app.route('/chat', methods=['POST']) # 只接受 POST 请求
def handle_chat():
    """处理来自前端的聊天消息"""
    global chat_session
    # 确保 Gemini 已配置
    if not chat_session and not configure_gemini():
         return jsonify({'error': 'Gemini 服务配置失败，无法处理请求。'}), 500

    # 从 POST 请求的 JSON body 中获取用户消息
    user_message = request.json.get('message')
    if not user_message:
        return jsonify({'error': '未收到消息内容。'}), 400

    print(f"收到用户消息: {user_message}")

    try:
        # 使用 chat session 发送消息，它会自动管理历史记录
        print("正在向 Gemini 发送消息...")
        # 使用 stream=False 获取完整回复 (简单)
        # response = chat_session.send_message(user_message, stream=False)

        # --- 或者，使用 stream=True 实现打字机效果 (更高级) ---
        # 注意：如果使用 stream=True，前端也需要相应修改来处理流式数据
        # 这里暂时保持 stream=False 以简化
        response = chat_session.send_message(
            user_message,
            stream=False,
            safety_settings={'HARASSMENT': 'BLOCK_NONE'}, # 示例安全设置
            request_options={'timeout': 120} # 聊天可能需要更长超时时间
        )
        print("收到 Gemini 响应。")

        # 直接从响应中获取文本
        model_reply = response.text
        print(f"Gemini 回复: {model_reply[:100]}...") # 打印部分回复日志

        # 将回复发送回前端
        return jsonify({'reply': model_reply})

    except Exception as e:
        print(f"与 Gemini 交互时出错: {e}", file=sys.stderr)
        # 在历史记录中可能包含错误信息，检查最后一条记录
        error_details = "未知错误"
        if chat_session and chat_session.history:
             last_part = chat_session.history[-1].parts[0]
             # 检查是否有错误相关的属性或文本
             # 这个检查依赖于库的具体错误处理方式，可能需要调整
             if hasattr(last_part, 'error_code') or "error" in str(last_part).lower():
                  error_details = str(last_part)

        # 尝试从异常本身获取信息
        error_type = type(e).__name__
        error_msg = str(e)

        print(f"详细错误信息: {error_details}, 异常类型: {error_type}, 异常消息: {error_msg}")

        # 返回错误信息给前端
        return jsonify({'error': f'与 AI 交互时出错 ({error_type})。请稍后再试或检查服务器日志。'}), 500


# --- 主程序入口 ---
if __name__ == "__main__":
    print("启动 Flask Web 服务器...")
    # 尝试在启动时配置 Gemini，虽然在第一个请求时也会检查
    configure_gemini()
    app.run(debug=True, host='0.0.0.0', port=5000) # 使用 debug=True 进行开发
