import os
import sys
import httpx
import subprocess
from bs4 import BeautifulSoup
from langchain.agents import create_agent as create_react_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

if not DEEPSEEK_API_KEY:
    print("错误: 请设置环境变量 DEEPSEEK_API_KEY")
    print("例如: set DEEPSEEK_API_KEY=your-key-here")
    sys.exit(1)


@tool
def web_search(query: str) -> str:
    """搜索互联网获取最新知识点、题目解析或参考资料"""
    try:
        url = f"https://cn.bing.com/search?q={query}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        resp = httpx.get(url, headers=headers, timeout=15)
        soup = BeautifulSoup(resp.text, "html.parser")
        results = []
        for item in soup.select(".b_algo")[:5]:
            title = item.select_one("h2")
            snippet = item.select_one(".b_caption p")
            if title:
                results.append(title.get_text(strip=True))
            if snippet:
                results.append(snippet.get_text(strip=True))
        return "\n".join(results) if results else "未找到相关结果"
    except Exception as e:
        return f"搜索出错: {e}"


@tool
def calculator(expression: str) -> str:
    """计算数学表达式，例如 '2 + 2' 或 '3 * 4' 或 'sqrt(16)'"""
    import math
    allowed = {"sqrt": math.sqrt, "sin": math.sin, "cos": math.cos,
               "tan": math.tan, "log": math.log, "log10": math.log10,
               "pi": math.pi, "e": math.e, "floor": math.floor, "ceil": math.ceil}
    return str(eval(expression, {"__builtins__": {}}, allowed))


@tool
def run_python(code: str) -> str:
    """运行 Python 代码（用于验证算法、计算结果或演示编程示例）"""
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            return result.stdout.strip() or "执行成功（无输出）"
        return f"错误: {result.stderr.strip()}"
    except subprocess.TimeoutExpired:
        return "执行超时"
    except Exception as e:
        return f"执行出错: {e}"


tools = [web_search, calculator, run_python]

model = ChatOpenAI(
    model="deepseek-chat",
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
    temperature=0.5,
)

SYSTEM_PROMPT = """你是 AI 教育辅导助手，擅长以下学科：
- 数学（代数、几何、微积分、概率统计）
- 物理（力学、电磁学、热学）
- 化学（无机、有机、化学计算）
- 编程（Python、算法、数据结构）
- 语文、英语等文科

教学原则：
1. 分步讲解，由浅入深
2. 多问"为什么"，引导学生思考
3. 对复杂概念用比喻和实例说明
4. 必要时检查学生理解程度
5. 鼓励而非批评
6. 遇到不确定的知识主动搜索验证"""

agent_executor = create_react_agent(
    model=model,
    tools=tools,
    system_prompt=SYSTEM_PROMPT,
)


def main():
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = input("请输入你的学习问题: ")

    try:
        result = agent_executor.invoke({"messages": [("human", query)]})
        print(f"\n{'='*40}\n{result['messages'][-1].content}")
    except Exception as e:
        print(f"错误: {e}")


if __name__ == "__main__":
    main()
