import os
import sys
import httpx
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
    """搜索互联网获取最新信息"""
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
    """计算数学表达式，例如 '2 + 2' 或 '3 * 4'"""
    return str(eval(expression))


tools = [web_search, calculator]

model = ChatOpenAI(
    model="deepseek-chat",
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
    temperature=0.7,
)

agent_executor = create_react_agent(
    model=model,
    tools=tools,
)


def main():
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = input("请输入你的问题: ")

    try:
        result = agent_executor.invoke({"messages": [("human", query)]})
        print(f"\n=== 答案 ===\n{result['messages'][-1].content}")
    except Exception as e:
        print(f"错误: {e}")


if __name__ == "__main__":
    main()
