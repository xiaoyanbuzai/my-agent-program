import os
from dotenv import load_dotenv
from langchain.agents import create_agent

load_dotenv()

def get_weather(city: str) -> str:
    """Do something."""
    weather_data = {
        "北京" : "晴,25℃",
        "上海" : "多云,22℃",
        "广州" : "多云,28℃",
        "深圳" : "小雨,26℃",
        "杭州" : "雷阵雨,24℃",
    }
    if city not in weather_data:
        return f"错误：没有 {city} 的天气数据，请换一个城市。"
    return f"{city}:{weather_data[city]}"
def get_weather_forecast(city: str) -> str:
    """do something."""
    forecast_data = {
        "北京" : "晴,26℃",
        "上海" : "多云,23℃",
        "广州" : "小雨,27℃",
        "深圳" : "雷阵雨,25℃",
        "杭州" : "阴,24℃",
    }
    if city not in forecast_data:
        return f"错误：没有 {city} 的天气预报数据，请换一个城市。"
    return f"{city}明天的天气预报:{forecast_data[city]}"

def calculate(expression: str) -> str:
    """Calculate a mathematical expression."""
    try:
        result = eval(expression)
        return f"The result of {expression} is {result}."
    except Exception as e:
        return f"Error calculating expression: {e}"

agent = create_agent(
    model="openai:deepseek-flash", # 用 DeepSeek 就改成 "deepseek-chat" 并配好 base_url
    tools=[get_weather,calculate,get_weather_forecast],
    system_prompt="You are a helpful assistant",
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "广州明天的天气怎么样？"}]}
)

for msg in result['messages']:
    print('--')
    print(type(msg).__name__,":",msg.content)
    if hasattr(msg, "tool_calls") and msg.tool_calls:
        print("tool_calls:", msg.tool_calls)

# print(result["messages"][-1].content)