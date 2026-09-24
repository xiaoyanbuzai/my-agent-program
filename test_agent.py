import os
import json
from datetime import datetime, timedelta
from urllib.parse import urlencode
from urllib.request import urlopen
from zoneinfo import ZoneInfo

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


def get_yesterday_weather(city: str) -> str:
    """查询指定城市昨天的实际天气，返回最高/最低温度和降水情况。"""
    try:
        location_query = urlencode({"name": city, "count": 1, "language": "zh", "format": "json"})
        with urlopen(
            f"https://geocoding-api.open-meteo.com/v1/search?{location_query}",
            timeout=10,
        ) as response:
            locations = json.load(response).get("results", [])

        if not locations:
            return f"错误：找不到城市“{city}”，请提供更具体的城市名称。"

        location = locations[0]
        timezone_name = location.get("timezone", "UTC")
        local_today = datetime.now(ZoneInfo(timezone_name)).date()
        yesterday = local_today - timedelta(days=1)
        weather_query = urlencode(
            {
                "latitude": location["latitude"],
                "longitude": location["longitude"],
                "start_date": yesterday.isoformat(),
                "end_date": yesterday.isoformat(),
                "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_sum",
                "temperature_unit": "celsius",
                "precipitation_unit": "mm",
                "timezone": timezone_name,
            }
        )
        with urlopen(
            f"https://api.open-meteo.com/v1/forecast?{weather_query}",
            timeout=10,
        ) as response:
            weather = json.load(response)

        daily = weather["daily"]
        weather_code = daily["weather_code"][0]
        description = _weather_code_description(weather_code)
        maximum = daily["temperature_2m_max"][0]
        minimum = daily["temperature_2m_min"][0]
        precipitation = daily["precipitation_sum"][0]
        return (
            f"{location['name']}昨天（{yesterday}）：{description}，"
            f"最高 {maximum}°C，最低 {minimum}°C，降水量 {precipitation} mm。"
        )
    except (KeyError, IndexError, ValueError, TimeoutError, OSError, json.JSONDecodeError) as error:
        return f"查询“{city}”昨天的天气失败：{error}"


def _weather_code_description(code: int) -> str:
    """将 WMO 天气代码转换为用户可读的中文描述。"""
    descriptions = {
        0: "晴",
        1: "大部晴朗",
        2: "局部多云",
        3: "阴",
        45: "雾",
        48: "雾凇",
        51: "小毛毛雨",
        53: "毛毛雨",
        55: "大毛毛雨",
        61: "小雨",
        63: "中雨",
        65: "大雨",
        71: "小雪",
        73: "中雪",
        75: "大雪",
        80: "小阵雨",
        81: "中阵雨",
        82: "强阵雨",
        95: "雷雨",
        96: "雷雨伴冰雹",
        99: "强雷雨伴冰雹",
    }
    return descriptions.get(code, f"天气代码 {code}")

def calculate(expression: str) -> str:
    """Calculate a mathematical expression."""
    try:
        result = eval(expression)
        return f"The result of {expression} is {result}."
    except Exception as e:
        return f"Error calculating expression: {e}"

agent = create_agent(
    model="openai:deepseek-flash", # 用 DeepSeek 就改成 "deepseek-chat" 并配好 base_url
    tools=[get_weather, calculate, get_weather_forecast, get_yesterday_weather,_weather_code_description],
    system_prompt="You are a helpful assistant",
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "广州的天气，昨天的天气，明天的天气怎么样？"}]}
)

for msg in result['messages']:
    print('--')
    print(type(msg).__name__,":",msg.content)
    if hasattr(msg, "tool_calls") and msg.tool_calls:
        print("tool_calls:", msg.tool_calls)

# print(result["messages"][-1].content)