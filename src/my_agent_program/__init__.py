def main() -> None:
    """控制台入口：启动 Web 应用。"""
    import uvicorn

    uvicorn.run("web_app:app", host="127.0.0.1", port=8000, reload=False)
